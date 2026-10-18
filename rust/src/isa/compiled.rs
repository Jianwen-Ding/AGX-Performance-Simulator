//! Pre-resolved ("compiled") form of `exec::exec`: every field value, operand location, bit size and
//! flag is looked up once per static instruction instead of once per lane. Semantics, including the
//! lane order of reads, writes and errors, are those of the reference interpreter in `exec.rs`;
//! `cost::trace_reference` runs the reference and the tests require identical traces.

use super::desc::{fget, sign_extend, Fields, OpKind};
use super::exec::{CoreState, SIMD_WIDTH};
use super::fma;
use super::operand::*;
use super::table::{Desc, Ex};

/// Errors only mark the instruction as skipped, so a static message is enough (and keeps `Result` small).
type R<T> = Result<T, &'static str>;

#[derive(Clone, Debug)]
enum Loc {
    R16(usize),
    R32(usize),
    R64(usize),
    U16(usize),
    U32(usize),
    U64(usize),
    Imm(i128),
    FImm(f64),
    Bad(&'static str),
}

#[derive(Clone, Debug)]
struct Src {
    loc: Loc,
    /// `get_bit_size()`; None where applegpu raises
    size: Option<u32>,
    /// MulSrcDesc / AddSrcDesc with a `.sx` register
    sx: bool,
    abs: bool,
    neg: bool,
}

impl Src {
    fn new(kind: &OpKind, o: &Operand, nh: usize) -> Src {
        let ok = |last: usize| last < nh;
        let loc = match o {
            Operand::Reg16(n, _) if ok(*n as usize) => Loc::R16(*n as usize),
            Operand::Reg32(n, _) if ok(2 * *n as usize + 1) => Loc::R32(*n as usize),
            Operand::Reg64(n, _) if ok(2 * *n as usize + 3) => Loc::R64(*n as usize),
            Operand::Reg16(..) | Operand::Reg32(..) | Operand::Reg64(..) => Loc::Bad("IndexError: register"),
            Operand::UReg16(n, _) => Loc::U16(*n as usize),
            Operand::UReg32(n, _) => Loc::U32(*n as usize),
            Operand::UReg64(n, _) => Loc::U64(*n as usize),
            Operand::Imm(ImmVal::Int(v), _) => Loc::Imm(*v as i128),
            Operand::Imm(ImmVal::Float(x), _) => Loc::FImm(*x),
            Operand::Tuple(..) => Loc::Bad("NotImplementedError: get_thread"),
            _ => Loc::Bad("AttributeError: get_thread"),
        };
        let sx = matches!(kind, OpKind::AluSrc { sx: true, .. }) && o.flags() & F_SX != 0;
        Src { loc, size: o.bit_size().ok(), sx, abs: o.flags() & F_ABS != 0, neg: o.flags() & F_NEG != 0 }
    }
    fn imm(v: i128) -> Src {
        Src { loc: Loc::Imm(v), size: Some(16), sx: false, abs: false, neg: false }
    }
    #[inline(always)]
    fn raw(&self, st: &CoreState, t: usize) -> R<i128> {
        Ok(match self.loc {
            // indices validated against the register file in `Compiled::new`
            Loc::R16(n) => r16(st, n, t),
            Loc::R32(n) => r16(st, 2 * n, t) | (r16(st, 2 * n + 1, t) << 16),
            Loc::R64(n) => r16(st, 2 * n, t) | (r16(st, 2 * n + 1, t) << 16) | (r16(st, 2 * n + 2, t) << 32) | (r16(st, 2 * n + 3, t) << 48),
            Loc::U16(n) => u16v(st, n)?,
            Loc::U32(n) => u16v(st, 2 * n)? | (u16v(st, 2 * n + 1)? << 16),
            Loc::U64(n) => u16v(st, 2 * n)? | (u16v(st, 2 * n + 1)? << 16) | (u16v(st, 2 * n + 2)? << 32) | (u16v(st, 2 * n + 3)? << 48),
            Loc::Imm(v) => v,
            Loc::FImm(_) => return Err("TypeError: float operand in integer operation".into()),
            Loc::Bad(e) => return Err(e),
        })
    }
    /// `get` for every lane at once, the operand's location decoded once (registers are read for inactive lanes
    /// too, which has no effect). Only a uniform, immediate or invalid operand can fail, and the same way in every
    /// lane, so this fails exactly when `get` at the first active lane would.
    #[inline(always)]
    fn fetch(&self, st: &CoreState, out: &mut [i128; SIMD_WIDTH]) -> R<()> {
        match self.loc {
            Loc::R16(n) => {
                for (t, o) in out.iter_mut().enumerate() {
                    *o = r16(st, n, t);
                }
            }
            Loc::R32(n) => {
                for (t, o) in out.iter_mut().enumerate() {
                    *o = r16(st, 2 * n, t) | (r16(st, 2 * n + 1, t) << 16);
                }
            }
            Loc::R64(_) => {
                for (t, o) in out.iter_mut().enumerate() {
                    *o = self.raw(st, t)?;
                }
            }
            // uniforms, immediates and invalid operands: the same in every lane
            _ => {
                let v = self.raw(st, 0)?;
                out.fill(v);
            }
        }
        if self.sx {
            let s = self.size.ok_or("bit size")?;
            for o in out.iter_mut() {
                *o = sign_extend(*o, s);
            }
        }
        Ok(())
    }
    /// `evaluate_thread` (sign-extending `.sx` registers of MulSrc/AddSrc).
    #[inline(always)]
    fn get(&self, st: &CoreState, t: usize) -> R<i128> {
        let v = self.raw(st, t)?;
        if self.sx {
            return Ok(sign_extend(v, self.size.ok_or("bit size")?));
        }
        Ok(v)
    }
    #[inline(always)]
    fn size(&self) -> R<u32> {
        self.size.ok_or("get_bit_size on non-register operand")
    }
    /// `evaluate_thread_float`
    #[inline(always)]
    fn get_float(&self, st: &CoreState, t: usize) -> R<u64> {
        let mut r = match self.loc {
            Loc::FImm(x) => x.to_bits(),
            Loc::Imm(x) => (x as f64).to_bits(),
            _ => {
                let bits = self.raw(st, t)? as u64;
                match self.size()? {
                    16 => fma::f16_to_f64(bits, false),
                    32 => fma::f32_to_f64(bits, true),
                    _ => return Err("NotImplementedError: 64-bit float source".into()),
                }
            }
        };
        if self.abs {
            r &= !(1u64 << 63);
        }
        if self.neg {
            r ^= 1u64 << 63;
        }
        Ok(r)
    }
}

/// Source lane of A for each lane of a 10-byte fmadd lane form (`None`: A reads 0). Only the measured subset is modeled
/// (playground findings 2026-10-11-fmadd-lane-offset and 2026-10-18-fmadd-followups); everything else is an error so nothing
/// is silently guessed: bit 52 (unknown operation), bit 74, combined grouping/cross bits, k > 16 (fmadd32 reads undefined
/// values), grouping/cross modes with k != 1, bits 64-67/69-71 with k != 1, and an fmadd32 lane whose source falls outside its
/// 2-/4-lane group (undefined value). An active lane whose source lane is masked off reads 0 (measured, all k tested).
/// Timing-dependent behavior is not modeled: the cross-half read (lane 16 reading lane 15) happens on hardware only with bit 68
/// and at least two instructions after the write of A (about 90% of the time), or whenever a `wait` follows the fmadd even
/// without bit 68. The model reads 0 across halves without bit 68 and crosses with it.
fn lane_src(half: bool, l: u64, g: u64, x: u64, exec: &[bool; SIMD_WIDTH]) -> R<[Option<usize>; SIMD_WIDTH]> {
    const B68: u64 = 1 << 4;
    const B72: u64 = 1 << 8;
    const B73: u64 = 1 << 9;
    const NOOP: u64 = 0b1111 | (0b111 << 5); // bits 64-67, 69-71
    if x != 0 {
        return Err("unsupported: fmadd bit 52 operation");
    }
    let extra = g & NOOP;
    if extra != 0 && (l != 1 || extra.count_ones() != 1 || g & !NOOP != 0) {
        return Err("unsupported: fmadd lane bits 64-71 outside the measured k=1 single flips");
    }
    let mode = g & !NOOP;
    if l > 16 || (mode != 0 && l != 1) {
        return Err("unsupported: fmadd lane offset outside the measured set (default mode k <= 16; other modes k = 1)");
    }
    let group = match mode {
        0 | B68 => if mode == B68 { 32 } else { 16 },
        B72 => 2,
        B73 => 4,
        _ => return Err("unsupported: fmadd lane mode bits combination"),
    };
    let k = l as usize;
    let mut out = [None; SIMD_WIDTH];
    for t in 0..SIMD_WIDTH {
        let start = t / group * group;
        if t >= start + k && k < group {
            out[t] = Some(t - k);
        } else if group <= 4 && !half && exec[t] {
            return Err("unsupported: fmadd32 lane source outside its group reads an undefined value");
        }
        // Measured (playground 2026-10-18-fmadd-followups): an active lane whose source lane is masked off reads 0.
        if exec[t] {
            if let Some(s) = out[t] {
                if !exec[s] {
                    out[t] = None;
                }
            }
        }
    }
    Ok(out)
}

#[inline(always)]
fn r16(st: &CoreState, r: usize, t: usize) -> i128 {
    debug_assert!(r < st.reg16s.len() && t < SIMD_WIDTH);
    // SAFETY: r is validated against the register file size when the instruction is compiled, and
    // the register file never changes size; t < SIMD_WIDTH in every lane loop.
    unsafe { *st.reg16s.get_unchecked(r).get_unchecked(t) as i128 }
}

#[inline(always)]
fn w16(st: &mut CoreState, r: usize, t: usize, v: i128) {
    debug_assert!(r < st.reg16s.len() && t < SIMD_WIDTH);
    // SAFETY: as for r16
    unsafe { *st.reg16s.get_unchecked_mut(r).get_unchecked_mut(t) = (v & 0xFFFF) as u16 }
}

#[inline]
fn u16v(st: &CoreState, r: usize) -> R<i128> {
    st.uniforms.get(r).map(|&v| v as i128).ok_or_else(|| "IndexError: uniform".into())
}

#[derive(Clone, Debug)]
enum Dst {
    R16(usize),
    R32(usize),
    R64(usize),
    /// partly or wholly outside the register file: applegpu's checked writes (low half first)
    Checked(Operand),
    Bad(&'static str),
}

impl Dst {
    fn new(o: &Operand, nh: usize) -> Dst {
        match o {
            Operand::Reg16(n, _) if (*n as usize) < nh => Dst::R16(*n as usize),
            Operand::Reg32(n, _) if 2 * (*n as usize) + 1 < nh => Dst::R32(*n as usize),
            Operand::Reg64(n, _) if 2 * (*n as usize) + 3 < nh => Dst::R64(*n as usize),
            Operand::Reg16(..) | Operand::Reg32(..) | Operand::Reg64(..) => Dst::Checked(o.clone()),
            Operand::Tuple(..) => Dst::Bad("NotImplementedError: set_thread"),
            _ => Dst::Bad("AttributeError: set_thread"),
        }
    }
    fn size(o: &Operand) -> Option<u32> {
        o.bit_size().ok()
    }
    #[inline(always)]
    fn set(&self, st: &mut CoreState, t: usize, v: i128) -> R<()> {
        match self {
            &Dst::R16(n) => w16(st, n, t, v),
            &Dst::R32(n) => {
                w16(st, 2 * n, t, v);
                w16(st, 2 * n + 1, t, v >> 16);
            }
            &Dst::R64(n) => {
                w16(st, 2 * n, t, v);
                w16(st, 2 * n + 1, t, v >> 16);
                w16(st, 2 * n + 2, t, v >> 32);
                w16(st, 2 * n + 3, t, v >> 48);
            }
            Dst::Checked(o) => {
                return match o {
                    Operand::Reg16(n, _) => st.set_reg16(*n as usize, t, v),
                    Operand::Reg32(n, _) => st.set_reg32(*n as usize, t, v),
                    Operand::Reg64(n, _) => st.set_reg64(*n as usize, t, v),
                    _ => unreachable!(),
                }
                .map_err(|_| "IndexError: register")
            }
            &Dst::Bad(e) => return Err(e),
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Cmp {
    float: bool,
    a: Src,
    b: Src,
    cc: u64,
    ccn: bool,
}

impl Cmp {
    #[inline(always)]
    fn eval(&self, st: &CoreState, t: usize) -> R<bool> {
        let c = if self.float {
            let a = f64::from_bits(self.a.get_float(st, t)?);
            let b = f64::from_bits(self.b.get_float(st, t)?);
            match self.cc {
                0b000 => a == b,
                0b001 => a < b,
                0b010 => a > b,
                0b101 => a >= b,
                0b110 => a <= b,
                _ => return Err("NotImplementedError: fcompare".into()),
            }
        } else {
            let mut a = self.a.get(st, t)?;
            let mut b = self.b.get(st, t)?;
            let (asz, bsz) = (self.a.size()?, self.b.size()?);
            if self.cc & 0b100 != 0 {
                a = sign_extend(a, asz);
                b = sign_extend(b, bsz);
            }
            match self.cc & 0b11 {
                0 => a == b,
                1 => a < b,
                2 => a > b,
                _ => return Err("NotImplementedError: icompare".into()),
            }
        };
        Ok(c != self.ccn)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bf {
    Bfi,
    Bfeil,
    Extr,
    Shlhi,
    Shrhi,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fl {
    FMAdd32,
    FMAdd16,
    FAdd32,
    FAdd16,
    FMul32,
    FMul16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ctl {
    If,
    While,
    Else,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sh {
    Down,
    Up,
    RotateUp,
    Xor,
}

#[derive(Clone, Debug)]
enum K {
    /// raises regardless of the exec mask
    Raise(&'static str),
    /// raises when some lane is active
    RaiseIfActive(&'static str),
    Nop,
    MovImm { d: Dst, v: i128 },
    GetSr { d: Dst, ok: bool },
    IAdd { a: Src, b: Src, d: Dst, shift: u64, sat: Option<bool>, neg: bool, signed: bool, dsize: Option<u32> },
    IMAdd { a: Src, b: Src, c: Src, d: Dst, shift: u64, sat: Option<bool>, neg: bool, signed: bool, dsize: Option<u32>, ab_ok: bool },
    Bitfield { op: Bf, a: Src, b: Src, c: Src, d: Dst, m: u32 },
    Asr { high: bool, a: Src, b: Src, d: Dst },
    Bitop { a: Src, b: Src, d: Dst, tt: [u64; 4] },
    Bitrev { a: Src, d: Dst },
    Popcount { a: Src, d: Dst },
    Ffs { a: Src, d: Dst },
    Float { op: Fl, a: Src, b: Src, c: Option<Src>, d: Dst, sat: bool, d16: Option<bool> },
    /// 10-byte fmadd with a lane offset on source A (`lane_src` documents the measured subset)
    FloatLane { half: bool, a: Src, b: Src, c: Src, d: Dst, sat: bool, d16: Option<bool>, l: u64, g: u64, x: u64 },
    Cmpsel { cmp: Cmp, x: Src, y: Src, d: Dst },
    PopExec { n: i128 },
    Mask { ctl: Ctl, cmp: Cmp, n: i128 },
    Ballot { cmp: Option<Cmp>, d: Dst },
    SimdShuffle { a: Src, b: Src, d: Dst },
    Shuffle { op: Sh, a: Src, b: Src, d: Dst },
    DeviceLoad { at: bool, a: usize, ot: bool, o: u64, ou: bool, fmt: u64, s: u64, r: usize, rt: u64, mask: u64 },
}

#[derive(Clone, Debug)]
pub struct Compiled {
    k: K,
}

fn f(fields: &Fields, name: &str) -> u64 {
    fget(fields, name).unwrap_or(0)
}

impl Compiled {
    /// `nh`: number of 16-bit registers in the CoreState the instruction will run on.
    pub fn new(d: &Desc, fields: &Fields, ops: &[Operand], nh: usize) -> Compiled {
        let get = |name: &str| -> Option<(&OpKind, &Operand)> {
            d.operands.iter().zip(ops).find(|(od, _)| od.name == name).map(|(od, o)| (&od.kind, o))
        };
        let src = |name: &str| -> Src {
            match get(name) {
                Some((OpKind::Immediate { .. }, _)) => Src::imm(f(fields, name) as i128),
                Some((k, o)) => Src::new(k, o, nh),
                None => Src { loc: Loc::Bad("KeyError: operand"), size: None, sx: false, abs: false, neg: false },
            }
        };
        let dst = |name: &str| match get(name) {
            Some((_, o)) => Dst::new(o, nh),
            None => Dst::Bad("KeyError: operand"),
        };
        let dsize = |name: &str| get(name).and_then(|(_, o)| Dst::size(o));
        let cmp = |float: bool| Cmp { float, a: src("A"), b: src("B"), cc: f(fields, "cc"), ccn: f(fields, "ccn") != 0 };
        let k = match d.ex {
            Ex::None => K::Raise("AttributeError: no exec"),
            Ex::MaskedTodo => K::RaiseIfActive("AssertionError: TODO"),
            Ex::Wait => K::Nop,
            Ex::MovImm(name) => K::MovImm { d: dst("D"), v: f(fields, name) as i128 },
            Ex::GetSr => K::GetSr { d: dst("D"), ok: f(fields, "SR") == 80 },
            Ex::IAdd => {
                let (a, b) = (src("A"), src("B"));
                let shift = f(fields, "s");
                let ds = dsize("D");
                let sat = match (a.size, b.size, ds) {
                    (Some(x), Some(y), Some(z)) => Some(f(fields, "S") != 0 && shift == 0 && x <= 32 && y <= 32 && z <= 32),
                    _ => None,
                };
                let signed = f(fields, "As") != 0 || f(fields, "Bs") != 0;
                K::IAdd { a, b, d: dst("D"), shift, sat, neg: f(fields, "N") != 0, signed, dsize: ds }
            }
            Ex::IMAdd => {
                let (a, b, c) = (src("A"), src("B"), src("C"));
                let shift = f(fields, "s");
                let ds = dsize("D");
                let ab_ok = a.size.is_some() && b.size.is_some();
                let sat = match (c.size, ds) {
                    (Some(x), Some(z)) => Some(f(fields, "S") != 0 && shift == 0 && x <= 32 && z <= 32),
                    _ => None,
                };
                let signed = f(fields, "As") != 0 || f(fields, "Bs") != 0 || f(fields, "Cs") != 0;
                K::IMAdd { a, b, c, d: dst("D"), shift, sat, neg: f(fields, "N") != 0, signed, dsize: ds, ab_ok }
            }
            Ex::Bfi | Ex::Bfeil | Ex::Extr | Ex::Shlhi | Ex::Shrhi => K::Bitfield {
                op: match d.ex {
                    Ex::Bfi => Bf::Bfi,
                    Ex::Bfeil => Bf::Bfeil,
                    Ex::Extr => Bf::Extr,
                    Ex::Shlhi => Bf::Shlhi,
                    _ => Bf::Shrhi,
                },
                a: src("A"),
                b: src("B"),
                c: src("C"),
                d: dst("D"),
                m: f(fields, "m") as u32,
            },
            Ex::Asr | Ex::Asrh => K::Asr { high: d.ex == Ex::Asrh, a: src("A"), b: src("B"), d: dst("D") },
            Ex::Bitop => K::Bitop {
                a: src("A"),
                b: src("B"),
                d: dst("D"),
                tt: [f(fields, "tt0"), f(fields, "tt1"), f(fields, "tt2"), f(fields, "tt3")],
            },
            Ex::Bitrev => K::Bitrev { a: src("A"), d: dst("D") },
            Ex::Popcount => K::Popcount { a: src("A"), d: dst("D") },
            Ex::Ffs => K::Ffs { a: src("A"), d: dst("D") },
            Ex::FMAdd32 | Ex::FMAdd16 | Ex::FAdd32 | Ex::FAdd16 | Ex::FMul32 | Ex::FMul16 => {
                let op = match d.ex {
                    Ex::FMAdd32 => Fl::FMAdd32,
                    Ex::FMAdd16 => Fl::FMAdd16,
                    Ex::FAdd32 => Fl::FAdd32,
                    Ex::FAdd16 => Fl::FAdd16,
                    Ex::FMul32 => Fl::FMul32,
                    _ => Fl::FMul16,
                };
                let c = if matches!(op, Fl::FMAdd32 | Fl::FMAdd16) { Some(src("C")) } else { None };
                K::Float { op, a: src("A"), b: src("B"), c, d: dst("D"), sat: f(fields, "S") != 0, d16: dsize("D").map(|s| s == 16) }
            }
            Ex::FMAddLane32 | Ex::FMAddLane16 => K::FloatLane {
                half: d.ex == Ex::FMAddLane16, a: src("A"), b: src("B"), c: src("C"), d: dst("D"), sat: f(fields, "S") != 0,
                d16: dsize("D").map(|s| s == 16), l: f(fields, "L"), g: f(fields, "G"), x: f(fields, "X"),
            },
            Ex::ICmpsel | Ex::FCmpsel => K::Cmpsel { cmp: cmp(d.ex == Ex::FCmpsel), x: src("X"), y: src("Y"), d: dst("D") },
            Ex::PopExec => K::PopExec { n: f(fields, "n") as i128 },
            Ex::IfCmp { float } => K::Mask { ctl: Ctl::If, cmp: cmp(float), n: f(fields, "n") as i128 },
            Ex::WhileCmp { float } => K::Mask { ctl: Ctl::While, cmp: cmp(float), n: f(fields, "n") as i128 },
            Ex::ElseCmp { float } => K::Mask { ctl: Ctl::Else, cmp: cmp(float), n: f(fields, "n") as i128 },
            Ex::ICmpBallot => K::Ballot { cmp: Some(cmp(false)), d: dst("D") },
            Ex::FCmpBallot => K::Ballot { cmp: None, d: dst("D") },
            Ex::SimdShuffle => K::SimdShuffle { a: src("A"), b: src("B"), d: dst("D") },
            Ex::SimdShuffleDown | Ex::SimdShuffleUp | Ex::SimdShuffleRotateUp | Ex::SimdShuffleXor => K::Shuffle {
                op: match d.ex {
                    Ex::SimdShuffleDown => Sh::Down,
                    Ex::SimdShuffleUp => Sh::Up,
                    Ex::SimdShuffleRotateUp => Sh::RotateUp,
                    _ => Sh::Xor,
                },
                a: src("A"),
                b: src("B"),
                d: dst("D"),
            },
            Ex::DeviceLoad => K::DeviceLoad {
                at: f(fields, "At") != 0,
                a: (f(fields, "A") >> 1) as usize,
                ot: f(fields, "Ot") != 0,
                o: f(fields, "O"),
                ou: f(fields, "Ou") != 0,
                fmt: f(fields, "F"),
                s: f(fields, "s"),
                r: f(fields, "R") as usize,
                rt: f(fields, "Rt"),
                mask: f(fields, "mask"),
            },
        };
        Compiled { k }
    }

    /// Masked lane loop: `body` on each active lane in order (MaskedInstructionDesc.exec).
    #[inline]
    fn lanes(st: &mut CoreState, mut body: impl FnMut(&mut CoreState, usize) -> R<()>) -> R<()> {
        for t in 0..SIMD_WIDTH {
            if st.exec[t] {
                body(st, t)?;
            }
        }
        Ok(())
    }

    /// `lanes` for a body that starts by reading `a` and `b` (in that order): the operands are fetched for all
    /// lanes first. A lane reads and writes only its own registers, so this is the same as reading them per lane.
    #[inline(always)]
    fn lanes2(st: &mut CoreState, a: &Src, b: &Src, mut body: impl FnMut(&mut CoreState, usize, i128, i128) -> R<()>) -> R<()> {
        if !st.exec.iter().any(|&x| x) {
            return Ok(());
        }
        let (mut av, mut bv) = ([0i128; SIMD_WIDTH], [0i128; SIMD_WIDTH]);
        a.fetch(st, &mut av)?;
        b.fetch(st, &mut bv)?;
        for t in 0..SIMD_WIDTH {
            if st.exec[t] {
                body(st, t, av[t], bv[t])?;
            }
        }
        Ok(())
    }
    /// `lanes2` with a third operand `c`, read after `a` and `b`
    #[inline(always)]
    fn lanes3(st: &mut CoreState, a: &Src, b: &Src, c: &Src, mut body: impl FnMut(&mut CoreState, usize, i128, i128, i128) -> R<()>) -> R<()> {
        if !st.exec.iter().any(|&x| x) {
            return Ok(());
        }
        let (mut av, mut bv, mut cv) = ([0i128; SIMD_WIDTH], [0i128; SIMD_WIDTH], [0i128; SIMD_WIDTH]);
        a.fetch(st, &mut av)?;
        b.fetch(st, &mut bv)?;
        c.fetch(st, &mut cv)?;
        for t in 0..SIMD_WIDTH {
            if st.exec[t] {
                body(st, t, av[t], bv[t], cv[t])?;
            }
        }
        Ok(())
    }

    pub fn exec(&self, st: &mut CoreState) -> Result<(), String> {
        self.run(st).map_err(str::to_string)
    }

    fn run(&self, st: &mut CoreState) -> R<()> {
        match &self.k {
            K::Raise(e) => Err(e),
            K::RaiseIfActive(e) => {
                if st.exec.iter().any(|&x| x) { Err(e) } else { Ok(()) }
            }
            K::Nop => Ok(()),
            K::MovImm { d, v } => Self::lanes(st, |st, t| d.set(st, t, *v)),
            K::GetSr { d, ok } => Self::lanes(st, |st, t| {
                if *ok { d.set(st, t, t as i128) } else { Err("AssertionError: get_sr TODO".into()) }
            }),
            K::IAdd { a, b, d, shift, sat, neg, signed, dsize } => Self::lanes2(st, a, b, |st, t, av, mut bv| {
                let sat = sat.ok_or("get_bit_size")?;
                if *neg {
                    bv = -bv;
                }
                bv = if *shift < 5 { bv << shift } else { 0 };
                let mut r = av + bv;
                if sat {
                    r = saturate(r, dsize.unwrap(), *signed);
                }
                d.set(st, t, r)
            }),
            K::IMAdd { a, b, c, d, shift, sat, neg, signed, dsize, ab_ok } => Self::lanes3(st, a, b, c, |st, t, av, bv, mut cv| {
                if !ab_ok {
                    return Err("get_bit_size".into());
                }
                let sat = sat.ok_or("get_bit_size")?;
                if *neg {
                    cv = -cv;
                }
                cv = if *shift < 5 { cv << shift } else { 0 };
                let mut r = av * bv + cv;
                if sat {
                    r = saturate(r, dsize.unwrap(), *signed);
                }
                d.set(st, t, r)
            }),
            K::Bitfield { op, a, b, c, d, m } => Self::lanes3(st, a, b, c, |st, t, av, bv, cv| {
                let (av, bv) = (av as u128, bv as u128);
                let s = (cv & 0x7F) as u32;
                let mask: u128 = if *m != 0 { (1u128 << m) - 1 } else { 0xFFFFFFFF };
                let r = match op {
                    Bf::Bfi => (av & !shl(mask, s)) | shl(bv & mask, s),
                    Bf::Bfeil => (av & !mask) | (shr(bv, super::exec::bfeil_shift(s, *m as u32)) & mask),
                    Bf::Extr => shr((bv << 32) | av, s) & mask,
                    Bf::Shlhi => {
                        let sm = shl(mask, s.saturating_sub(32));
                        let hi = if s >= 32 { shl(bv, s - 32) } else { bv >> (32 - s) };
                        (hi & sm) | (av & !sm)
                    }
                    Bf::Shrhi => {
                        let sm = (mask << 32) >> s.min(32);
                        (shr(bv << 32, s) & sm) | (av & !sm)
                    }
                };
                d.set(st, t, r as i128)
            }),
            K::Asr { high, a, b, d } => Self::lanes2(st, a, b, |st, t, av, bv| {
                let x = sign_extend(av, a.size()?);
                let x = if *high { x << 32 } else { x };
                d.set(st, t, x >> ((bv & 0x7F) as u32).min(127))
            }),
            K::Bitop { a, b, d, tt } => {
                // the truth table as all-ones / all-zero masks: the same r as or-ing the selected terms. In 64 bits:
                // every destination keeps at most the low 64, and bitwise operations do not carry between bits.
                let k = |i: usize| if tt[i] != 0 { u64::MAX } else { 0 };
                let (k0, k1, k2, k3) = (k(0), k(1), k(2), k(3));
                let pass = *tt == [0, 0, 1, 1] || *tt == [1, 1, 0, 0];
                Self::lanes2(st, a, b, |st, t, av, bv| {
                    let (x, y) = (av as u64, bv as u64);
                    let r = if pass { av } else { ((!x & !y & k0) | (x & !y & k1) | (!x & y & k2) | (x & y & k3)) as i128 };
                    d.set(st, t, r)
                })
            }
            K::Bitrev { a, d } => Self::lanes(st, |st, t| {
                let av = a.get(st, t)?;
                d.set(st, t, ((av as u32).reverse_bits()) as i128)
            }),
            K::Popcount { a, d } => Self::lanes(st, |st, t| {
                let av = a.get(st, t)?;
                d.set(st, t, (av as u32).count_ones() as i128)
            }),
            K::Ffs { a, d } => Self::lanes(st, |st, t| {
                let av = a.get(st, t)?;
                let x = (av as u64) & 0x1_FFFF_FFFF; // bits 0..=32
                d.set(st, t, if x == 0 { -1 } else { 63 - x.leading_zeros() as i128 })
            }),
            K::Float { op, a, b, c, d, sat, d16 } => {
                let mut memo: Option<(u64, u64, u64, u64)> = None;
                Self::lanes(st, |st, t| {
                let av = a.get_float(st, t)?;
                let bv = b.get_float(st, t)?;
                let (x, y, z) = match op {
                    Fl::FMAdd32 | Fl::FMAdd16 => (av, bv, c.as_ref().unwrap().get_float(st, t)?),
                    Fl::FAdd32 | Fl::FAdd16 => (av, fma::F64_ONE, bv),
                    _ => (av, bv, 0),
                };
                let d16 = d16.ok_or("get_bit_size")?;
                // pure function of (x, y, z): lanes usually share their inputs, so remember the last one
                let v = match memo {
                    Some((mx, my, mz, mv)) if (mx, my, mz) == (x, y, z) => mv,
                    _ => {
                        let mut r = fma::fma_rto_narrow(x, y, z);
                        if matches!(op, Fl::FMAdd32 | Fl::FAdd32 | Fl::FMul32) {
                            r = fma::f32_to_f64(fma::f64_to_f32(r, true), false);
                        }
                        if *sat {
                            r = fma::saturate64(r);
                        }
                        let v = if d16 { fma::f64_to_f16(r, false) } else { fma::f64_to_f32(r, true) };
                        memo = Some((x, y, z, v));
                        v
                    }
                };
                d.set(st, t, v as i128)
                })
            }
            K::FloatLane { half, a, b, c, d, sat, d16, l, g, x } => {
                if !st.exec.iter().any(|&e| e) {
                    return Ok(());
                }
                let src_lane = lane_src(*half, *l, *g, *x, &st.exec)?;
                let d16 = d16.ok_or("get_bit_size")?;
                // source A is read from other lanes: snapshot every lane before any destination write
                let mut av = [0u64; SIMD_WIDTH];
                let mut bv = [0u64; SIMD_WIDTH];
                let mut cv = [0u64; SIMD_WIDTH];
                for t in 0..SIMD_WIDTH {
                    bv[t] = b.get_float(st, t)?;
                    cv[t] = c.get_float(st, t)?;
                    av[t] = match src_lane[t] {
                        Some(s) => a.get_float(st, s)?,
                        None => 0f64.to_bits(),
                    };
                }
                for t in 0..SIMD_WIDTH {
                    if st.exec[t] {
                        let mut r = fma::fma_rto_narrow(av[t], bv[t], cv[t]);
                        if !*half {
                            r = fma::f32_to_f64(fma::f64_to_f32(r, true), false);
                        }
                        if *sat {
                            r = fma::saturate64(r);
                        }
                        let v = if d16 { fma::f64_to_f16(r, false) } else { fma::f64_to_f32(r, true) };
                        d.set(st, t, v as i128)?;
                    }
                }
                Ok(())
            }
            K::Cmpsel { cmp, x, y, d } => Self::lanes(st, |st, t| {
                let v = if cmp.eval(st, t)? { x.get(st, t)? } else { y.get(st, t)? };
                d.set(st, t, v)
            }),
            K::PopExec { n } => {
                for t in 0..SIMD_WIDTH {
                    let v = (st.reg16s[0][t] as i128 - n).max(0);
                    st.reg16s[0][t] = (v & 0xFFFF) as u16;
                }
                update_exec(st);
                Ok(())
            }
            K::Mask { ctl, cmp, n } => {
                for t in 0..SIMD_WIDTH {
                    let mut v = st.reg16s[0][t] as i128;
                    match ctl {
                        Ctl::If => {
                            if v != 0 {
                                v += n;
                            } else if !cmp.eval(st, t)? {
                                v = 1;
                            }
                        }
                        Ctl::While => {
                            if v < *n {
                                v = if cmp.eval(st, t)? { 0 } else { *n };
                            }
                        }
                        Ctl::Else => {
                            if v == 0 {
                                v = *n;
                            } else if v == 1 {
                                v = if cmp.eval(st, t)? { 0 } else { 1 };
                            }
                        }
                    }
                    st.reg16s[0][t] = (v & 0xFFFF) as u16;
                }
                update_exec(st);
                Ok(())
            }
            K::Ballot { cmp, d } => {
                let mut result = 0i128;
                for t in 0..SIMD_WIDTH {
                    if st.exec[t] && cmp.as_ref().map_or(Ok(true), |c| c.eval(st, t))? {
                        result |= 1 << t;
                    }
                }
                for t in 0..SIMD_WIDTH {
                    if st.exec[t] {
                        d.set(st, t, result)?;
                    }
                }
                Ok(())
            }
            K::SimdShuffle { a, b, d } => {
                let mut quad_values = [0i128; 8];
                for (qi_out, qs) in (0..SIMD_WIDTH).step_by(4).enumerate() {
                    let mut qi = 0usize;
                    for t in qs..qs + 4 {
                        qi |= (b.get(st, t)? & 3) as usize;
                    }
                    quad_values[qi_out] = a.get(st, qs + qi)?;
                }
                for t in 0..SIMD_WIDTH {
                    if st.exec[t] {
                        let index = b.get(st, t)?;
                        let r = if index < SIMD_WIDTH as i128 { quad_values[(index >> 2) as usize] } else { a.get(st, t)? };
                        d.set(st, t, r)?;
                    }
                }
                Ok(())
            }
            K::Shuffle { op, a, b, d } => {
                let mut av = [0i128; SIMD_WIDTH];
                let mut bv = [0i128; SIMD_WIDTH];
                for t in 0..SIMD_WIDTH {
                    av[t] = a.get(st, t)?;
                }
                for t in 0..SIMD_WIDTH {
                    bv[t] = b.get(st, t)?;
                }
                let mut quads = [[0i128; 4]; 8];
                for (qn, qs) in (0..SIMD_WIDTH).step_by(4).enumerate() {
                    let mut q = 0i128;
                    for t in qs..qs + 4 {
                        q |= bv[t] & 3;
                    }
                    let q = q as usize;
                    for i in 0..4 {
                        let lane = match op {
                            Sh::Down => (i + q) & 3,
                            Sh::Xor => (i ^ q) & 3,
                            _ => (i as isize - q as isize).rem_euclid(4) as usize,
                        };
                        quads[qn][i] = av[qs + lane];
                    }
                }
                for t in 0..SIMD_WIDTH {
                    if !st.exec[t] {
                        continue;
                    }
                    let index = bv[t];
                    let ti = t as i128;
                    let r = match op {
                        Sh::Down => {
                            if index + ti < 32 { quads[((index + ti) >> 2) as usize][t & 3] } else { av[t] }
                        }
                        Sh::Up => {
                            if ti - index >= 0 { quads[((ti - index) >> 2) as usize][t & 3] } else { av[t] }
                        }
                        Sh::RotateUp => quads[((ti - index).rem_euclid(32) >> 2) as usize][t & 3],
                        Sh::Xor => {
                            if (ti ^ index) < 32 { quads[(((ti ^ index) & 31) >> 2) as usize][t & 3] } else { av[t] }
                        }
                    };
                    d.set(st, t, r)?;
                }
                Ok(())
            }
            &K::DeviceLoad { at, a, ot, o, ou, fmt, s, r, rt, mask } => Self::lanes(st, |st, t| {
                let mut address: i128 = if at {
                    return Err("TypeError: Uniforms.get_reg64() takes 2 positional arguments".into());
                } else {
                    st.get_reg64(a, t).map_err(|_| "IndexError: register")?
                };
                let mut offset: i128 = if ot {
                    sign_extend(o as i128, 16)
                } else {
                    let v = st.get_reg32((o >> 1) as usize, t).map_err(|_| "IndexError: register")?;
                    if !ou { sign_extend(v, 32) } else { v }
                };
                let mut shift = if s == 3 { 2 } else { s };
                let bit_packed = matches!(fmt, 8 | 12 | 13);
                if bit_packed {
                    shift = 2;
                }
                offset <<= shift;
                let item_size: i128 = match fmt {
                    1 | 3 | 6 | 7 => 2,
                    2 => 4,
                    _ => 1,
                };
                address &= !(item_size - 1);
                let mut register = if rt == 1 { r >> 1 } else { r };
                if rt != 1 && fmt == 2 {
                    return Err("AssertionError: illegal instruction".into());
                }
                let mut m = mask;
                if bit_packed {
                    address &= !3;
                    m = 0xF;
                }
                let mut values = [0i128; 4];
                let mut nv = 0;
                for i in 0..4 {
                    if m & (1 << i) != 0 {
                        let la = address + (offset + i) * item_size;
                        if item_size != 4 {
                            return Err("AttributeError: _Memory has no get_byte/get_u16".into());
                        }
                        let mem = st.device_memory.as_ref().ok_or("AttributeError: no device memory")?;
                        values[nv] = mem.get_u32(la).map_err(|_| "emulated load outside table")? as i128;
                        nv += 1;
                    }
                }
                for &v in &values[..nv] {
                    if fmt >= 4 {
                        return Err("unsupported device_load format in emulator".into());
                    }
                    if rt == 1 {
                        st.set_reg32(register, t, v).map_err(|_| "IndexError: register")?;
                    } else {
                        st.set_reg16(register, t, v).map_err(|_| "IndexError: register")?;
                    }
                    register += 1;
                }
                Ok(())
            }),
        }
    }
}

#[inline]
fn shl(x: u128, s: u32) -> u128 {
    if s >= 128 { 0 } else { x << s }
}
#[inline]
fn shr(x: u128, s: u32) -> u128 {
    if s >= 128 { 0 } else { x >> s }
}

fn saturate(result: i128, dest_size: u32, signed: bool) -> i128 {
    let (lo, hi) = if signed {
        (-(1i128 << (dest_size - 1)), (1i128 << (dest_size - 1)) - 1)
    } else {
        (0, (1i128 << dest_size) - 1)
    };
    result.clamp(lo, hi)
}

fn update_exec(st: &mut CoreState) {
    for t in 0..SIMD_WIDTH {
        st.exec[t] = st.reg16s[0][t] == 0;
    }
}
