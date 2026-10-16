//! Functional semantics for the instructions the applegpu-derived emulator lacks or gets wrong, as measured on
//! the M2 Max (playground findings: float semantics, sub-op spaces; the hardware test suite). Used only by the
//! functional executor (`func.rs`); the cost model's trace keeps applegpu's exact behaviour.
//!
//! fp32: denormal inputs read as ±0, a result whose exact value is below the smallest normal becomes ±0 (before
//! rounding), round to nearest even, fused fma, canonical NaN 0x7fc00000. fp16: full IEEE with denormals, fused,
//! canonical NaN 0x7e00. `.sat`: clamp to [0, 1], NaN -> +0.

use crate::isa::exec::{CoreState, SIMD_WIDTH};
use crate::isa::fma::{f16_to_f64, f64_to_f16};
use crate::isa::operand::{ImmVal, Operand, F_ABS, F_NEG, F_SX};
use crate::isa::{fget, Fields};

type R<T> = Result<T, String>;

const F32_MIN_NORMAL: f64 = 1.1754943508222875e-38; // 2^-126
pub const NAN32: u32 = 0x7fc0_0000;
pub const NAN16: u32 = 0x7e00;

/// Raw bits and bit size of a source operand for lane t (no modifiers applied).
pub fn raw(st: &CoreState, o: &Operand, t: usize) -> R<(u64, u32)> {
    Ok(match o {
        Operand::Reg16(n, _) => (st.get_reg16(*n as usize, t)? as u64, 16),
        Operand::Reg32(n, _) => (st.get_reg32(*n as usize, t)? as u64, 32),
        Operand::Reg64(n, _) => (st.get_reg64(*n as usize, t)? as u64, 64),
        Operand::UReg16(n, _) => (st.ureg16(*n as usize)? as u64, 16),
        Operand::UReg32(n, _) => (st.ureg32(*n as usize)? as u64, 32),
        Operand::UReg64(n, _) => (st.ureg64(*n as usize)? as u64, 64),
        Operand::Imm(ImmVal::Int(v), _) => (*v as u64, 32),
        Operand::Imm(ImmVal::Float(x), _) => ((*x as f32).to_bits() as u64, 32),
        other => return Err(format!("operand {:?} not readable", other)),
    })
}

/// Integer value of a source: zero-extended, or sign-extended with `.sx`.
pub fn int(st: &CoreState, o: &Operand, t: usize) -> R<(i128, u32)> {
    let (b, sz) = raw(st, o, t)?;
    let mut v = b as i128;
    if o.flags() & F_SX != 0 && sz < 64 && v >> (sz - 1) & 1 == 1 {
        v -= 1i128 << sz;
    }
    Ok((v, sz))
}

fn f32_in(bits: u32) -> f64 {
    let e = (bits >> 23) & 0xff;
    if e == 0 {
        // denormal or zero: ±0 (DAZ)
        return if bits >> 31 != 0 { -0.0 } else { 0.0 };
    }
    f32::from_bits(bits) as f64
}

/// Float value of a source (16-bit registers are fp16, everything else fp32), modifiers applied.
pub fn float(st: &CoreState, o: &Operand, t: usize, daz: bool) -> R<f64> {
    let (b, sz) = raw(st, o, t)?;
    let mut x = match (o, sz) {
        (Operand::Imm(ImmVal::Float(x), _), _) => *x,
        (_, 16) => f64::from_bits(f16_to_f64(b, false)),
        _ => {
            if daz { f32_in(b as u32) } else { f32::from_bits(b as u32) as f64 }
        }
    };
    if o.flags() & F_ABS != 0 {
        x = x.abs();
    }
    if o.flags() & F_NEG != 0 {
        x = -x;
    }
    Ok(x)
}

/// Round an f64 (exact, or rounded to odd) to fp32 with the hardware's flush and NaN rules.
pub fn to_f32(x: f64) -> u32 {
    if x.is_nan() {
        return NAN32;
    }
    if x != 0.0 && x.abs() < F32_MIN_NORMAL {
        return if x < 0.0 { 0x8000_0000 } else { 0 };
    }
    (x as f32).to_bits()
}

pub fn to_f16(x: f64) -> u32 {
    if x.is_nan() {
        return NAN16;
    }
    f64_to_f16(x.to_bits(), false) as u32
}

pub fn sat(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(0.0, 1.0).max(0.0) + 0.0
    }
}

fn fma_odd(a: f64, b: f64, c: f64) -> f64 {
    f64::from_bits(crate::isa::fma::fma_rto_narrow(a.to_bits(), b.to_bits(), c.to_bits()))
}

/// Write a value to a destination operand for lane t (Reg16: low 16 bits, Reg32: 32, Reg64: 64).
pub fn put(st: &mut CoreState, d: &Operand, t: usize, v: u64) -> R<()> {
    match d {
        Operand::Reg16(n, _) => st.set_reg16(*n as usize, t, (v & 0xffff) as i128),
        Operand::Reg32(n, _) => st.set_reg32(*n as usize, t, (v & 0xffff_ffff) as i128),
        Operand::Reg64(n, _) => st.set_reg64(*n as usize, t, v as i128),
        other => Err(format!("destination {:?} not writable", other)),
    }
}

fn dsize(d: &Operand) -> u32 {
    match d {
        Operand::Reg16(..) => 16,
        Operand::Reg64(..) => 64,
        _ => 32,
    }
}

/// An fp32 source resolved once per instruction: a 32-bit lane register (DAZ, then abs / neg), or a value that is
/// the same in every lane (uniform or immediate: `float` of lane 0).
#[derive(Clone, Copy)]
enum FSrc {
    L(usize, bool, bool),
    K(f64),
}

fn fsrc_of(st: &CoreState, o: &Operand) -> R<Option<FSrc>> {
    Ok(match o {
        Operand::Reg32(n, _) => Some(FSrc::L(*n as usize, o.flags() & F_ABS != 0, o.flags() & F_NEG != 0)),
        Operand::UReg16(..) | Operand::UReg32(..) | Operand::Imm(..) => Some(FSrc::K(float(st, o, 0, true)?)),
        _ => None,
    })
}

#[inline]
fn fsrc_get(st: &CoreState, s: FSrc, t: usize) -> R<f64> {
    Ok(match s {
        FSrc::L(n, abs, neg) => {
            let mut x = f32_in(st.get_reg32(n, t)? as u32);
            if abs {
                x = x.abs();
            }
            if neg {
                x = -x;
            }
            x
        }
        FSrc::K(x) => x,
    })
}

/// fp32 fadd / fmul / fmadd with 32-bit register, uniform or immediate sources and a 32-bit destination: `float_arith`
/// without the per-lane operand dispatch (same values: `float` with DAZ, then the same rounding).
fn float_arith_fast(kind: u8, s: bool, ops: &[Operand], st: &mut CoreState) -> R<bool> {
    let Operand::Reg32(dn, _) = ops[0] else { return Ok(false) };
    if !st.exec.iter().any(|&e| e) {
        return Ok(true); // no lane writes (the general path reads nothing either)
    }
    let (Some(a), Some(b)) = (fsrc_of(st, &ops[1])?, fsrc_of(st, &ops[2])?) else { return Ok(false) };
    let c = if kind == 2 {
        match fsrc_of(st, &ops[3])? {
            Some(c) => Some(c),
            None => return Ok(false),
        }
    } else {
        None
    };
    for t in 0..SIMD_WIDTH {
        if !st.exec[t] {
            continue;
        }
        let av = fsrc_get(st, a, t)?;
        let bv = fsrc_get(st, b, t)?;
        let mut x = match c {
            None if kind == 0 => av + bv,
            None => av * bv,
            Some(c) => fma_odd(av, bv, fsrc_get(st, c, t)?),
        };
        if s {
            x = sat(x);
        }
        st.set_reg32(dn as usize, t, to_f32(x) as i128)?;
    }
    Ok(true)
}

/// fadd/fmul/fmadd 16/32 (`ops` = D, A, B[, C])
pub fn float_arith(name: &str, f: &Fields, ops: &[Operand], st: &mut CoreState) -> R<()> {
    let half = name.ends_with("16");
    let s = fget(f, "S").unwrap_or(0) != 0;
    let kind = if name.starts_with("fadd") { 0u8 } else if name.starts_with("fmul") { 1 } else { 2 };
    if !half && float_arith_fast(kind, s, ops, st)? {
        return Ok(());
    }
    let out16 = half || dsize(&ops[0]) == 16;
    for t in 0..SIMD_WIDTH {
        if !st.exec[t] {
            continue;
        }
        let a = float(st, &ops[1], t, !half)?;
        let b = float(st, &ops[2], t, !half)?;
        let mut x = if kind == 0 {
            a + b // exact in f64 for fp32/fp16 operands, except far-apart exponents where double rounding is harmless
        } else if kind == 1 {
            a * b // exact (48-bit product)
        } else {
            let c = float(st, &ops[3], t, !half)?;
            fma_odd(a, b, c)
        };
        if s {
            x = sat(x);
        }
        let bits = if out16 { to_f16(x) } else { to_f32(x) };
        put(st, &ops[0], t, bits as u64)?;
    }
    Ok(())
}

/// Float condition codes (bits 0-2) with the invert bit (3).
pub fn fcond(cc: u64, a: f64, b: f64) -> bool {
    let nan = a.is_nan() || b.is_nan();
    let r = match cc & 7 {
        0 => a == b,
        1 => a < b,
        2 => a > b,
        // "NaN loses": a NaN operand makes the other one win (min/max selection)
        3 => (a < b && !nan) || (b.is_nan() && !a.is_nan()),
        5 => a >= b,
        6 => a <= b,
        7 => (a > b && !nan) || (b.is_nan() && !a.is_nan()),
        _ => false, // 4: never
    };
    r != (cc & 8 != 0)
}

/// Integer condition codes: bits 0-1 (eq, lt, gt, never), bit 2 signed, bit 3 invert.
pub fn icond(cc: u64, a: i128, asz: u32, b: i128, bsz: u32) -> bool {
    let sx = |v: i128, sz: u32| if sz < 64 && v >> (sz - 1) & 1 == 1 { v - (1i128 << sz) } else { v };
    let (a, b) = if cc & 4 != 0 { (sx(a, asz), sx(b, bsz)) } else { (a, b) };
    let r = match cc & 3 {
        0 => a == b,
        1 => a < b,
        2 => a > b,
        _ => false,
    };
    r != (cc & 8 != 0)
}

fn cc_of(f: &Fields) -> u64 {
    fget(f, "cc").unwrap_or(0) | fget(f, "ccn").unwrap_or(0) << 3
}

pub fn compare(float_cmp: bool, f: &Fields, a: &Operand, b: &Operand, st: &CoreState, t: usize) -> R<bool> {
    compare_cc(float_cmp, cc_of(f), a, b, st, t)
}

/// `compare` with the condition code already read from the fields (lane loops read it once per instruction)
#[inline]
pub fn compare_cc(float_cmp: bool, cc: u64, a: &Operand, b: &Operand, st: &CoreState, t: usize) -> R<bool> {
    if float_cmp {
        Ok(fcond(cc, float(st, a, t, true)?, float(st, b, t, true)?))
    } else {
        let (x, xs) = int(st, a, t)?;
        let (y, ys) = int(st, b, t)?;
        Ok(icond(cc, x & ((1i128 << xs) - 1), xs, y & ((1i128 << ys) - 1), ys))
    }
}

/// A source operand resolved once per instruction: a lane register (16 / 32-bit) or a value that is the same in every
/// lane (uniform, immediate). `raw` without the per-lane operand match.
#[derive(Clone, Copy)]
enum Src {
    L16(usize),
    L32(usize),
    K(u64, u32),
}

fn src_of(st: &CoreState, o: &Operand) -> R<Option<Src>> {
    Ok(match o {
        Operand::Reg16(n, _) => Some(Src::L16(*n as usize)),
        Operand::Reg32(n, _) => Some(Src::L32(*n as usize)),
        Operand::UReg16(..) | Operand::UReg32(..) | Operand::Imm(ImmVal::Int(_), _) => {
            let (v, sz) = raw(st, o, 0)?;
            Some(Src::K(v, sz))
        }
        _ => None,
    })
}

#[inline(always)]
fn src_get(st: &CoreState, s: Src, t: usize) -> R<(u64, u32)> {
    Ok(match s {
        Src::L16(n) => (st.get_reg16(n, t)? as u64, 16),
        Src::L32(n) => (st.get_reg32(n, t)? as u64, 32),
        Src::K(v, sz) => (v, sz),
    })
}

/// icmpsel with 16 / 32-bit register, uniform or immediate operands: the common case, without per-lane operand
/// dispatch. Same result as the general path (compare masks to the operand size and sign-extends for signed codes).
/// Every operand is at most 32 bits wide, so the compare runs in i64; the condition code is decoded once.
fn icmpsel_fast(cc: u64, ops: &[Operand], st: &mut CoreState) -> R<bool> {
    let (Some(a), Some(b), Some(x), Some(y)) = (src_of(st, &ops[1])?, src_of(st, &ops[2])?, src_of(st, &ops[3])?, src_of(st, &ops[4])?)
    else {
        return Ok(false);
    };
    let (dn, d16) = match ops[0] {
        Operand::Reg16(n, _) => (n as usize, true),
        Operand::Reg32(n, _) => (n as usize, false),
        _ => return Ok(false),
    };
    let signed = cc & 4 != 0;
    let inv = cc & 8 != 0;
    let op = cc & 3;
    #[inline]
    fn ext(v: u64, sz: u32, signed: bool) -> i64 {
        let m = v & ((1u64 << sz) - 1);
        if signed && m >> (sz - 1) & 1 == 1 { m as i64 - (1i64 << sz) } else { m as i64 }
    }
    // a constant operand is extended once
    let ka = if let Src::K(v, sz) = a { Some(ext(v, sz, signed)) } else { None };
    let kb = if let Src::K(v, sz) = b { Some(ext(v, sz, signed)) } else { None };
    // Every register in range (checked once): lanes read and write the halves directly, with no per-lane checks.
    // Otherwise the general loop below, which reports an out-of-range register at the lane that first touches it.
    let n16 = st.reg16s.len();
    let fits = |s: Src| match s { Src::L16(n) => n < n16, Src::L32(n) => 2 * n + 1 < n16, Src::K(..) => true };
    if fits(a) && fits(b) && fits(x) && fits(y) && if d16 { dn < n16 } else { 2 * dn + 1 < n16 } {
        #[inline(always)]
        fn get(r: &[[u16; SIMD_WIDTH]], s: Src, t: usize) -> (u64, u32) {
            match s {
                Src::L16(n) => (r[n][t] as u64, 16),
                Src::L32(n) => (r[2 * n][t] as u64 | (r[2 * n + 1][t] as u64) << 16, 32),
                Src::K(v, sz) => (v, sz),
            }
        }
        // operands gathered for every lane first (a lane only reads and writes its own registers)
        let r = &st.reg16s;
        let mut c = [inv; SIMD_WIDTH];
        if op != 3 {
            // the operand kind decided once, each loop with a fixed size
            let ext_all = |s: Src, k: Option<i64>| -> [i64; SIMD_WIDTH] {
                let mut o = [0i64; SIMD_WIDTH];
                match (k, s) {
                    (Some(k), _) => o.fill(k),
                    (None, Src::L16(n)) => {
                        for (t, x) in o.iter_mut().enumerate() {
                            *x = ext(r[n][t] as u64, 16, signed);
                        }
                    }
                    (None, Src::L32(n)) => {
                        for (t, x) in o.iter_mut().enumerate() {
                            *x = ext(r[2 * n][t] as u64 | (r[2 * n + 1][t] as u64) << 16, 32, signed);
                        }
                    }
                    (None, s) => {
                        for (t, x) in o.iter_mut().enumerate() {
                            let (v, sz) = get(r, s, t);
                            *x = ext(v, sz, signed);
                        }
                    }
                }
                o
            };
            let (av, bv) = (ext_all(a, ka), ext_all(b, kb));
            for t in 0..SIMD_WIDTH {
                c[t] = (match op { 0 => av[t] == bv[t], 1 => av[t] < bv[t], _ => av[t] > bv[t] }) != inv;
            }
        }
        for t in 0..SIMD_WIDTH {
            if !st.exec[t] {
                continue;
            }
            let (v, _) = get(&st.reg16s, if c[t] { x } else { y }, t);
            if d16 {
                st.reg16s[dn][t] = v as u16;
            } else {
                st.reg16s[2 * dn][t] = v as u16;
                st.reg16s[2 * dn + 1][t] = (v >> 16) as u16;
            }
        }
        return Ok(true);
    }
    for t in 0..SIMD_WIDTH {
        if !st.exec[t] {
            continue;
        }
        let c = if op == 3 {
            inv
        } else {
            let av = match ka { Some(k) => k, None => { let (v, sz) = src_get(st, a, t)?; ext(v, sz, signed) } };
            let bv = match kb { Some(k) => k, None => { let (v, sz) = src_get(st, b, t)?; ext(v, sz, signed) } };
            (match op { 0 => av == bv, 1 => av < bv, _ => av > bv }) != inv
        };
        let (v, _) = src_get(st, if c { x } else { y }, t)?;
        if d16 {
            st.set_reg16(dn, t, (v & 0xffff) as i128)?;
        } else {
            st.set_reg32(dn, t, (v & 0xffff_ffff) as i128)?;
        }
    }
    Ok(true)
}

/// icmpsel / fcmpsel: D = cond(A, B) ? X : Y  (ops: D, A, B, X, Y)
pub fn cmpsel(float_cmp: bool, f: &Fields, ops: &[Operand], st: &mut CoreState) -> R<()> {
    let cc = cc_of(f);
    if !float_cmp && icmpsel_fast(cc, ops, st)? {
        return Ok(());
    }
    for t in 0..SIMD_WIDTH {
        if !st.exec[t] {
            continue;
        }
        let c = compare_cc(float_cmp, cc, &ops[1], &ops[2], st, t)?;
        let (v, _) = raw(st, if c { &ops[3] } else { &ops[4] }, t)?;
        put(st, &ops[0], t, v)?;
    }
    Ok(())
}

/// icmp/fcmp ballot (quad: each lane receives the 4-bit mask of its quad... variant chosen by the suite)
pub fn ballot(float_cmp: bool, quad: bool, f: &Fields, ops: &[Operand], st: &mut CoreState) -> R<()> {
    let mut bits = 0u64;
    let cc = cc_of(f);
    for t in 0..SIMD_WIDTH {
        if st.exec[t] && compare_cc(float_cmp, cc, &ops[1], &ops[2], st, t)? {
            bits |= 1 << t;
        }
    }
    for t in 0..SIMD_WIDTH {
        if st.exec[t] {
            let v = if quad { (bits >> (t & !3)) & 0xf } else { bits };
            put(st, &ops[0], t, v)?;
        }
    }
    Ok(())
}

/// if / else / while exec-mask update (applegpu semantics, r0l nesting counter), any condition code
pub fn mask_ctl(kind: &str, float_cmp: bool, f: &Fields, ops: &[Operand], st: &mut CoreState) -> R<()> {
    let n = fget(f, "n").unwrap_or(1) as i128;
    let cc = cc_of(f);
    let k = match kind { "if" => 0u8, "while" => 1, _ => 2 };
    // operands: r0l (destination), A, B
    let (a, b) = (&ops[1], &ops[2]);
    // integer compares on 16 / 32-bit registers, uniforms or immediates: operands resolved once (as icmpsel_fast)
    let fast = if float_cmp { None } else { match (src_of(st, a)?, src_of(st, b)?) { (Some(x), Some(y)) => Some((x, y)), _ => None } };
    let cmp = |st: &CoreState, t: usize| -> R<bool> {
        match fast {
            Some((x, y)) => {
                let (av, asz) = src_get(st, x, t)?;
                let (bv, bsz) = src_get(st, y, t)?;
                Ok(icond(cc, (av & ((1u64 << asz) - 1)) as i128, asz, (bv & ((1u64 << bsz) - 1)) as i128, bsz))
            }
            None => compare_cc(float_cmp, cc, a, b, st, t),
        }
    };
    for t in 0..SIMD_WIDTH {
        let mut v = st.get_reg16(0, t)?;
        match k {
            0 => {
                if v != 0 {
                    v += n;
                } else if !cmp(st, t)? {
                    v = 1;
                }
            }
            1 => {
                if v < n {
                    v = if cmp(st, t)? { 0 } else { n };
                }
            }
            _ => {
                if v == 0 {
                    v = n;
                } else if v == 1 {
                    v = if cmp(st, t)? { 0 } else { 1 };
                }
            }
        }
        st.set_reg16(0, t, v & 0xffff)?;
    }
    for t in 0..SIMD_WIDTH {
        st.exec[t] = st.get_reg16(0, t)? == 0;
    }
    Ok(())
}

// ---- rounding, interleave, convert, cross-lane (fitted on the hardware suite) --------------------------

/// floor / ceil / trunc / rint: fp32 (denormal in = 0, tiny out flushed) or fp16, canonical NaN, sign of zero kept.
pub fn round_op(name: &str, ops: &[Operand], st: &mut CoreState) -> R<()> {
    let d16 = dsize(&ops[0]) == 16;
    for t in 0..SIMD_WIDTH {
        if !st.exec[t] {
            continue;
        }
        let x = float(st, &ops[1], t, true)?;
        let y = if x.is_nan() || x.is_infinite() {
            x
        } else {
            let r = match name {
                "floor" => x.floor(),
                "ceil" => x.ceil(),
                "trunc" => x.trunc(),
                _ => x.round_ties_even(),
            };
            r.copysign(x)
        };
        put(st, &ops[0], t, if d16 { to_f16(y) } else { to_f32(y) } as u64)?;
    }
    Ok(())
}

/// intl: interleave the low 16 bits of A (even result bits) and B (odd result bits).
pub fn intl(ops: &[Operand], st: &mut CoreState) -> R<()> {
    let spread = |v: u64| (0..16).fold(0u64, |r, i| r | ((v >> i) & 1) << (2 * i));
    for t in 0..SIMD_WIDTH {
        if st.exec[t] {
            let a = raw(st, &ops[1], t)?.0;
            let b = raw(st, &ops[2], t)?.0;
            put(st, &ops[0], t, spread(a) | spread(b) << 1)?;
        }
    }
    Ok(())
}

/// Round an exact integer to fp32 / fp16 bits, to nearest even or toward zero.
fn int_to_float(v: i64, half: bool, rtz: bool) -> u64 {
    let near = if half { f64_to_f16((v as f64).to_bits(), false) } else { (v as f32).to_bits() as u64 };
    if !rtz {
        return near;
    }
    // round toward zero still overflows to ±inf past the fp16 exponent range: 65535 -> 65504 but 65536 -> inf
    // (coverage convert u32_to_f mode 8 / 9 with a 16-bit destination)
    if half && v.unsigned_abs() >= 1 << 16 {
        return if v < 0 { 0xfc00 } else { 0x7c00 };
    }
    let mag = |b: u64| -> f64 {
        if half { f64::from_bits(f16_to_f64(b, false)).abs() } else { (f32::from_bits(b as u32) as f64).abs() }
    };
    // nearest rounded away from zero: step one representable value toward zero (inf -> largest finite)
    if mag(near) > (v as f64).abs() { near - 1 } else { near }
}

/// convert. The field applegpu names `round` (bit 26) is the direction (0: float -> integer, 1: integer -> float);
/// mode = size << 2 | rte << 1 | signed with size 0/1/2 = 8/16/32 bits. Float -> integer saturates to the type,
/// NaN -> 0, and the result is zero-extended from the type width; integer -> float rounds to the destination
/// precision (fp16 for a 16-bit destination).
pub fn convert(f: &Fields, ops: &[Operand], st: &mut CoreState) -> R<()> {
    let mode = fget(f, "mode").unwrap_or(0);
    let to_float = fget(f, "round").unwrap_or(0) & 1 != 0;
    let signed = mode & 1 != 0;
    let rte = mode & 2 != 0;
    let w = match mode >> 2 {
        0 => 8,
        1 => 16,
        2 => 32,
        _ => return Err(format!("convert mode {} not modelled", mode)),
    };
    let d16 = dsize(&ops[0]) == 16;
    for t in 0..SIMD_WIDTH {
        if !st.exec[t] {
            continue;
        }
        let out = if to_float {
            let (b, _) = raw(st, &ops[1], t)?;
            let mut v = (b & ((1u64 << w) - 1)) as i64;
            if signed && v >> (w - 1) & 1 == 1 {
                v -= 1i64 << w;
            }
            int_to_float(v, d16, !rte)
        } else {
            let x = float(st, &ops[1], t, true)?;
            let (lo, hi) = if signed { (-(1i64 << (w - 1)), (1i64 << (w - 1)) - 1) } else { (0, (1i64 << w) - 1) };
            let v = if x.is_nan() {
                0
            } else {
                let r = if rte { x.round_ties_even() } else { x.trunc() };
                if r <= lo as f64 { lo } else if r >= hi as f64 { hi } else { r as i64 }
            };
            (v as u64) & ((1u64 << w) - 1)
        };
        put(st, &ops[0], t, out)?;
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum XOp {
    IAdd,
    And,
    Or,
    Xor,
    MinU,
    MaxU,
    MinS,
    MaxS,
    FAdd,
    FMul,
    MinF,
    MaxF,
}

fn xop(name: &str) -> Option<XOp> {
    Some(match name {
        "iadd" => XOp::IAdd,
        "and" => XOp::And,
        "or" => XOp::Or,
        "xor" => XOp::Xor,
        "min.u" => XOp::MinU,
        "max.u" => XOp::MaxU,
        "min.s" => XOp::MinS,
        "max.s" => XOp::MaxS,
        "fadd" => XOp::FAdd,
        "fmul" => XOp::FMul,
        "min.f" => XOp::MinF,
        "max.f" => XOp::MaxF,
        _ => return None,
    })
}

fn daz_bits(b: u32) -> f64 {
    f32_in(b)
}

/// fp32 value -> hardware fp32 bits -> value as the next operation reads it (flush to zero both ways)
fn fround(x: f64) -> u32 {
    to_f32(x)
}

/// One combining step on raw 32-bit values (floats: exact op, one rounding, denormals read as zero; min/max select
/// the raw operand, comparing denormals as zero, the left operand on ties).
fn xapply(op: XOp, x: u32, y: u32) -> u32 {
    let s = |v: u32| v as i32;
    match op {
        XOp::IAdd => x.wrapping_add(y),
        XOp::And => x & y,
        XOp::Or => x | y,
        XOp::Xor => x ^ y,
        XOp::MinU => x.min(y),
        XOp::MaxU => x.max(y),
        XOp::MinS => if s(y) < s(x) { y } else { x },
        XOp::MaxS => if s(y) > s(x) { y } else { x },
        XOp::FAdd => fround(fma_odd(daz_bits(x), 1.0, daz_bits(y))),
        XOp::FMul => fround(daz_bits(x) * daz_bits(y)),
        XOp::MinF | XOp::MaxF => {
            let (fx, fy) = (daz_bits(x), daz_bits(y));
            if fx.is_nan() {
                // NaN loses; of two NaNs the left one is kept
                return if fy.is_nan() { x } else { y };
            }
            if fy.is_nan() || fx == fy {
                return x;
            }
            if (fx < fy) == (op == XOp::MinF) { x } else { y }
        }
    }
}

/// xapply on 16-bit values (integers wrap at 16 bits, floats are fp16 with denormals kept, one rounding).
fn xapply16(op: XOp, x: u32, y: u32) -> u32 {
    let s = |v: u32| v as u16 as i16;
    let h = |v: u32| f64::from_bits(f16_to_f64(v as u64, false));
    match op {
        XOp::IAdd => x.wrapping_add(y) & 0xffff,
        XOp::And => x & y,
        XOp::Or => x | y,
        XOp::Xor => x ^ y,
        XOp::MinU => x.min(y),
        XOp::MaxU => x.max(y),
        XOp::MinS => if s(y) < s(x) { y } else { x },
        XOp::MaxS => if s(y) > s(x) { y } else { x },
        XOp::FAdd => to_f16(h(x) + h(y)),
        XOp::FMul => to_f16(h(x) * h(y)),
        XOp::MinF | XOp::MaxF => {
            let (fx, fy) = (h(x), h(y));
            if fx.is_nan() {
                return if fy.is_nan() { x } else { y };
            }
            if fy.is_nan() || fx == fy {
                return x;
            }
            if (fx < fy) == (op == XOp::MinF) { x } else { y }
        }
    }
}

fn identity16(op: XOp) -> u32 {
    match op {
        XOp::IAdd | XOp::Or | XOp::Xor | XOp::MaxU => 0,
        XOp::And | XOp::MinU => 0xffff,
        XOp::MinS => 0x7fff,
        XOp::MaxS => 0x8000,
        XOp::FAdd => 0x8000,
        XOp::FMul => 0x3c00,
        XOp::MinF | XOp::MaxF => NAN16,
    }
}

fn identity(op: XOp) -> u32 {
    match op {
        XOp::IAdd | XOp::Or | XOp::Xor | XOp::MaxU => 0,
        XOp::And | XOp::MinU => u32::MAX,
        XOp::MinS => 0x7fff_ffff,
        XOp::MaxS => 0x8000_0000,
        XOp::FAdd => 0x8000_0000,
        XOp::FMul => 0x3f80_0000,
        XOp::MinF | XOp::MaxF => NAN32,
    }
}

fn comb(op: XOp, w16: bool, x: Option<u32>, y: Option<u32>) -> Option<u32> {
    match (x, y) {
        (None, v) | (v, None) => v,
        (Some(a), Some(b)) => Some(if w16 { xapply16(op, a, b) } else { xapply(op, a, b) }),
    }
}

fn canon(op: XOp, w16: bool, v: u32) -> u32 {
    // min / max return an operand's raw bits, NaN included (random/prog_077: all-NaN quad -> 0xffffffff)
    let is_float = matches!(op, XOp::FAdd | XOp::FMul);
    if !is_float {
        v
    } else if w16 {
        if f64::from_bits(f16_to_f64(v as u64, false)).is_nan() { NAN16 } else { v }
    } else if f32::from_bits(v).is_nan() {
        NAN32
    } else {
        v
    }
}

/// simd_/quad_ reductions and exclusive prefix ops (`ops` = D, A). Inactive lanes take no part.
/// Reductions: pairwise tree over the quad / SIMD group. Quad prefix: lane-order chain. SIMD prefix: Sklansky scan
/// over the 32 lanes; lane i receives the inclusive value of lane i-1 (lane 0 and lanes with nothing before them: the
/// identity). 16-bit sources combine at 16 bits (fp16, denormals kept, rounded every step); 64-bit sources are read as
/// their low 32 bits (coverage/expected alu, alu_masked).
pub fn cross_lane(name: &str, ops: &[Operand], st: &mut CoreState) -> R<()> {
    let quad = name.starts_with("quad_");
    let rest = name.split_once('_').map(|x| x.1).unwrap_or("");
    let (prefix, opname) = match rest.strip_prefix("prefix_") {
        Some(o) => (true, o),
        None => (false, rest),
    };
    let op = xop(opname).ok_or_else(|| format!("{} not modelled", name))?;
    // A 16-bit source is combined at 16 bits (fp16 for the float ops). A 64-bit source is read as its low 32 bits.
    let w16 = raw(st, &ops[1], 0)?.1 == 16;
    let mut x: [Option<u32>; SIMD_WIDTH] = [None; SIMD_WIDTH];
    for t in 0..SIMD_WIDTH {
        if st.exec[t] {
            let b = raw(st, &ops[1], t)?.0 as u32;
            // fp32 fadd / fmul read denormals as signed zero (also when a lane's value passes through unchanged)
            let flush = !w16 && matches!(op, XOp::FAdd | XOp::FMul) && (b >> 23) & 0xff == 0;
            x[t] = Some(if flush { b & 0x8000_0000 } else { b });
        } else if matches!(op, XOp::MinF | XOp::MaxF) {
            // inactive lanes take part as the canonical NaN (random/prog_106: a lone active NaN comes out canonical,
            // while four active NaNs keep their raw bits, prog_077)
            x[t] = Some(if w16 { NAN16 } else { NAN32 });
        }
    }
    let comb = |x, y| comb(op, w16, x, y);
    let identity = |op| if w16 { identity16(op) } else { identity(op) };
    let w = if quad { 4 } else { 32 };
    let mut out: [Option<u32>; SIMD_WIDTH] = [None; SIMD_WIDTH];
    if !prefix {
        for lo in (0..SIMD_WIDTH).step_by(w) {
            let mut vs: Vec<Option<u32>> = x[lo..lo + w].to_vec();
            while vs.len() > 1 {
                vs = vs.chunks(2).map(|p| comb(p[0], p[1])).collect();
            }
            for o in &mut out[lo..lo + w] {
                *o = vs[0];
            }
        }
    } else {
        // inclusive values per lane
        let mut inc = x;
        if quad {
            for q in 0..8 {
                for i in 1..4 {
                    inc[4 * q + i] = comb(inc[4 * q + i - 1], x[4 * q + i]);
                }
            }
        } else {
            // Sklansky network over the 32 lanes: at block size b, every lane in the upper half of a 2b block takes
            // the last inclusive value of the lower half on its left (fp16 recordings fit this order exactly)
            let mut b = 1;
            while b < SIMD_WIDTH {
                let prev = inc;
                for t in 0..SIMD_WIDTH {
                    if (t / b) % 2 == 1 {
                        inc[t] = comb(prev[(t / b) * b - 1], prev[t]);
                    }
                }
                b *= 2;
            }
        }
        for t in 0..SIMD_WIDTH {
            let first = if quad { t % 4 == 0 } else { t == 0 };
            out[t] = Some(if first { identity(op) } else { inc[t - 1].unwrap_or(identity(op)) });
        }
    }
    for t in 0..SIMD_WIDTH {
        if st.exec[t] {
            if let Some(v) = out[t] {
                put(st, &ops[0], t, canon(op, w16, v) as u64)?;
            }
        }
    }
    Ok(())
}

/// quad_shuffle family (`ops` = D, A, B): index B per lane; an index that leaves the quad keeps the lane's own value
/// (rotate_up wraps).
pub fn quad_shuffle(name: &str, ops: &[Operand], st: &mut CoreState) -> R<()> {
    let mut a = [0u64; SIMD_WIDTH];
    let mut b = [0i64; SIMD_WIDTH];
    for t in 0..SIMD_WIDTH {
        a[t] = raw(st, &ops[1], t)?.0;
        b[t] = (raw(st, &ops[2], t)?.0 & 0xffff) as i64;
    }
    for t in 0..SIMD_WIDTH {
        if !st.exec[t] {
            continue;
        }
        let (q, i) = (t & !3, (t & 3) as i64);
        let j = match name {
            "quad_shuffle" => Some(b[t]),
            "quad_shuffle_down" => Some(i + b[t]),
            "quad_shuffle_up" => Some(i - b[t]),
            "quad_shuffle_xor" => Some(i ^ b[t]),
            "quad_shuffle_rotate_up" => Some((i - b[t]).rem_euclid(4)),
            _ => None,
        }
        .ok_or_else(|| format!("{} not modelled", name))?;
        let v = if (0..4).contains(&j) { a[q + j as usize] } else { a[t] };
        put(st, &ops[0], t, v)?;
    }
    Ok(())
}

// ---- transcendentals --------------------------------------------------------------------------------------
//
// fp16 forms are correctly rounded (every sampled lane of the suite). fp32 forms are the hardware's approximation:
// rcp / rsqrt / rsqrt_special / sin_pt_2 come from recorded exact tables, log2 and exp2 from the recorded fixed-point
// samples (`transtab`), for fp32 and fp16 sources alike: every fp32 transcendental is exact. Special values, the denormal rules (fp32 inputs flushed to zero, fp16
// denormals kept), `.sat` and sin_pt_1 are exact.

/// Exact value (in f64) of the function, before rounding to the destination format.
fn trans_f64(name: &str, x: f64) -> f64 {
    match name {
        "rcp" => 1.0 / x,
        "rsqrt" | "rsqrt_special" => {
            if x < 0.0 {
                f64::NAN
            } else if x == 0.0 {
                // rsqrt(±0) = ±inf; rsqrt_special(±0) = 1
                if name == "rsqrt" { f64::INFINITY.copysign(x) } else { 1.0 }
            } else if x.is_infinite() {
                if name == "rsqrt" { 0.0 } else { 1.0 }
            } else {
                1.0 / x.sqrt()
            }
        }
        "log2" => {
            if x == 0.0 {
                f64::NEG_INFINITY
            } else if x < 0.0 {
                f64::NAN
            } else {
                x.log2()
            }
        }
        "exp2" => x.exp2(),
        "sin_pt_1" => {
            let ax = x.abs();
            if x.is_nan() || ax > 4.0 {
                f64::NAN
            } else if x == 0.0 {
                0.0
            } else if ax <= 1.0 {
                x
            } else if ax <= 3.0 {
                (2.0 - ax) * x.signum()
            } else {
                (ax - 4.0) * x.signum()
            }
        }
        "sin_pt_2" => {
            if x.is_nan() || x.abs() > 1.0 {
                f64::NAN
            } else if x == 0.0 {
                std::f64::consts::FRAC_PI_2
            } else {
                (std::f64::consts::FRAC_PI_2 * x).sin() / x
            }
        }
        _ => f64::NAN,
    }
}

/// `ops` = D, A. `.sat` clamps the result to [0, 1] (NaN -> +0), also on table results. fp32 destinations take the
/// recorded table when one covers the operand (fp32 or fp16 source: the unit computes on the fp32 value).
pub fn transcendental(name: &str, f: &Fields, ops: &[Operand], st: &mut CoreState) -> R<()> {
    let d16 = dsize(&ops[0]) == 16;
    let s = fget(f, "S").unwrap_or(0) != 0;
    for t in 0..SIMD_WIDTH {
        if !st.exec[t] {
            continue;
        }
        // fp32 sources: denormals read as zero; fp16 sources keep denormals (exact in fp32)
        let x = float(st, &ops[1], t, true)?;
        let table = if d16 || x.is_nan() { None } else { crate::transtab::lookup(name, (x as f32).to_bits()) };
        let mut y = match table {
            Some(b) => f32::from_bits(b) as f64,
            None => trans_f64(name, x),
        };
        if s {
            y = sat(y);
        }
        let bits = match table {
            Some(b) if !s => b,
            _ if d16 => to_f16(y),
            _ => to_f32(y),
        };
        put(st, &ops[0], t, bits as u64)?;
    }
    Ok(())
}
