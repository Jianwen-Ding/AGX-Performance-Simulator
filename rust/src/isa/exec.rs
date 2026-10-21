//! applegpu's emulator: `CoreState` and each descriptor's `exec` / `exec_thread`, including where
//! applegpu raises (the cost model's trace treats an instruction that raises as skipped).

use super::desc::{fget, fx, sign_extend, Fields, OpKind};
use super::fma;
use super::operand::*;
use super::table::{Desc, Ex};

pub const SIMD_WIDTH: usize = 32;
pub const EXEC_OVERFLOW_UNRESOLVED: &str = "execution-counter overflow activation spacing unresolved; use wait 0 or pop_exec 0 before further work (AGX_EXPERIMENTAL_EXEC_OVERFLOW=counter-sync keeps the known-incomplete immediate-sync hypothesis)";

pub struct Memory {
    pub base: i128,
    pub words: Vec<u32>,
}

impl Memory {
    /// agx_loop._Memory.get_u32
    pub fn get_u32(&self, address: i128) -> Result<u32, String> {
        let d = address - self.base;
        let (i, r) = (d.div_euclid(4), d.rem_euclid(4));
        if r != 0 || i < 0 || i >= self.words.len() as i128 {
            return Err(format!("emulated load outside table: {:#x}", address));
        }
        Ok(self.words[i as usize])
    }
}

pub struct CoreState {
    pub reg16s: Vec<[u16; SIMD_WIDTH]>,
    pub exec: [bool; SIMD_WIDTH],
    /// Physical lanes still alive; mask reconvergence cannot revive a stopped or absent lane.
    pub live: [bool; SIMD_WIDTH],
    /// A masked lane wrapped to zero in IF/push; its activation timing is not modelled.
    pub overflow_activation_pending: bool,
    /// Explicit hypothesis for tests/experiments; this is not hardware-accurate timing.
    pub allow_overflow_activation_hypothesis: bool,
    /// 512 16-bit values (u0..u255 are 32-bit; applegpu keeps 256 halves, Mesa uses up to u255); `get_reg64` takes no thread argument.
    pub uniforms: Vec<u16>,
    pub device_memory: Option<Memory>,
}

type R<T> = Result<T, String>;

impl CoreState {
    pub fn new(num_registers: usize, device_memory: Option<Memory>) -> Self {
        CoreState {
            reg16s: vec![[0; SIMD_WIDTH]; num_registers * 2],
            exec: [true; SIMD_WIDTH],
            live: [true; SIMD_WIDTH],
            overflow_activation_pending: false,
            allow_overflow_activation_hypothesis: false,
            uniforms: vec![0; 512],
            device_memory,
        }
    }
    /// Conservatively refuse instruction-spacing-sensitive overflow activation.
    /// Only the directly measured wait0/pop0 settlement is accepted by default;
    /// empirically sufficient ALU spacing is deliberately not guessed here.
    #[inline]
    pub fn check_overflow_activation(&mut self, settles: bool) -> Result<(), &'static str> {
        if settles {
            self.overflow_activation_pending = false;
        } else if self.overflow_activation_pending && !self.allow_overflow_activation_hypothesis
            && std::env::var("AGX_EXPERIMENTAL_EXEC_OVERFLOW").ok().as_deref() != Some("counter-sync") {
            return Err(EXEC_OVERFLOW_UNRESOLVED);
        }
        Ok(())
    }
    #[inline(always)]
    fn idx(&self, r: usize) -> R<usize> {
        if r < self.reg16s.len() { Ok(r) } else { Err("IndexError: register".into()) }
    }
    #[inline(always)]
    pub fn get_reg16(&self, r: usize, t: usize) -> R<i128> {
        Ok(self.reg16s[self.idx(r)?][t] as i128)
    }
    #[inline(always)]
    pub fn set_reg16(&mut self, r: usize, t: usize, v: i128) -> R<()> {
        let i = self.idx(r)?;
        self.reg16s[i][t] = (v & 0xFFFF) as u16;
        Ok(())
    }
    #[inline(always)]
    pub fn get_reg32(&self, r: usize, t: usize) -> R<i128> {
        if 2 * r + 1 < self.reg16s.len() {
            // both halves in range: one check (the general path below reports the same errors otherwise)
            return Ok(self.reg16s[2 * r][t] as i128 | (self.reg16s[2 * r + 1][t] as i128) << 16);
        }
        Ok(self.get_reg16(2 * r, t)? | (self.get_reg16(2 * r + 1, t)? << 16))
    }
    #[inline(always)]
    pub fn set_reg32(&mut self, r: usize, t: usize, v: i128) -> R<()> {
        if 2 * r + 1 < self.reg16s.len() {
            self.reg16s[2 * r][t] = (v & 0xFFFF) as u16;
            self.reg16s[2 * r + 1][t] = ((v >> 16) & 0xFFFF) as u16;
            return Ok(());
        }
        // applegpu writes the low half first; an out-of-range high half raises after it
        self.set_reg16(2 * r, t, v & 0xFFFF)?;
        self.set_reg16(2 * r + 1, t, (v >> 16) & 0xFFFF)
    }
    pub fn get_reg64(&self, r: usize, t: usize) -> R<i128> {
        Ok(self.get_reg32(r, t)? | (self.get_reg32(r + 1, t)? << 32))
    }
    pub fn set_reg64(&mut self, r: usize, t: usize, v: i128) -> R<()> {
        self.set_reg32(r, t, v & 0xFFFFFFFF)?;
        self.set_reg32(r + 1, t, (v >> 32) & 0xFFFFFFFF)
    }
    pub fn ureg16(&self, r: usize) -> R<i128> {
        self.uniforms.get(r).map(|&v| v as i128).ok_or_else(|| "IndexError: uniform".into())
    }
    pub fn ureg32(&self, r: usize) -> R<i128> {
        Ok(self.ureg16(2 * r)? | (self.ureg16(2 * r + 1)? << 16))
    }
    pub fn ureg64(&self, r: usize) -> R<i128> {
        Ok(self.ureg32(r)? | (self.ureg32(r + 1)? << 32))
    }
}

/// Integer or float value of an operand for one lane (Python `get_thread`; float only for
/// floating-point immediates).
#[derive(Clone, Copy, Debug)]
enum Val {
    I(i128),
    F(f64),
}

fn get_thread(op: &Operand, st: &CoreState, t: usize) -> R<Val> {
    Ok(Val::I(match op {
        Operand::Reg16(n, _) => st.get_reg16(*n as usize, t)?,
        Operand::Reg32(n, _) => st.get_reg32(*n as usize, t)?,
        Operand::Reg64(n, _) => st.get_reg64(*n as usize, t)?,
        Operand::UReg16(n, _) => st.ureg16(*n as usize)?,
        Operand::UReg32(n, _) => st.ureg32(*n as usize)?,
        Operand::UReg64(n, _) => st.ureg64(*n as usize)?,
        Operand::Imm(ImmVal::Int(v), _) => *v as i128,
        Operand::Imm(ImmVal::Float(x), _) => return Ok(Val::F(*x)),
        Operand::Tuple(..) => return Err("NotImplementedError: get_thread".into()),
        _ => return Err(format!("AttributeError: get_thread on {}", op.type_name())),
    }))
}

fn int(v: Val) -> R<i128> {
    match v {
        Val::I(i) => Ok(i),
        Val::F(_) => Err("TypeError: float operand in integer operation".into()),
    }
}

fn set_thread(op: &Operand, st: &mut CoreState, t: usize, v: i128) -> R<()> {
    match op {
        Operand::Reg16(n, _) => st.set_reg16(*n as usize, t, v),
        Operand::Reg32(n, _) => st.set_reg32(*n as usize, t, v),
        Operand::Reg64(n, _) => st.set_reg64(*n as usize, t, v),
        Operand::Tuple(..) => Err("NotImplementedError: set_thread".into()),
        _ => Err(format!("AttributeError: set_thread on {}", op.type_name())),
    }
}

struct Ctx<'a> {
    d: &'a Desc,
    f: &'a Fields,
    ops: &'a [Operand],
}

impl Ctx<'_> {
    fn op(&self, name: &str) -> R<(&OpKind, &Operand)> {
        for (od, o) in self.d.operands.iter().zip(self.ops) {
            if od.name == name {
                return Ok((&od.kind, o));
            }
        }
        Err(format!("KeyError: operand {}", name))
    }
    /// `operands[name].evaluate_thread`: MulSrcDesc sign-extends `.sx` registers.
    fn eval(&self, name: &str, st: &CoreState, t: usize) -> R<i128> {
        let (k, o) = self.op(name)?;
        if let OpKind::Immediate { .. } = k {
            return Ok(fx(self.f, name)? as i128); // ImmediateDesc.evaluate_thread
        }
        let v = get_thread(o, st, t)?;
        if let OpKind::AluSrc { sx: true, .. } = k {
            let v = int(v)?;
            let size = o.bit_size()?;
            if o.flags() & F_SX != 0 {
                return Ok(sign_extend(v, size));
            }
            return Ok(v);
        }
        int(v)
    }
    fn eval_float(&self, name: &str, st: &CoreState, t: usize) -> R<u64> {
        let (_, o) = self.op(name)?;
        let mut r = match o {
            Operand::Imm(ImmVal::Float(x), _) => x.to_bits(),
            Operand::Imm(ImmVal::Int(x), _) => (*x as f64).to_bits(),
            _ => {
                let bits = int(get_thread(o, st, t)?)? as u64;
                match o.bit_size()? {
                    16 => fma::f16_to_f64(bits, false),
                    32 => fma::f32_to_f64(bits, true),
                    _ => return Err("NotImplementedError: 64-bit float source".into()),
                }
            }
        };
        if o.flags() & F_ABS != 0 {
            r &= !(1u64 << 63);
        }
        if o.flags() & F_NEG != 0 {
            r ^= 1u64 << 63;
        }
        Ok(r)
    }
    fn bits(&self, name: &str) -> R<u32> {
        self.op(name)?.1.bit_size()
    }
    fn set(&self, name: &str, st: &mut CoreState, t: usize, v: i128) -> R<()> {
        let (_, o) = self.op(name)?;
        set_thread(o, st, t, v)
    }
    fn icompare(&self, st: &CoreState, t: usize) -> R<bool> {
        let mut a = self.eval("A", st, t)?;
        let mut b = self.eval("B", st, t)?;
        let a_size = self.bits("A")?;
        let b_size = self.bits("B")?;
        let cc = fx(self.f, "cc")?;
        if cc & 0b100 != 0 {
            a = sign_extend(a, a_size);
            b = sign_extend(b, b_size);
        }
        let c = match cc & 0b11 {
            0 => a == b,
            1 => a < b,
            2 => a > b,
            _ => return Err("NotImplementedError: icompare".into()),
        };
        Ok(if fget(self.f, "ccn").unwrap_or(0) != 0 { !c } else { c })
    }
    fn fcompare(&self, st: &CoreState, t: usize) -> R<bool> {
        let a = f64::from_bits(self.eval_float("A", st, t)?);
        let b = f64::from_bits(self.eval_float("B", st, t)?);
        let c = match fx(self.f, "cc")? {
            0b000 => a == b,
            0b001 => a < b,
            0b010 => a > b,
            0b101 => a >= b,
            0b110 => a <= b,
            _ => return Err("NotImplementedError: fcompare".into()),
        };
        Ok(if fget(self.f, "ccn").unwrap_or(0) != 0 { !c } else { c })
    }
    fn compare(&self, float: bool, st: &CoreState, t: usize) -> R<bool> {
        if float { self.fcompare(st, t) } else { self.icompare(st, t) }
    }
    fn set_float_result(&self, st: &mut CoreState, t: usize, mut r: u64) -> R<()> {
        if fx(self.f, "S")? != 0 {
            r = fma::saturate64(r);
        }
        let v = if self.bits("D")? == 16 { fma::f64_to_f16(r, false) } else { fma::f64_to_f32(r, true) };
        self.set("D", st, t, v as i128)
    }
}

#[inline]
/// Shift amount of bfeil: 7 bits, as the recorded full-width form (`bfeil r14, r6, r7, r8`, no mask field) shows.
/// For the form with an immediate mask (what Mesa emits for ubitfield_extract) there is no recording yet;
/// AGX_BFEIL_MASKED_SHIFT32=1 makes it take the shift modulo 32 instead (the alternative to test on the M2:
/// viterbi, shiftfuse and lz4dec decide between them).
pub fn bfeil_shift(s: u32, m: u32) -> u32 {
    static MOD32: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if m != 0 && *MOD32.get_or_init(|| std::env::var("AGX_BFEIL_MASKED_SHIFT32").map_or(false, |v| v == "1")) {
        s & 31
    } else {
        s
    }
}

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

fn exec_thread(c: &Ctx, ex: Ex, st: &mut CoreState, t: usize) -> R<()> {
    let f = c.f;
    match ex {
        Ex::MovImm(name) => {
            let a = fx(f, name)? as i128;
            c.set("D", st, t, a)
        }
        Ex::GetSr => {
            if fx(f, "SR")? == 80 {
                c.set("D", st, t, t as i128)
            } else {
                Err("AssertionError: get_sr TODO".into())
            }
        }
        Ex::IAdd => {
            let a = c.eval("A", st, t)?;
            let mut b = c.eval("B", st, t)?;
            let shift = fx(f, "s")?;
            let (a_size, b_size, d_size) = (c.bits("A")?, c.bits("B")?, c.bits("D")?);
            let sat = fx(f, "S")? != 0 && shift == 0 && a_size <= 32 && b_size <= 32 && d_size <= 32;
            if fx(f, "N")? != 0 {
                b = -b;
            }
            b = if shift < 5 { b << shift } else { 0 };
            let mut result = a + b;
            if sat {
                let signed = fx(f, "As")? != 0 || fx(f, "Bs")? != 0;
                result = saturate(result, d_size, signed);
            }
            c.set("D", st, t, result)
        }
        Ex::IMAdd => {
            let a = c.eval("A", st, t)?;
            let b = c.eval("B", st, t)?;
            let mut cc = c.eval("C", st, t)?;
            let shift = fx(f, "s")?;
            let (_a, _b, c_size, d_size) = (c.bits("A")?, c.bits("B")?, c.bits("C")?, c.bits("D")?);
            let sat = fx(f, "S")? != 0 && shift == 0 && c_size <= 32 && d_size <= 32;
            if fx(f, "N")? != 0 {
                cc = -cc;
            }
            cc = if shift < 5 { cc << shift } else { 0 };
            let mut result = a * b + cc;
            if sat {
                let signed = fx(f, "As")? != 0 || fx(f, "Bs")? != 0 || fx(f, "Cs")? != 0;
                result = saturate(result, d_size, signed);
            }
            c.set("D", st, t, result)
        }
        Ex::Bfi | Ex::Bfeil | Ex::Extr | Ex::Shlhi | Ex::Shrhi => {
            let a = c.eval("A", st, t)? as u128;
            let b = c.eval("B", st, t)? as u128;
            let cv = c.eval("C", st, t)?;
            let m = fx(f, "m")? as u32;
            let s = (cv & 0x7F) as u32;
            let mask: u128 = if m != 0 { (1u128 << m) - 1 } else { 0xFFFFFFFF };
            // big-int expressions evaluated modulo 2^128 (only the low 64 bits reach a register)
            let r = match ex {
                Ex::Bfi => (a & !shl(mask, s)) | shl(b & mask, s),
                Ex::Bfeil => (a & !mask) | (shr(b, bfeil_shift(s, m)) & mask),
                Ex::Extr => shr((b << 32) | a, s) & mask,
                Ex::Shlhi => {
                    let sm = shl(mask, s.saturating_sub(32));
                    let hi = if s >= 32 { shl(b, s - 32) } else { b >> (32 - s) };
                    (hi & sm) | (a & !sm)
                }
                _ => {
                    let sm = (mask << 32) >> s.min(32);
                    (shr(b << 32, s) & sm) | (a & !sm)
                }
            };
            c.set("D", st, t, r as i128)
        }
        Ex::Asr | Ex::Asrh => {
            let a = c.eval("A", st, t)?;
            let b = c.eval("B", st, t)?;
            let a_size = c.bits("A")?;
            let s = (b & 0x7F) as u32;
            let x = sign_extend(a, a_size);
            let x = if ex == Ex::Asrh { x << 32 } else { x };
            c.set("D", st, t, x >> s.min(127))
        }
        Ex::Bitop => {
            let a = c.eval("A", st, t)?;
            let b = c.eval("B", st, t)?;
            let tt = [fx(f, "tt0")?, fx(f, "tt1")?, fx(f, "tt2")?, fx(f, "tt3")?];
            let result = if tt == [0, 0, 1, 1] || tt == [1, 1, 0, 0] {
                a
            } else {
                let mut r = 0i128;
                if tt[0] != 0 {
                    r |= !a & !b;
                }
                if tt[1] != 0 {
                    r |= a & !b;
                }
                if tt[2] != 0 {
                    r |= !a & b;
                }
                if tt[3] != 0 {
                    r |= a & b;
                }
                r
            };
            c.set("D", st, t, result)
        }
        Ex::Bitrev => {
            let a = c.eval("A", st, t)?;
            let mut r = 0i128;
            for i in 0..32 {
                if a & (1 << i) != 0 {
                    r |= 1 << (31 - i);
                }
            }
            c.set("D", st, t, r)
        }
        Ex::Popcount => {
            let a = c.eval("A", st, t)?;
            c.set("D", st, t, (0..32).filter(|i| a & (1 << i) != 0).count() as i128)
        }
        Ex::Ffs => {
            let a = c.eval("A", st, t)?;
            let r = (0..=32).rev().find(|i| a & (1i128 << i) != 0).map(|i| i as i128).unwrap_or(-1);
            c.set("D", st, t, r)
        }
        Ex::FMAdd32 | Ex::FMAdd16 | Ex::FAdd32 | Ex::FAdd16 | Ex::FMul32 | Ex::FMul16 => {
            let a = c.eval_float("A", st, t)?;
            let b = c.eval_float("B", st, t)?;
            let (x, y, z) = match ex {
                Ex::FMAdd32 | Ex::FMAdd16 => (a, b, c.eval_float("C", st, t)?),
                Ex::FAdd32 | Ex::FAdd16 => (a, fma::F64_ONE, b),
                _ => (a, b, 0),
            };
            let mut r = fma::fma_rto_narrow(x, y, z);
            if matches!(ex, Ex::FMAdd32 | Ex::FAdd32 | Ex::FMul32) {
                r = fma::f32_to_f64(fma::f64_to_f32(r, true), false);
            }
            c.set_float_result(st, t, r)
        }
        Ex::ICmpsel | Ex::FCmpsel => {
            let cmp = c.compare(ex == Ex::FCmpsel, st, t)?;
            let v = c.eval(if cmp { "X" } else { "Y" }, st, t)?;
            c.set("D", st, t, v)
        }
        Ex::DeviceLoad => device_load(c, st, t),
        _ => Err("AssertionError: TODO exec_thread".into()),
    }
}

fn device_load(c: &Ctx, st: &mut CoreState, t: usize) -> R<()> {
    let f = c.f;
    let mut address: i128 = if fx(f, "At")? != 0 {
        return Err("TypeError: Uniforms.get_reg64() takes 2 positional arguments".into());
    } else {
        st.get_reg64((fx(f, "A")? >> 1) as usize, t)?
    };
    let mut offset: i128 = if fx(f, "Ot")? != 0 {
        sign_extend(fx(f, "O")? as i128, 16)
    } else {
        let o = st.get_reg32((fx(f, "O")? >> 1) as usize, t)?;
        if fx(f, "Ou")? == 0 { sign_extend(o, 32) } else { o }
    };
    let fmt = fx(f, "F")?;
    let mut shift = fx(f, "s")?;
    if shift == 3 {
        shift = 2;
    }
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
    let mut register = fx(f, "R")? as usize;
    let rt = fx(f, "Rt")?;
    if rt == 1 {
        register >>= 1;
    }
    if rt != 1 && fmt == 2 {
        return Err("AssertionError: illegal instruction".into());
    }
    let mut mask = fx(f, "mask")?;
    if bit_packed {
        address &= !3;
        mask = 0xF;
    }
    let mut values = vec![];
    for i in 0..4 {
        if mask & (1 << i) != 0 {
            let la = address + (offset + i) * item_size;
            if item_size != 4 {
                return Err("AttributeError: _Memory has no get_byte/get_u16".into());
            }
            let mem = st.device_memory.as_ref().ok_or("AttributeError: no device memory")?;
            values.push(mem.get_u32(la)? as i128);
        }
    }
    for v in values {
        if fmt >= 4 {
            return Err("unsupported device_load format in emulator".into());
        }
        if rt == 1 {
            st.set_reg32(register, t, v)?;
        } else {
            st.set_reg16(register, t, v)?;
        }
        register += 1;
    }
    Ok(())
}

fn update_exec(st: &mut CoreState) {
    for t in 0..SIMD_WIDTH {
        st.exec[t] = st.live[t] && st.reg16s[0][t] == 0;
    }
}

/// `desc.exec(n, corestate)` with the descriptor's pre-decoded fields and operands.
pub fn exec(d: &Desc, f: &Fields, ops: &[Operand], st: &mut CoreState) -> R<()> {
    let settles = d.ex == Ex::Wait && fget(f, "i") == Some(0)
        || d.ex == Ex::PopExec && fget(f, "n") == Some(0);
    st.check_overflow_activation(settles).map_err(str::to_string)?;
    let c = Ctx { d, f, ops };
    match d.ex {
        Ex::None => Err(format!("AttributeError: {} has no exec", d.name)),
        Ex::Wait => Ok(()),
        Ex::PopExec => {
            let n = fx(f, "n")? as i128;
            for t in 0..SIMD_WIDTH {
                if !st.live[t] { continue; }
                let v = (st.get_reg16(0, t)? - n).max(0);
                st.set_reg16(0, t, v)?;
            }
            update_exec(st);
            Ok(())
        }
        Ex::IfCmp { float } | Ex::WhileCmp { float } | Ex::ElseCmp { float } => {
            let n = fx(f, "n")? as i128;
            for t in 0..SIMD_WIDTH {
                if !st.live[t] { continue; }
                let mut v = st.get_reg16(0, t)?;
                match d.ex {
                    Ex::IfCmp { .. } => {
                        if v != 0 {
                            v += n;
                        } else if !c.compare(float, st, t)? {
                            v = 1;
                        }
                    }
                    Ex::WhileCmp { .. } => {
                        if v < n {
                            v = if c.compare(float, st, t)? { 0 } else { n };
                        }
                    }
                    _ => {
                        if v == 0 {
                            v = n;
                        } else if v == 1 {
                            v = if c.compare(float, st, t)? { 0 } else { 1 };
                        }
                    }
                }
                if matches!(d.ex, Ex::IfCmp { .. }) && !st.exec[t] && v > 0xFFFF && v & 0xFFFF == 0 {
                    st.overflow_activation_pending = true;
                }
                st.set_reg16(0, t, v)?;
            }
            update_exec(st);
            Ok(())
        }
        Ex::ICmpBallot | Ex::FCmpBallot => {
            let mut result = 0i128;
            for t in 0..SIMD_WIDTH {
                if st.exec[t] && (d.ex == Ex::FCmpBallot || c.icompare(st, t)?) {
                    result |= 1 << t;
                }
            }
            for t in 0..SIMD_WIDTH {
                if st.exec[t] {
                    c.set("D", st, t, result)?;
                }
            }
            Ok(())
        }
        Ex::SimdShuffle => {
            let mut quad_values = [0i128; 8];
            let mut quad_live = [false; 8];
            for (qn, qs) in (0..SIMD_WIDTH).step_by(4).enumerate() {
                let mut qi = 0usize;
                for t in qs..qs + 4 {
                    if st.live[t] { qi |= (c.eval("B", st, t)? & 3) as usize; }
                }
                quad_live[qn] = st.live[qs + qi];
                if quad_live[qn] { quad_values[qn] = c.eval("A", st, qs + qi)?; }
            }
            for t in 0..SIMD_WIDTH {
                if st.exec[t] {
                    let index = c.eval("B", st, t)?;
                    let r = if index < SIMD_WIDTH as i128 && quad_live[(index >> 2) as usize] {
                        quad_values[(index >> 2) as usize]
                    } else { c.eval("A", st, t)? };
                    c.set("D", st, t, r)?;
                }
            }
            Ok(())
        }
        Ex::SimdShuffleDown | Ex::SimdShuffleUp | Ex::SimdShuffleRotateUp | Ex::SimdShuffleXor => {
            let mut av = [0i128; SIMD_WIDTH];
            let mut bv = [0i128; SIMD_WIDTH];
            for t in 0..SIMD_WIDTH {
                if st.live[t] { av[t] = c.eval("A", st, t)?; }
            }
            for t in 0..SIMD_WIDTH {
                if st.live[t] { bv[t] = c.eval("B", st, t)?; }
            }
            let mut quads = [[0i128; 4]; 8];
            let mut quad_live = [[false; 4]; 8];
            for (qn, qs) in (0..SIMD_WIDTH).step_by(4).enumerate() {
                let mut q = 0i128;
                for t in qs..qs + 4 {
                    if st.live[t] { q |= bv[t] & 3; }
                }
                let q = q as usize;
                let lane = |i: usize| match d.ex {
                    Ex::SimdShuffleDown => (i + q) & 3,
                    Ex::SimdShuffleXor => (i ^ q) & 3,
                    _ => (i as isize - q as isize).rem_euclid(4) as usize,
                };
                for i in 0..4 {
                    quads[qn][i] = av[qs + lane(i)];
                    quad_live[qn][i] = st.live[qs + lane(i)];
                }
            }
            for t in 0..SIMD_WIDTH {
                if !st.exec[t] {
                    continue;
                }
                let index = bv[t];
                let ti = t as i128;
                let read = |qn: usize| if quad_live[qn][t & 3] { quads[qn][t & 3] } else { av[t] };
                let r = match d.ex {
                    Ex::SimdShuffleDown => {
                        if index + ti < 32 { read(((index + ti) >> 2) as usize) } else { av[t] }
                    }
                    Ex::SimdShuffleUp => {
                        if ti - index >= 0 { read(((ti - index) >> 2) as usize) } else { av[t] }
                    }
                    Ex::SimdShuffleRotateUp => read(((ti - index).rem_euclid(32) >> 2) as usize),
                    _ => {
                        if (ti ^ index) < 32 { read((((ti ^ index) & 31) >> 2) as usize) } else { av[t] }
                    }
                };
                c.set("D", st, t, r)?;
            }
            Ok(())
        }
        ex => {
            // MaskedInstructionDesc.exec: exec_thread on each active lane in order
            for t in 0..SIMD_WIDTH {
                if st.exec[t] {
                    if matches!(ex, Ex::MaskedTodo | Ex::FMAddLane32 | Ex::FMAddLane16 | Ex::FMAddB52 | Ex::SimdMatrix32 | Ex::SimdMatrix16) {
                        return Err("AssertionError: TODO".into());
                    }
                    exec_thread(&c, ex, st, t)?;
                }
            }
            Ok(())
        }
    }
}
