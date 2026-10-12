//! Functional semantics for the instructions the applegpu-derived emulator lacks or gets wrong, as measured on
//! the M2 Max (playground findings: float semantics, sub-op spaces; the hardware test suite). Used only by the
//! functional executor (`func.rs`); the cost model's trace keeps applegpu's exact behaviour.
//!
//! fp32: denormal inputs read as ±0, a result whose exact value is below the smallest normal becomes ±0 (before
//! rounding), round to nearest even, fused fma, canonical NaN 0x7fc00000. fp16: full IEEE with denormals, fused,
//! canonical NaN 0x7e00. `.sat`: clamp to [0, 1], NaN -> +0.

use crate::isa::exec::{CoreState, SIMD_WIDTH};
use crate::isa::fma::{bfma64, f16_to_f64, f64_to_f16, ROUND_TO_ODD};
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
    f64::from_bits(bfma64(a.to_bits(), b.to_bits(), c.to_bits(), &ROUND_TO_ODD))
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

/// fadd/fmul/fmadd 16/32 (`ops` = D, A, B[, C])
pub fn float_arith(name: &str, f: &Fields, ops: &[Operand], st: &mut CoreState) -> R<()> {
    let half = name.ends_with("16");
    let s = fget(f, "S").unwrap_or(0) != 0;
    for t in 0..SIMD_WIDTH {
        if !st.exec[t] {
            continue;
        }
        let a = float(st, &ops[1], t, !half)?;
        let b = float(st, &ops[2], t, !half)?;
        let mut x = if name.starts_with("fadd") {
            a + b // exact in f64 for fp32/fp16 operands, except far-apart exponents where double rounding is harmless
        } else if name.starts_with("fmul") {
            a * b // exact (48-bit product)
        } else {
            let c = float(st, &ops[3], t, !half)?;
            fma_odd(a, b, c)
        };
        if s {
            x = sat(x);
        }
        let bits = if half || dsize(&ops[0]) == 16 { to_f16(x) } else { to_f32(x) };
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
    let cc = cc_of(f);
    if float_cmp {
        Ok(fcond(cc, float(st, a, t, true)?, float(st, b, t, true)?))
    } else {
        let (x, xs) = int(st, a, t)?;
        let (y, ys) = int(st, b, t)?;
        Ok(icond(cc, x & ((1i128 << xs) - 1), xs, y & ((1i128 << ys) - 1), ys))
    }
}

/// icmpsel / fcmpsel: D = cond(A, B) ? X : Y  (ops: D, A, B, X, Y)
pub fn cmpsel(float_cmp: bool, f: &Fields, ops: &[Operand], st: &mut CoreState) -> R<()> {
    for t in 0..SIMD_WIDTH {
        if !st.exec[t] {
            continue;
        }
        let c = compare(float_cmp, f, &ops[1], &ops[2], st, t)?;
        let (v, _) = raw(st, if c { &ops[3] } else { &ops[4] }, t)?;
        put(st, &ops[0], t, v)?;
    }
    Ok(())
}

/// icmp/fcmp ballot (quad: each lane receives the 4-bit mask of its quad... variant chosen by the suite)
pub fn ballot(float_cmp: bool, quad: bool, f: &Fields, ops: &[Operand], st: &mut CoreState) -> R<()> {
    let mut bits = 0u64;
    for t in 0..SIMD_WIDTH {
        if st.exec[t] && compare(float_cmp, f, &ops[1], &ops[2], st, t)? {
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
    // operands: r0l (destination), A, B
    let (a, b) = (&ops[1], &ops[2]);
    for t in 0..SIMD_WIDTH {
        let mut v = st.get_reg16(0, t)?;
        match kind {
            "if" => {
                if v != 0 {
                    v += n;
                } else if !compare(float_cmp, f, a, b, st, t)? {
                    v = 1;
                }
            }
            "while" => {
                if v < n {
                    v = if compare(float_cmp, f, a, b, st, t)? { 0 } else { n };
                }
            }
            _ => {
                if v == 0 {
                    v = n;
                } else if v == 1 {
                    v = if compare(float_cmp, f, a, b, st, t)? { 0 } else { 1 };
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
