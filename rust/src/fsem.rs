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

fn comb(op: XOp, x: Option<u32>, y: Option<u32>) -> Option<u32> {
    match (x, y) {
        (None, v) | (v, None) => v,
        (Some(a), Some(b)) => Some(xapply(op, a, b)),
    }
}

fn canon(op: XOp, v: u32) -> u32 {
    // min / max return an operand's raw bits, NaN included (random/prog_077: all-NaN quad -> 0xffffffff)
    let is_float = matches!(op, XOp::FAdd | XOp::FMul);
    if is_float && f32::from_bits(v).is_nan() { NAN32 } else { v }
}

/// simd_/quad_ reductions and exclusive prefix ops (`ops` = D, A). Inactive lanes take no part.
/// Reductions: pairwise tree over the quad / SIMD group. Quad prefix: lane-order chain. SIMD prefix: per-quad
/// inclusive chain for lanes 0-2 and a tree total for lane 3, then a Kogge-Stone scan over quads (strides 1, 2, 4);
/// lane i receives the inclusive value of lane i-1 (lane 0 and lanes with nothing before them: the identity).
pub fn cross_lane(name: &str, ops: &[Operand], st: &mut CoreState) -> R<()> {
    let quad = name.starts_with("quad_");
    let rest = name.split_once('_').map(|x| x.1).unwrap_or("");
    let (prefix, opname) = match rest.strip_prefix("prefix_") {
        Some(o) => (true, o),
        None => (false, rest),
    };
    let op = xop(opname).ok_or_else(|| format!("{} not modelled", name))?;
    let mut x: [Option<u32>; SIMD_WIDTH] = [None; SIMD_WIDTH];
    for t in 0..SIMD_WIDTH {
        if st.exec[t] {
            let (b, sz) = raw(st, &ops[1], t)?;
            if sz != 32 {
                return Err(format!("{} with a {}-bit source not modelled", name, sz));
            }
            let b = b as u32;
            // fadd / fmul read denormals as signed zero (also when a lane's value passes through unchanged)
            let flush = matches!(op, XOp::FAdd | XOp::FMul) && (b >> 23) & 0xff == 0;
            x[t] = Some(if flush { b & 0x8000_0000 } else { b });
        } else if matches!(op, XOp::MinF | XOp::MaxF) {
            // inactive lanes take part as the canonical NaN (random/prog_106: a lone active NaN comes out canonical,
            // while four active NaNs keep their raw bits, prog_077)
            x[t] = Some(NAN32);
        }
    }
    let w = if quad { 4 } else { 32 };
    let mut out: [Option<u32>; SIMD_WIDTH] = [None; SIMD_WIDTH];
    if !prefix {
        for lo in (0..SIMD_WIDTH).step_by(w) {
            let mut vs: Vec<Option<u32>> = x[lo..lo + w].to_vec();
            while vs.len() > 1 {
                vs = vs.chunks(2).map(|p| comb(op, p[0], p[1])).collect();
            }
            for o in &mut out[lo..lo + w] {
                *o = vs[0];
            }
        }
    } else {
        // inclusive values per lane
        let mut inc: [Option<u32>; SIMD_WIDTH] = [None; SIMD_WIDTH];
        let mut tots = [None; 8];
        for q in 0..8 {
            let a = &x[4 * q..4 * q + 4];
            let s01 = comb(op, a[0], a[1]);
            let v2 = comb(op, s01, a[2]);
            let v3 = if quad { comb(op, v2, a[3]) } else { comb(op, s01, comb(op, a[2], a[3])) };
            inc[4 * q..4 * q + 4].copy_from_slice(&[a[0], s01, v2, v3]);
            tots[q] = v3;
        }
        if !quad {
            let mut d = 1;
            while d < 8 {
                let (pv, py) = (inc, tots);
                for q in d..8 {
                    for i in 0..4 {
                        inc[4 * q + i] = comb(op, py[q - d], pv[4 * q + i]);
                    }
                    tots[q] = comb(op, py[q - d], py[q]);
                }
                d *= 2;
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
                put(st, &ops[0], t, canon(op, v) as u64)?;
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
// fp16 forms are correctly rounded (every sampled lane of the suite). fp32 forms are the hardware's ~26-bit
// approximation, which the suite does not pin down (10-15% of lanes differ by 1-3 ulp from correct rounding);
// they are modelled as correctly rounded unless a recorded table (`transcendental_table`) supplies the exact
// mantissa mapping. Special values, the denormal rules (fp32 inputs flushed to zero, fp16 denormals kept) and
// sin_pt_1 are exact.

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

pub fn transcendental(name: &str, ops: &[Operand], st: &mut CoreState) -> R<()> {
    let d16 = dsize(&ops[0]) == 16;
    for t in 0..SIMD_WIDTH {
        if !st.exec[t] {
            continue;
        }
        let (bits, sz) = raw(st, &ops[1], t)?;
        // fp32 sources: denormals read as zero; fp16 sources keep denormals
        let x = float(st, &ops[1], t, true)?;
        let y = if !d16 && sz == 32 {
            match crate::transtab::lookup(name, bits as u32, &ops[1]) {
                Some(b) => {
                    put(st, &ops[0], t, b as u64)?;
                    continue;
                }
                None => trans_f64(name, x),
            }
        } else {
            trans_f64(name, x)
        };
        put(st, &ops[0], t, if d16 { to_f16(y) } else { to_f32(y) } as u64)?;
    }
    Ok(())
}
