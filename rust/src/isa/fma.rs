//! Port of applegpu's `fma.py` (itself based on Go's src/math/fma.go): bit-exact f16/f32/f64
//! conversions and a 64-bit fused multiply-add with selectable rounding, as used by the emulator.
//!
//! Copyright (c) 2009-2019 The Go Authors; Copyright (c) 2021, Dougall Johnson (BSD-3-Clause).

const F16_BIAS: i64 = 15;
const F16_FRACTION_BITS: u32 = 10;
const F16_FRACTION_MASK: u64 = (1 << F16_FRACTION_BITS) - 1;
const F16_EXPONENT_MASK: i64 = (1 << 5) - 1;
const F16_SIGN_SHIFT: u32 = 15;

const F32_BIAS: i64 = 127;
const F32_FRACTION_BITS: u32 = 23;
const F32_FRACTION_MASK: u64 = (1 << F32_FRACTION_BITS) - 1;
const F32_EXPONENT_MASK: i64 = (1 << 8) - 1;
const F32_SIGN_SHIFT: u32 = 31;

const F64_BIAS: i64 = 1023;
const F64_FRACTION_BITS: u32 = 52;
const F64_FRACTION_MASK: u64 = (1 << F64_FRACTION_BITS) - 1;
const F64_EXPONENT_MASK: i64 = (1 << 11) - 1;
const F64_SIGN_SHIFT: u32 = 63;

pub const F64_ONE: u64 = (F64_BIAS as u64) << F64_FRACTION_BITS;
pub const F64_NAN_BITS: u64 = 0x7FF8000000000000;
pub const F64_INFINITY_BITS: u64 = 0x7FF0000000000000;

const ZERO_EXP: i64 = -(F64_BIAS * 8);
const INF_EXP: i64 = F64_BIAS * 8;

pub type Rounding = [u64; 8];
pub const ROUND_NEAREST_EVEN: Rounding = [0, 0, 0, 1, 0, 0, 1, 1];
pub const ROUND_TO_ODD: Rounding = [0, 1, 1, 1, 0, 0, 0, 0];

pub fn saturate64(bits: u64) -> u64 {
    let s = bits >> F64_SIGN_SHIFT;
    let e = ((bits >> F64_FRACTION_BITS) as i64) & F64_EXPONENT_MASK;
    let f = bits & F64_FRACTION_MASK;
    if s != 0 || (e == F64_EXPONENT_MASK && f != 0) {
        return 0;
    }
    if bits > F64_ONE {
        return F64_ONE;
    }
    bits
}

#[inline]
fn do_rounding(m: u128, mode: &Rounding) -> u128 {
    (m >> 2) + mode[(m & 7) as usize] as u128
}

/// `(v >> shift) | (any shifted-out bit set)`, for any non-negative shift (Python big-int semantics).
#[inline]
fn shr_compress(v: u128, shift: i64) -> u128 {
    if shift <= 0 {
        return v;
    }
    if shift >= 128 {
        return (v != 0) as u128;
    }
    let flag = (v & ((1u128 << shift) - 1) != 0) as u128;
    (v >> shift) | flag
}

pub fn f16_to_f64(bits: u64, ftz: bool) -> u64 {
    let s = bits >> F16_SIGN_SHIFT;
    let mut e = ((bits >> F16_FRACTION_BITS) as i64) & F16_EXPONENT_MASK;
    let mut f = bits & F16_FRACTION_MASK;
    if e == F16_EXPONENT_MASK {
        e = F64_EXPONENT_MASK;
    } else if e == 0 {
        if f == 0 || ftz {
            e = 0;
            f = 0;
        } else {
            while f & (1 << F16_FRACTION_BITS) == 0 {
                f <<= 1;
                e -= 1;
            }
            e += 1;
            f &= F16_FRACTION_MASK;
            e = e - F16_BIAS + F64_BIAS;
        }
    } else {
        e = e - F16_BIAS + F64_BIAS;
    }
    (s << F64_SIGN_SHIFT) | ((e as u64) << F64_FRACTION_BITS) | (f << (F64_FRACTION_BITS - F16_FRACTION_BITS))
}

pub fn f32_to_f64(bits: u64, ftz: bool) -> u64 {
    let ef = (bits >> F32_FRACTION_BITS) & F32_EXPONENT_MASK as u64;
    if ef != 0 && ef != F32_EXPONENT_MASK as u64 {
        // normal: the hardware widening is exact and gives the same bits
        return (f32::from_bits(bits as u32) as f64).to_bits();
    }
    let s = bits >> F32_SIGN_SHIFT;
    let mut e = ((bits >> F32_FRACTION_BITS) as i64) & F32_EXPONENT_MASK;
    let mut f = bits & F32_FRACTION_MASK;
    if e == F32_EXPONENT_MASK {
        e = F64_EXPONENT_MASK;
    } else if e == 0 {
        if f == 0 || ftz {
            e = 0;
            f = 0;
        } else {
            while f & (1 << F32_FRACTION_BITS) == 0 {
                f <<= 1;
                e -= 1;
            }
            e += 1;
            f &= F32_FRACTION_MASK;
            e = e - F32_BIAS + F64_BIAS;
        }
    } else {
        e = e - F32_BIAS + F64_BIAS;
    }
    (s << F64_SIGN_SHIFT) | ((e as u64) << F64_FRACTION_BITS) | (f << (F64_FRACTION_BITS - F32_FRACTION_BITS))
}

fn f64_to_small(bits: u64, ftz: bool, bias: i64, frac_bits: u32, exp_mask: i64, sign_shift: u32) -> u64 {
    let s = (bits >> F64_SIGN_SHIFT) as u128;
    let mut e = ((bits >> F64_FRACTION_BITS) as i64) & F64_EXPONENT_MASK;
    let mut f = (bits & F64_FRACTION_MASK) as u128;
    if e == F64_EXPONENT_MASK {
        e = exp_mask;
    } else if e == 0 && f == 0 {
        e = 0;
    } else {
        e = e - F64_BIAS + bias;
        if ftz {
            if e <= 0 {
                f = 0;
                e = 0;
            }
        } else if e < -(frac_bits as i64) {
            e = 0;
            f = 1;
        } else if e <= 0 {
            f |= 1u128 << F64_FRACTION_BITS;
            f = shr_compress(f, -e + 1);
            e = 0;
        }
        if e >= exp_mask {
            e = exp_mask;
            f = 0;
        }
    }
    let f = shr_compress(f, (F64_FRACTION_BITS - frac_bits - 2) as i64);
    ((s << sign_shift) + ((e as u128) << frac_bits) + do_rounding(f, &ROUND_NEAREST_EVEN)) as u64
}

pub fn f64_to_f32(bits: u64, ftz: bool) -> u64 {
    f64_to_small(bits, ftz, F32_BIAS, F32_FRACTION_BITS, F32_EXPONENT_MASK, F32_SIGN_SHIFT)
}

pub fn f64_to_f16(bits: u64, ftz: bool) -> u64 {
    f64_to_small(bits, ftz, F16_BIAS, F16_FRACTION_BITS, F16_EXPONENT_MASK, F16_SIGN_SHIFT)
}

fn split(b: u64) -> (u64, i64, u128) {
    let sign = b >> 63;
    let exp = ((b >> 52) as i64) & F64_EXPONENT_MASK;
    let mut mantissa = (b & F64_FRACTION_MASK) as u128;
    let e;
    if exp == F64_EXPONENT_MASK {
        e = INF_EXP;
        mantissa |= 1 << 52;
    } else if exp == 0 && mantissa == 0 {
        e = ZERO_EXP;
    } else if exp == 0 {
        let shift = (mantissa as u64).leading_zeros() as i64 - 11;
        mantissa <<= shift;
        e = 1 - shift;
    } else {
        mantissa |= 1 << 52;
        e = exp;
    }
    (sign, e, mantissa)
}

fn is_nan(b: u64) -> bool {
    let m = b & F64_FRACTION_MASK;
    let e = ((b >> F64_FRACTION_BITS) as i64) & F64_EXPONENT_MASK;
    e == F64_EXPONENT_MASK && m != 0
}

fn is_inf(e: i64, m: u128) -> bool {
    e == INF_EXP && (m & ((1 << 52) - 1)) == 0
}

/// applegpu `bfma64` (ARM64_NANS = False).
pub fn bfma64(bx: u64, by: u64, bz: u64, rounding: &Rounding) -> u64 {
    let (xs, xe, xm) = split(bx);
    let (ys, ye, ym) = split(by);
    let (mut zs, mut ze, mut zm) = split(bz);
    for v in [bz, bx, by] {
        if is_nan(v) {
            return F64_NAN_BITS;
        }
    }
    if is_inf(xe, xm) && ym == 0 {
        return F64_NAN_BITS;
    }
    if xm == 0 && is_inf(ye, ym) {
        return F64_NAN_BITS;
    }
    let mut ps = xs ^ ys;
    let mut pe = xe + ye - F64_BIAS + 1;
    let mut pm: u128 = (xm * ym) << 21;
    if (is_inf(xe, xm) || is_inf(ye, ym)) && is_inf(ze, zm) && ps != zs {
        return F64_NAN_BITS;
    }
    zm <<= 74;
    if (xm == 0 || ym == 0) && zm == 0 && zs == ps {
        return zs << 63;
    }
    if (pm >> 126) & 1 == 0 {
        pm <<= 1;
        pe -= 1;
    }
    if pe < ze || (pe == ze && pm < zm) {
        std::mem::swap(&mut ps, &mut zs);
        std::mem::swap(&mut pe, &mut ze);
        std::mem::swap(&mut pm, &mut zm);
    }
    zm = shr_compress(zm, pe - ze);
    let mut m: u128;
    if ps == zs {
        // pm + zm may need 128 bits: both are below 2^127
        pm += zm;
        let carry = (pm >> 127) as i64;
        if carry == 0 {
            pe -= 1;
        }
        m = shr_compress(pm, 64 + carry);
    } else {
        pm -= zm;
        let nz = pm.leading_zeros() as i64;
        pe -= nz;
        let shifted = if nz == 0 { pm >> 1 } else if nz - 1 >= 128 { 0 } else { pm << (nz - 1) };
        // nz == 0 cannot occur (the larger operand is normalised to bit 126)
        m = shr_compress(shifted, 64);
    }
    if pe > 1022 + F64_BIAS || (pe == 1022 + F64_BIAS && (m + (1 << 9)) >> 63 == 1) {
        return (ps << 63) | F64_INFINITY_BITS;
    }
    if pe < 0 {
        m = shr_compress(m, -pe);
        pe = 0;
    }
    if m == 0 {
        ps = 0;
    }
    m = shr_compress(m, 8);
    m = do_rounding(m, rounding);
    if m == 0 {
        pe = 0;
    }
    ((ps as u128) << 63).wrapping_add((pe as u128) << 52).wrapping_add(m) as u64
}

/// `bfma64(x, y, z, ROUND_TO_ODD)` for x and y with at most 24 significant bits (f32 / f16 values widened to f64,
/// or 1.0), on the hardware FPU: the product is then exact in f64, the sum's rounding error comes from TwoSum, and an
/// inexact sum is moved to the odd neighbour, which is the round-to-odd result. Anything outside plain finite non-zero
/// normal values (NaN, infinities, zeros of the product, tiny or huge magnitudes) takes the exact software path.
#[inline]
pub fn fma_rto_narrow(x: u64, y: u64, z: u64) -> u64 {
    const LOW29: u64 = (1 << 29) - 1;
    let (a, b, c) = (f64::from_bits(x), f64::from_bits(y), f64::from_bits(z));
    if (x | y) & LOW29 == 0 {
        let p = a * b;
        let ap = p.abs();
        if ap > 1e-280 && ap < 1e280 {
            if c == 0.0 {
                return p.to_bits();
            }
            let ac = c.abs();
            if ac > 1e-280 && ac < 1e280 {
                let s = p + c;
                if s.abs() > 1e-280 {
                    let bb = s - p;
                    let e = (p - (s - bb)) + (c - bb);
                    let mut bits = s.to_bits();
                    if e != 0.0 && bits & 1 == 0 {
                        if (e > 0.0) == (s > 0.0) {
                            bits += 1;
                        } else {
                            bits -= 1;
                        }
                    }
                    return bits;
                }
            }
        }
    }
    bfma64(x, y, z, &ROUND_TO_ODD)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fma_basics() {
        let one = 1.0f64.to_bits();
        let two = 2.0f64.to_bits();
        assert_eq!(bfma64(one, two, one, &ROUND_NEAREST_EVEN), 3.0f64.to_bits());
        assert_eq!(f64_to_f32(1.5f64.to_bits(), true), 1.5f32.to_bits() as u64);
        assert_eq!(f32_to_f64(1.5f32.to_bits() as u64, true), 1.5f64.to_bits());
        assert_eq!(f64_to_f16(1.5f64.to_bits(), false), 0x3e00);
        assert_eq!(f16_to_f64(0x3e00, false), 1.5f64.to_bits());
    }

    /// the hardware-FPU round-to-odd fma equals the software one on f32 / f16 operands, specials included
    #[test]
    fn fma_rto_narrow_matches() {
        let mut s: u64 = 0x9E3779B97F4A7C15;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let special = [0u32, 0x80000000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000, 0x7fc00000, 0x00000001, 0x007fffff,
            0x00800000, 0x7f7fffff, 0xff7fffff, 0x33800000, 0x4b800000];
        let mut pick = |r: u64| -> u64 {
            let b = match r % 5 {
                0 => special[(r >> 8) as usize % special.len()],
                1 => (r >> 16) as u32 & 0x80ffffff | 0x3f000000, // near 1, both signs
                2 => 0x3f800000 ^ ((r >> 20) as u32 & 0xff), // ulp-level neighbours of 1
                _ => (r >> 13) as u32,
            };
            f32_to_f64(b as u64, r & (1 << 40) != 0)
        };
        for i in 0..3_000_000u64 {
            let (x, y) = (pick(rnd()), pick(rnd()));
            let z = if i % 7 == 0 { 0 } else if i % 11 == 0 { (-(f64::from_bits(x) * f64::from_bits(y))).to_bits() } else { pick(rnd()) };
            let y = if i % 13 == 0 { F64_ONE } else { y };
            assert_eq!(fma_rto_narrow(x, y, z), bfma64(x, y, z, &ROUND_TO_ODD), "{:016x} {:016x} {:016x}", x, y, z);
        }
    }

    /// the fast normal-number widening equals the bitwise one (that is also the reference for the other cases)
    #[test]
    fn f32_to_f64_fast_path() {
        for hi in 0..=0x1ffu64 {
            for lo in [0u64, 1, 0x3ff, 0x400000, 0x7fffff, 0x123457] {
                let bits = hi << 23 | lo;
                let e = (bits >> 23) & 0xff;
                if e == 0 || e == 0xff {
                    continue;
                }
                let s = bits >> 31;
                let slow = (s << 63) | (((e as i64 - 127 + 1023) as u64) << 52) | (lo << 29);
                assert_eq!(f32_to_f64(bits, true), slow);
                assert_eq!(f32_to_f64(bits, false), slow);
            }
        }
    }
}
