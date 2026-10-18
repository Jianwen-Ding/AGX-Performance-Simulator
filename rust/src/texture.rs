//! Texture-unit conversion rules measured on the M2 Max through Mesa's GLES driver (playground
//! novel-results/2026-10-18-gles-texture). Pure functions only: not yet wired to `texture_load` / `texture_sample`,
//! which also need a texture / sampler descriptor decoder.

/// unorm8 -> float: `k / 255`.
pub fn unorm8(k: u8) -> f32 {
    k as f32 / 255.0
}

/// snorm8 -> float: `max(k / 127, -1)` (-128 and -127 both give -1).
pub fn snorm8(k: i8) -> f32 {
    (k as f32 / 127.0).max(-1.0)
}

/// sRGB8 -> float: the sRGB curve rounded to a 12-bit fraction, `round(srgb(k) * 4095) / 4095`.
pub fn srgb8(k: u8) -> f32 {
    let x = k as f64 / 255.0;
    let s = if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) };
    ((s * 4095.0).round() / 4095.0) as f32
}

/// Bilinear filter of four unorm8 texels with weights `fx`, `fy` in [0, 1): the exact result rounded half-up to
/// 1/16 of an 8-bit step and returned as `n / 4080`.
pub fn bilinear_unorm8(t00: u8, t10: u8, t01: u8, t11: u8, fx: f64, fy: f64) -> f32 {
    let (a, b, c, d) = (t00 as f64, t10 as f64, t01 as f64, t11 as f64);
    let exact = (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (c * (1.0 - fx) + d * fx) * fy;
    ((exact * 16.0 + 0.5).floor() / 4080.0) as f32
}

/// Texel index of an unnormalised coordinate under the three measured address modes.
pub fn wrap_clamp(x: i32, n: i32) -> i32 {
    x.clamp(0, n - 1)
}
pub fn wrap_repeat(x: i32, n: i32) -> i32 {
    x.rem_euclid(n)
}
pub fn wrap_mirror(x: i32, n: i32) -> i32 {
    let p = x.rem_euclid(2 * n);
    if p < n { p } else { 2 * n - 1 - p }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions() {
        assert_eq!(unorm8(255), 1.0);
        assert_eq!(snorm8(-128), -1.0);
        assert_eq!(snorm8(-127), -1.0);
        assert_eq!(snorm8(127), 1.0);
        // measured: k = 1 -> 1/4095, k = 5 -> 6/4095, k = 128 -> 884/4095
        assert_eq!(srgb8(1), 1.0 / 4095.0);
        assert_eq!(srgb8(5), 6.0 / 4095.0);
        assert_eq!(srgb8(128), 884.0 / 4095.0);
        assert_eq!(srgb8(255), 1.0);
    }

    #[test]
    fn bilinear_measured_points() {
        // cell 0 of the hardware run: texels 124 41 39 160, fy = 0.375: a=0 -> 92.125/255, a=1 -> 92.125/255, a=2 -> 92.0625/255
        let f = |a: u32| bilinear_unorm8(124, 41, 39, 160, a as f64 / 256.0, 0.375);
        assert_eq!(f(0), (92.125f64 * 16.0 / 4080.0) as f32);
        assert_eq!(f(1), f(0));
        assert_eq!(f(2), (92.0625f64 * 16.0 / 4080.0) as f32);
    }

    #[test]
    fn wraps() {
        assert_eq!((wrap_clamp(-3, 16), wrap_clamp(20, 16)), (0, 15));
        assert_eq!((wrap_repeat(-1, 16), wrap_repeat(17, 16)), (15, 1));
        assert_eq!((wrap_mirror(-1, 16), wrap_mirror(16, 16), wrap_mirror(31, 16), wrap_mirror(32, 16)), (0, 15, 0, 0));
    }
}
