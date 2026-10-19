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

// ---------------------------------------------------------------------------------------------------------------
// Texture / sampler descriptor decode (Mesa cmdbuf.xml `Texture` / `Sampler` structs) and sampling.
// Measured semantics: playground novel-results/2026-10-18-gles-texture, 2026-10-19-texture-edge.

/// Decoded 24-byte `agx_texture_packed` descriptor (linear layouts only for addressing).
#[derive(Debug, Clone, PartialEq)]
pub struct TexDesc {
    pub dim: u32,
    pub channels: u32,
    pub type_: u32,
    pub swizzle: [u32; 4],
    pub width: u32,
    pub height: u32,
    pub first_level: u32,
    pub last_level: u32,
    pub addr: u64,
    pub stride: u64,
    pub srgb: bool,
    pub layout: u32,
}

fn bits(b: &[u8], start: u32, size: u32) -> u64 {
    let mut v = 0u64;
    for i in 0..size {
        let bit = start + i;
        v |= (((b[(bit / 8) as usize] >> (bit % 8)) & 1) as u64) << i;
    }
    v
}

/// Decode a 24-byte texture descriptor. Errors on compressed / tiled / multisample layouts
/// (not needed for the CUDA-relevant paths); multi-channel formats beyond the table below
/// are reported, not guessed.
pub fn decode_texdesc(b: &[u8]) -> Result<TexDesc, String> {
    if b.len() < 24 {
        return Err(format!("texture descriptor holds {} bytes, need 24", b.len()));
    }
    let layout = bits(b, 4, 2) as u32;
    if layout != 0 {
        return Err(format!("non-linear texture layout {} not modelled", layout));
    }
    Ok(TexDesc {
        dim: bits(b, 0, 4) as u32,
        channels: bits(b, 6, 7) as u32,
        type_: bits(b, 13, 3) as u32,
        swizzle: [bits(b, 16, 3) as u32, bits(b, 19, 3) as u32, bits(b, 22, 3) as u32, bits(b, 25, 3) as u32],
        width: bits(b, 28, 14) as u32 + 1,
        height: bits(b, 42, 14) as u32 + 1,
        first_level: bits(b, 56, 4) as u32,
        last_level: bits(b, 60, 4) as u32,
        addr: bits(b, 66, 36) << 4,
        stride: bits(b, 110, 18), // raw field: row pitch is (stride + 1) * 16, see tex_stride
        srgb: bits(b, 108, 1) != 0,
        layout,
    })
}

/// Decoded 8-byte sampler descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct SampDesc {
    pub min_filter: u32,
    pub mag_filter: u32,
    pub mip_filter: u32,
    pub wrap: [u32; 3],
    pub border: u32,
}

pub fn decode_sampler(b: &[u8]) -> Result<SampDesc, String> {
    if b.len() < 8 {
        return Err(format!("sampler descriptor holds {} bytes, need 8", b.len()));
    }
    Ok(SampDesc {
        min_filter: bits(b, 25, 2) as u32,
        mag_filter: bits(b, 23, 2) as u32,
        mip_filter: bits(b, 27, 2) as u32,
        wrap: [bits(b, 29, 3) as u32, bits(b, 32, 3) as u32, bits(b, 35, 3) as u32],
        border: bits(b, 55, 2) as u32,
    })
}

/// (bytes per texel, kind) for the CUDA-relevant formats. kind: 0 = unorm, 1 = snorm, 2 = uint,
/// 3 = sint, 4 = float. Channels enum values from Mesa cmdbuf.xml.
pub fn tex_format(channels: u32, type_: u32) -> Result<(usize, u32), String> {
    let bpp = match channels {
        0x00 => 1, // R8
        0x09 => 2, // R16
        0x21 => 4, // R32
        0x23 => 4, // R16G16
        0x28 => 4, // R8G8B8A8
        0x32 => 8, // R16G16B16A16
        0x38 => 16, // R32G32B32A32
        _ => return Err(format!("channels {:#x} not modelled", channels)),
    };
    let _ = type_;
    Ok((bpp, type_))
}

/// Byte offset of texel (x, y) relative to the image base, or None if out of range.
/// The row pitch is (stride field + 1) * 16 (Mesa writes pitch - 16, shr 4).
pub fn texel_off(desc: &TexDesc, bpp: usize, x: i64, y: i64) -> Option<u64> {
    if !(0 <= x && x < desc.width as i64 && 0 <= y && y < desc.height as i64) {
        return None;
    }
    Some(y as u64 * tex_stride(desc) + x as u64 * bpp as u64)
}

pub fn apply_swizzle(v: [f32; 4], s: [u32; 4]) -> [f32; 4] {
    let mut o = [0.0; 4];
    for i in 0..4 {
        o[i] = match s[i] {
            0 => v[0],
            1 => v[1],
            2 => v[2],
            3 => v[3],
            4 => 1.0,
            _ => 0.0,
        };
    }
    o
}

fn f32_le(b: &[u8]) -> f32 {
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
fn u16_le(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}
fn i16_le(b: &[u8]) -> i16 {
    i16::from_le_bytes([b[0], b[1]])
}

/// Convert one texel to float (sample path): unorm / snorm / float per the measured rules;
/// sRGB decodes with the 12-bit table. Integer formats must use `texel_int`.
pub fn texel_float(raw: &[u8], channels: u32, type_: u32, srgb: bool) -> Result<[f32; 4], String> {
    // component count and element size from the channel format
    let (comps, esize) = match channels {
        0x00 => (1, 1), // R8
        0x09 => (1, 2), // R16
        0x21 => (1, 4), // R32
        0x23 => (2, 2), // R16G16
        0x28 => (4, 1), // R8G8B8A8
        0x32 => (4, 2), // R16G16B16A16
        0x38 => (4, 4), // R32G32B32A32
        _ => return Err(format!("channels {:#x} not modelled", channels)),
    };
    let mut v = [0.0f32; 4];
    for i in 0..comps {
        let b = &raw[i * esize..(i + 1) * esize];
        v[i] = match (type_, esize) {
            (0, 1) => unorm8(b[0]),
            (1, 1) => snorm8(b[0] as i8),
            (0, 2) => u16_le(b) as f32 / 65535.0,
            (1, 2) => (i16_le(b) as f32 / 32767.0).max(-1.0),
            (4, 2) => f16_to_f32(u16_le(b)),
            (4, 4) => f32_le(b),
            _ => return Err(format!("type {} with {}-byte elements not modelled", type_, esize)),
        };
    }
    if srgb {
        if type_ != 0 || (channels != 0x28 && channels != 0x00) {
            return Err("sRGB only modelled for unorm8".into());
        }
        for i in 0..comps {
            v[i] = srgb8((v[i] * 255.0).round().clamp(0.0, 255.0) as u8);
        }
    }
    Ok(v)
}

pub fn f16_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32 / 1024.0;
    if e == 0 {
        s * m * 2f32.powi(-14)
    } else if e == 31 {
        if m == 0.0 { s * f32::INFINITY } else { f32::NAN }
    } else {
        s * (1.0 + m) * 2f32.powi(e - 15)
    }
}

/// Convert one texel to integers (image / texelFetch integer path): zero- or sign-extended.
pub fn texel_int(raw: &[u8], channels: u32, type_: u32) -> Result<[u32; 4], String> {
    let (comps, esize) = match channels {
        0x00 => (1, 1),
        0x09 => (1, 2),
        0x21 => (1, 4),
        0x23 => (2, 2),
        0x28 => (4, 1),
        0x32 => (4, 2),
        0x38 => (4, 4),
        _ => return Err(format!("channels {:#x} not modelled", channels)),
    };
    if type_ != 2 && type_ != 3 {
        return Err(format!("integer fetch needs uint/sint type, got {}", type_));
    }
    let mut v = [0u32; 4];
    for i in 0..comps {
        let b = &raw[i * esize..(i + 1) * esize];
        v[i] = match (type_, esize) {
            (2, 1) => b[0] as u32,
            (3, 1) => (b[0] as i8) as i32 as u32,
            (2, 2) => u16_le(b) as u32,
            (3, 2) => (i16_le(b) as i32) as u32,
            (2, 4) => u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            (3, 4) => (i32::from_le_bytes([b[0], b[1], b[2], b[3]])) as u32,
            _ => return Err("unreachable".into()),
        };
    }
    Ok(v)
}

/// Wrap mode application for normalized coordinates. Returns None for modes that need a
/// border colour (2-bit Border colour 3 = custom) or are otherwise unhandled.
pub fn apply_wrap(mode: u32, x: i32, n: i32) -> Option<i32> {
    if n <= 0 {
        return None;
    }
    match mode {
        0 => Some(wrap_clamp(x, n)),
        1 => Some(wrap_repeat(x, n)),
        2 => Some(wrap_mirror(x, n)),
        // 3 = clamp-to-border, 4 = GL clamp, 5 = mirrored clamp: need the border colour
        _ => None,
    }
}

/// Nearest-neighbour sample of texel (x, y) with wrapping; out of range without wrapping
/// (texelFetch) reads zero (measured).
pub fn fetch_texel(
    desc: &TexDesc,
    texels: &[u8],
    bpp: usize,
    x: i64,
    y: i64,
    wrap: Option<(u32, u32)>,
) -> Option<[f32; 4]> {
    let (xi, yi) = match wrap {
        Some((wx, wy)) => (apply_wrap(wx, x as i32, desc.width as i32)? as i64,
                           apply_wrap(wy, y as i32, desc.height as i32)? as i64),
        None => {
            if x < 0 || y < 0 || x >= desc.width as i64 || y >= desc.height as i64 {
                return None;
            }
            (x, y)
        }
    };
    let off = texel_off(desc, bpp, xi, yi)? as usize;
    let raw = texels.get(off..off + bpp)?;
    let mut v = texel_float(raw, desc.channels, desc.type_, desc.srgb).ok()?;
    v = apply_swizzle(v, desc.swizzle);
    Some(v)
}

/// Bilinear sample at normalized (u, v) with sampler wrap modes; the unorm path uses the
/// measured 12-bit rounding, other formats filter in float32.
pub fn sample_linear(
    desc: &TexDesc,
    samp: &SampDesc,
    texels: &[u8],
    bpp: usize,
    u: f32,
    v: f32,
) -> Result<[f32; 4], String> {
    let w = desc.width as f64;
    let h = desc.height as f64;
    let x = u as f64 * w - 0.5;
    let y = v as f64 * h - 0.5;
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    let wx = samp.wrap[0];
    let wy = samp.wrap[1];
    let t = |dx: i32, dy: i32| -> Result<[f32; 4], String> {
        let xi = apply_wrap(wx, x0 + dx, desc.width as i32)
            .ok_or_else(|| format!("wrap mode {} not modelled", wx))? as i64;
        let yi = apply_wrap(wy, y0 + dy, desc.height as i32)
            .ok_or_else(|| format!("wrap mode {} not modelled", wy))? as i64;
        let off = texel_off(desc, bpp, xi, yi).ok_or("texel out of range")? as usize;
        let raw = texels.get(off..off + bpp).ok_or("texel out of image")?;
        let c = texel_float(raw, desc.channels, desc.type_, desc.srgb)?;
        Ok(apply_swizzle(c, desc.swizzle))
    };
    let c00 = t(0, 0)?;
    let c10 = t(1, 0)?;
    let c01 = t(0, 1)?;
    let c11 = t(1, 1)?;
    // unorm8 path: exact measured rule (12-bit grid, half-up on 1/16 steps)
    if desc.channels == 0x28 && desc.type_ == 0 && !desc.srgb {
        let ch = |c: [f32; 4], i: usize| (c[i] * 255.0).round() as u8;
        let mut out = [0.0f32; 4];
        for i in 0..4 {
            out[i] = bilinear_unorm8(ch(c00, i), ch(c10, i), ch(c01, i), ch(c11, i), fx, fy);
        }
        return Ok(out);
    }
    let mut out = [0.0f32; 4];
    for i in 0..4 {
        let exact = (c00[i] as f64 * (1.0 - fx) + c10[i] as f64 * fx) * (1.0 - fy)
            + (c01[i] as f64 * (1.0 - fx) + c11[i] as f64 * fy) * fy;
        out[i] = exact as f32;
    }
    Ok(out)
}

pub fn apply_swizzle_u32(v: [u32; 4], s: [u32; 4]) -> [u32; 4] {
    let mut o = [0u32; 4];
    for i in 0..4 {
        o[i] = match s[i] {
            0 => v[0],
            1 => v[1],
            2 => v[2],
            3 => v[3],
            4 => 1,
            _ => 0,
        };
    }
    o
}

/// Decoded 24-byte `agx_pbe_packed` descriptor (image-store path): same addressing info as
/// `TexDesc` at shifted positions; swizzles are 2 bits (R/G/B/A only). Stride is a raw byte
/// count with Mesa's -16 bias (hk_image_view.c: the driver writes linear pitch minus 16),
/// so the row pitch is field + 16.
#[derive(Debug, Clone, PartialEq)]
pub struct PbeDesc {
    pub width: u32,
    pub height: u32,
    pub addr: u64,
    pub stride: u64,
    pub channels: u32,
    pub type_: u32,
    pub swizzle: [u32; 4],
    pub srgb: bool,
}

pub fn decode_pbe(b: &[u8]) -> Result<PbeDesc, String> {
    if b.len() < 24 {
        return Err(format!("pbe descriptor holds {} bytes, need 24", b.len()));
    }
    if bits(b, 4, 2) != 0 {
        return Err("non-linear pbe layout not modelled".into());
    }
    if bits(b, 59, 1) != 0 {
        return Err("compressed pbe not modelled".into());
    }
    Ok(PbeDesc {
        width: bits(b, 24, 14) as u32 + 1,
        height: bits(b, 38, 14) as u32 + 1,
        addr: bits(b, 64, 36) << 4,
        stride: bits(b, 104, 21) + 16,
        channels: bits(b, 6, 7) as u32,
        type_: bits(b, 13, 3) as u32,
        swizzle: [bits(b, 16, 2) as u32, bits(b, 18, 2) as u32, bits(b, 20, 2) as u32, bits(b, 22, 2) as u32],
        srgb: bits(b, 125, 1) != 0,
    })
}

/// Row pitch of a linear texture descriptor: Mesa writes (pitch - 16) >> 4 (hk_image_view.c),
/// so hardware pitch is field * 16 + 16. Self-consistent with synthesis that writes pitch/16 - 1.
pub fn tex_stride(desc: &TexDesc) -> u64 {
    (desc.stride + 1) * 16
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
    fn texdesc_roundtrip() {
        // hand-packed descriptor: 2D linear R32F 64x32 at VA 0x123456000, stride 256, identity swizzle
        let mut b = [0u8; 24];
        let mut put = |start: u32, size: u32, mut v: u64| {
            for i in 0..size {
                if (v & 1) != 0 {
                    b[((start + i) / 8) as usize] |= 1 << ((start + i) % 8);
                }
                v >>= 1;
            }
        };
        put(0, 4, 2); // dim 2D
        put(4, 2, 0); // linear
        put(6, 7, 0x21); // R32
        put(13, 3, 4); // float
        put(16, 3, 0); put(19, 3, 1); put(22, 3, 2); put(25, 3, 3);
        put(28, 14, 64 - 1);
        put(42, 14, 32 - 1);
        put(66, 36, 0x123456000 >> 4);
        put(110, 18, (256 - 16) >> 4); // Mesa writes linear pitch minus 16
        let d = decode_texdesc(&b).unwrap();
        assert_eq!((d.dim, d.channels, d.type_, d.width, d.height), (2, 0x21, 4, 64, 32));
        assert_eq!((d.addr, d.stride), (0x123456000, 15));
        assert_eq!(texel_off(&d, 4, 0, 0), Some(0));
        assert_eq!(texel_off(&d, 4, 63, 31), Some(31 * 256 + 63 * 4));
        assert_eq!(texel_off(&d, 4, 64, 0), None);
        assert_eq!(texel_float(&[0x00, 0x00, 0x80, 0x3f], 0x21, 4, false).unwrap(), [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(texel_int(&[0xff, 0xff, 0xff, 0xff], 0x21, 2).unwrap(), [0xffffffff, 0, 0, 0]);
    }

    #[test]
    fn wraps() {
        assert_eq!((wrap_clamp(-3, 16), wrap_clamp(20, 16)), (0, 15));
        assert_eq!((wrap_repeat(-1, 16), wrap_repeat(17, 16)), (15, 1));
        assert_eq!((wrap_mirror(-1, 16), wrap_mirror(16, 16), wrap_mirror(31, 16), wrap_mirror(32, 16)), (0, 15, 0, 0));
    }
}
