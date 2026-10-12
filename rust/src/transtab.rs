//! Recorded tables for the fp32 transcendentals (exact hardware mantissa mappings).
//!
//! The hardware computes rcp / rsqrt / rsqrt_special / log2 / exp2 / sin_pt_2 in fp32 with a ~26-bit
//! approximation that the test suite's samples cannot pin down. `scripts/record_transcendentals.py` (hardware)
//! records the result for every fp32 mantissa; tables found in `testsuite/transcendental/<op>.bin.gz` (one int8 per
//! input: hardware bits minus the reference `(1/x) as f32` / `(1/sqrt(x)) as f32` computed in f64, in the order
//! the script documents) make these instructions exact. log2 is exact from the raw samples
//! (`samples/log2_1`, `samples/log2_0.5`: a fixed-point e + F(m), see `log2`). exp2 / sin_pt_2 have no exact
//! model; without a table the simulator rounds the exact value correctly.

use std::collections::HashMap;
use std::io::Read;
use std::sync::OnceLock;

/// Inputs covered by a table: rcp: mantissas at one exponent (results scale by 2^-e); rsqrt / rsqrt_special:
/// mantissas at an even and an odd exponent. Other functions are not separable by exponent and have no table yet.
struct Table {
    /// hardware - reference, for x in [1, 2) (rcp) or [1, 4) (rsqrt: 2^24 entries, [1,2) then [2,4))
    delta: Vec<i8>,
}

/// Reference result bits for a table input x (the formula the recording script uses).
fn reference(name: &str, x: f64) -> u32 {
    let y = if name == "rcp" { 1.0 / x } else { 1.0 / x.sqrt() };
    (y as f32).to_bits()
}

/// Hardware result bits for table entry `i` (input bits `xbits`).
fn entry(name: &str, t: &Table, i: usize, xbits: u32) -> Option<u32> {
    let d = *t.delta.get(i)? as i64;
    Some((reference(name, f32::from_bits(xbits) as f64) as i64 + d) as u32)
}

fn tables() -> &'static HashMap<String, Table> {
    static T: OnceLock<HashMap<String, Table>> = OnceLock::new();
    T.get_or_init(|| {
        let mut m = HashMap::new();
        let dir = std::env::var("AGX_TRANS_TABLES")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| crate::cost::repo_root().join("testsuite/transcendental"));
        for op in ["rcp", "rsqrt", "rsqrt_special"] {
            let p = dir.join(format!("{}.bin.gz", op));
            let Ok(f) = std::fs::File::open(&p) else { continue };
            let mut b = vec![];
            if flate2::read::GzDecoder::new(f).read_to_end(&mut b).is_err() {
                continue;
            }
            let delta = b.into_iter().map(|x| x as i8).collect();
            m.insert(op.to_string(), Table { delta });
        }
        m
    })
}

/// Exact fp32 result bits from a recorded table, for normal finite inputs whose result is normal. `b` is the fp32
/// operand the unit sees: source modifiers applied, fp32 denormals already flushed, fp16 sources widened exactly.
pub fn lookup(name: &str, b: u32) -> Option<u32> {
    if name == "log2" {
        return log2(b);
    }
    let t = tables().get(name)?;
    let e = ((b >> 23) & 0xff) as i32;
    if e == 0 || e == 255 {
        return None;
    }
    let sign = b & 0x8000_0000;
    let mant = b & 0x7f_ffff;
    match name {
        "rcp" => {
            // table: x = 1.mant (exponent 127) -> result bits; result exponent shifts by -(e - 127)
            let r = entry(name, t, mant as usize, 127 << 23 | mant)?;
            let re = ((r >> 23) & 0xff) as i32 - (e - 127);
            if !(1..255).contains(&re) || (r >> 23) & 0xff == 0 {
                return None;
            }
            Some(sign | (re as u32) << 23 | (r & 0x7f_ffff))
        }
        "rsqrt" | "rsqrt_special" => {
            if sign != 0 {
                return None;
            }
            // table index: (e odd -> [2,4): 2^23 + mant) since x = 2^(e-127) * 1.mant
            let odd = ((e - 127) & 1) as usize;
            let r = entry(name, t, odd << 23 | mant as usize, (127 + odd as u32) << 23 | mant)?;
            let shift = (e - 127 - odd as i32) / 2;
            let re = ((r >> 23) & 0xff) as i32 - shift;
            if !(1..255).contains(&re) {
                return None;
            }
            Some((re as u32) << 23 | (r & 0x7f_ffff))
        }
        _ => None,
    }
}

/// log2 fixed-point tables from the hardware samples (`samples/log2_1` and `samples/log2_0.5`, raw result bits for
/// x = 1.m and x = 0.5 * 1.m over every mantissa m). The hardware computes log2 as a fixed-point sum
/// e + F(m) with 23 fraction bits (F = the [1,2) sample, exactly; [2,4) results are round(1 + F(m)) for all 2^23
/// mantissas) and truncates that sum to fp32 (toward zero). Negative exponents add one unit (2^-23) to the fraction (the [0.5,1)
/// sample is -1 + F(m) + 2^-23 for all but three mantissas: m = 0 and two with a finer value, taken as recorded),
/// so a negative exponent uses e + 1 + G(m) with G the [0.5,1) sample.
struct Log2 {
    f: Vec<f32>,
    g: Vec<f32>,
}

fn log2_tables() -> Option<&'static Log2> {
    static T: OnceLock<Option<Log2>> = OnceLock::new();
    T.get_or_init(|| {
        let dir = std::env::var("AGX_TRANS_TABLES")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| crate::cost::repo_root().join("testsuite/transcendental"));
        let read = |n: &str| -> Option<Vec<f32>> {
            let f = std::fs::File::open(dir.join("samples").join(n)).ok()?;
            let mut b = vec![];
            flate2::read::GzDecoder::new(f).read_to_end(&mut b).ok()?;
            if b.len() != 4 << 23 {
                return None;
            }
            Some(b.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        };
        Some(Log2 { f: read("log2_1.bin.gz")?, g: read("log2_0.5.bin.gz")? })
    })
    .as_ref()
}

/// Exact fp32 log2 of a positive normal fp32 operand `b` (None: no samples, or not a positive normal).
pub fn log2(b: u32) -> Option<u32> {
    let e = ((b >> 23) & 0xff) as i32;
    if b >> 31 != 0 || e == 0 || e == 255 {
        return None;
    }
    let t = log2_tables()?;
    let m = (b & 0x7f_ffff) as usize;
    let e = e - 127;
    // exact in f64: |e| < 128 and 23-bit (or recorded fp32) fractions
    let v = if e >= 0 { e as f64 + t.f[m] as f64 } else { (e + 1) as f64 + t.g[m] as f64 };
    // the sum is truncated (rounded toward zero) to fp32: every lane of the suite's log2 cases, at all exponents
    let r = v as f32;
    Some(if (r as f64).abs() > v.abs() { r.to_bits() - 1 } else { r.to_bits() })
}
