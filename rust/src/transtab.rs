//! Optional recorded tables for the fp32 transcendentals (exact hardware mantissa mappings).
//!
//! The hardware computes rcp / rsqrt / rsqrt_special / log2 / exp2 / sin_pt_2 in fp32 with a ~26-bit
//! approximation that the test suite's samples cannot pin down. `scripts/record_transcendentals.py` (hardware)
//! records the result for every fp32 mantissa; tables found in `testsuite/transcendental/<op>.bin.gz` (one int8 per
//! input: hardware bits minus the reference `(1/x) as f32` / `(1/sqrt(x)) as f32` computed in f64, in the order
//! the script documents) make these instructions exact. Without a table the simulator rounds the exact value
//! correctly.

use std::collections::HashMap;
use std::io::Read;
use std::sync::OnceLock;

use crate::isa::operand::{Operand, F_ABS, F_NEG};

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

/// Exact fp32 result bits from a recorded table, for normal finite inputs whose result is normal.
pub fn lookup(name: &str, bits: u32, src: &Operand) -> Option<u32> {
    let t = tables().get(name)?;
    let mut b = bits;
    if src.flags() & F_ABS != 0 {
        b &= 0x7fff_ffff;
    }
    if src.flags() & F_NEG != 0 {
        b ^= 0x8000_0000;
    }
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
