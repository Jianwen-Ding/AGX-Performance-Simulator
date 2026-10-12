//! AGX performance simulator: a Rust port of `scripts/agx_cost.py` (cycle cost model) together
//! with the applegpu ISA layer it depends on and the kernel generators used by the offline verifier.

pub mod cost;
pub mod dispatch;
pub mod func;
pub mod fsem;
pub mod isa;
pub mod kernels;
pub mod pyrandom;

/// Python `round(x, nd)` followed by `repr` (as `json.dumps` prints it).
pub fn py_round_repr(x: f64, nd: usize) -> String {
    let s = format!("{:.*}", nd, x);
    let v: f64 = s.parse().unwrap();
    isa::operand::py_float(v)
}
