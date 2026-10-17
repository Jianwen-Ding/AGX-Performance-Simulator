//! Isolate the agx_run memory-dump path; compare byte-wise reads with the bulk path.
//! cargo run --release --example memory_dump -- [MiB] [repeats]
use agx_sim::func::Memory;
use std::{hint::black_box, time::Instant};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mib: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(16);
    let repeats: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(8);
    assert!(mib > 0 && repeats > 0);
    let n = mib * 1024 * 1024;
    let mut m = Memory::default();
    m.map(0x1000, (0..n).map(|i| i as u8).collect());
    let start = Instant::now();
    let mut reference = vec![];
    for _ in 0..repeats {
        reference = (0..n).map(|k| m.read(black_box(0x1000 + k as u64), 1).unwrap() as u8).collect::<Vec<_>>();
        black_box(&reference);
    }
    let scalar = start.elapsed().as_secs_f64();
    let start = Instant::now();
    for _ in 0..repeats {
        let bytes = m.read_bytes(black_box(0x1000), black_box(n)).unwrap();
        assert_eq!(bytes, reference);
        black_box(bytes);
    }
    let bulk = start.elapsed().as_secs_f64();
    println!("bytes={n} repeats={repeats} scalar_ms={:.3} bulk_ms={:.3} speedup={:.2}",
        scalar * 1000.0, bulk * 1000.0, scalar / bulk);
}
