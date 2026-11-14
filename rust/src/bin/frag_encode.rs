//! `frag_encode TEMPLATE_HEX K=V ...`: re-encode one instruction with some fields replaced (the way `kernels::store` does), for
//! instructions the text assembler cannot emit (zs_emit, sample_mask, ld/st_tile with other registers). Prints the bytes as hex.
//! Example: frag_encode 09110004f0040000 R=16 F=4   (st_tile of r8.. as u8norm)
use agx_sim::isa;
use std::collections::HashMap;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.is_empty() {
        eprintln!("usage: frag_encode TEMPLATE_HEX [FIELD=VALUE ...]");
        std::process::exit(2);
    }
    let bytes: Vec<u8> = (0..a[0].len()).step_by(2).map(|i| u8::from_str_radix(&a[0][i..i + 2], 16).expect("hex")).collect();
    let n = isa::opcode_to_number(&bytes);
    let d = isa::get_instruction_descriptor(n).expect("template does not decode");
    let mut f: HashMap<&'static str, i64> = d.decode_fields(n).into_iter().map(|(k, v)| (k, v as i64)).collect();
    for kv in &a[1..] {
        let (k, v) = kv.split_once('=').expect("FIELD=VALUE");
        let key: &'static str = *f.keys().find(|x| **x == k).unwrap_or_else(|| panic!("{} has no field {} (fields: {:?})", d.name, k, f.keys().collect::<Vec<_>>()));
        let v = if let Some(h) = v.strip_prefix("0x") { i64::from_str_radix(h, 16).unwrap() } else { v.parse().expect("number") };
        f.insert(key, v);
    }
    let x = d.encode_fields(&f).expect("encode");
    let m = isa::get_instruction_descriptor(x).expect("re-decode");
    let back: HashMap<&'static str, i64> = m.decode_fields(x).into_iter().map(|(k, v)| (k, v as i64)).collect();
    assert!(std::ptr::eq(m, d) && back == f, "encoding does not round-trip");
    let size = m.decode_size(x);
    println!("{}", (0..size).map(|i| format!("{:02x}", (x >> (8 * i)) as u8)).collect::<String>());
    eprintln!("{} {:?}", d.name, back);
}
