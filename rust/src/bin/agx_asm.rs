//! `agx_asm`: assemble AGX instructions with the applegpu-compatible assembler (offline; no applegpu needed).
//!
//! Reads one request per line on stdin and prints one line per request:
//!   `TEXT` or `TEXT\tWANT`  assemble (WANT = expected disassembly, as agx_loop.encode(text, want)) -> hex or `ERROR msg`
//!   `@F=V,F=V\tTEXT`        assemble TEXT, then set decode fields F (or raw bits `bSTART_SIZE`) to V (raw encodings the assembler has no
//!                           syntax for, e.g. condition codes without a name) -> `hex\tdisassembly` or `ERROR msg`
//!                           (TEXT may be `!HEX`: patch an existing encoding)
//!   `?TEXT`                 assemble (or `?!HEX` decode the first instruction) -> `name\tsize\tdisassembly\tF=V F=V ...`
//!   `!HEX`                  disassemble a whole body -> `name size text | name size text | ...` or `ERROR at <offset>: msg`
//! Used by `scripts/coverage.py validate` to check the coverage catalog without the GPU or applegpu.

use agx_sim::isa;
use std::collections::HashMap;
use std::io::{BufRead, Write};

fn unhex(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err("odd hex length".into());
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| e.to_string())).collect()
}

fn disassemble(code: &[u8]) -> Result<String, String> {
    let mut parts = vec![];
    let mut p = 0;
    while p < code.len() {
        let mut buf = code[p..].to_vec();
        buf.resize(buf.len().max(16), 0);
        let (d, n, size) = isa::decode_one(&buf).ok_or_else(|| format!("at {}: undecodable {}", p, hexs(&code[p..(p + 8).min(code.len())])))?;
        if size == 0 || p + size > code.len() {
            return Err(format!("at {}: {} truncated", p, d.name));
        }
        let text = isa::disassemble_normalized(d, n).unwrap_or_else(|e| format!("<{}>", e));
        parts.push(format!("{} {} {}", d.name, size, text));
        p += size;
    }
    Ok(parts.join(" | "))
}

fn decoded(text: &str) -> Result<(&'static isa::Desc, u128), String> {
    let b = match text.strip_prefix('!') {
        Some(h) => { let mut b = unhex(h.trim())?; b.resize(16, 0); b }
        None => isa::assemble_line(text)?,
    };
    let n = isa::opcode_to_number(&b);
    let d = isa::get_instruction_descriptor(n).ok_or_else(|| format!("inconsistent assembly: {}", text))?;
    Ok((d, d.mask_instr(n)))
}

fn fields(text: &str) -> Result<String, String> {
    let (d, n) = decoded(text)?;
    let f: Vec<String> = d.decode_fields(n).iter().map(|(k, v)| format!("{}={}", k, v)).collect();
    let dis = isa::disassemble_normalized(d, n).unwrap_or_else(|e| format!("<{}>", e));
    Ok(format!("{}\t{}\t{}\t{}", d.name, d.decode_size(n), dis, f.join(" ")))
}

/// Field-edit an assembled instruction. Keys are decode fields (the result must decode as the same instruction with
/// exactly those fields) or raw bit ranges `bSTART_SIZE` (any decode accepted; the disassembly shows what it is).
fn patch(spec: &str, text: &str) -> Result<String, String> {
    let (d, n) = decoded(text)?;
    let mut f: HashMap<&'static str, i64> = d.decode_fields(n).into_iter().map(|(k, v)| (k, v as i64)).collect();
    let mut raw: Vec<(u32, u32, u128)> = vec![];
    for kv in spec.split(',') {
        let (k, v) = kv.split_once('=').ok_or_else(|| format!("bad field spec {:?}", kv))?;
        let v = if let Some(h) = v.strip_prefix("0x") { i64::from_str_radix(h, 16) } else if let Some(b) = v.strip_prefix("0b") { i64::from_str_radix(b, 2) } else { v.parse() }
            .map_err(|e| format!("{}: {}", kv, e))?;
        if let Some(r) = k.strip_prefix('b').and_then(|r| r.split_once('_')) {
            let (st, sz) = (r.0.parse::<u32>().map_err(|e| e.to_string())?, r.1.parse::<u32>().map_err(|e| e.to_string())?);
            raw.push((st, sz, v as u128));
            continue;
        }
        let key = *f.keys().find(|x| **x == k).ok_or_else(|| format!("{} has no field {:?}", d.name, k))?;
        f.insert(key, v);
    }
    let mut x = d.patch_fields(n, &f)?;
    for &(st, sz, v) in &raw {
        let m = ((1u128 << sz) - 1) << st;
        x = (x & !m) | ((v << st) & m);
    }
    if !raw.is_empty() && d.sizes.0 != d.sizes.1 {
        // like patch_fields: the long form when any bit beyond the short form is set
        x &= !(1u128 << d.length_bit_pos);
        if x >> (8 * d.sizes.0) != 0 {
            x |= 1u128 << d.length_bit_pos;
        }
    }
    let m = isa::get_instruction_descriptor(x).ok_or("patched encoding does not decode")?;
    if raw.is_empty() {
        if m.name != d.name {
            return Err(format!("patched encoding decodes as {}", m.name));
        }
        let back: HashMap<&'static str, i64> = m.decode_fields(x).into_iter().map(|(k, v)| (k, v as i64)).collect();
        if back != f {
            return Err("patched fields do not round-trip".into());
        }
    }
    let x = m.mask_instr(x);
    let text = isa::disassemble_normalized(m, x).unwrap_or_else(|e| format!("<{}>", e));
    Ok(format!("{}\t{}", hexs(&m.to_bytes(x)), text))
}

fn hexs(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout());
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let res = if let Some(h) = line.strip_prefix('!') {
            unhex(h.trim()).and_then(|b| disassemble(&b))
        } else if let Some(t) = line.strip_prefix('?') {
            fields(t)
        } else if let Some(r) = line.strip_prefix('@') {
            match r.split_once('\t') {
                Some((f, t)) => patch(f, t),
                None => Err("expected @F=V,...<TAB>TEXT".into()),
            }
        } else {
            let (text, want) = match line.split_once('\t') {
                Some((t, w)) => (t, Some(w)),
                None => (line.as_str(), None),
            };
            isa::encode(text, want).map(|b| hexs(&b))
        };
        match res {
            Ok(s) => writeln!(out, "{}", s),
            Err(e) => writeln!(out, "ERROR {}", e.replace('\n', " ")),
        }
        .unwrap();
        out.flush().unwrap();
    }
}
