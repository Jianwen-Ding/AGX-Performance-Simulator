//! AGX (G13/G14) instruction set: a Rust port of the parts of dougallj/applegpu (BSD-3-Clause,
//! revision 4c5bae6) that the cost model uses — decoding, disassembly, field encoding, the
//! single-line assembler and the per-instruction emulator.

pub mod desc;
pub mod exec;
pub mod fma;
pub mod operand;
pub mod table;

pub use desc::{fget, fx, Fields};
pub use operand::{py_str, ImmVal, Operand};
use std::collections::HashMap;
use std::sync::OnceLock;
pub use table::{Desc, Ex, Mn};

pub const MAX_OPCODE_LEN: usize = 12;

pub fn descriptors() -> &'static [Desc] {
    static D: OnceLock<Vec<Desc>> = OnceLock::new();
    D.get_or_init(table::build)
}

/// applegpu `opcode_to_number`: up to 12 bytes, little endian.
pub fn opcode_to_number(b: &[u8]) -> u128 {
    let mut n = 0u128;
    for (i, &c) in b.iter().take(MAX_OPCODE_LEN).enumerate() {
        n |= (c as u128) << (8 * i);
    }
    n
}

impl Desc {
    pub fn decode_size(&self, n: u128) -> usize {
        if (n >> self.length_bit_pos) & 1 != 0 { self.sizes.1 } else { self.sizes.0 }
    }
    pub fn mask_instr(&self, n: u128) -> u128 {
        let bytes = self.decode_size(n);
        if bytes >= 16 { n } else { n & ((1u128 << (8 * bytes)) - 1) }
    }
    pub fn matches(&self, n: u128) -> bool {
        self.mask_instr(n) & self.mask == self.bits
    }
    pub fn decode_fields(&self, n: u128) -> Fields {
        let n = self.mask_instr(n);
        self.merged
            .iter()
            .map(|(name, subs)| {
                let mut v = 0u64;
                for &(start, size, shift) in subs {
                    v |= (((n >> start) & ((1u128 << size) - 1)) as u64) << shift;
                }
                (*name, v)
            })
            .collect()
    }
    pub fn fields_to_operands(&self, f: &Fields) -> Result<Vec<Operand>, String> {
        self.operands.iter().map(|o| o.decode(f)).collect()
    }
    pub fn decode_operands(&self, n: u128) -> Result<Vec<Operand>, String> {
        self.fields_to_operands(&self.decode_fields(n))
    }
    pub fn fields_to_mnem(&self, f: &Fields) -> String {
        match self.mn {
            Mn::Sat { alt } => {
                let base = match alt {
                    Some(a) if fget(f, "N").unwrap_or(0) != 0 => a,
                    _ => self.name,
                };
                if fget(f, "S").unwrap_or(0) != 0 { format!("{}.sat", base) } else { base.to_string() }
            }
            _ => self.name.to_string(),
        }
    }

    fn map_to_alias(&self, mnem: String, ops: Vec<Operand>) -> Result<(String, Vec<Operand>), String> {
        let s = |o: &Operand| py_str(o);
        Ok(match self.mn {
            Mn::Bitop => {
                let tt = s(&ops[0])?;
                const BIN: &[(&str, &str)] =
                    &[("0001", "and"), ("0111", "or"), ("0110", "xor"), ("1110", "nand"), ("1000", "nor"), ("1001", "xnor")];
                const UN: &[(&str, &str)] = &[("1010", "not"), ("0101", "mov")];
                if let Some((_, a)) = BIN.iter().find(|(k, _)| *k == tt) {
                    return Ok((a.to_string(), ops[1..].to_vec()));
                }
                if s(&ops[3])? == "0" {
                    if let Some((_, a)) = UN.iter().find(|(k, _)| *k == tt) {
                        return Ok((a.to_string(), ops[1..3].to_vec()));
                    }
                }
                if tt == "0011" || tt == "1100" {
                    return Ok(("bitop_mov_a".into(), ops));
                }
                (mnem, ops)
            }
            Mn::IfFcmp | Mn::ElseIcmp => {
                if s(&ops[1])? == "eq" && s(&ops[2])? == "0.0" && s(&ops[3])? == "0.0" {
                    if self.mn == Mn::ElseIcmp {
                        return Ok(("else_exec".into(), vec![ops[0].clone(), ops[4].clone()]));
                    }
                    if s(&ops[4])? == "0" {
                        return Ok(("update_exec".into(), vec![ops[0].clone()]));
                    }
                    return Ok(("push_exec".into(), vec![ops[0].clone(), ops[4].clone()]));
                }
                (mnem, ops)
            }
            Mn::LdstTile => {
                let load = matches!(ops[0], Operand::Int(x) if x != 0);
                ((if load { "ld_tile" } else { "st_tile" }).into(), ops[1..].to_vec())
            }
            Mn::Map => match &ops[0] {
                Operand::Str(x) if x == "unmap" => ("unmap".into(), ops[1..].to_vec()),
                Operand::Str(x) if x == "map" => ("map".into(), ops[1..].to_vec()),
                _ => ("TODO.map".into(), ops),
            },
            _ => (mnem, ops),
        })
    }

    /// `str(desc.disassemble(n))` (pc=None, so branch targets print as `pc+N` / `pc-N`).
    pub fn disassemble(&self, n: u128) -> Result<String, String> {
        let f = self.decode_fields(n);
        let mnem = self.fields_to_mnem(&f);
        let ops = self.fields_to_operands(&f)?;
        let (mnem, ops) = self.map_to_alias(mnem, ops)?;
        let mut parts = vec![];
        for o in &ops {
            let s = py_str(o)?;
            if !s.is_empty() {
                parts.push(s);
            }
        }
        Ok(format!("{:<16} {}", mnem, parts.join(", ")))
    }

    /// `encode_fields`: every merged field must be given.
    pub fn encode_fields(&self, fields: &HashMap<&'static str, i64>) -> Result<u128, String> {
        let mut have: Vec<&str> = fields.keys().copied().collect();
        have.sort();
        let mut want: Vec<&str> = self.merged.iter().map(|(n, _)| *n).collect();
        want.sort();
        if have != want {
            return Err(format!("AssertionError: fields {:?} != {:?}", have, want));
        }
        self.patch_fields(self.bits, fields)
    }

    pub fn patch_fields(&self, encoded: u128, fields: &HashMap<&'static str, i64>) -> Result<u128, String> {
        let mut e = encoded;
        for (name, value) in fields {
            let (_, subs) = self.merged.iter().find(|(n, _)| n == name).ok_or_else(|| format!("KeyError: {}", name))?;
            for &(start, size, shift) in subs {
                let mask = (1i128 << size) - 1;
                let raw = ((*value as i128) >> shift) & mask;
                e = (e & !((mask as u128) << start)) | ((raw as u128) << start);
            }
        }
        if self.sizes.0 != self.sizes.1 {
            e &= !(1u128 << self.length_bit_pos);
            if e & (0xFFFFu128 << (self.sizes.0 * 8)) != 0 {
                e |= 1u128 << self.length_bit_pos;
            }
        }
        if !self.matches(e) {
            return Err("AssertionError: encoded instruction does not match".into());
        }
        Ok(self.mask_instr(e))
    }

    pub fn to_bytes(&self, n: u128) -> Vec<u8> {
        (0..self.decode_size(n)).map(|i| (n >> (8 * i)) as u8).collect()
    }

    /// applegpu `fields_for_mnem`.
    fn fields_for_mnem(&self, mnem: &str, ops: &[String]) -> Result<Option<HashMap<&'static str, i64>>, String> {
        let mut f = HashMap::new();
        Ok(match self.mn {
            Mn::Sat { alt } => {
                let (base, sat) = match mnem.strip_suffix(".sat") {
                    Some(b) => (b, 1),
                    None => (mnem, 0),
                };
                let ok = if let Some(a) = alt {
                    if base == a {
                        f.insert("N", 1);
                        true
                    } else if base == self.name {
                        f.insert("N", 0);
                        true
                    } else {
                        false
                    }
                } else {
                    base == self.name
                };
                if ok {
                    f.insert("S", sat);
                    Some(f)
                } else {
                    None
                }
            }
            Mn::MovImm16 | Mn::MovImm32 => {
                if mnem == self.name && !ops.is_empty() {
                    let r = operand::try_parse_register(&ops[0]);
                    let want = if self.mn == Mn::MovImm16 {
                        matches!(r, Some(Operand::Reg16(..)))
                    } else {
                        matches!(r, Some(Operand::Reg32(..)))
                    };
                    if want {
                        let second = ops.get(1).ok_or("IndexError: mov_imm operands")?;
                        if operand::try_parse_integer(second).is_some() {
                            return Ok(Some(f));
                        }
                    }
                }
                None
            }
            Mn::Bitop => {
                const ALIASES: &[&str] = &["and", "or", "xor", "nand", "nor", "xnor", "not", "mov"];
                if mnem == "bitop" || mnem == "not" || ALIASES.contains(&mnem) { Some(f) } else { None }
            }
            _ => {
                if mnem == self.name { Some(f) } else { None }
            }
        })
    }

    fn rewrite_operands_strings(&self, mnem: &str, ops: Vec<String>) -> Vec<String> {
        if self.mn != Mn::Bitop {
            return ops;
        }
        const BIN: &[(&str, &str)] =
            &[("0001", "and"), ("0111", "or"), ("0110", "xor"), ("1110", "nand"), ("1000", "nor"), ("1001", "xnor")];
        const UN: &[(&str, &str)] = &[("1010", "not"), ("0101", "mov")];
        if let Some((k, _)) = BIN.iter().find(|(_, v)| *v == mnem) {
            let mut r = vec![k.to_string()];
            r.extend(ops);
            return r;
        }
        if let Some((k, _)) = UN.iter().find(|(_, v)| *v == mnem) {
            let mut r = vec![k.to_string()];
            r.extend(ops);
            r.push("0".into());
            return r;
        }
        ops
    }
}

pub fn get_instruction_descriptor(n: u128) -> Option<&'static Desc> {
    descriptors().iter().find(|d| d.matches(n))
}

pub fn desc_by_name(name: &str) -> Option<&'static Desc> {
    descriptors().iter().find(|d| d.name == name)
}

/// One instruction at the start of `code`: (descriptor, masked number, size). None if undecodable.
pub fn decode_one(code: &[u8]) -> Option<(&'static Desc, u128, usize)> {
    let n = opcode_to_number(code);
    let d = get_instruction_descriptor(n)?;
    let size = d.decode_size(n);
    Some((d, d.mask_instr(n), size))
}

/// `' '.join(str(d.disassemble(n)).split())`
pub fn disassemble_normalized(d: &Desc, n: u128) -> Result<String, String> {
    Ok(d.disassemble(n)?.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// applegpu `assemble.assemble_line`.
pub fn assemble_line(line: &str) -> Result<Vec<u8>, String> {
    let line = line.trim();
    let (mnem, rest) = match line.split_once(char::is_whitespace) {
        Some((m, r)) => (m, Some(r.trim_start())),
        None => (line, None),
    };
    let ops: Vec<String> = match rest {
        None => vec![],
        Some(r) => r.split(',').map(|s| s.trim().to_string()).collect(),
    };
    let mut found = None;
    for d in descriptors() {
        if let Some(f) = d.fields_for_mnem(mnem, &ops)? {
            found = Some((d, f));
            break;
        }
    }
    let (d, mut fields) = found.ok_or_else(|| format!("unknown mnem {:?}", mnem))?;
    let ops = d.rewrite_operands_strings(mnem, ops);
    for (od, s) in d.operands.iter().zip(ops.iter()) {
        od.encode_string(&mut fields, s)?;
    }
    for od in d.operands.iter().skip(ops.len()) {
        od.encode_string(&mut fields, "")?;
    }
    let n = d.encode_fields(&fields)?;
    Ok(d.to_bytes(n))
}

/// agx_loop.encode: assemble one line and require the decode to round-trip to `want` (or the text).
pub fn encode(text: &str, want: Option<&str>) -> Result<Vec<u8>, String> {
    let b = assemble_line(text)?;
    let n = opcode_to_number(&b);
    let d = get_instruction_descriptor(n).ok_or_else(|| format!("inconsistent assembly: {}", text))?;
    if d.decode_size(n) != b.len() {
        return Err(format!("inconsistent assembly: {}", text));
    }
    let shown = disassemble_normalized(d, n)?;
    let want = want.unwrap_or(text).split_whitespace().collect::<Vec<_>>().join(" ");
    if shown != want {
        return Err(format!("round-trip mismatch: {:?} -> {:?}", text, shown));
    }
    Ok(b)
}

/// Infallible `encode` for fixed, known-good kernel text.
pub fn e(text: &str) -> Vec<u8> {
    encode(text, None).unwrap_or_else(|err| panic!("{}", err))
}

pub fn e2(text: &str, want: &str) -> Vec<u8> {
    encode(text, Some(want)).unwrap_or_else(|err| panic!("{}", err))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(text: &str) {
        let b = e(text);
        let (d, n, size) = decode_one(&b).unwrap();
        assert_eq!(size, b.len());
        assert_eq!(disassemble_normalized(d, n).unwrap(), text);
    }

    #[test]
    fn round_trips() {
        for t in [
            "iadd r5, r5, r24", "isub r5, r5, r24", "iadd r5, r5, r24, lsl 2", "imadd r5, r5, r25, r24",
            "imadd r5, r5, r25, 0", "and r5, r5, r24", "xor r5, r5, r24", "bfi r5, r5, r24, r26", "asr r5, r5, r26",
            "bitrev r5, r5", "popcount r5, r5", "ffs r5, r5", "intl r5, r5, r24", "icmpsel ult, r5, r5, r24, r5, r25",
            "fcmpsel lt, r5, r5, r24, r5, r25", "mov_imm r5, 305419896, 0b0", "iadd r5, r5, 1", "iadd r5, r5, r24.cache",
            "fadd32 r5, r5, r24", "fadd32 r5, r5, 0.125", "fadd32 r5, r5.cache, r24.cache", "fmadd32 r5, r5, 0.5, 0.125",
            "fadd16 r5l, r5l, r24l", "fmadd16 r5l, r5l, r25l, r24l", "convert f_to_u32, r5, r5, rtz",
            "convert u32_to_f, r5, r5, rte", "floor r5, r5", "rcp r5, r5", "rsqrt_special r5, r5", "sin_pt_1 r5, r5",
            "simd_shuffle r5, r5, r26l", "simd_shuffle_down r5, r5, r26l", "quad_shuffle r5, r5, r26l",
            "simd_iadd r5, r5", "simd_prefix_iadd r5, r5", "simd_min.u r5, r5", "quad_iadd r5, r5",
            "icmp_ballot r5, ult, r5, r24", "fcmp_ballot r5, lt, r5, r24", "iadd r3, 1, r3.sx",
            "while_icmp r0l, ult, r3, r4, 1", "pop_exec r0l, 1", "jmp_exec_any pc-40", "wait 0", "wait 1",
            "device_load 0, i32, x, r6, u8_u9, r26, unsigned", "iadd r2, r5, 0", "iadd r5_r6, r5_r6, r24_r25",
            "imadd r5_r6, r5, r25, r24_r25",
        ] {
            rt(t);
        }
        let b = e2("if_fcmp r0l, eq, 0.0, 0.0, 1", "push_exec r0l, 1");
        assert_eq!(b.len(), 6);
    }
}
