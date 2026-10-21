//! Source lines of AGX code: the DWARF line table (`.debug_line`, versions 4 and 5) of an AGX ELF object as the
//! LLVM backend writes it, with its `R_AGX_ABS32` / `R_AGX_ABS64` relocations applied, so that a PC in `.text` maps
//! to `file:line:column`.

use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub address: u64,
    pub file: String,
    pub line: u32,
    pub column: u32,
    /// The first address after the sequence this row ends (no source location).
    pub end_sequence: bool,
}

#[derive(Debug, Default)]
pub struct LineTable {
    /// Every row, sorted by address (end-of-sequence rows included).
    pub rows: Vec<Row>,
}

struct Section<'a> {
    name: String,
    kind: u32,
    data: &'a [u8],
    link: u32,
    info: u32,
}

fn rd(b: &[u8], off: usize, n: usize) -> Result<u64, String> {
    let s = b.get(off..off + n).ok_or("truncated ELF or DWARF data")?;
    Ok(s.iter().rev().fold(0u64, |a, &x| (a << 8) | x as u64))
}

fn sections(elf: &[u8]) -> Result<Vec<Section<'_>>, String> {
    if elf.get(..4) != Some(b"\x7fELF") || elf.get(4) != Some(&2) || elf.get(5) != Some(&1) {
        return Err("not a little-endian ELF64 file".into());
    }
    let shoff = rd(elf, 0x28, 8)? as usize;
    let shentsize = rd(elf, 0x3a, 2)? as usize;
    let shnum = rd(elf, 0x3c, 2)? as usize;
    let shstrndx = rd(elf, 0x3e, 2)? as usize;
    let mut raw = Vec::new();
    for i in 0..shnum {
        let h = shoff + i * shentsize;
        let (name, kind) = (rd(elf, h, 4)? as usize, rd(elf, h + 4, 4)? as u32);
        let (off, size) = (rd(elf, h + 0x18, 8)? as usize, rd(elf, h + 0x20, 8)? as usize);
        let (link, info) = (rd(elf, h + 0x28, 4)? as u32, rd(elf, h + 0x2c, 4)? as u32);
        let data = if kind == 8 { &[][..] } else { elf.get(off..off + size).ok_or("section outside the file")? };
        raw.push((name, kind, data, link, info));
    }
    let strtab = raw.get(shstrndx).ok_or("no section name table")?.2;
    raw.into_iter()
        .map(|(name, kind, data, link, info)| {
            let end = strtab[name..].iter().position(|&c| c == 0).ok_or("bad section name")?;
            Ok(Section { name: String::from_utf8_lossy(&strtab[name..name + end]).into(), kind, data, link, info })
        })
        .collect()
}

/// The contents of section `idx` with its RELA relocations applied.
fn relocated(secs: &[Section], idx: usize) -> Result<Vec<u8>, String> {
    let mut out = secs[idx].data.to_vec();
    for rela in secs.iter().filter(|s| s.kind == 4 && s.info as usize == idx) {
        let symtab = secs.get(rela.link as usize).ok_or("bad relocation symbol table")?.data;
        for e in rela.data.chunks_exact(24) {
            let off = rd(e, 0, 8)? as usize;
            let info = rd(e, 8, 8)?;
            let addend = rd(e, 16, 8)?;
            let sym = (info >> 32) as usize;
            let value = rd(symtab, sym * 24 + 8, 8)?.wrapping_add(addend);
            let n = match info as u32 {
                1 => 8,
                2 => 4,
                0 => continue,
                t => return Err(format!("unsupported relocation type {t} in {}", secs[idx].name)),
            };
            let dst = out.get_mut(off..off + n).ok_or("relocation outside its section")?;
            dst.copy_from_slice(&value.to_le_bytes()[..n]);
        }
    }
    Ok(out)
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl Reader<'_> {
    fn u(&mut self, n: usize) -> Result<u64, String> {
        let v = rd(self.b, self.p, n)?;
        self.p += n;
        Ok(v)
    }
    fn uleb(&mut self) -> Result<u64, String> {
        let (mut v, mut s) = (0u64, 0);
        loop {
            let c = self.u(1)?;
            if s < 64 {
                v |= (c & 0x7f) << s;
            }
            s += 7;
            if c & 0x80 == 0 {
                return Ok(v);
            }
        }
    }
    fn sleb(&mut self) -> Result<i64, String> {
        let (mut v, mut s) = (0i64, 0);
        loop {
            let c = self.u(1)?;
            if s < 64 {
                v |= ((c & 0x7f) as i64) << s;
            }
            s += 7;
            if c & 0x80 == 0 {
                if s < 64 && c & 0x40 != 0 {
                    v |= -1i64 << s;
                }
                return Ok(v);
            }
        }
    }
    fn cstr(&mut self) -> Result<String, String> {
        let end = self.b[self.p..].iter().position(|&c| c == 0).ok_or("unterminated string")?;
        let s = String::from_utf8_lossy(&self.b[self.p..self.p + end]).into();
        self.p += end + 1;
        Ok(s)
    }
}

fn str_at(b: &[u8], off: usize) -> Result<String, String> {
    Reader { b, p: off }.cstr()
}

/// A DWARF 5 entry list (directories or files): the path and the directory index of each entry.
fn entries(r: &mut Reader, strs: &HashMap<&str, Vec<u8>>) -> Result<Vec<(String, u64)>, String> {
    let nfmt = r.u(1)? as usize;
    let mut fmt = Vec::new();
    for _ in 0..nfmt {
        fmt.push((r.uleb()?, r.uleb()?));
    }
    let n = r.uleb()?;
    let mut out = Vec::new();
    for _ in 0..n {
        let (mut path, mut dir) = (String::new(), 0);
        for &(content, form) in &fmt {
            let (s, v): (Option<String>, u64) = match form {
                0x08 => (Some(r.cstr()?), 0),
                0x1f | 0x0e => {
                    let off = r.u(4)? as usize;
                    let sec = if form == 0x1f { ".debug_line_str" } else { ".debug_str" };
                    (Some(str_at(strs.get(sec).ok_or(format!("no {sec}"))?, off)?), 0)
                }
                0x0b => (None, r.u(1)?),
                0x05 => (None, r.u(2)?),
                0x06 => (None, r.u(4)?),
                0x07 => (None, r.u(8)?),
                0x0f => (None, r.uleb()?),
                0x1e => {
                    r.p += 16;
                    (None, 0)
                }
                0x09 => {
                    let n = r.uleb()? as usize;
                    r.p += n;
                    (None, 0)
                }
                f => return Err(format!("unsupported form {f:#x} in a line table header")),
            };
            match content {
                1 => path = s.unwrap_or_default(),
                2 => dir = v,
                _ => {}
            }
        }
        out.push((path, dir));
    }
    Ok(out)
}

fn join(dir: &str, file: &str) -> String {
    if dir.is_empty() || file.starts_with('/') { file.to_string() } else { format!("{}/{}", dir.trim_end_matches('/'), file) }
}

impl LineTable {
    pub fn from_elf(elf: &[u8]) -> Result<LineTable, String> {
        let secs = sections(elf)?;
        let find = |n: &str| secs.iter().position(|s| s.name == n);
        let Some(line) = find(".debug_line") else { return Ok(LineTable::default()) };
        let mut strs = HashMap::new();
        for n in [".debug_line_str", ".debug_str"] {
            if let Some(i) = find(n) {
                strs.insert(n, relocated(&secs, i)?);
            }
        }
        let data = relocated(&secs, line)?;
        let mut rows = Vec::new();
        let mut r = Reader { b: &data, p: 0 };
        while r.p < data.len() {
            let len = r.u(4)? as usize;
            let end = r.p + len;
            let version = r.u(2)?;
            if version >= 5 {
                r.u(2)?; // address size, segment selector size
            }
            let hlen = r.u(4)? as usize;
            let prog = r.p + hlen;
            let min_inst = r.u(1)?;
            if version >= 4 {
                r.u(1)?; // maximum operations per instruction
            }
            let default_stmt = r.u(1)?;
            let _ = default_stmt;
            let line_base = r.u(1)? as i8 as i64;
            let line_range = r.u(1)?.max(1);
            let opcode_base = r.u(1)?;
            let lens: Vec<u64> = (1..opcode_base).map(|_| r.u(1)).collect::<Result<_, _>>()?;
            let files: Vec<String> = if version >= 5 {
                let dirs = entries(&mut r, &strs)?;
                let files = entries(&mut r, &strs)?;
                files.iter().map(|(f, d)| join(dirs.get(*d as usize).map_or("", |x| x.0.as_str()), f)).collect()
            } else {
                let mut dirs = vec![String::new()];
                loop {
                    let d = r.cstr()?;
                    if d.is_empty() {
                        break;
                    }
                    dirs.push(d);
                }
                let mut files = vec![String::new()]; // 1-based
                loop {
                    let f = r.cstr()?;
                    if f.is_empty() {
                        break;
                    }
                    let d = r.uleb()? as usize;
                    r.uleb()?;
                    r.uleb()?;
                    files.push(join(dirs.get(d).map_or("", |x| x.as_str()), &f));
                }
                files
            };
            r.p = prog;
            let first_file = if version >= 5 { 0 } else { 1 };
            let (mut addr, mut file, mut ln, mut col) = (0u64, first_file, 1i64, 0u64);
            let name = |f: u64| files.get(f as usize).cloned().unwrap_or_else(|| format!("<file {f}>"));
            let push = |rows: &mut Vec<Row>, addr: u64, file: u64, ln: i64, col: u64, end: bool| {
                rows.push(Row { address: addr, file: name(file), line: ln as u32, column: col as u32, end_sequence: end })
            };
            while r.p < end {
                let op = r.u(1)?;
                if op >= opcode_base {
                    let adj = op - opcode_base;
                    addr += (adj / line_range) * min_inst;
                    ln += line_base + (adj % line_range) as i64;
                    push(&mut rows, addr, file, ln, col, false);
                    continue;
                }
                match op {
                    0 => {
                        let n = r.uleb()? as usize;
                        let next = r.p + n;
                        match r.u(1)? {
                            1 => {
                                push(&mut rows, addr, file, ln, col, true);
                                (addr, file, ln, col) = (0, first_file, 1, 0);
                            }
                            2 => addr = r.u(n - 1)?,
                            _ => {}
                        }
                        r.p = next;
                    }
                    1 => push(&mut rows, addr, file, ln, col, false),
                    2 => addr += r.uleb()? * min_inst,
                    3 => ln += r.sleb()?,
                    4 => file = r.uleb()?,
                    5 => col = r.uleb()?,
                    8 => addr += ((255 - opcode_base) / line_range) * min_inst,
                    9 => addr += r.u(2)?,
                    6 | 7 | 10 | 11 => {}
                    _ => {
                        for _ in 0..lens[op as usize - 1] {
                            r.uleb()?;
                        }
                    }
                }
            }
            r.p = end;
        }
        rows.sort_by_key(|row| (row.address, !row.end_sequence));
        Ok(LineTable { rows })
    }

    /// The source location of the instruction at `pc`: the last row at or below it, unless a sequence ends there.
    pub fn lookup(&self, pc: u64) -> Option<&Row> {
        // At an address where one sequence ends and another starts, the end sorts first.
        let i = self.rows.partition_point(|row| row.address <= pc);
        let row = self.rows[..i].last()?;
        (!row.end_sequence).then_some(row)
    }

    /// `file:line:column` of `pc`, if known.
    pub fn describe(&self, pc: u64) -> Option<String> {
        self.lookup(pc).map(|r| format!("{}:{}:{}", r.file, r.line, r.column))
    }
}

/// The PC an `agx_run` error names: "... at pc N" (the last one), else the first "at N" or "at 0xN" (an
/// instruction's position comes before the message of what it did, which may name addresses too).
pub fn error_pc(message: &str) -> Option<u64> {
    let rest = match message.rfind(" at pc ") {
        Some(i) => &message[i + 7..],
        None => &message[message.find(" at ")? + 4..],
    };
    let tok: String = rest.chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
    if let Some(h) = tok.strip_prefix("0x") { u64::from_str_radix(h, 16).ok() } else { tok.parse().ok() }
}

#[cfg(test)]
mod tests {
    use super::*;

    // tests/data/lines.o: llc -mtriple=agx -mcpu=agx-g13g -filetype=obj of a three-statement kernel with
    // line-tables-only debug info (k.cu lines 2, 3 and 4 at .text offsets 0, 8 and 0x10; .text is 0x22 bytes).
    #[test]
    fn line_table_of_an_agx_object() {
        let t = LineTable::from_elf(include_bytes!("../tests/data/lines.o")).unwrap();
        assert_eq!(t.describe(0).as_deref(), Some("/src/k.cu:2:5"));
        assert_eq!(t.describe(6).as_deref(), Some("/src/k.cu:2:5"));
        assert_eq!(t.describe(8).as_deref(), Some("/src/k.cu:3:7"));
        assert_eq!(t.describe(0x10).as_deref(), Some("/src/k.cu:4:1"));
        assert_eq!(t.describe(0x22), None, "past the end of the sequence");
    }

    #[test]
    fn the_pc_of_an_error() {
        assert_eq!(error_pc("overflow at 0x30000 at pc 8"), Some(8));
        assert_eq!(error_pc("undecodable instruction at 12"), Some(12));
        assert_eq!(error_pc("device_store at 0x1a (0102): write to unmapped address 0x30000"), Some(0x1a));
        assert_eq!(error_pc("step limit reached"), None);
    }
}
