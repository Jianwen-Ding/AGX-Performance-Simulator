//! Mesa ISA mode: only the instruction forms Mesa's asahi compiler emits for G13 (M1) and G14 (M2).
//!
//! The forms come from `llvm/utils/agx-mesa-isa/mesa_isa.py` in the LLVM AGX fork, which reads them off Mesa's
//! packer (`src/asahi/compiler/agx_pack.c`) and the operand values its compiler creates; `mesa_table.rs` is generated
//! from it. A form is one opcode at one length: its fields must hold values Mesa packs and every other bit must equal
//! the form's fixed value. The checker here follows `check_form` in that script and `AGXMesaISA.cpp` in the backend.
//!
//! `agx_run`, `agx_dispatch` and `agx_cost` take `--mesa-isa GEN` (or `AGX_MESA_ISA=GEN`): the functional simulator
//! then fails on the first executed instruction Mesa would not emit, and the cost model on any instruction of the
//! program.

use std::sync::atomic::{AtomicU8, Ordering};

/// The device variants Mesa packs differently. Only bit 45 of device atomics differs: Mesa sets it for a G13 with
/// more than one cluster (M1 Pro, Max, Ultra) and for two-die parts (M1 Ultra, M2 Ultra).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MesaGen {
    G13 = 0,
    G13X = 1,
    G14 = 2,
    G14D = 3,
}

impl MesaGen {
    /// `g13`, `g13x`, `g14`, `g14d`, or an LLVM processor name (`agx-g14c`, `agx-g13d`, ...).
    pub fn parse(s: &str) -> Result<MesaGen, String> {
        let s = s.trim().to_ascii_lowercase();
        let s = s.strip_prefix("agx-").unwrap_or(&s);
        Ok(match s {
            "g13" | "g13g" => MesaGen::G13,
            "g13x" | "g13s" | "g13c" | "g13d" => MesaGen::G13X,
            "g14" | "g14g" | "g14s" | "g14c" => MesaGen::G14,
            "g14d" => MesaGen::G14D,
            _ => return Err(format!("--mesa-isa: unknown GPU {:?} (g13, g13x, g14, g14d or agx-g1Nx)", s)),
        })
    }
}

pub struct MesaField {
    pub rule: u8,
    pub ranges: [(i32, i32); 4],
    pub p: [u16; 3],
}

pub struct MesaForm {
    pub name: &'static str,
    pub length: usize,
    pub mask: u128,
    pub value: u128,
    pub long_mask: u128,
    pub first: usize,
    pub count: usize,
    pub same: u32,
    pub gens: u8,
    pub dst: usize,
}

use super::mesa_table::{FIELDS, FORMS, SETS};

static MODE: AtomicU8 = AtomicU8::new(u8::MAX);

/// Selects Mesa ISA mode for this process (None turns it off).
pub fn set_mode(g: Option<MesaGen>) {
    MODE.store(g.map_or(u8::MAX, |g| g as u8), Ordering::Relaxed);
}

/// The selected mode: `set_mode`, else the `AGX_MESA_ISA` environment variable.
pub fn mode() -> Option<MesaGen> {
    static ENV: std::sync::OnceLock<Option<MesaGen>> = std::sync::OnceLock::new();
    match MODE.load(Ordering::Relaxed) {
        0 => Some(MesaGen::G13),
        1 => Some(MesaGen::G13X),
        2 => Some(MesaGen::G14),
        3 => Some(MesaGen::G14D),
        _ => *ENV.get_or_init(|| {
            std::env::var("AGX_MESA_ISA").ok().filter(|v| !v.is_empty()).map(|v| {
                MesaGen::parse(&v).unwrap_or_else(|e| panic!("AGX_MESA_ISA: {}", e))
            })
        }),
    }
}

/// Handles the `--mesa-isa GEN` option of the tools: returns true when `a[i]` was it.
pub fn take_option(a: &[String], i: usize) -> Result<bool, String> {
    if a[i] != "--mesa-isa" { return Ok(false); }
    let v = a.get(i + 1).ok_or("--mesa-isa needs a GPU (g13, g13x, g14, g14d)")?;
    set_mode(Some(MesaGen::parse(v)?));
    Ok(true)
}

const ANY: u8 = 0;
const SET: u8 = 1;
const RANGE: u8 = 2;
const EVEN: u8 = 3;
const ALU_DST: u8 = 4;
const ALU_SRC: u8 = 5;
const MEM_REG: u8 = 6;
const MEM_BASE: u8 = 7;
const MEM_INDEX: u8 = 8;
const LOCAL_BASE: u8 = 9;
const LOCAL_INDEX: u8 = 10;
const ZERO_INDEX: u8 = 11;
const ATOMIC_DST: u8 = 12;
const LOCAL_ATOMIC_DST: u8 = 13;
const TILE_REG: u8 = 14;
const ITER_DST: u8 = 15;
const SAMPLE_MASK_SRC: u8 = 16;
const TEX_EXT: u8 = 17;

const K_IADD: u16 = 1;
const K_HALF: u16 = 3;
const K_CMPSEL: u16 = 4;
const SRC_WIDTH: [u32; 5] = [10, 11, 12, 11, 9];

const SIZE16: u32 = 1;
const SIZE32: u32 = 2;
const SIZE64: u32 = 4;

#[derive(Clone, Copy, PartialEq, Default)]
enum Ty {
    #[default]
    None,
    Imm,
    Uniform,
    Reg,
}

#[derive(Clone, Copy, Default)]
struct Op {
    ty: Ty,
    size: u32,
}

fn extract(n: u128, f: &MesaField) -> u64 {
    let (mut v, mut shift) = (0u64, 0);
    for &(lo, hi) in &f.ranges {
        if hi < lo { break; }
        let w = (hi - lo + 1) as u32;
        v |= (((n >> lo) & ((1u128 << w) - 1)) as u64) << shift;
        shift += w;
    }
    v
}

fn in_set(i: u16, v: u64) -> bool {
    SETS[i as usize].iter().any(|&x| x as u64 == v)
}

fn aligned(v: u32, size: u32) -> bool {
    match size {
        SIZE32 => v % 2 == 0,
        SIZE64 => v % 4 == 0,
        _ => true,
    }
}

/// An ALU source as Mesa's agx_pack_alu_src / agx_pack_cmpsel_src encode it: (type, size, value), or None for an
/// encoding Mesa never produces.
fn decode_alu_src(kind: u16, v: u64) -> Option<(Ty, u32, u32)> {
    let width = SRC_WIDTH[kind as usize];
    let short = v & ((1 << width) - 1);
    let ext = v >> width;
    let (v6, b67, b8) = (short & 63, (short >> 6) & 3, (short >> 8) & 1);
    let value = (v6 | (ext << 6)) as u32;
    if kind == K_CMPSEL {
        return if b8 == 1 && b67 == 0 {
            Some((Ty::Imm, 0, value))
        } else if b8 == 1 && b67 >> 1 == 1 {
            Some((Ty::Uniform, 0, value | ((b67 as u32 & 1) << 8)))
        } else if b8 == 0 && b67 != 0 {
            Some((Ty::Reg, 0, value))
        } else {
            None
        };
    }
    let b9 = if kind == K_HALF { 0 } else { (short >> 9) & 1 };
    if b67 == 0 && b8 == 0 && b9 == 0 { return Some((Ty::Imm, 0, value)); }
    if b8 == 1 && b9 == 0 {
        return Some((Ty::Uniform, if b67 >> 1 == 1 { SIZE32 } else { SIZE16 }, value | ((b67 as u32 & 1) << 8)));
    }
    if b67 == 0 { return None; }
    let size = if b9 == 1 { if b8 == 1 { SIZE64 } else { SIZE32 } } else { SIZE16 };
    Some((Ty::Reg, size, value))
}

fn check_field(form: &MesaForm, i: usize, v: u64, ops: &mut [Op]) -> Result<(), &'static str> {
    let f = &FIELDS[form.first + i];
    let ok = |c: bool, why: &'static str| if c { Ok(()) } else { Err(why) };
    match f.rule {
        ANY | TILE_REG | ITER_DST => Ok(()),
        SET => ok(in_set(f.p[0], v), "field value"),
        RANGE => ok(v >= f.p[0] as u64 && v <= f.p[1] as u64, "field range"),
        EVEN => ok(v % 2 == 0 && v < 256, "register"),
        ALU_DST => {
            let (cache, s32) = (v & 1, (v >> 1) & 1);
            let mut reg = (v >> 2) as u32;
            let mut size = SIZE16;
            if s32 == 1 {
                size = if reg & 1 == 1 { SIZE64 } else { SIZE32 };
                reg &= !1;
                if size == SIZE64 && reg % 4 != 0 { return Err("destination alignment"); }
            }
            if size & f.p[0] as u32 == 0 { return Err("destination size"); }
            if cache == 1 && f.p[1] == 0 { return Err("destination cache hint"); }
            ops[i] = Op { ty: Ty::Reg, size };
            Ok(())
        }
        ALU_SRC => {
            let kind = f.p[0];
            let (ty, size, value) = decode_alu_src(kind, v).ok_or("source encoding")?;
            if kind == K_CMPSEL {
                let size = if form.dst < form.count { ops[form.dst].size } else { SIZE32 };
                if ty == Ty::Reg && !aligned(value, size) { return Err("source alignment"); }
                ops[i] = Op { ty, size };
                return Ok(());
            }
            if f.p[2] != 0 {
                if ty != Ty::Imm { return Err("source must be an immediate"); }
                if !in_set(f.p[2] - 1, value as u64) { return Err("immediate source value"); }
            }
            if matches!(ty, Ty::Reg | Ty::Uniform) && size & f.p[1] as u32 == 0 { return Err("source size"); }
            if ty == Ty::Reg && !aligned(value, size) { return Err("source alignment"); }
            let short = v & ((1 << SRC_WIDTH[kind as usize]) - 1);
            if kind == K_IADD && (short >> 10) & 1 == 1 && (ty == Ty::Imm || size == SIZE64) {
                return Err("source sign extension");
            }
            ops[i] = Op { ty, size };
            Ok(())
        }
        MEM_REG => ok(v >> 8 == 0 || (v & 0xff) % 2 == 0, "register alignment"),
        MEM_BASE => ok((v & 0xff) % 2 == 0, "base alignment"),
        MEM_INDEX => ok(v >> 16 != 0 || ((v & 0xffff) % 2 == 0 && (v & 0xffff) < 0x100), "index register"),
        LOCAL_BASE => ok(!(v >> 8 == 2 && v & 0xff != 0), "local base immediate"),
        LOCAL_INDEX => ok(v >> 16 != 0 || (v & 0xffff) < 0x100, "local index register"),
        ZERO_INDEX => ok(v == 1 << 16, "local atomic index"),
        ATOMIC_DST => {
            if v >> 8 == 0 { ok(v & 0xff == 0, "atomic destination") } else { ok((v & 0xff) % 2 == 0, "atomic destination alignment") }
        }
        LOCAL_ATOMIC_DST => {
            if v >> 9 == 0 {
                ok(v & 0x1ff == 0, "atomic destination")
            } else {
                ok((v >> 8) & 1 == 0 || (v & 0xff) % 2 == 0, "atomic destination alignment")
            }
        }
        SAMPLE_MASK_SRC => ok((v >> 6) & 1 == 0, "sample mask source"),
        TEX_EXT => ok((v >> 5) & 1 == 0, "texture kill bit"),
        _ => Err("rule"),
    }
}

fn check_form(form: &MesaForm, n: u128, gen: MesaGen) -> Result<(), &'static str> {
    if n & form.mask != form.value { return Err("fixed bits"); }
    let mut ops = [Op::default(); 24];
    for i in 0..form.count {
        check_field(form, i, extract(n, &FIELDS[form.first + i]), &mut ops)?;
    }
    if form.same != 0 {
        let mut size = 0;
        for (i, o) in ops.iter().enumerate().take(form.count) {
            if form.same & (1 << i) == 0 || o.ty == Ty::Imm || o.size == 0 { continue; }
            if size != 0 && size != o.size { return Err("operand sizes differ"); }
            size = o.size;
        }
    }
    if form.long_mask != 0 && n & form.long_mask == 0 { return Err("long form not needed"); }
    // Last, so that a field Mesa never packs is reported first.
    if form.gens & (1 << gen as u8) == 0 { return Err("generation"); }
    Ok(())
}

/// Checks one whole instruction. Ok gives the Mesa form's name; Err says what differs from the closest form.
pub fn check(bytes: &[u8], gen: MesaGen) -> Result<&'static str, String> {
    let n = super::opcode_to_number(bytes);
    let mut closest: Option<String> = None;
    for form in FORMS {
        if form.length != bytes.len() { continue; }
        match check_form(form, n, gen) {
            Ok(()) => return Ok(form.name),
            Err("fixed bits") => {}
            Err("generation") => {
                closest.get_or_insert_with(|| format!("{}: not emitted for this GPU", form.name));
            }
            Err(why) => {
                if closest.as_deref().map_or(true, |c| c.ends_with("not emitted for this GPU")) {
                    closest = Some(format!("{}: {}", form.name, why));
                }
            }
        }
    }
    Err(closest.unwrap_or_else(|| "no Mesa instruction has this opcode and length".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    /// Instructions packed by Mesa's own agx_pack.c (llvm/utils/agx-mesa-isa/tools/fuzz.sh), with their generation.
    #[test]
    fn accepts_mesa_packed_instructions() {
        let corpus = include_str!("mesa_corpus.txt");
        let mut n = 0;
        for line in corpus.lines().filter(|l| !l.starts_with('#') && !l.is_empty()) {
            let mut it = line.split_whitespace();
            let (gen, op, h) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
            let gen = MesaGen::parse(gen).unwrap();
            if let Err(e) = check(&hex(h), gen) { panic!("{} {} rejected: {}", op, h, e); }
            n += 1;
        }
        assert!(n > 1000, "corpus has {} instructions", n);
    }

    /// Compares with mesa_isa.py's engine: AGX_MESA_DIFF=FILE with lines "gen hex accepted(0/1)".
    #[test]
    #[ignore]
    fn agrees_with_reference_engine() {
        let path = std::env::var("AGX_MESA_DIFF").expect("AGX_MESA_DIFF");
        let mut bad = 0;
        for line in std::fs::read_to_string(path).unwrap().lines() {
            let mut it = line.split_whitespace();
            let (g, h, want) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap() == "1");
            let got = check(&hex(h), MesaGen::parse(g).unwrap()).is_ok();
            if got != want {
                bad += 1;
                if bad < 10 { eprintln!("{} {}: reference {} rust {}", g, h, want, got); }
            }
        }
        assert_eq!(bad, 0);
    }

    #[test]
    fn rejects_instructions_mesa_never_emits() {
        // wait 0 and stop are Mesa's; the 0x78 block barrier and the shifted shlhi are not.
        assert!(check(&hex("3800"), MesaGen::G14).is_ok());
        assert!(check(&hex("8800"), MesaGen::G14).is_ok());
        assert!(check(&hex("7800"), MesaGen::G14).is_err());
        // device atomic add r0, [u0_u1, 0]: bit 45 belongs to two-die parts only.
        let atomic: u128 = 0x15 | 1 << 26 | 1 << 31 | 5 << 44 | 1 << 27 | 1 << 24;
        let bytes = |n: u128| n.to_le_bytes()[..8].to_vec();
        assert!(check(&bytes(atomic), MesaGen::G14).is_ok());
        assert!(check(&bytes(atomic | 1 << 45), MesaGen::G14).is_err());
        assert!(check(&bytes(atomic | 1 << 45), MesaGen::G14D).is_ok());
        // atomic op 1 (sub, the G14 float add with bit 45) is never translated by Mesa.
        assert!(check(&bytes(atomic | 1 << 6), MesaGen::G13).is_err());
    }
}
