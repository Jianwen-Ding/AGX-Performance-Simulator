//! applegpu's `InstructionDesc` / `OperandDesc` framework: constant bits, raw and merged fields,
//! operand decode / encode_string, mnemonic selection and disassembly aliases.

use super::operand::*;
use std::collections::HashMap;

pub type Fields = Vec<(&'static str, u64)>;

pub fn fget(f: &Fields, name: &str) -> Option<u64> {
    f.iter().find(|(k, _)| *k == name).map(|(_, v)| *v)
}

pub fn fx(f: &Fields, name: &str) -> Result<u64, String> {
    fget(f, name).ok_or_else(|| format!("KeyError: {}", name))
}

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

pub fn sign_extend(v: i128, bits: u32) -> i128 {
    let v = v & ((1i128 << bits) - 1);
    if v & (1i128 << (bits - 1)) != 0 {
        v | (-1i128 << bits)
    } else {
        v
    }
}

/// Operand descriptor kinds (one per applegpu OperandDesc class / configuration).
#[derive(Clone, Debug)]
pub enum OpKind {
    ImplicitR0L,
    AluDst { allow32: bool, allow64: bool, paired: bool },
    AluSrc { allow32: bool, allow64: bool, paired: bool, sx: bool, float: bool },
    CmpselSrc,
    TruthTable,
    Binary { size: u32 },
    Immediate { size: u32 },
    Reg32F,
    UReg64F,
    Enum { size: u32, values: &'static [(u64, &'static str)] },
    Shift,
    Mask,
    BranchOffset { size: u32 },
    StackAdjustment,
    StackReg32,
    Cond { float: bool, has_ccn: bool },
    MemoryShift,
    MemoryIndex,
    ThreadgroupIndex,
    AsyncMemoryReg,
    AsyncMemoryBase,
    MemoryBase,
    SampleMask { flags_type: u32 },
    DiscardMask,
    ZS,
    MemoryReg,
    TgMemoryReg { optional: bool },
    TileToMemoryReg,
    TgMemoryBase,
    SReg32D,
    ExReg32,
    ExReg64,
    ExReg16,
    VarRegister { triple: bool },
    UVSIndex,
    CFD,
    CFPerspective,
    IterSampleID,
    TextureStateDest,
    SamplerStateDest,
    SampleUReg,
    Texture,
    Sampler,
    Coords,
    SampleCompareOff,
    Lod,
    PBELod,
    Reg32_4_4,
    PBESourceReg,
    AtomicSource,
    AtomicDestination,
}

#[derive(Clone, Debug)]
pub struct OpDesc {
    pub name: &'static str,
    pub kind: OpKind,
    /// derived field names used by decode (`name + 't'`, ...)
    pub t: &'static str,
    pub x2: &'static str,
    pub fields: Vec<(u32, u32, &'static str)>,
    pub merged: Vec<(&'static str, Vec<(u32, u32, &'static str)>)>,
}

impl OpDesc {
    fn new(name: &str, kind: OpKind) -> Self {
        OpDesc {
            name: leak(name.to_string()),
            kind,
            t: leak(format!("{}t", name)),
            x2: leak(format!("{}s", name)),
            fields: vec![],
            merged: vec![],
        }
    }
    fn field(mut self, start: u32, size: u32, name: String) -> Self {
        self.fields.push((start, size, leak(name)));
        self
    }
    fn merged(mut self, name: &str, subs: Vec<(u32, u32, String)>) -> Self {
        self.merged.push((leak(name.to_string()), subs.into_iter().map(|(a, b, c)| (a, b, leak(c))).collect()));
        self
    }
    fn m2(self, off: u32, offx: u32) -> Self {
        let n = self.name.to_string();
        self.merged(&n.clone(), vec![(off, 6, n.clone()), (offx, 2, n + "x")])
    }
}

// ---- operand descriptor constructors (argument order as in applegpu) --------------------------

pub fn implicit_r0l(name: &str) -> OpDesc {
    OpDesc::new(name, OpKind::ImplicitR0L).field(7, 1, format!("{}t", name))
}
fn alu_dst_base(name: &str, bit_off_ex: u32, allow32: bool, allow64: bool, paired: bool) -> OpDesc {
    OpDesc::new(name, OpKind::AluDst { allow32, allow64, paired })
        .merged(name, vec![(9, 6, name.into()), (bit_off_ex, 2, format!("{}x", name))])
        .field(7, 2, format!("{}t", name))
}
pub fn alu_dst(name: &str, ex: u32) -> OpDesc { alu_dst_base(name, ex, true, false, false) }
pub fn paired_alu_dst(name: &str, ex: u32) -> OpDesc { alu_dst_base(name, ex, true, false, true) }
pub fn alu_dst64(name: &str, ex: u32) -> OpDesc { alu_dst_base(name, ex, true, true, false) }
pub fn float_dst(name: &str, ex: u32) -> OpDesc { alu_dst_base(name, ex, true, false, false).field(6, 1, "S".into()) }
pub fn float_dst16(name: &str, ex: u32) -> OpDesc { alu_dst_base(name, ex, false, false, false).field(6, 1, "S".into()) }

#[allow(clippy::too_many_arguments)]
fn alu_src_base(name: &str, off: u32, ex: u32, type_size: u32, allow32: bool, allow64: bool, paired: bool, sx: bool,
                float: Option<Option<u32>>) -> OpDesc {
    let mut d = OpDesc::new(name, OpKind::AluSrc { allow32, allow64, paired, sx, float: float.is_some() })
        .merged(name, vec![(off, 6, name.into()), (ex, 2, format!("{}x", name))])
        .field(off + 6, type_size, format!("{}t", name));
    if sx {
        d = d.field(off + 10, 1, format!("{}s", name));
    }
    if let Some(m) = float {
        d = d.field(m.unwrap_or(off + 6 + type_size), 2, format!("{}m", name));
    }
    d
}
pub fn alu_src(n: &str, o: u32, e: u32) -> OpDesc { alu_src_base(n, o, e, 4, true, false, false, false, None) }
pub fn alu_src64(n: &str, o: u32, e: u32) -> OpDesc { alu_src_base(n, o, e, 4, true, true, false, false, None) }
pub fn alu_src16(n: &str, o: u32, e: u32) -> OpDesc { alu_src_base(n, o, e, 4, false, false, false, false, None) }
pub fn mul_src(n: &str, o: u32, e: u32) -> OpDesc { alu_src_base(n, o, e, 4, true, false, false, true, None) }
pub fn add_src(n: &str, o: u32, e: u32) -> OpDesc { alu_src_base(n, o, e, 4, true, true, false, true, None) }
pub fn float_src(n: &str, o: u32, e: u32) -> OpDesc { alu_src_base(n, o, e, 4, true, false, false, false, Some(None)) }
pub fn paired_float_src(n: &str, o: u32, e: u32, m: Option<u32>) -> OpDesc {
    alu_src_base(n, o, e, 4, true, false, true, false, Some(m))
}
pub fn float_src16(n: &str, o: u32, e: u32) -> OpDesc { alu_src_base(n, o, e, 3, true, false, false, false, Some(None)) }

pub fn cmpsel_src(name: &str, off: u32, ex: u32) -> OpDesc {
    OpDesc::new(name, OpKind::CmpselSrc)
        .merged(name, vec![(off, 6, name.into()), (ex, 2, format!("{}x", name))])
        .field(off + 6, 3, format!("{}t", name))
}
pub fn truth_table(name: &str) -> OpDesc {
    OpDesc::new(name, OpKind::TruthTable)
        .field(26, 1, format!("{}0", name))
        .field(27, 1, format!("{}1", name))
        .field(38, 1, format!("{}2", name))
        .field(39, 1, format!("{}3", name))
}
fn field_desc(name: &str, kind: OpKind, start: u32, size: u32) -> OpDesc {
    OpDesc::new(name, kind).field(start, size, name.into())
}
fn field_desc_list(name: &str, kind: OpKind, subs: &[(u32, u32, &str)]) -> OpDesc {
    OpDesc::new(name, kind).merged(name, subs.iter().map(|&(a, b, c)| (a, b, c.to_string())).collect())
}
pub fn binary(name: &str, start: u32, size: u32) -> OpDesc { field_desc(name, OpKind::Binary { size }, start, size) }
pub fn immediate(name: &str, start: u32, size: u32) -> OpDesc { field_desc(name, OpKind::Immediate { size }, start, size) }
pub fn immediate_list(name: &str, subs: &[(u32, u32, &str)]) -> OpDesc {
    let size = subs.iter().map(|s| s.1).sum();
    field_desc_list(name, OpKind::Immediate { size }, subs)
}
pub fn reg32_desc(name: &str, start: u32, size: u32) -> OpDesc { field_desc(name, OpKind::Reg32F, start, size) }
pub fn ureg64_desc(name: &str, start: u32, size: u32) -> OpDesc { field_desc(name, OpKind::UReg64F, start, size) }
pub fn enum_desc(name: &str, start: u32, size: u32, values: &'static [(u64, &'static str)]) -> OpDesc {
    field_desc(name, OpKind::Enum { size, values }, start, size)
}
pub fn enum_list(name: &str, subs: &[(u32, u32, &str)], values: &'static [(u64, &'static str)]) -> OpDesc {
    let size = subs.iter().map(|s| s.1).sum();
    field_desc_list(name, OpKind::Enum { size, values }, subs)
}
pub fn shift_desc(name: &str) -> OpDesc {
    OpDesc::new(name, OpKind::Shift).merged("s", vec![(39, 1, "s1".into()), (52, 2, "s2".into())])
}
pub fn mask_desc(name: &str) -> OpDesc {
    OpDesc::new(name, OpKind::Mask).merged(
        name,
        vec![(38, 2, format!("{}1", name)), (50, 2, format!("{}2", name)), (63, 1, format!("{}3", name))],
    )
}
pub fn branch_offset(name: &str, start: u32, size: u32) -> OpDesc { field_desc(name, OpKind::BranchOffset { size }, start, size) }
pub fn stack_adjustment(name: &str) -> OpDesc {
    OpDesc::new(name, OpKind::StackAdjustment).merged(
        name,
        vec![(20, 4, format!("{}1", name)), (32, 4, format!("{}2", name)), (56, 8, format!("{}3", name))],
    )
}
pub fn stack_reg32(name: &str, parts: &[(u32, u32, &str)]) -> OpDesc {
    OpDesc::new(name, OpKind::StackReg32).merged(name, parts.iter().map(|&(a, b, c)| (a, b, c.to_string())).collect())
}
pub fn cond(float: bool, cc_off: u32, cc_n_off: Option<u32>) -> OpDesc {
    let mut d = OpDesc::new("cc", OpKind::Cond { float, has_ccn: cc_n_off.is_some() }).field(cc_off, 3, "cc".into());
    if let Some(n) = cc_n_off {
        d = d.field(n, 1, "ccn".into());
    }
    d
}
pub fn icond(cc_off: u32, cc_n_off: Option<u32>) -> OpDesc { cond(false, cc_off, cc_n_off) }
pub fn fcond(cc_off: u32, cc_n_off: Option<u32>) -> OpDesc { cond(true, cc_off, cc_n_off) }
pub fn memory_shift(name: &str) -> OpDesc { OpDesc::new(name, OpKind::MemoryShift).field(42, 2, name.into()) }
pub fn memory_index(name: &str) -> OpDesc {
    OpDesc::new(name, OpKind::MemoryIndex)
        .merged(name, vec![(20, 4, format!("{}l", name)), (32, 4, format!("{}h", name)), (56, 8, format!("{}x", name))])
        .field(24, 1, format!("{}t", name))
}
pub fn threadgroup_index(name: &str) -> OpDesc {
    OpDesc::new(name, OpKind::ThreadgroupIndex)
        .merged(name, vec![(28, 6, name.into()), (48, 10, format!("{}x", name))])
        .field(34, 1, format!("{}t", name))
}
pub fn async_memory_reg(name: &str) -> OpDesc {
    OpDesc::new(name, OpKind::AsyncMemoryReg).merged(name, vec![(9, 6, name.into())]).field(8, 1, format!("{}t", name))
}
pub fn async_memory_base(name: &str) -> OpDesc {
    OpDesc::new(name, OpKind::AsyncMemoryBase)
        .merged(name, vec![(16, 4, format!("{}l", name)), (36, 4, format!("{}h", name))])
        .field(25, 1, format!("{}t", name))
}
pub fn memory_base(name: &str) -> OpDesc {
    OpDesc::new(name, OpKind::MemoryBase)
        .merged(name, vec![(16, 4, format!("{}l", name)), (36, 4, format!("{}h", name))])
        .field(27, 1, format!("{}t", name))
}
pub fn sample_mask(name: &str, off: u32, offx: u32, offt: u32, flags_type: u32) -> OpDesc {
    OpDesc::new(name, OpKind::SampleMask { flags_type }).m2(off, offx).field(offt, flags_type, format!("{}t", name))
}
pub fn discard_mask(name: &str) -> OpDesc {
    OpDesc::new(name, OpKind::DiscardMask).m2(16, 26).field(23, 1, format!("{}t", name))
}
pub fn zs_desc(name: &str) -> OpDesc {
    OpDesc::new(name, OpKind::ZS).m2(16, 26).field(29, 1, "z".into()).field(30, 1, "s".into())
}
pub fn memory_reg(name: &str, off: u32, offx: u32, offt: u32) -> OpDesc {
    OpDesc::new(name, OpKind::MemoryReg).m2(off, offx).field(offt, 1, format!("{}t", name))
}
pub fn tg_memory_reg(name: &str, offa: Option<u32>) -> OpDesc {
    let mut d = OpDesc::new(name, OpKind::TgMemoryReg { optional: offa.is_some() })
        .m2(9, 60)
        .field(8, 1, format!("{}t", name));
    if let Some(a) = offa {
        d = d.field(a, 1, format!("{}a", name));
    }
    d
}
pub fn tile_to_memory_reg(name: &str) -> OpDesc { OpDesc::new(name, OpKind::TileToMemoryReg).m2(9, 56) }
pub fn tg_memory_base(name: &str) -> OpDesc {
    OpDesc::new(name, OpKind::TgMemoryBase).m2(16, 58).field(22, 2, format!("{}t", name))
}
pub fn sreg32_desc(name: &str, start: u32, start_ex: u32) -> OpDesc { OpDesc::new(name, OpKind::SReg32D).m2(start, start_ex) }
fn ex_reg(name: &str, kind: OpKind, start: u32, start_ex: u32, bits: u32) -> OpDesc {
    OpDesc::new(name, kind).merged(name, vec![(start, bits, name.into()), (start_ex, 2, format!("{}x", name))])
}
pub fn ex_reg32(name: &str, s: u32, e: u32) -> OpDesc { ex_reg(name, OpKind::ExReg32, s, e, 5) }
pub fn var_register(name: &str, triple: bool) -> OpDesc {
    OpDesc::new(name, OpKind::VarRegister { triple })
        .m2(9, 56)
        .field(8, 1, format!("{}t", name))
        .field(30, 2, "count".into())
}
pub fn uvs_index(name: &str) -> OpDesc { OpDesc::new(name, OpKind::UVSIndex).m2(16, 26).field(23, 1, format!("{}t", name)) }
pub fn cf_desc(name: &str, off: u32, offx: u32) -> OpDesc {
    OpDesc::new(name, OpKind::CFD).m2(off, offx).field(23, 1, format!("{}t", name))
}
pub fn cf_perspective(name: &str, off: u32, offx: u32) -> OpDesc { OpDesc::new(name, OpKind::CFPerspective).m2(off, offx) }
pub fn iter_sample_id(name: &str, off: u32) -> OpDesc { OpDesc::new(name, OpKind::IterSampleID).field(off, 8, name.into()) }
pub fn texture_state_dest(name: &str) -> OpDesc { OpDesc::new(name, OpKind::TextureStateDest).field(8, 7, name.into()) }
pub fn sampler_state_dest(name: &str) -> OpDesc { OpDesc::new(name, OpKind::SamplerStateDest).field(8, 5, name.into()) }
pub fn sample_ureg(name: &str, start: u32) -> OpDesc {
    OpDesc::new(name, OpKind::SampleUReg).merged(name, vec![(start, 5, name.into())])
}
pub fn texture_desc(name: &str, offx: u32) -> OpDesc {
    OpDesc::new(name, OpKind::Texture).m2(32, offx).field(38, 2, format!("{}t", name))
}
pub fn sampler_desc(name: &str) -> OpDesc { OpDesc::new(name, OpKind::Sampler).m2(56, 92).field(62, 1, format!("{}t", name)) }
pub fn coords_desc(name: &str, offx: u32) -> OpDesc {
    OpDesc::new(name, OpKind::Coords)
        .m2(16, offx)
        .field(22, 1, format!("{}t", name))
        .field(47, 1, format!("{}s", name))
}
pub fn sample_compare_off(name: &str) -> OpDesc {
    OpDesc::new(name, OpKind::SampleCompareOff).m2(80, 94).field(91, 1, format!("{}t", name))
}
pub fn lod_desc(name: &str, off: u32, offx: u32) -> OpDesc { OpDesc::new(name, OpKind::Lod).m2(off, offx) }
pub fn pbe_lod(name: &str) -> OpDesc { OpDesc::new(name, OpKind::PBELod).m2(24, 60).field(31, 1, format!("{}t", name)) }
pub fn reg32_4_4(name: &str, start: u32, start2: u32) -> OpDesc {
    OpDesc::new(name, OpKind::Reg32_4_4).merged(name, vec![(start, 4, name.into()), (start2, 4, format!("{}x", name))])
}
pub fn pbe_source_reg(name: &str) -> OpDesc {
    OpDesc::new(name, OpKind::PBESourceReg).m2(9, 56).field(8, 1, format!("{}t", name))
}
pub fn atomic_source(name: &str, off: u32) -> OpDesc {
    OpDesc::new(name, OpKind::AtomicSource).merged(name, vec![(off, 8, name.into())])
}
pub fn atomic_destination(name: &str) -> OpDesc {
    OpDesc::new(name, OpKind::AtomicDestination).m2(10, 40).field(47, 1, format!("{}t", name))
}

pub const TEX_SIZES: &[(u64, (u32, u32))] = &[
    (0, (1, 0)), (1, (1, 1)), (2, (2, 0)), (3, (2, 1)), (4, (2, 1)), (5, (3, 0)), (6, (3, 0)), (7, (3, 1)), (8, (2, 2)),
];

fn tex_size(n: u64) -> Result<(u32, u32), String> {
    TEX_SIZES.iter().find(|(k, _)| *k == n).map(|(_, v)| *v).ok_or_else(|| format!("KeyError: tex type {}", n))
}

const ICOND: &[(u64, &str)] = &[
    (0b0000, "ueq"), (0b0001, "ult"), (0b0010, "ugt"), (0b0100, "seq"), (0b0101, "slt"), (0b0110, "sgt"),
    (0b1000, "nueq"), (0b1001, "ugte"), (0b1010, "ulte"), (0b1100, "nseq"), (0b1101, "sgte"), (0b1110, "slte"),
];
const FCOND: &[(u64, &str)] = &[
    (0b000, "eq"), (0b001, "lt"), (0b010, "gt"), (0b011, "ltn"), (0b101, "gte"), (0b110, "lte"), (0b111, "gtn"),
    (0b1000, "neq"), (0b1001, "nlt"), (0b1011, "nltn"), (0b1010, "ngt"), (0b1101, "ngte"), (0b1110, "nlte"), (0b1111, "ngtn"),
];

fn tuple16(start: u32, count: u32) -> Operand { Operand::Tuple((0..count).map(|i| Operand::Reg16(start + i, 0)).collect(), 0) }
fn tuple32(start: u32, count: u32) -> Operand { Operand::Tuple((0..count).map(|i| Operand::Reg32(start + i, 0)).collect(), 0) }

fn hint(mut r: Operand, bits: u64) -> Result<Operand, String> {
    match bits {
        0b10 => r.add_flag(F_CACHE)?,
        0b11 => r.add_flag(F_DISCARD)?,
        0b01 => {}
        _ => return Err("AssertionError: hint bits".into()),
    }
    Ok(r)
}

fn as_u32(v: u64) -> u32 { v as u32 }

impl OpDesc {
    /// Python `decode(fields)`.
    pub fn decode(&self, f: &Fields) -> Result<Operand, String> {
        let name = self.name;
        let v = || fx(f, name);
        let t = || fx(f, self.t);
        Ok(match &self.kind {
            OpKind::ImplicitR0L => {
                let mut r = Operand::Reg16(0, 0);
                if t()? & 1 != 0 {
                    r.add_flag(F_CACHE)?;
                }
                r
            }
            &OpKind::AluDst { allow32, allow64, paired } => {
                let flags = t()?;
                let value = as_u32(v()?);
                let mut r = if flags & 2 != 0 && allow32 {
                    if value & 1 != 0 && allow64 {
                        Operand::Reg64(value >> 1, 0)
                    } else if paired {
                        tuple32(value >> 1, 2)
                    } else {
                        Operand::Reg32(value >> 1, 0)
                    }
                } else if paired {
                    tuple16(value, 2)
                } else {
                    Operand::Reg16(value, 0)
                };
                if flags & 1 != 0 {
                    r.add_flag(F_CACHE)?;
                }
                r
            }
            &OpKind::AluSrc { allow32, allow64, paired, sx, float } => {
                let flags = t()?;
                let mut value = as_u32(v()?);
                let mut r = if flags == 0 {
                    if float {
                        Operand::Imm(ImmVal::Float(decode_float_immediate(value as u64)), 0)
                    } else {
                        Operand::Imm(ImmVal::Int(value as i64), 0)
                    }
                } else if flags & 0b1100 == 0b0100 {
                    value |= ((flags & 1) as u32) << 8;
                    if flags & 2 != 0 { Operand::UReg32(value >> 1, 0) } else { Operand::UReg16(value, 0) }
                } else if flags & 0b11 != 0 {
                    let hi = flags & 0b1100;
                    let r = if allow64 && hi == 0b1100 {
                        if value & 1 != 0 {
                            return Err("AssertionError: odd 64-bit source".into());
                        }
                        Operand::Reg64(value >> 1, 0)
                    } else if allow32 && (hi == 0b1000 || hi == 0b1100) {
                        if paired { tuple32(value >> 1, 2) } else { Operand::Reg32(value >> 1, 0) }
                    } else if hi == 0 || hi == 0b1000 || hi == 0b1100 {
                        if paired { tuple16(value, 2) } else { Operand::Reg16(value, 0) }
                    } else {
                        Operand::Null
                    };
                    if r == Operand::Null { r } else { hint(r, flags & 0b11)? }
                } else {
                    Operand::Null // print('TODO: ...') and return None
                };
                if sx && r.is_register() && fx(f, self.x2)? & 1 != 0 {
                    r.add_flag(F_SX)?;
                }
                if float {
                    let m = fx(f, leak_m(name))?;
                    if m & 1 != 0 {
                        r.add_flag(F_ABS)?;
                    }
                    if m & 2 != 0 {
                        r.add_flag(F_NEG)?;
                    }
                }
                r
            }
            OpKind::CmpselSrc => {
                let flags = t()?;
                let value = as_u32(v()?);
                let is32 = fx(f, "Dt")? & 2 != 0;
                match flags {
                    0b100 => Operand::Imm(ImmVal::Int(value as i64), 0),
                    1..=3 => {
                        let r = if is32 {
                            if value & 1 != 0 {
                                return Err("AssertionError: odd cmpsel source".into());
                            }
                            Operand::Reg32(value >> 1, 0)
                        } else {
                            Operand::Reg16(value, 0)
                        };
                        hint(r, flags & 0b11)?
                    }
                    0b110 | 0b111 => {
                        if is32 {
                            if value & 1 != 0 {
                                return Err("AssertionError: odd cmpsel source".into());
                            }
                            Operand::UReg32(value >> 1, 0)
                        } else {
                            Operand::UReg16(value, 0)
                        }
                    }
                    _ => Operand::Null,
                }
            }
            OpKind::TruthTable => {
                let mut s = String::new();
                for i in 0..4 {
                    s += &fx(f, &format!("{}{}", name, i))?.to_string();
                }
                Operand::Str(s)
            }
            &OpKind::Binary { size } => Operand::Str(format!("0b{:0w$b}", v()?, w = size as usize)),
            OpKind::Immediate { .. } => Operand::Int(v()? as i64),
            OpKind::Reg32F => Operand::Reg32(as_u32(v()?), 0),
            OpKind::UReg64F => Operand::UReg64(as_u32(v()?), 0),
            OpKind::Enum { values, .. } => {
                let x = v()?;
                match values.iter().find(|(k, _)| *k == x) {
                    Some((_, s)) => Operand::Str(s.to_string()),
                    None => Operand::Int(x as i64),
                }
            }
            OpKind::Shift => {
                let s = fx(f, "s")?;
                Operand::Str(if s != 0 { format!("lsl {}", s) } else { String::new() })
            }
            OpKind::Mask => {
                let m = v()?;
                Operand::Str(if m != 0 { format!("mask 0x{:X}", (1u128 << m) - 1) } else { String::new() })
            }
            &OpKind::BranchOffset { size } => Operand::RelOff(sign_extend(v()? as i128, size) as i64),
            OpKind::StackAdjustment => Operand::Imm(ImmVal::Int(sign_extend(v()? as i128, 16) as i64), 0),
            OpKind::StackReg32 => Operand::Reg32(as_u32(v()? >> 1), 0),
            &OpKind::Cond { float, .. } => {
                let x = fx(f, "cc")? | (fget(f, "ccn").unwrap_or(0) << 3);
                let table = if float { FCOND } else { ICOND };
                match table.iter().find(|(k, _)| *k == x) {
                    Some((_, s)) => Operand::Str(s.to_string()),
                    None => Operand::Int(x as i64),
                }
            }
            OpKind::MemoryShift => {
                let s = v()?;
                Operand::Str(if s != 0 { format!("lsl {}", s) } else { String::new() })
            }
            OpKind::MemoryIndex => {
                let value = v()?;
                if t()? != 0 {
                    Operand::Int(sign_extend(value as i128, 16) as i64)
                } else {
                    if value & 1 != 0 || value >= 0x100 {
                        return Err("AssertionError: memory index register".into());
                    }
                    Operand::Reg32(as_u32(value >> 1), 0)
                }
            }
            OpKind::ThreadgroupIndex => {
                let value = v()?;
                if t()? != 0 {
                    Operand::Int(sign_extend(value as i128, 16) as i64)
                } else {
                    if value >= 0x100 {
                        return Err("AssertionError: threadgroup index register".into());
                    }
                    Operand::Reg16(as_u32(value & 0xFF), 0)
                }
            }
            OpKind::AsyncMemoryReg => {
                let value = as_u32(v()?);
                if t()? != 0 { Operand::UReg64(value >> 1, 0) } else { Operand::Reg64(value >> 1, 0) }
            }
            OpKind::AsyncMemoryBase => {
                let value = as_u32(v()?);
                let count = if fx(f, "F")? != 0 { 5 } else { 3 };
                if t()? != 0 {
                    Operand::Tuple((0..count).map(|i| Operand::UReg32(value + i, 0)).collect(), 0)
                } else {
                    tuple32(value, count)
                }
            }
            OpKind::MemoryBase => {
                let value = as_u32(v()?);
                if value & 1 != 0 {
                    return Err("AssertionError: odd memory base".into());
                }
                if t()? != 0 { Operand::UReg64(value >> 1, 0) } else { Operand::Reg64(value >> 1, 0) }
            }
            &OpKind::SampleMask { flags_type } => {
                let flags = t()?;
                let value = as_u32(v()?);
                if flags >= 2 {
                    return Err("AssertionError: sample mask flags".into());
                }
                if (flags == 0) == (flags_type == 2) { Operand::Imm(ImmVal::Int(value as i64), 0) } else { Operand::Reg16(value, 0) }
            }
            OpKind::DiscardMask => {
                let value = as_u32(v()?);
                if t()? == 0 { Operand::Reg16(value, 0) } else { Operand::Imm(ImmVal::Int(value as i64), 0) }
            }
            OpKind::ZS => {
                let value = as_u32(v()?);
                let count = 2 * (fx(f, "z")? != 0) as u32 + (fx(f, "s")? != 0) as u32;
                if count == 0 {
                    return Err("AssertionError: zs_emit count".into());
                }
                tuple16(value, count)
            }
            OpKind::MemoryReg => {
                let value = as_u32(v()?);
                let count = fx(f, "mask")?.count_ones();
                if t()? == 0 { tuple16(value, count) } else { tuple32(value >> 1, count) }
            }
            &OpKind::TgMemoryReg { optional } => {
                if optional && fx(f, leak_a(name))? == 0 {
                    return Ok(Operand::Null);
                }
                let value = as_u32(v()?);
                let count = if optional { 1 } else { fx(f, "mask")?.count_ones() };
                if t()? == 0 { tuple16(value, count) } else { tuple32(value >> 1, count) }
            }
            OpKind::TileToMemoryReg => Operand::Reg16(as_u32(v()?), 0),
            OpKind::TgMemoryBase => {
                let flags = t()?;
                let value = as_u32(v()?);
                match flags {
                    0 => Operand::Reg16(value, 0),
                    0b10 => Operand::Imm(ImmVal::Int(0), 0),
                    _ => Operand::UReg16(value | ((flags as u32 >> 1) << 8), 0),
                }
            }
            OpKind::SReg32D => Operand::SReg32(as_u32(v()?)),
            OpKind::ExReg32 => Operand::Reg32(as_u32(v()?), 0),
            OpKind::ExReg64 => Operand::Reg64(as_u32(v()?), 0),
            OpKind::ExReg16 => Operand::Reg16(as_u32(v()?), 0),
            &OpKind::VarRegister { triple } => {
                let value = as_u32(v()?);
                let mut count = fx(f, "count")? as u32;
                if count == 0 {
                    count = 4;
                }
                if triple {
                    count *= 3;
                }
                if t()? == 0 { tuple16(value, count) } else { tuple32(value >> 1, count) }
            }
            OpKind::UVSIndex => {
                let value = as_u32(v()?);
                if t()? == 0 { Operand::Reg16(value, 0) } else { Operand::Imm(ImmVal::Int(value as i64), 0) }
            }
            OpKind::CFD => {
                let value = as_u32(v()?);
                if t()? == 0 { Operand::CF(value) } else { Operand::Reg16(value, 0) }
            }
            OpKind::CFPerspective => Operand::CF(as_u32(v()?)),
            OpKind::IterSampleID => {
                let value = as_u32(v()?);
                let interp = fx(f, "interpolation")?;
                if interp & 1 != 0 {
                    if interp & 2 != 0 { Operand::Reg16(value, 0) } else { Operand::Imm(ImmVal::Int(value as i64), 0) }
                } else {
                    if value != 0 {
                        return Err("AssertionError: sample id".into());
                    }
                    Operand::Str(String::new())
                }
            }
            OpKind::TextureStateDest => Operand::TextureState(as_u32(v()?)),
            OpKind::SamplerStateDest => Operand::SamplerState(as_u32(v()?)),
            OpKind::SampleUReg => {
                if fx(f, "Tt")? & 2 != 0 { Operand::UReg64(as_u32(v()?) * 2, 0) } else { Operand::Null }
            }
            OpKind::Texture => {
                let value = as_u32(v()?);
                match t()? {
                    0 => Operand::TextureState(value),
                    1 => Operand::Reg16(value, 0),
                    2 => {
                        if value == 0 { Operand::Imm(ImmVal::Int(0), 0) } else { Operand::Reg16(value, 0) }
                    }
                    _ => Operand::Reg32(value >> 1, 0),
                }
            }
            OpKind::Sampler => {
                let value = as_u32(v()?);
                if t()? == 0 { Operand::SamplerState(value) } else { Operand::Reg16(value, 0) }
            }
            OpKind::Coords => {
                let value = as_u32(v()?);
                let (count, extra) = tex_size(fx(f, "n")?)?;
                let mut tup = if fx(f, self.x2)? != 0 {
                    tuple16(value, count + extra)
                } else if extra != 0 {
                    tuple16(value, count * 2 + 1)
                } else {
                    tuple32(value >> 1, count)
                };
                if t()? != 0 {
                    tup.add_flag(F_DISCARD)?;
                }
                tup
            }
            OpKind::SampleCompareOff => {
                let mut value = as_u32(v()?);
                let has_offset = t()? != 0;
                let has_compare = fx(f, "compare")? != 0;
                let mut regs = vec![];
                if has_compare {
                    regs.push(Operand::Reg32(value >> 1, 0));
                    value += 2;
                }
                if has_offset {
                    regs.push(Operand::Reg16(value, 0));
                }
                Operand::Tuple(regs, 0)
            }
            OpKind::Lod => {
                let value = as_u32(v()?);
                let lod = fx(f, "lod")?;
                if lod == 0 {
                    Operand::Imm(ImmVal::Int(0), 0)
                } else if lod == 0b100 || lod == 0b1100 {
                    let mut count = tex_size(fx(f, "n")?)?.0 * 2;
                    if lod == 0b1100 {
                        count += 1;
                    }
                    tuple32(value >> 1, count)
                } else if lod == 0b010 || lod == 0b001 {
                    Operand::UReg16(value, 0)
                } else {
                    if lod != 0b110 && lod != 0b101 {
                        return Err("AssertionError: lod".into());
                    }
                    Operand::Reg16(value, 0)
                }
            }
            OpKind::PBELod => {
                let value = as_u32(v()?);
                if t()? == 0 { Operand::Reg16(value, 0) } else { Operand::Imm(ImmVal::Int(value as i64), 0) }
            }
            OpKind::Reg32_4_4 => Operand::Reg32(as_u32(v()?) >> 1, 0),
            OpKind::PBESourceReg => {
                let value = as_u32(v()?);
                if t()? == 0 {
                    tuple16(value, 4)
                } else {
                    if value & 1 != 0 {
                        return Err("AssertionError: odd PBE source".into());
                    }
                    tuple32(value >> 1, 4)
                }
            }
            OpKind::AtomicSource => {
                let value = as_u32(v()?);
                if value & 1 != 0 {
                    return Err("AssertionError: odd atomic source".into());
                }
                if fx(f, "op")? == 3 { Operand::Reg64(value >> 1, 0) } else { Operand::Reg32(value >> 1, 0) }
            }
            OpKind::AtomicDestination => {
                let value = as_u32(v()?);
                if t()? == 1 {
                    if value & 1 != 0 {
                        return Err("AssertionError: odd atomic destination".into());
                    }
                    Operand::Reg32(value >> 1, 0)
                } else {
                    Operand::Null
                }
            }
        })
    }

    /// Python `encode_string(fields, opstr)` (assembler). Kinds without one raise.
    pub fn encode_string(&self, f: &mut HashMap<&'static str, i64>, opstr: &str) -> Result<(), String> {
        let name = self.name;
        let err = |what: &str| Err(format!("invalid {} {:?}", what, opstr));
        match &self.kind {
            OpKind::ImplicitR0L => match try_parse_register(opstr) {
                Some(Operand::Reg16(0, fl)) => {
                    f.insert(self.t, if fl & F_CACHE != 0 { 1 } else { 0 });
                    Ok(())
                }
                _ => err("ImplicitR0LDesc"),
            },
            &OpKind::AluDst { paired, .. } => {
                let reg = if paired {
                    match try_parse_register_tuple(opstr) {
                        Some(Operand::Tuple(regs, fl)) if regs.len() == 2 => with_flags(&regs[0], fl),
                        _ => return err("paired ALUDstDesc"),
                    }
                } else {
                    match try_parse_register(opstr) {
                        Some(r) => r,
                        None => return err("ALUDstDesc"),
                    }
                };
                let (value, mut flags) = match reg {
                    Operand::Reg16(n, _) => (n, 0),
                    Operand::Reg32(n, _) => (n << 1, 2),
                    Operand::Reg64(n, _) => ((n << 1) | 1, 2),
                    _ => return err("ALUDstDesc"),
                };
                if reg.flags() & F_CACHE != 0 {
                    flags |= 1;
                }
                f.insert(self.t, flags);
                f.insert(name, value as i64);
                Ok(())
            }
            &OpKind::AluSrc { paired, sx, float, .. } => {
                let mut op = None;
                if paired {
                    if let Some(Operand::Tuple(regs, fl)) = try_parse_register_tuple(opstr) {
                        if regs.len() == 2 {
                            op = Some(Operand::Tuple(regs, fl));
                        }
                    }
                } else if let Some(r) = try_parse_register(opstr) {
                    op = Some(r);
                }
                let op = match op {
                    Some(o) => o,
                    None => match try_parse_integer(opstr) {
                        Some(v) => Operand::Imm(ImmVal::Int(v), 0),
                        None => return err("ALUSrcDesc"),
                    },
                };
                // ALUSrcDesc.encode
                let mut o = op.clone();
                if paired {
                    match &op {
                        Operand::Reg16(..) | Operand::Reg32(..) => return Err(format!("invalid paired operand {:?}", op)),
                        Operand::Tuple(regs, fl) => o = with_flags(&regs[0], *fl),
                        _ => {}
                    }
                }
                let (mut flags, value): (i64, i64) = match &o {
                    Operand::UReg16(n, _) | Operand::UReg32(n, _) => {
                        let mut flags = 0b0100;
                        let value = if matches!(o, Operand::UReg32(..)) {
                            flags |= 2;
                            (*n as i64) << 1
                        } else {
                            *n as i64
                        };
                        flags |= (value >> 8) & 1;
                        (flags, value & 0xFF)
                    }
                    Operand::Reg16(n, _) | Operand::Reg32(n, _) | Operand::Reg64(n, _) => {
                        let (flags, value) = match &o {
                            Operand::Reg16(..) => (0b0000, *n as i64),
                            Operand::Reg32(..) => (0b1000, (*n as i64) << 1),
                            _ => (0b1100, (*n as i64) << 1),
                        };
                        let fl = o.flags();
                        let hintb = if fl & F_CACHE != 0 { 2 } else if fl & F_DISCARD != 0 { 3 } else { 1 };
                        (flags | hintb, value)
                    }
                    Operand::Imm(ImmVal::Int(v), _) => {
                        if !(0..256).contains(v) {
                            return Err(format!("out of range immediate {}", v));
                        }
                        (0, *v)
                    }
                    _ => return err("ALUSrcDesc"),
                };
                if flags < 0 {
                    flags = 0;
                }
                f.insert(self.t, flags);
                f.insert(name, value);
                if sx {
                    let s = if op.is_register() && op.flags() & F_SX != 0 { 1 } else { 0 };
                    f.insert(self.x2, s);
                }
                if float {
                    let mut m = 0;
                    if op.is_register() {
                        if op.flags() & F_ABS != 0 {
                            m |= 1;
                        }
                        if op.flags() & F_NEG != 0 {
                            m |= 2;
                        }
                    }
                    f.insert(leak_m(name), m);
                }
                Ok(())
            }
            OpKind::CmpselSrc => {
                let op = match try_parse_register(opstr) {
                    Some(r) => r,
                    None => match try_parse_integer(opstr) {
                        Some(v) => Operand::Imm(ImmVal::Int(v), 0),
                        None => return err("CmpselSrcDesc"),
                    },
                };
                let dt32 = f.get("Dt").copied().ok_or("KeyError: Dt")? & 2 != 0;
                let (flags, value) = match &op {
                    Operand::UReg16(n, _) | Operand::UReg32(n, _) => {
                        let is32 = matches!(op, Operand::UReg32(..));
                        if is32 != dt32 {
                            return err("CmpselSrcDesc (mismatch with dest)");
                        }
                        let value = if is32 { (*n as i64) << 1 } else { *n as i64 };
                        (0b110 | ((value >> 8) & 1), value & 0xFF)
                    }
                    Operand::Reg16(n, _) | Operand::Reg32(n, _) => {
                        let is32 = matches!(op, Operand::Reg32(..));
                        if is32 != dt32 {
                            return err("CmpselSrcDesc (mismatch with dest)");
                        }
                        let value = if is32 { (*n as i64) << 1 } else { *n as i64 };
                        let fl = op.flags();
                        (if fl & F_CACHE != 0 { 2 } else if fl & F_DISCARD != 0 { 3 } else { 1 }, value)
                    }
                    Operand::Imm(ImmVal::Int(v), _) => {
                        if !(0..256).contains(v) {
                            return Err(format!("out of range immediate {}", v));
                        }
                        (0b100, *v)
                    }
                    _ => return err("CmpselSrcDesc"),
                };
                f.insert(self.t, flags);
                f.insert(name, value);
                Ok(())
            }
            OpKind::TruthTable => {
                if opstr.len() != 4 || !opstr.bytes().all(|b| b == b'0' || b == b'1') {
                    return err("TruthTable");
                }
                for (i, c) in opstr.bytes().enumerate() {
                    f.insert(leak(format!("tt{}", i)), (c - b'0') as i64);
                }
                Ok(())
            }
            OpKind::Binary { size } | OpKind::Immediate { size } => {
                let mut value = match try_parse_integer(opstr) {
                    Some(v) => v,
                    None => return err("IntegerFieldDesc"),
                };
                if value < 0 || value >= (1i64 << size) {
                    value &= (1i64 << size) - 1;
                }
                f.insert(name, value);
                Ok(())
            }
            OpKind::Reg32F => match try_parse_register(opstr) {
                Some(Operand::Reg32(n, _)) => {
                    f.insert(name, n as i64);
                    Ok(())
                }
                _ => err("Reg32Desc"),
            },
            OpKind::UReg64F => match try_parse_register(opstr) {
                Some(Operand::UReg64(n, _)) => {
                    f.insert(name, n as i64);
                    Ok(())
                }
                _ => err("UReg64Desc"),
            },
            OpKind::Enum { values, .. } => {
                if let Some((k, _)) = values.iter().find(|(_, v)| *v == opstr) {
                    f.insert(name, *k as i64);
                    return Ok(());
                }
                match try_parse_integer(opstr) {
                    Some(v) => {
                        f.insert(name, v);
                        Ok(())
                    }
                    None => err("enum"),
                }
            }
            OpKind::Shift | OpKind::MemoryShift => {
                let s = if opstr.is_empty() {
                    0
                } else if let Some(r) = opstr.strip_prefix("lsl ") {
                    match try_parse_integer(r) {
                        Some(v) => v,
                        None => return err("ShiftDesc"),
                    }
                } else {
                    return err("ShiftDesc");
                };
                f.insert(if matches!(self.kind, OpKind::Shift) { "s" } else { name }, s);
                Ok(())
            }
            OpKind::Mask => {
                if opstr.is_empty() {
                    f.insert(name, 0);
                    return Ok(());
                }
                if let Some(r) = opstr.strip_prefix("mask ") {
                    if let Some(mask) = try_parse_integer(r) {
                        let b = format!("{:b}", mask + 1);
                        if b.matches('1').count() == 1 {
                            let mut m = b.len() as i64 - 1;
                            if 0 < m && m <= 32 {
                                if m == 32 {
                                    m = 0;
                                }
                                f.insert(name, m);
                                return Ok(());
                            }
                        }
                    }
                }
                err("MaskDesc")
            }
            &OpKind::BranchOffset { size } => {
                if opstr.starts_with("pc") {
                    let s: String = opstr.chars().filter(|&c| c != ' ').collect();
                    if s.starts_with("pc-") || s.starts_with("pc+") {
                        if let Some(value) = try_parse_integer(&s[2..]) {
                            let masked = value & ((1i64 << size) - 1);
                            if value as i128 != sign_extend(masked as i128, size) {
                                return err("out of range BranchOffsetDesc");
                            }
                            f.insert(name, masked);
                            return Ok(());
                        }
                    }
                }
                err("BranchOffsetDesc")
            }
            OpKind::StackAdjustment => {
                let value = match try_parse_integer(opstr) {
                    Some(v) => v,
                    None => return err("StackAdjustmentDesc"),
                };
                let masked = value & 0xFFFF;
                if sign_extend(masked as i128, 16) != value as i128 {
                    return err("StackAdjustmentDesc (out of range)");
                }
                f.insert(name, masked);
                Ok(())
            }
            OpKind::StackReg32 => match try_parse_register(opstr) {
                Some(Operand::Reg32(n, _)) => {
                    f.insert(name, (n as i64) << 1);
                    Ok(())
                }
                _ => err("StackReg32Desc"),
            },
            &OpKind::Cond { float, has_ccn } => {
                let table = if float { FCOND } else { ICOND };
                let cond = match table.iter().find(|(_, s)| *s == opstr) {
                    Some((k, _)) => *k as i64,
                    None => return err("condition"),
                };
                f.insert("cc", cond & 0b111);
                if has_ccn {
                    f.insert("ccn", cond >> 3);
                } else if cond > 0b111 {
                    return err("condition (no ccn fields)");
                }
                Ok(())
            }
            OpKind::MemoryIndex => {
                if let Some(Operand::Reg32(n, _)) = try_parse_register(opstr) {
                    f.insert(self.t, 0);
                    f.insert(name, (n as i64) << 1);
                    return Ok(());
                }
                if let Some(v) = try_parse_integer(opstr) {
                    if !(0..0x100).contains(&v) {
                        return Err("AssertionError: memory index range".into());
                    }
                    f.insert(self.t, 1);
                    f.insert(name, v);
                    return Ok(());
                }
                err("MemoryIndexDesc")
            }
            OpKind::ThreadgroupIndex => {
                if let Some(Operand::Reg16(n, _)) = try_parse_register(opstr) {
                    f.insert(self.t, 0);
                    f.insert(name, (n as i64) << 1);
                    return Ok(());
                }
                if let Some(v) = try_parse_integer(opstr) {
                    if !(-0x8000..0x8000).contains(&v) {
                        return Err("AssertionError: threadgroup index range".into());
                    }
                    f.insert(self.t, 1);
                    f.insert(name, v & 0xFFFF);
                    return Ok(());
                }
                err("ThreadgroupIndexDesc")
            }
            OpKind::MemoryBase => match try_parse_register(opstr) {
                Some(r @ (Operand::Reg64(..) | Operand::UReg64(..))) => {
                    let (n, u) = match r {
                        Operand::Reg64(n, _) => (n, 0),
                        Operand::UReg64(n, _) => (n, 1),
                        _ => unreachable!(),
                    };
                    f.insert(self.t, u);
                    f.insert(name, (n as i64) << 1);
                    Ok(())
                }
                _ => err("MemoryBaseDesc"),
            },
            OpKind::MemoryReg | OpKind::TgMemoryReg { .. } => {
                let regs: Vec<Option<Operand>> = opstr.split('_').map(try_parse_register).collect();
                let (flags, value) = if regs.iter().all(|r| matches!(r, Some(Operand::Reg32(..)))) {
                    (1, match &regs[0] { Some(Operand::Reg32(n, _)) => (*n as i64) << 1, _ => 0 })
                } else if regs.iter().all(|r| matches!(r, Some(Operand::Reg16(..)))) {
                    (0, match &regs[0] { Some(Operand::Reg16(n, _)) => *n as i64, _ => 0 })
                } else {
                    return err("MemoryRegDesc");
                };
                let num = |r: &Option<Operand>| match r {
                    Some(Operand::Reg32(n, _)) | Some(Operand::Reg16(n, _)) => *n,
                    _ => 0,
                };
                for w in regs.windows(2) {
                    if num(&w[1]) != num(&w[0]) + 1 {
                        return err("MemoryRegDesc (must be consecutive)");
                    }
                }
                if regs.is_empty() || regs.len() > 4 {
                    return err("MemoryRegDesc (1-4 values)");
                }
                if matches!(self.kind, OpKind::TgMemoryReg { .. }) {
                    f.insert("mask", (1i64 << regs.len()) - 1);
                }
                f.insert(name, value);
                f.insert(self.t, flags);
                Ok(())
            }
            OpKind::SReg32D => {
                let mut s = opstr;
                if s.contains(" (") && s.contains(')') {
                    s = s.split(" (").next().unwrap();
                }
                if let Some(r) = s.strip_prefix("sr") {
                    if let Ok(v) = r.trim().parse::<i64>() {
                        if (0..256).contains(&v) {
                            f.insert(name, v);
                            return Ok(());
                        }
                    }
                }
                err("SReg32Desc")
            }
            OpKind::TextureStateDest => match try_parse_register(opstr) {
                Some(Operand::TextureState(n)) => {
                    f.insert(name, n as i64);
                    Ok(())
                }
                _ => err("TextureStateDestDesc"),
            },
            OpKind::SamplerStateDest => match try_parse_register(opstr) {
                Some(Operand::SamplerState(n)) => {
                    f.insert(name, n as i64);
                    Ok(())
                }
                _ => err("SamplerStateDestDesc"),
            },
            OpKind::Texture => {
                let (value, flags) = match try_parse_register(opstr) {
                    Some(Operand::Reg32(n, _)) => ((n as i64) << 1, 1),
                    Some(Operand::TextureState(n)) => (n as i64, 0),
                    _ => return err("TextureDesc"),
                };
                f.insert(name, value);
                f.insert(self.t, flags);
                Ok(())
            }
            OpKind::Sampler => {
                let (value, flags) = match try_parse_register(opstr) {
                    Some(Operand::Reg16(n, _)) => (n as i64, 1),
                    Some(Operand::SamplerState(n)) => (n as i64, 0),
                    _ => return err("SamplerDesc"),
                };
                f.insert(name, value);
                f.insert(self.t, flags);
                Ok(())
            }
            _ => Err(format!("AttributeError: no encode_string for {:?}", self.kind)),
        }
    }
}

fn with_flags(r: &Operand, fl: u8) -> Operand {
    match r {
        Operand::Reg16(n, _) => Operand::Reg16(*n, fl),
        Operand::Reg32(n, _) => Operand::Reg32(*n, fl),
        Operand::Reg64(n, _) => Operand::Reg64(*n, fl),
        o => o.clone(),
    }
}

fn leak_m(name: &str) -> &'static str {
    // interned per operand name; names are a small fixed set
    static CACHE: std::sync::OnceLock<std::sync::Mutex<HashMap<String, &'static str>>> = std::sync::OnceLock::new();
    let m = CACHE.get_or_init(Default::default);
    let mut g = m.lock().unwrap();
    if let Some(v) = g.get(name) {
        return v;
    }
    let v = leak(format!("{}m", name));
    g.insert(name.to_string(), v);
    v
}

fn leak_a(name: &str) -> &'static str {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<HashMap<String, &'static str>>> = std::sync::OnceLock::new();
    let m = CACHE.get_or_init(Default::default);
    let mut g = m.lock().unwrap();
    if let Some(v) = g.get(name) {
        return v;
    }
    let v = leak(format!("{}a", name));
    g.insert(name.to_string(), v);
    v
}
