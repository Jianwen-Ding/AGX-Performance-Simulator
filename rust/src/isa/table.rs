//! applegpu's `instruction_descriptors`, in registration order (decode and the assembler both take
//! the first match).

use super::desc::*;

/// Emulator semantics (applegpu `exec` / `exec_thread`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ex {
    /// InstructionDesc without `exec`: always raises AttributeError.
    None,
    /// MaskedInstructionDesc without `exec_thread`: raises only if some lane is active.
    MaskedTodo,
    MovImm(&'static str),
    GetSr,
    IAdd,
    IMAdd,
    Bfi,
    Bfeil,
    Extr,
    Shlhi,
    Shrhi,
    Asr,
    Asrh,
    Bitop,
    Bitrev,
    Popcount,
    Ffs,
    FMAdd32,
    FMAdd16,
    FAdd32,
    FAdd16,
    FMul32,
    FMul16,
    PopExec,
    IfCmp { float: bool },
    WhileCmp { float: bool },
    ElseCmp { float: bool },
    ICmpsel,
    FCmpsel,
    ICmpBallot,
    FCmpBallot,
    SimdShuffle,
    SimdShuffleDown,
    SimdShuffleUp,
    SimdShuffleRotateUp,
    SimdShuffleXor,
    Wait,
    DeviceLoad,
}

/// Mnemonic behaviour (fields_to_mnem / fields_for_mnem / map_to_alias / rewrite_operands_strings).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mn {
    Plain,
    /// SaturatableInstructionDesc; `alt` is the N=1 base name (isub / imsub), if any.
    Sat { alt: Option<&'static str> },
    MovImm16,
    MovImm32,
    Bitop,
    IfFcmp,
    ElseIcmp,
    LdstTile,
    Map,
}

#[derive(Debug)]
pub struct Desc {
    pub name: &'static str,
    pub sizes: (usize, usize),
    pub length_bit_pos: u32,
    pub mask: u128,
    pub bits: u128,
    pub fields: Vec<(u32, u32, &'static str)>,
    pub fields_mask: u128,
    /// merged field name -> [(start, size, shift)]
    pub merged: Vec<(&'static str, Vec<(u32, u32, u32)>)>,
    pub operands: Vec<OpDesc>,
    pub ex: Ex,
    pub mn: Mn,
}

impl Desc {
    fn new(name: &'static str, size: (usize, usize), lbp: u32, ex: Ex, mn: Mn) -> Self {
        Desc { name, sizes: size, length_bit_pos: lbp, mask: 0, bits: 0, fields: vec![], fields_mask: 0, merged: vec![],
               operands: vec![], ex, mn }
    }
    fn add_raw_field(&mut self, start: u32, size: u32, name: &'static str) {
        let mask = ((1u128 << size) - 1) << start;
        assert!(self.mask & mask == 0, "{} field {} overlaps a constant", self.name, name);
        assert!(self.fields_mask & mask == 0, "{} field {} overlaps a field", self.name, name);
        assert!(!self.fields.iter().any(|f| f.2 == name), "{} duplicate field {}", self.name, name);
        self.fields_mask |= mask;
        self.fields.push((start, size, name));
    }
    fn add_merged(&mut self, name: &'static str, subs: &[(u32, u32, &'static str)]) {
        let mut pairs = vec![];
        let mut shift = 0;
        for &(start, size, sub) in subs {
            self.add_raw_field(start, size, sub);
            pairs.push((start, size, shift));
            shift += size;
        }
        self.merged.push((name, pairs));
    }
    fn c(&mut self, start: u32, size: u32, value: u128) -> &mut Self {
        let mask = (1u128 << size) - 1;
        assert!(value & !mask == 0);
        assert!(self.mask & (mask << start) == 0, "{} constant overlap", self.name);
        self.mask |= mask << start;
        self.bits |= value << start;
        self
    }
    fn f(&mut self, start: u32, size: u32, name: &'static str) -> &mut Self {
        self.add_merged(name, &[(start, size, name)]);
        self
    }
    fn op(&mut self, o: OpDesc) -> &mut Self {
        for &(start, size, name) in &o.fields.clone() {
            self.add_merged(name, &[(start, size, name)]);
        }
        for (name, subs) in &o.merged.clone() {
            self.add_merged(name, subs);
        }
        self.operands.push(o);
        self
    }
}

const MEMORY_FORMATS: &[(u64, &str)] = &[
    (0, "i8"), (1, "i16"), (2, "i32"), (3, "f16"), (4, "u8norm"), (5, "s8norm"), (6, "u16norm"), (7, "s16norm"),
    (8, "rgb10a2"), (10, "srgba8"), (12, "rg11b10f"), (13, "rgb9e5"), (14, "rg11b10f.rtz"), (15, "rgb9e5.rtz"),
];
const MASK_DESCRIPTIONS: &[(u64, &str)] = &[
    (0, ""), (1, "x"), (2, "y"), (3, "xy"), (4, "z"), (5, "xz"), (6, "yz"), (7, "xyz"), (8, "w"), (9, "xw"), (10, "yw"),
    (11, "xyw"), (12, "zw"), (13, "xzw"), (14, "yzw"), (15, "xyzw"),
];
const SAMPLE_MASK_DESCRIPTIONS: &[(u64, &str)] = &[
    (1, "x"), (2, "y"), (3, "xy"), (4, "z"), (5, "xz"), (6, "yz"), (7, "xyz"), (8, "w"), (9, "xw"), (10, "yw"),
    (11, "xyw"), (12, "zw"), (13, "xzw"), (14, "yzw"), (15, "xyzw"),
];
const CONVERT_MODES: &[(u64, &str)] = &[
    (0, "u8_to_f"), (1, "s8_to_f"), (4, "f_to_u16"), (5, "f_to_s16"), (6, "u16_to_f"), (7, "s16_to_f"), (8, "f_to_u32"),
    (9, "f_to_s32"), (10, "u32_to_f"), (11, "s32_to_f"),
];
const CONVERT_ROUND: &[(u64, &str)] = &[(0, "rtz"), (1, "rte")];
const ATOMIC_OPCODES: &[(u64, &str)] = &[
    (0, "add"), (1, "sub"), (2, "xchg"), (3, "cmpxchg"), (4, "umin"), (5, "imin"), (6, "umax"), (7, "imax"), (8, "and"),
    (9, "or"), (10, "xor"),
];
const TEX_TYPES: &[(u64, &str)] = &[
    (0, "tex_1d"), (1, "tex_1d_array"), (2, "tex_2d"), (3, "tex_2d_array"), (4, "tex_2d_ms"), (5, "tex_3d"),
    (6, "tex_cube"), (7, "tex_cube_array"), (8, "tex_2d_ms_array"),
];
const SIGNEDNESS: &[(u64, &str)] = &[(0, "signed"), (1, "unsigned")];
const FORWARDING: &[(u64, &str)] = &[(0, ""), (1, "forward")];
const ELIDE: &[(u64, &str)] = &[(0, ""), (1, "elide")];
const INTERPOLATION: &[(u64, &str)] = &[(0, "center"), (1, "sample"), (2, "centroid"), (3, "sample")];
const ASYNC_F: &[(u64, &str)] = &[(0, "copy_1d"), (1, "copy_2d")];
const SAMPLE_MODE: &[(u64, &str)] = &[(0b00, ""), (0b10, "query_lod")];
const SAMPLE_Q6: &[(u64, &str)] = &[
    (0b00000, "none"), (0b00001, "compare"), (0b00010, "gather_r"), (0b00011, "gather_depth"), (0b00110, "gather_g"),
    (0b01010, "gather_b"), (0b01110, "gather_a"),
];
const LOD_MODES: &[(u64, &str)] = &[
    (0b0000, "auto_lod"), (0b0001, "auto_lod_bias"), (0b0010, "lod_min"), (0b0101, "auto_lod_bias"), (0b0110, "lod_min"),
    (0b0100, "lod_grad"), (0b1100, "lod_grad_min"),
];
const ROUND_PBE: &[(u64, &str)] = &[(0, "rte"), (1, "rtz")];
const MAP_OP: &[(u64, &str)] = &[(0, "unmap"), (1, "map")];
const MAP_TARGET: &[(u64, &str)] = &[(0, "target0"), (1, "target1")];

pub fn build() -> Vec<Desc> {
    let mut v: Vec<Desc> = Vec::new();
    use Ex as E;
    let s1 = |n: usize| (n, n);
    macro_rules! push {
        ($d:expr) => {{
            v.push($d);
        }};
    }
    // MovImm16
    let mut d = Desc::new("mov_imm", (4, 6), 15, E::MovImm("imm16"), Mn::MovImm16);
    d.c(0, 7, 0b1100010).op(alu_dst("D", 44)).c(8, 1, 0).op(immediate("imm16", 16, 16));
    push!(d);
    // MovImm32
    let mut d = Desc::new("mov_imm", (6, 8), 15, E::MovImm("imm32"), Mn::MovImm32);
    d.c(0, 7, 0b1100010).op(alu_dst("D", 60)).c(8, 1, 1).op(immediate("imm32", 16, 32)).op(binary("kill", 62, 1));
    push!(d);
    // MovFromSr
    let mut d = Desc::new("get_sr", s1(4), 15, E::GetSr, Mn::Plain);
    d.c(0, 7, 0b1110010).c(15, 1, 0).op(alu_dst("D", 28)).op(sreg32_desc("SR", 16, 26));
    push!(d);
    // IAdd
    let mut d = Desc::new("iadd", s1(8), 15, E::IAdd, Mn::Sat { alt: Some("isub") });
    d.f(6, 1, "S").c(0, 6, 0b001110).c(15, 1, 0).f(27, 1, "N");
    d.op(alu_dst64("D", 44)).op(add_src("A", 16, 42)).op(add_src("B", 28, 40)).op(shift_desc("shift"));
    push!(d);
    // IMAdd
    let mut d = Desc::new("imadd", s1(8), 15, E::IMAdd, Mn::Sat { alt: Some("imsub") });
    d.f(6, 1, "S").c(0, 6, 0b011110).c(15, 1, 0).f(27, 1, "N");
    d.op(alu_dst64("D", 60)).op(mul_src("A", 16, 58)).op(mul_src("B", 28, 56)).op(add_src("C", 40, 54)).op(shift_desc("shift"));
    push!(d);
    // Convert
    let mut d = Desc::new("convert", s1(6), 15, E::None, Mn::Plain);
    d.c(0, 7, 0b0111110).c(15, 1, 1).c(38, 2, 0).c(42, 2, 0);
    d.op(enum_desc("mode", 16, 6, CONVERT_MODES)).c(22, 4, 0).op(alu_dst("D", 44)).op(alu_src("src", 28, 40));
    d.op(enum_desc("round", 26, 2, CONVERT_ROUND));
    push!(d);
    // Bitfield family
    for (name, b15, b26, ex) in [("bfi", 0, 0, E::Bfi), ("bfeil", 1, 0, E::Bfeil), ("extr", 0, 1, E::Extr),
                                 ("shlhi", 0, 2, E::Shlhi), ("shrhi", 1, 2, E::Shrhi)] {
        let mut d = Desc::new(name, s1(8), 15, ex, Mn::Plain);
        d.c(0, 7, 0x2E).op(alu_dst("D", 60)).op(alu_src("A", 16, 58)).op(alu_src("B", 28, 56));
        d.op(alu_src("C", 40, 54)).op(mask_desc("m")).c(15, 1, b15).c(26, 2, b26);
        push!(d);
    }
    for (name, b26, ex) in [("asr", 1, E::Asr), ("asrh", 3, E::Asrh)] {
        let mut d = Desc::new(name, s1(8), 15, ex, Mn::Plain);
        d.c(0, 7, 0x2E).op(alu_dst("D", 60)).op(alu_src("A", 16, 58)).op(alu_src("B", 28, 56)).c(15, 1, 1).c(26, 2, b26);
        push!(d);
    }
    // Bitop
    let mut d = Desc::new("bitop", s1(6), 15, E::Bitop, Mn::Bitop);
    d.c(0, 7, 0x7E).c(15, 1, 0).op(truth_table("tt")).op(alu_dst("D", 44)).op(alu_src("A", 16, 42)).op(alu_src("B", 28, 40));
    push!(d);
    // IUnary / IBinary
    let iunary = |name: &'static str, ex: Ex| {
        let mut d = Desc::new(name, s1(6), 15, ex, Mn::Plain);
        d.c(0, 7, 0b0111110).c(15, 1, 0).op(alu_dst("D", 44)).op(alu_src("A", 16, 42)).c(28, 10, 0);
        d
    };
    let mut d = iunary("bitrev", E::Bitrev);
    d.c(38, 2, 0).c(26, 2, 0b01);
    push!(d);
    let mut d = iunary("popcount", E::Popcount);
    d.c(38, 2, 0).c(26, 2, 0b10);
    push!(d);
    let mut d = Desc::new("intl", s1(6), 15, E::MaskedTodo, Mn::Plain);
    d.c(0, 7, 0b0111110).c(15, 1, 0).op(alu_dst("D", 44)).op(alu_src("A", 16, 42)).op(alu_src("B", 28, 40));
    d.c(38, 2, 0).c(26, 2, 0);
    push!(d);
    let mut d = iunary("ffs", E::Ffs);
    d.c(38, 2, 0).c(26, 2, 0b11);
    push!(d);
    let mut d = iunary("iunop", E::MaskedTodo);
    d.op(binary("op1", 38, 2)).op(binary("op2", 26, 2));
    push!(d);
    // Float arithmetic
    let mut d = Desc::new("fmadd32", (6, 8), 15, E::FMAdd32, Mn::Sat { alt: None });
    d.c(0, 6, 0b111010).op(float_dst("D", 60)).op(float_src("A", 16, 58)).op(float_src("B", 28, 56)).op(float_src("C", 40, 54));
    push!(d);
    let mut d = Desc::new("fmadd16", (6, 8), 15, E::FMAdd16, Mn::Sat { alt: None });
    d.c(0, 6, 0b110110).op(float_dst16("D", 60)).op(float_src16("A", 16, 58)).op(float_src16("B", 28, 56));
    d.op(float_src16("C", 40, 54));
    push!(d);
    for (name, opc, half, ex) in [("fadd32", 0b101010, false, E::FAdd32), ("fadd16", 0b100110, true, E::FAdd16),
                                   ("fmul32", 0b011010, false, E::FMul32), ("fmul16", 0b010110, true, E::FMul16)] {
        let mut d = Desc::new(name, (4, 6), 15, ex, Mn::Sat { alt: None });
        d.c(0, 6, opc);
        if half {
            d.op(float_dst16("D", 44)).op(float_src16("A", 16, 42)).op(float_src16("B", 28, 40));
        } else {
            d.op(float_dst("D", 44)).op(float_src("A", 16, 42)).op(float_src("B", 28, 40));
        }
        push!(d);
    }
    let funary = |name: &'static str| {
        let mut d = Desc::new(name, (4, 6), 15, E::MaskedTodo, Mn::Sat { alt: None });
        d.c(0, 6, 0b001010).op(float_dst("D", 44)).op(float_src("A", 16, 42)).c(34, 8, 0);
        d
    };
    for (name, op) in [("floor", 0b000000), ("ceil", 0b010000), ("trunc", 0b100000), ("rint", 0b110000),
                       ("rcp", 0b001000), ("rsqrt", 0b001001), ("rsqrt_special", 0b000001), ("sin_pt_1", 0b001010),
                       ("sin_pt_2", 0b001110), ("log2", 0b001100), ("exp2", 0b001101)] {
        let mut d = funary(name);
        d.c(28, 6, op);
        push!(d);
    }
    for (name, op) in [("dfdx", 0b000100), ("dfdy", 0b000110)] {
        let mut d = funary(name);
        d.c(28, 6, op).f(46, 1, "kill");
        push!(d);
    }
    let mut d = funary("funop");
    d.op(binary("op", 28, 6));
    push!(d);
    // Flow control
    let mut d = Desc::new("ret", s1(2), 15, E::None, Mn::Plain);
    d.c(0, 7, 0b0010100).op(reg32_desc("reg32", 9, 7));
    push!(d);
    let mut d = Desc::new("stop", s1(2), 15, E::None, Mn::Plain);
    d.c(0, 16, 0x88);
    push!(d);
    let mut d = Desc::new("trap", s1(2), 15, E::None, Mn::Plain);
    d.c(0, 16, 0x08);
    push!(d);
    let mut d = Desc::new("call", s1(2), 15, E::None, Mn::Plain);
    d.c(0, 7, 0b0000100).op(reg32_desc("reg32", 9, 7));
    push!(d);
    let mut d = Desc::new("jmp_incomplete", s1(4), 15, E::None, Mn::Plain);
    d.c(0, 16, 0).c(24, 8, 0).op(branch_offset("off", 16, 8));
    push!(d);
    let mut d = Desc::new("jmp_exec_any", s1(6), 15, E::MaskedTodo, Mn::Plain);
    d.c(0, 16, 0b1100000000000000).op(branch_offset("off", 16, 32));
    push!(d);
    let mut d = Desc::new("jmp_exec_none", s1(6), 15, E::MaskedTodo, Mn::Plain);
    d.c(0, 16, 0b1100000000100000).op(branch_offset("off", 16, 32));
    push!(d);
    let mut d = Desc::new("call", s1(6), 15, E::MaskedTodo, Mn::Plain);
    d.c(0, 16, 0xC010).op(branch_offset("off", 16, 32));
    push!(d);
    let mut d = Desc::new("pop_exec", s1(6), 15, E::PopExec, Mn::Plain);
    d.c(0, 7, 0x52).c(9, 2, 3).c(13, 35, 0).op(implicit_r0l("D")).op(immediate("n", 11, 2));
    push!(d);
    let icmp_mask = |name: &'static str, op9: u128, ex: Ex, mn: Mn| {
        let mut d = Desc::new(name, s1(6), 15, ex, mn);
        d.c(0, 7, 0x52).op(implicit_r0l("D")).op(icond(13, Some(8))).op(alu_src("A", 16, 42)).op(alu_src("B", 28, 40));
        d.c(26, 2, 0).c(38, 2, 0).c(44, 2, 0).op(immediate("n", 11, 2)).c(9, 2, op9);
        d
    };
    let fcmp_mask = |name: &'static str, op9: u128, ex: Ex, mn: Mn| {
        let mut d = Desc::new(name, s1(6), 15, ex, mn);
        d.c(0, 7, 0x42).op(implicit_r0l("D")).op(fcond(13, Some(8))).op(float_src("A", 16, 42)).op(float_src("B", 28, 40));
        d.c(44, 2, 0).op(immediate("n", 11, 2)).c(9, 2, op9);
        d
    };
    push!(icmp_mask("if_icmp", 0, E::IfCmp { float: false }, Mn::Plain));
    push!(fcmp_mask("if_fcmp", 0, E::IfCmp { float: true }, Mn::IfFcmp));
    push!(icmp_mask("while_icmp", 2, E::WhileCmp { float: false }, Mn::Plain));
    push!(fcmp_mask("while_fcmp", 2, E::WhileCmp { float: true }, Mn::Plain));
    push!(icmp_mask("else_icmp", 1, E::ElseCmp { float: false }, Mn::ElseIcmp));
    push!(fcmp_mask("else_fcmp", 1, E::ElseCmp { float: true }, Mn::Plain));
    // Select
    let mut d = Desc::new("icmpsel", (8, 10), 15, E::ICmpsel, Mn::Plain);
    d.c(0, 7, 0x12).op(icond(61, None)).op(alu_dst("D", 76)).op(alu_src("A", 16, 74)).op(alu_src("B", 28, 72));
    d.op(cmpsel_src("X", 40, 70)).op(cmpsel_src("Y", 52, 68));
    push!(d);
    let mut d = Desc::new("fcmpsel", (8, 10), 15, E::FCmpsel, Mn::Plain);
    d.c(0, 7, 0x02).op(fcond(61, None)).op(alu_dst("D", 76)).op(float_src("A", 16, 74)).op(float_src("B", 28, 72));
    d.op(cmpsel_src("X", 40, 70)).op(cmpsel_src("Y", 52, 68));
    push!(d);
    // Ballots
    let mut d = Desc::new("icmp_ballot", s1(8), 15, E::ICmpBallot, Mn::Plain);
    d.c(0, 7, 0b0110010).op(alu_dst("D", 44)).op(icond(61, Some(47))).op(alu_src("A", 16, 42)).op(alu_src("B", 28, 40));
    d.c(26, 2, 0).c(38, 2, 0).c(48, 13, 1);
    push!(d);
    let mut d = Desc::new("icmp_quad_ballot", s1(8), 15, E::None, Mn::Plain);
    d.c(0, 7, 0b0110010).op(alu_dst("D", 44)).op(icond(61, Some(47))).op(alu_src("A", 16, 42)).op(alu_src("B", 28, 40));
    d.c(26, 2, 0).c(38, 2, 0).c(48, 13, 0);
    push!(d);
    let mut d = Desc::new("fcmp_ballot", s1(8), 15, E::FCmpBallot, Mn::Plain);
    d.c(0, 7, 0b0100010).op(alu_dst("D", 44)).op(fcond(61, Some(47))).op(float_src("A", 16, 42)).op(float_src("B", 28, 40));
    d.c(48, 13, 1);
    push!(d);
    let mut d = Desc::new("fcmp_quad_ballot", s1(8), 15, E::None, Mn::Plain);
    d.c(0, 7, 0b0100010).op(alu_dst("D", 44)).op(fcond(61, Some(47))).op(float_src("A", 16, 42)).op(float_src("B", 28, 40));
    d.c(48, 13, 0);
    push!(d);
    // SIMD shuffles
    for (name, b47, b38, ex) in [("simd_shuffle", 0, 0b00, E::SimdShuffle), ("simd_shuffle_down", 0, 0b11, E::SimdShuffleDown),
                                 ("simd_shuffle_up", 0, 0b10, E::SimdShuffleUp),
                                 ("simd_shuffle_rotate_up", 1, 0b10, E::SimdShuffleRotateUp),
                                 ("simd_shuffle_xor", 0, 0b01, E::SimdShuffleXor)] {
        let mut d = Desc::new(name, s1(6), 15, ex, Mn::Plain);
        d.c(0, 7, 0b01101111).c(15, 1, 0).op(alu_dst("D", 44)).op(alu_src("A", 16, 42)).op(alu_src16("B", 28, 40));
        d.c(47, 1, b47).c(38, 2, b38).c(26, 2, 0b01);
        push!(d);
    }
    for (name, b26) in [("simd_matrix_fmadd32", 1), ("simd_matrix_fmadd16", 0)] {
        let mut d = Desc::new(name, s1(8), 15, E::MaskedTodo, Mn::Plain);
        d.c(0, 7, 0x6f).c(15, 1, 0).c(27, 1, 1).c(26, 1, b26).op(paired_alu_dst("D", 60));
        d.op(paired_float_src("A", 16, 58, Some(52))).op(paired_float_src("B", 28, 56, None));
        d.op(paired_float_src("C", 40, 54, None)).c(63, 1, 1);
        push!(d);
    }
    for (op1, op2, op3, name) in [(Some(0), 0b00, 0b00, "quad_shuffle"), (Some(0), 0b11, 0b00, "quad_shuffle_down"),
                                  (Some(0), 0b10, 0b00, "quad_shuffle_up"), (Some(1), 0b10, 0b00, "quad_shuffle_rotate_up"),
                                  (Some(0), 0b01, 0b00, "quad_shuffle_xor"), (None, 0, 0, "simd_shuf_op")] {
        let mut d = Desc::new(name, s1(6), 15, E::None, Mn::Plain);
        d.c(0, 7, 0b01101111).c(15, 1, 0);
        match op1 {
            None => {
                d.op(binary("op1", 47, 1)).op(binary("op2", 38, 2)).op(binary("op3", 26, 2));
            }
            Some(o1) => {
                d.c(47, 1, o1).c(38, 2, op2).c(26, 2, op3);
            }
        }
        d.op(alu_dst("D", 44)).op(alu_src64("A", 16, 42)).op(alu_src64("B", 28, 40));
        push!(d);
    }
    let reductions: &[(Option<u128>, Option<u128>, &'static str)] = &[
        (Some(0), Some(0b0000000000000000), "quad_and"), (Some(0), Some(0b0001000000000000), "quad_or"),
        (Some(0), Some(0b0010000000000000), "quad_xor"), (Some(1), Some(0b0000000000000000), "quad_iadd"),
        (Some(0), Some(0b0000000000000100), "quad_fadd"), (Some(0), Some(0b0001000000000100), "quad_fmul"),
        (Some(1), Some(0b0000000000010000), "quad_prefix_iadd"), (Some(0), Some(0b0000000000010100), "quad_prefix_fadd"),
        (Some(0), Some(0b0001000000010100), "quad_prefix_fmul"), (Some(1), Some(0b0110000000000000), "quad_min.u"),
        (Some(1), Some(0b0111000000000000), "quad_max.u"), (Some(1), Some(0b0010000000000000), "quad_min.s"),
        (Some(1), Some(0b0011000000000000), "quad_max.s"), (Some(0), Some(0b0010000000000100), "quad_min.f"),
        (Some(0), Some(0b0011000000000100), "quad_max.f"), (Some(0), Some(0b0000000000001000), "simd_and"),
        (Some(0), Some(0b0001000000001000), "simd_or"), (Some(0), Some(0b0010000000001000), "simd_xor"),
        (Some(1), Some(0b0000000000001000), "simd_iadd"), (Some(0), Some(0b0000000000001100), "simd_fadd"),
        (Some(0), Some(0b0001000000001100), "simd_fmul"), (Some(1), Some(0b0000000000011000), "simd_prefix_iadd"),
        (Some(0), Some(0b0000000000011100), "simd_prefix_fadd"), (Some(0), Some(0b0001000000011100), "simd_prefix_fmul"),
        (Some(0), Some(0b0000000000011000), "simd_prefix_and"), (Some(0), Some(0b0001000000011000), "simd_prefix_or"),
        (Some(0), Some(0b0010000000011000), "simd_prefix_xor"), (Some(1), Some(0b0010000000011000), "simd_prefix_min.s"),
        (Some(1), Some(0b0110000000011000), "simd_prefix_min.u"), (Some(1), Some(0b0011000000011000), "simd_prefix_max.s"),
        (Some(1), Some(0b0111000000011000), "simd_prefix_max.u"), (Some(1), Some(0b0110000000001000), "simd_min.u"),
        (Some(1), Some(0b0111000000001000), "simd_max.u"), (Some(1), Some(0b0010000000001000), "simd_min.s"),
        (Some(1), Some(0b0011000000001000), "simd_max.s"), (Some(0), Some(0b0010000000001100), "simd_min.f"),
        (Some(0), Some(0b0011000000001100), "simd_max.f"), (None, None, "simd_op"),
    ];
    for &(op1, op2, name) in reductions {
        let mut d = Desc::new(name, s1(6), 15, E::None, Mn::Plain);
        d.c(0, 7, 0b01101111).c(15, 1, 1);
        match op1 {
            None => {
                d.op(binary("op1", 47, 1));
            }
            Some(x) => {
                d.c(47, 1, x);
            }
        }
        match op2 {
            None => {
                d.op(binary("op2", 26, 16));
            }
            Some(x) => {
                d.c(26, 16, x);
            }
        }
        d.op(alu_dst("D", 44)).op(alu_src64("A", 16, 42));
        push!(d);
    }
    for (name, c) in [("wait_pix", 0b01001000), ("signal_pix", 0b01011000)] {
        let mut d = Desc::new(name, s1(4), 15, E::None, Mn::Plain);
        d.c(0, 8, c).op(immediate("i", 8, 10)).op(immediate("j", 22, 2));
        push!(d);
    }
    let mut d = Desc::new("wait", s1(2), 15, E::Wait, Mn::Plain);
    d.c(0, 8, 0b111000).op(immediate("i", 8, 1));
    push!(d);
    // ld/st tile
    let mut d = Desc::new("TODO.ld/st_tile", (8, 10), 15, E::None, Mn::LdstTile);
    d.c(0, 6, 0x09).op(immediate("load", 6, 1)).op(tg_memory_reg("R", None)).op(enum_desc("F", 24, 4, MEMORY_FORMATS));
    d.op(immediate("u0", 35, 1)).op(enum_desc("mask", 36, 4, MASK_DESCRIPTIONS));
    d.op(immediate_list("O", &[(28, 7, "A"), (40, 2, "Ax")])).op(sample_mask("S", 42, 56, 22, 2));
    d.op(immediate_list("C", &[(16, 6, "C"), (58, 2, "Cx")]));
    push!(d);
    // iter / iterproj
    for (name, projected) in [("iter", false), ("iterproj", true)] {
        let mut d = Desc::new(name, (4, 8), 15, E::None, Mn::Plain);
        d.c(0, 6, 0x21).c(6, 1, projected as u128).c(7, 1, 0).op(var_register("D", false)).op(cf_desc("I", 16, 58));
        if projected {
            d.op(cf_perspective("J", 24, 60));
        }
        d.op(iter_sample_id("S", 32)).op(enum_desc("forwarding", 22, 1, FORWARDING)).op(enum_desc("elide", 46, 1, ELIDE));
        d.op(enum_desc("interpolation", 48, 2, INTERPOLATION)).op(binary("kill", 52, 1)).op(immediate("unk1", 40, 6));
        d.op(immediate("unk2", 47, 1)).op(immediate("unk3", 50, 2)).op(immediate("unk4", 53, 3)).op(immediate("unk5", 62, 2));
        push!(d);
    }
    let mut d = Desc::new("TODO.ldcf", (4, 8), 15, E::None, Mn::Plain);
    d.c(0, 6, 0x21).c(6, 2, 0b10).op(var_register("D", true)).op(cf_desc("I", 16, 58));
    push!(d);
    let mut d = Desc::new("uniform_store", (6, 8), 47, E::None, Mn::Plain);
    d.op(immediate("unk", 25, 2)).c(0, 7, 0b1000101).c(9, 1, 0).c(28, 2, 0b11).c(50, 2, 0);
    d.op(enum_desc("F", 7, 2, MEMORY_FORMATS)).op(enum_desc("mask", 52, 4, MASK_DESCRIPTIONS)).op(immediate("b", 44, 3));
    d.op(memory_reg("R", 10, 40, 49)).c(16, 4, 0).c(36, 4, 0).c(27, 1, 1).op(memory_index("O")).op(memory_shift("s"));
    push!(d);
    let mut d = Desc::new("tex_state_store", s1(8), 15, E::None, Mn::Plain);
    d.c(0, 8, 0b11101101).c(20, 1, 1).op(texture_state_dest("TS")).op(ureg64_desc("B", 58, 6)).op(reg32_desc("O", 27, 7));
    push!(d);
    let mut d = Desc::new("sampler_state_store", s1(8), 15, E::None, Mn::Plain);
    d.c(0, 8, 0b10101101).c(20, 1, 1).op(sampler_state_dest("SS")).op(ureg64_desc("B", 58, 6)).op(reg32_desc("O", 27, 7));
    push!(d);
    for (op, mnem) in [(0b01010001, "no_var"), (0b00010001, "st_var"), (0b10010001, "st_var_final")] {
        let mut d = Desc::new(mnem, s1(4), 15, E::None, Mn::Plain);
        d.c(0, 10, op).c(22, 1, 0).op(immediate("u", 31, 1)).op(ex_reg32("r", 10, 24)).op(uvs_index("I"));
        push!(d);
    }
    for (name, bit) in [("async_store", 1u128), ("async_load", 0)] {
        let mut d = Desc::new(name, s1(8), 15, E::MaskedTodo, Mn::Plain);
        d.c(0, 7, 0b0100101 | (bit << 6)).op(enum_desc("F", 47, 1, ASYNC_F)).op(async_memory_reg("R"));
        d.op(async_memory_base("A"));
        push!(d);
    }
    let device_ls = |name: &'static str, bit: u128, ex: Ex| {
        let mut d = Desc::new(name, (6, 8), 47, ex, Mn::Plain);
        d.op(immediate("g", 30, 1)).c(0, 7, 0b0000101 | (bit << 6));
        d.op(enum_list("F", &[(7, 3, "F"), (48, 1, "Fx")], MEMORY_FORMATS)).op(enum_desc("mask", 52, 4, MASK_DESCRIPTIONS));
        d.op(memory_reg("R", 10, 40, 49)).op(memory_base("A")).op(memory_index("O")).op(enum_desc("Ou", 25, 1, SIGNEDNESS));
        d.op(memory_shift("s"));
        d
    };
    let mut d = device_ls("device_load", 0, E::DeviceLoad);
    d.c(26, 1, 1).c(28, 2, 0).c(44, 3, 4).c(50, 2, 0);
    push!(d);
    let mut d = device_ls("device_store", 1, E::MaskedTodo);
    d.op(immediate("u6", 44, 1)).c(26, 1, 1).c(28, 2, 0).c(45, 2, 2).c(50, 2, 0);
    push!(d);
    let mut d = device_ls("device_load.TODO", 0, E::MaskedTodo);
    d.op(immediate("u1", 26, 1)).op(immediate("u3", 28, 2)).op(immediate("u4", 44, 3)).op(immediate("u5", 50, 2));
    push!(d);
    let mut d = device_ls("device_store.TODO", 1, E::MaskedTodo);
    d.op(immediate("u6", 44, 1)).op(immediate("u1", 26, 1)).op(immediate("u3", 28, 2)).op(immediate("u4", 45, 2));
    d.op(immediate("u5", 50, 2));
    push!(d);
    for (name, bit) in [("stack_store", 1u128), ("stack_load", 0)] {
        let mut d = Desc::new(name, (6, 8), 47, E::None, Mn::Plain);
        d.c(0, 8, (bit << 7) | 0b00110101);
        if bit == 0 {
            d.op(memory_reg("R", 10, 40, 49));
        }
        d.op(enum_list("F", &[(8, 2, "F"), (50, 2, "Fx")], MEMORY_FORMATS)).op(immediate("i1", 26, 1));
        d.op(immediate("i2", 36, 3)).op(enum_desc("mask", 52, 4, MASK_DESCRIPTIONS)).op(immediate("i5", 44, 3));
        if bit == 1 {
            d.op(memory_reg("R", 10, 40, 49));
        }
        d.op(memory_index("O")).op(immediate("i6", 30, 1));
        push!(d);
    }
    let mut d = Desc::new("stack_get_ptr", (6, 8), 47, E::None, Mn::Plain);
    d.c(0, 8, 0b00110101).c(16, 4, 0b0001).c(48, 2, 0b10).op(immediate("i0", 8, 2)).op(immediate("i1", 26, 1));
    d.op(immediate("i2", 36, 3)).op(immediate("i4", 50, 6)).op(immediate("i3", 44, 3));
    d.op(stack_reg32("R", &[(10, 6, "R"), (40, 2, "Rx")]));
    push!(d);
    let mut d = Desc::new("TODO.stack_adjust", (6, 8), 47, E::None, Mn::Plain);
    d.c(0, 8, 0b10110101).c(16, 4, 0b0001).c(24, 2, 0b01).op(immediate("i0", 8, 2)).op(immediate("i1", 26, 1));
    d.op(immediate("i2", 36, 3)).op(immediate("i3", 44, 3)).op(immediate("i4", 50, 6)).op(stack_adjustment("v"));
    push!(d);
    for (name, bit) in [("threadgroup_load", 1u128), ("threadgroup_store", 0)] {
        let mut d = Desc::new(name, (6, 8), 15, E::None, Mn::Plain);
        d.c(0, 4, 0b1001).c(5, 2, 0b01 | (bit << 1)).op(enum_desc("F", 24, 4, MEMORY_FORMATS));
        d.op(enum_desc("mask", 36, 4, MASK_DESCRIPTIONS)).op(tg_memory_reg("R", None)).op(tg_memory_base("A"));
        d.op(threadgroup_index("O"));
        push!(d);
    }
    for (name, c) in [("texture_sample", 0x31), ("texture_load", 0x71)] {
        let mut d = Desc::new(name, (8, 12), 15, E::None, Mn::Plain);
        d.op(immediate("compare", 23, 1)).op(enum_desc("mode", 30, 2, SAMPLE_MODE)).op(binary("q3", 43, 4));
        d.op(binary("slot", 63, 1)).op(enum_desc("q6", 86, 5, SAMPLE_Q6)).op(enum_desc("mask", 48, 4, SAMPLE_MASK_DESCRIPTIONS));
        d.op(binary("kill", 69, 2)).op(memory_reg("R", 9, 72, 8)).op(sample_ureg("U", 64)).op(texture_desc("T", 78));
        d.op(sampler_desc("S")).op(enum_list("n", &[(40, 3, "n"), (71, 1, "nx")], TEX_TYPES)).op(coords_desc("C", 74));
        d.op(enum_desc("lod", 52, 4, LOD_MODES)).op(lod_desc("D", 24, 76)).op(sample_compare_off("O"));
        d.c(0, 8, c);
        push!(d);
    }
    let mut d = Desc::new("threadgroup_barrier", s1(2), 15, E::None, Mn::Plain);
    d.c(0, 8, 0x68);
    push!(d);
    let mut d = Desc::new("atomic", s1(8), 15, E::None, Mn::Plain);
    d.c(0, 6, 0x15).op(enum_desc("op", 6, 4, ATOMIC_OPCODES)).op(immediate("g", 30, 1)).op(atomic_destination("R"));
    d.op(memory_base("A")).op(memory_index("O")).op(enum_desc("Ou", 25, 1, SIGNEDNESS)).op(atomic_source("S", 48));
    d.op(immediate("u1", 26, 1)).op(immediate("u2", 28, 2)).op(immediate("u3", 31, 1)).op(immediate("u4", 42, 3));
    d.op(immediate("u5", 45, 2));
    push!(d);
    let mut d = Desc::new("threadgroup_atomic", (6, 10), 15, E::None, Mn::Plain);
    d.c(0, 6, 0x19).op(enum_desc("op", 24, 4, ATOMIC_OPCODES)).op(tg_memory_reg("R", Some(38))).op(tg_memory_base("A"));
    d.op(threadgroup_index("O")).op(atomic_source("S", 64)).op(immediate("u1", 6, 2)).op(immediate("u3", 35, 3));
    d.op(immediate("u4", 40, 8)).op(immediate("u5", 72, 8));
    push!(d);
    for (name, size, c, csize) in [("TODO.after_sampling1", 2, 0x101C, 16), ("TODO.after_sampling2", 2, 0x62CC, 16)] {
        let mut d = Desc::new(name, s1(size), 15, E::None, Mn::Plain);
        d.c(0, csize, c);
        push!(d);
    }
    let mut d = Desc::new("TODO.unk30C0", s1(6), 15, E::None, Mn::Plain);
    d.c(0, 16, 0xC030).op(immediate("imm", 16, 32));
    push!(d);
    let mut d = Desc::new("TODO.unk40C0", s1(6), 15, E::None, Mn::Plain);
    d.c(0, 48, 0xC040);
    push!(d);
    let mut d = Desc::new("TODO.unkC0000000", s1(4), 15, E::None, Mn::Plain);
    d.c(0, 32, 0xC0);
    push!(d);
    let mut d = Desc::new("TODO.unk51", s1(4), 15, E::None, Mn::Plain);
    d.c(0, 31, 0x51);
    push!(d);
    let mut d = Desc::new("TODO.unkE800", s1(2), 15, E::None, Mn::Plain);
    d.c(0, 16, 0xE8);
    push!(d);
    let mut d = Desc::new("TODO.unk0C00", s1(2), 15, E::None, Mn::Plain);
    d.c(0, 16, 0x0C);
    push!(d);
    let mut d = Desc::new("doorbell", s1(2), 15, E::None, Mn::Plain);
    d.c(0, 8, 0x28).op(immediate("imm", 8, 8));
    push!(d);
    let mut d = Desc::new("jmp_if_skipping_doorbell", s1(4), 15, E::None, Mn::Plain);
    d.c(0, 16, 0x20).op(branch_offset("off", 16, 8)).c(24, 8, 0);
    push!(d);
    let mut d = Desc::new("image_write_block", (6, 10), 15, E::None, Mn::Plain);
    d.c(0, 8, 0xB1).op(tile_to_memory_reg("R")).op(coords_desc("C", 58)).op(pbe_lod("D")).op(texture_desc("T", 62));
    d.op(enum_list("n", &[(40, 3, "n"), (55, 1, "nx")], TEX_TYPES)).c(48, 5, 0).op(enum_desc("round", 53, 1, ROUND_PBE));
    d.op(enum_list("F", &[(8, 1, "F"), (64, 3, "Fx")], MEMORY_FORMATS)).c(68, 10, 0).op(immediate("q0", 23, 1));
    d.op(immediate("q1", 43, 4)).op(immediate("q2", 67, 1));
    push!(d);
    let mut d = Desc::new("TODO.stack_adjust2", (6, 8), 47, E::None, Mn::Plain);
    d.c(0, 8, 0b10110101).c(16, 4, 0b0001).c(24, 2, 0b00).op(immediate("i0", 8, 2)).op(immediate("i1", 26, 1));
    d.op(immediate("i2", 36, 3)).op(immediate("i3", 44, 3)).op(immediate("i4", 50, 6)).op(reg32_4_4("r", 20, 32));
    push!(d);
    let mut d = Desc::new("image_write", (6, 8), 15, E::MaskedTodo, Mn::Plain);
    d.c(0, 8, 0xF1).op(pbe_source_reg("R")).op(coords_desc("C", 58)).op(pbe_lod("D")).op(sample_ureg("U", 48));
    d.op(texture_desc("T", 62)).op(enum_list("n", &[(40, 3, "n"), (55, 1, "nx")], TEX_TYPES));
    d.op(enum_desc("round", 53, 1, ROUND_PBE)).op(immediate("u1", 23, 1)).op(immediate("u2", 30, 1));
    d.op(immediate("u3", 43, 4)).op(immediate("u5", 54, 1));
    push!(d);
    let mut d = Desc::new("TODO.sr_thing", s1(4), 15, E::MaskedTodo, Mn::Plain);
    d.c(0, 7, 0b1110010).c(15, 1, 1).op(alu_dst("D", 28)).op(sreg32_desc("SR", 16, 26));
    push!(d);
    let mut d = Desc::new("TODO.pop_exec2", s1(6), 15, E::MaskedTodo, Mn::Plain);
    d.c(0, 7, 0x52).c(9, 2, 3).c(13, 10, 0).c(23, 1, 1).op(implicit_r0l("D")).op(immediate("n", 11, 2));
    push!(d);
    let mut d = Desc::new("map", s1(8), 15, E::MaskedTodo, Mn::Map);
    d.c(0, 8, 0x75).c(10, 1, 0).c(47, 1, 1).op(enum_desc("op", 16, 4, MAP_OP)).op(enum_desc("target", 36, 1, MAP_TARGET));
    d.c(24, 2, 1).c(27, 3, 0).c(31, 1, 0).op(ex_reg32("R", 11, 40)).op(stack_adjustment("v"));
    d.op(binary("q1", 8, 2)).op(binary("q2", 30, 1)).op(binary("q3", 37, 3)).op(binary("q4", 42, 5)).op(binary("q5", 48, 8));
    push!(d);
    let mut d = Desc::new("TODO.unk75_alt", s1(8), 15, E::MaskedTodo, Mn::Plain);
    d.c(0, 8, 0x75).c(10, 1, 0).c(47, 1, 1).op(immediate("q16", 16, 4)).op(immediate("q17", 24, 2));
    d.op(immediate("q18", 27, 3)).op(immediate("q19", 31, 1)).op(ex_reg32("R", 11, 40)).op(stack_adjustment("v"));
    d.op(binary("q1", 8, 2));
    push!(d);
    let mut d = Desc::new("sample_mask", s1(4), 15, E::MaskedTodo, Mn::Plain);
    d.c(0, 8, 0xC1).op(sample_mask("S", 9, 24, 8, 1)).op(discard_mask("M")).c(15, 1, 0);
    push!(d);
    let mut d = Desc::new("zs_emit", s1(4), 15, E::MaskedTodo, Mn::Plain);
    d.c(0, 8, 0x41).c(15, 1, 0).op(sample_mask("S", 9, 24, 8, 1)).op(zs_desc("T")).op(immediate("u0", 22, 2));
    push!(d);
    let mut d = Desc::new("memory_barrier", s1(2), 15, E::MaskedTodo, Mn::Plain);
    d.c(0, 8, 0xF5).op(immediate("a", 10, 2)).op(immediate("b", 8, 2)).op(immediate("c", 12, 4));
    push!(d);
    v
}
