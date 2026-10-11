//! Decoded operands, with Python `str()` formatting identical to applegpu's operand classes.

use std::fmt;

pub const F_CACHE: u8 = 1;
pub const F_DISCARD: u8 = 2;
pub const F_SX: u8 = 4;
pub const F_ABS: u8 = 8;
pub const F_NEG: u8 = 16;

/// Operand flags. applegpu appends them in a fixed order (cache/discard, sx, abs, neg), so a bit set
/// printed in that order reproduces its `'.'.join(...)` output.
pub fn flag_names(f: u8) -> impl Iterator<Item = &'static str> {
    [(F_CACHE, "cache"), (F_DISCARD, "discard"), (F_SX, "sx"), (F_ABS, "abs"), (F_NEG, "neg")]
        .into_iter()
        .filter(move |(b, _)| f & b != 0)
        .map(|(_, n)| n)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImmVal {
    Int(i64),
    Float(f64),
}

/// One decoded operand. `Int`/`Str` are the plain Python ints and strings some operand
/// descriptors return; `Null` is Python `None`.
#[derive(Clone, Debug, PartialEq)]
pub enum Operand {
    Reg16(u32, u8),
    Reg32(u32, u8),
    Reg64(u32, u8),
    UReg16(u32, u8),
    UReg32(u32, u8),
    UReg64(u32, u8),
    SReg32(u32),
    TextureState(u32),
    SamplerState(u32),
    CF(u32),
    Tuple(Vec<Operand>, u8),
    Imm(ImmVal, u8),
    RelOff(i64),
    Int(i64),
    Str(String),
    Null,
}

impl Operand {
    pub fn flags(&self) -> u8 {
        match self {
            Operand::Reg16(_, f) | Operand::Reg32(_, f) | Operand::Reg64(_, f) | Operand::UReg16(_, f)
            | Operand::UReg32(_, f) | Operand::UReg64(_, f) | Operand::Tuple(_, f) | Operand::Imm(_, f) => *f,
            _ => 0,
        }
    }

    /// Append a flag (Python `op.flags.append`). Errors on operands without a flags list.
    pub fn add_flag(&mut self, flag: u8) -> Result<(), String> {
        match self {
            Operand::Reg16(_, f) | Operand::Reg32(_, f) | Operand::Reg64(_, f) | Operand::UReg16(_, f)
            | Operand::UReg32(_, f) | Operand::UReg64(_, f) | Operand::Tuple(_, f) | Operand::Imm(_, f) => {
                *f |= flag;
                Ok(())
            }
            Operand::SReg32(_) | Operand::TextureState(_) | Operand::SamplerState(_) | Operand::CF(_) => Ok(()),
            _ => Err("operand has no flags".into()),
        }
    }

    /// isinstance(op, Register): every register class (not tuples, immediates or None).
    pub fn is_register(&self) -> bool {
        matches!(
            self,
            Operand::Reg16(..) | Operand::Reg32(..) | Operand::Reg64(..) | Operand::UReg16(..) | Operand::UReg32(..)
                | Operand::UReg64(..) | Operand::SReg32(_) | Operand::TextureState(_) | Operand::SamplerState(_)
                | Operand::CF(_)
        )
    }

    pub fn is_ureg(&self) -> bool {
        matches!(self, Operand::UReg16(..) | Operand::UReg32(..) | Operand::UReg64(..))
    }

    /// `get_bit_size()`; errors where Python raises (tuples, None, strings).
    pub fn bit_size(&self) -> Result<u32, String> {
        match self {
            Operand::Reg16(..) | Operand::UReg16(..) => Ok(16),
            Operand::Reg32(..) | Operand::UReg32(..) | Operand::SReg32(_) | Operand::TextureState(_)
            | Operand::SamplerState(_) | Operand::CF(_) => Ok(32),
            Operand::Reg64(..) | Operand::UReg64(..) => Ok(64),
            Operand::Imm(..) | Operand::RelOff(_) => Ok(16),
            _ => Err("get_bit_size on non-register operand".into()),
        }
    }

    /// agx_cost.regs_of: register halves (16-bit units) named by the operand.
    pub fn regs_of(&self, out: &mut Vec<u32>) {
        match self {
            Operand::Reg16(n, _) => out.push(*n),
            Operand::Reg32(n, _) => {
                out.push(2 * n);
                out.push(2 * n + 1)
            }
            Operand::Reg64(n, _) => out.extend(2 * n..2 * n + 4),
            Operand::Tuple(regs, _) => {
                for r in regs {
                    r.regs_of(out)
                }
            }
            _ => {}
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Operand::Reg16(..) => "Reg16",
            Operand::Reg32(..) => "Reg32",
            Operand::Reg64(..) => "Reg64",
            Operand::UReg16(..) => "UReg16",
            Operand::UReg32(..) => "UReg32",
            Operand::UReg64(..) => "UReg64",
            Operand::SReg32(_) => "SReg32",
            Operand::TextureState(_) => "TextureState",
            Operand::SamplerState(_) => "SamplerState",
            Operand::CF(_) => "CF",
            Operand::Tuple(..) => "RegisterTuple",
            Operand::Imm(..) => "Immediate",
            Operand::RelOff(_) => "RelativeOffset",
            Operand::Int(_) => "int",
            Operand::Str(_) => "str",
            Operand::Null => "NoneType",
        }
    }
}

/// Python `repr(float)` (shortest round-trip digits, `1e-05` style exponents).
pub fn py_float(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let a = x.abs();
    if (1e-4..1e16).contains(&a) {
        let s = format!("{}", x);
        if s.contains('.') { s } else { s + ".0" }
    } else {
        let s = format!("{:e}", x);
        let (m, e) = s.split_once('e').unwrap();
        let (sign, digits) = if let Some(d) = e.strip_prefix('-') { ('-', d) } else { ('+', e) };
        format!("{}e{}{:0>2}", m, sign, digits)
    }
}

fn write_reg(f: &mut fmt::Formatter, base: &str, flags: u8) -> fmt::Result {
    f.write_str(base)?;
    for n in flag_names(flags) {
        write!(f, ".{}", n)?;
    }
    Ok(())
}

/// Register name tables bound the index exactly like applegpu's lists (IndexError otherwise).
pub fn reg_name(op: &Operand) -> Result<String, String> {
    let bad = || Err(format!("register index out of range: {:?}", op));
    Ok(match op {
        Operand::Reg16(n, _) => {
            if *n >= 256 { return bad(); }
            format!("r{}{}", n / 2, if n % 2 == 0 { 'l' } else { 'h' })
        }
        Operand::Reg32(n, _) => {
            if *n >= 128 { return bad(); }
            format!("r{}", n)
        }
        Operand::Reg64(n, _) => {
            if *n >= 128 { return bad(); }
            format!("r{}_r{}", n, n + 1)
        }
        Operand::UReg16(n, _) => {
            if *n >= 512 { return bad(); }
            format!("u{}{}", n / 2, if n % 2 == 0 { 'l' } else { 'h' })
        }
        Operand::UReg32(n, _) => {
            if *n >= 256 { return bad(); }
            format!("u{}", n)
        }
        Operand::UReg64(n, _) => {
            if *n >= 256 { return bad(); }
            format!("u{}_u{}", n, n + 1)
        }
        _ => unreachable!(),
    })
}

pub const SR_NAMES: &[(u32, &str)] = &[
    (0, "threadgroup_position_in_grid.x"), (1, "threadgroup_position_in_grid.y"), (2, "threadgroup_position_in_grid.z"),
    (4, "threads_per_threadgroup.x"), (5, "threads_per_threadgroup.y"), (6, "threads_per_threadgroup.z"),
    (8, "dispatch_threads_per_threadgroup.x"), (9, "dispatch_threads_per_threadgroup.y"),
    (10, "dispatch_threads_per_threadgroup.z"), (20, "core_index"), (21, "vm_slot"),
    (48, "thread_position_in_threadgroup.x"), (49, "thread_position_in_threadgroup.y"),
    (50, "thread_position_in_threadgroup.z"), (51, "thread_index_in_threadgroup"), (52, "thread_index_in_simdgroup"),
    (53, "simdgroup_index_in_threadgroup"), (56, "active_thread_index_in_quadgroup"),
    (58, "active_thread_index_in_simdgroup"), (60, "internal_coverage_mask"), (124, "input_sample_mask"),
    (62, "backfacing"), (63, "is_active_thread"), (80, "thread_position_in_grid.x"), (81, "thread_position_in_grid.y"),
    (82, "thread_position_in_grid.z"), (144, "opfifo_cmd"), (146, "opfifo_data_l"), (147, "opfifo_data_h"),
];

/// Python `str(operand)`. Errors where applegpu would raise IndexError on a register name.
pub fn py_str(op: &Operand) -> Result<String, String> {
    struct D<'a>(&'a Operand);
    impl fmt::Display for D<'_> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self.0 {
                op @ (Operand::Reg16(..) | Operand::Reg32(..) | Operand::Reg64(..) | Operand::UReg16(..)
                | Operand::UReg32(..) | Operand::UReg64(..)) => {
                    write_reg(f, &reg_name(op).map_err(|_| fmt::Error)?, op.flags())
                }
                Operand::SReg32(n) => {
                    write!(f, "sr{}", n)?;
                    if let Some((_, name)) = SR_NAMES.iter().find(|(k, _)| k == n) {
                        write!(f, " ({})", name)?;
                    }
                    Ok(())
                }
                Operand::TextureState(n) => write!(f, "ts{}", n),
                Operand::SamplerState(n) => write!(f, "ss{}", n),
                Operand::CF(n) => write!(f, "cf{}", n),
                Operand::Tuple(regs, flags) => {
                    let mut first = true;
                    for r in regs {
                        if !first {
                            f.write_str("_")?;
                        }
                        first = false;
                        write!(f, "{}", D(r))?;
                    }
                    for n in flag_names(*flags) {
                        write!(f, ".{}", n)?;
                    }
                    Ok(())
                }
                Operand::Imm(v, flags) => {
                    match v {
                        ImmVal::Int(i) => write!(f, "{}", i)?,
                        ImmVal::Float(x) => f.write_str(&py_float(*x))?,
                    }
                    for n in flag_names(*flags) {
                        write!(f, ".{}", n)?;
                    }
                    Ok(())
                }
                Operand::RelOff(v) => {
                    if *v >= 0 { write!(f, "pc+{}", v) } else { write!(f, "pc-{}", -v) }
                }
                Operand::Int(i) => write!(f, "{}", i),
                Operand::Str(s) => f.write_str(s),
                Operand::Null => f.write_str("None"),
            }
        }
    }
    let mut s = String::new();
    use std::fmt::Write;
    write!(s, "{}", D(op)).map_err(|_| format!("register index out of range in {:?}", op))?;
    Ok(s)
}

/// applegpu `try_parse_register`: `$`-prefix cache hint, `.flag` suffixes.
pub fn try_parse_register(s: &str) -> Option<Operand> {
    let mut flags = 0u8;
    let s = if let Some(r) = s.strip_prefix('$') {
        flags |= F_CACHE;
        r
    } else {
        s
    };
    let mut parts = s.split('.');
    let base = parts.next()?;
    let mut op = parse_reg_name(base)?;
    for p in parts {
        let fl = match p {
            "abs" => F_ABS,
            "neg" => F_NEG,
            "sx" => F_SX,
            "cache" => F_CACHE,
            "discard" => F_DISCARD,
            _ => return None,
        };
        flags |= fl;
    }
    // flags on ts/ss/cf are kept by applegpu but never printed or encoded
    let _ = op.add_flag(flags);
    Some(op)
}

fn parse_num(s: &str) -> Option<u32> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) || (s.len() > 1 && s.starts_with('0')) {
        return None;
    }
    s.parse().ok()
}

/// Exact names from applegpu's `registers_by_name` table.
fn parse_reg_name(s: &str) -> Option<Operand> {
    let b = s.as_bytes();
    if b.is_empty() {
        return None;
    }
    match b[0] {
        b'r' | b'u' => {
            let uniform = b[0] == b'u';
            let rest = &s[1..];
            let limit = if uniform { 256 } else { 128 };
            if let Some((a, c)) = rest.split_once('_') {
                // r{i}_r{i+1} (64-bit) only
                let n = parse_num(a)?;
                let p = if uniform { 'u' } else { 'r' };
                let m = parse_num(c.strip_prefix(p)?)?;
                if n >= limit || m != n + 1 {
                    return None;
                }
                return Some(if uniform { Operand::UReg64(n, 0) } else { Operand::Reg64(n, 0) });
            }
            if let Some(n) = rest.strip_suffix('l').and_then(parse_num) {
                if n >= limit {
                    return None;
                }
                return Some(if uniform { Operand::UReg16(2 * n, 0) } else { Operand::Reg16(2 * n, 0) });
            }
            if let Some(n) = rest.strip_suffix('h').and_then(parse_num) {
                if n >= limit {
                    return None;
                }
                return Some(if uniform { Operand::UReg16(2 * n + 1, 0) } else { Operand::Reg16(2 * n + 1, 0) });
            }
            let n = parse_num(rest)?;
            if n >= limit {
                return None;
            }
            Some(if uniform { Operand::UReg32(n, 0) } else { Operand::Reg32(n, 0) })
        }
        b't' | b's' | b'c' => {
            let (p, mk): (&str, fn(u32) -> Operand) = match b[0] {
                b't' => ("ts", Operand::TextureState),
                b's' => ("ss", Operand::SamplerState),
                _ => ("cf", Operand::CF),
            };
            let n = parse_num(s.strip_prefix(p)?)?;
            if n >= 256 {
                return None;
            }
            Some(mk(n))
        }
        _ => None,
    }
}

/// applegpu `try_parse_register_tuple`.
pub fn try_parse_register_tuple(s: &str) -> Option<Operand> {
    let mut flags = 0u8;
    let s = if let Some(r) = s.strip_prefix('$') {
        flags |= F_CACHE;
        r
    } else {
        s
    };
    let mut parts = s.split('.');
    let regs_s = parts.next()?;
    let mut regs = Vec::new();
    for r in regs_s.split('_') {
        regs.push(try_parse_register(r)?); // None in the list fails both isinstance checks
    }
    let all32 = regs.iter().all(|r| matches!(r, Operand::Reg32(..)));
    let all16 = regs.iter().all(|r| matches!(r, Operand::Reg16(..)));
    if !all32 && !all16 {
        return None;
    }
    if regs.iter().any(|r| r.flags() != 0) {
        return None;
    }
    let num = |r: &Operand| match r {
        Operand::Reg32(n, _) | Operand::Reg16(n, _) => *n,
        _ => 0,
    };
    for w in regs.windows(2) {
        if num(&w[0]) + 1 != num(&w[1]) {
            return None;
        }
    }
    for p in parts {
        flags |= match p {
            "abs" => F_ABS,
            "neg" => F_NEG,
            "sx" => F_SX,
            "cache" => F_CACHE,
            "discard" => F_DISCARD,
            _ => return None,
        };
    }
    Some(Operand::Tuple(regs, flags))
}

pub fn decode_float_immediate(n: u64) -> f64 {
    let sign = if n & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((n & 0x70) >> 4) as i32;
    let f = (n & 0xF) as f64;
    if e == 0 {
        sign * f / 64.0
    } else {
        sign * ((0x10 | (n & 0xF)) as f64) * 2f64.powi(e - 7)
    }
}

/// applegpu `try_parse_integer` (float-immediate spellings first, then Python `int(s, base)`).
pub fn try_parse_integer(s: &str) -> Option<i64> {
    // float_immediate_lookup = {str(decode_float_immediate(i)): i}, later entries win
    static LOOKUP: std::sync::OnceLock<std::collections::HashMap<String, i64>> = std::sync::OnceLock::new();
    let lookup = LOOKUP.get_or_init(|| (0..256u64).map(|i| (py_float(decode_float_immediate(i)), i as i64)).collect());
    if let Some(&v) = lookup.get(s) {
        return Some(v);
    }
    let base = if s.contains("0b") {
        2
    } else if s.contains("0x") {
        16
    } else {
        10
    };
    py_int(s, base)
}

/// Python `int(s, base)` for the subset applegpu relies on (sign, optional base prefix, underscores).
fn py_int(s: &str, base: u32) -> Option<i64> {
    let t = s.trim();
    let (neg, t) = if let Some(r) = t.strip_prefix('-') {
        (true, r)
    } else if let Some(r) = t.strip_prefix('+') {
        (false, r)
    } else {
        (false, t)
    };
    let t = match base {
        2 => t.strip_prefix("0b").or_else(|| t.strip_prefix("0B")).unwrap_or(t),
        16 => t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")).unwrap_or(t),
        _ => t,
    };
    if t.is_empty() || t.starts_with('_') || t.ends_with('_') || t.contains("__") {
        return None;
    }
    let digits: String = t.chars().filter(|&c| c != '_').collect();
    if base == 10 && digits.len() > 1 && digits.starts_with('0') && digits.chars().any(|c| c != '0') {
        return None; // Python rejects leading zeros in base 10
    }
    let v = i64::from_str_radix(&digits, base).ok()?;
    Some(if neg { -v } else { v })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn floats() {
        assert_eq!(py_float(0.125), "0.125");
        assert_eq!(py_float(1.0), "1.0");
        assert_eq!(py_float(-0.0), "-0.0");
        assert_eq!(py_float(1e-5), "1e-05");
        assert_eq!(py_float(0.015625), "0.015625");
        assert_eq!(try_parse_integer("0.125"), Some(8));
        assert_eq!(try_parse_integer("0.0"), Some(0));
        assert_eq!(try_parse_integer("0b11"), Some(3));
        assert_eq!(try_parse_integer("-0x11"), Some(-17));
        assert_eq!(try_parse_integer("305419896"), Some(305419896));
        assert_eq!(try_parse_integer("1"), Some(1));
        assert_eq!(try_parse_integer("r3"), None);
    }
    #[test]
    fn regs() {
        assert_eq!(try_parse_register("r3l"), Some(Operand::Reg16(6, 0)));
        assert_eq!(try_parse_register("r3.sx"), Some(Operand::Reg32(3, F_SX)));
        assert_eq!(try_parse_register("u8_u9"), Some(Operand::UReg64(8, 0)));
        assert_eq!(try_parse_register("r20.cache"), Some(Operand::Reg32(20, F_CACHE)));
        assert_eq!(py_str(&Operand::Reg16(1, 0)).unwrap(), "r0h");
        assert_eq!(py_str(&Operand::Reg32(5, F_CACHE)).unwrap(), "r5.cache");
    }
}
