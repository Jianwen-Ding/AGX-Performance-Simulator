//! Fragment-stage context for the functional simulator: coefficient registers, pixel positions of the lanes of a
//! SIMD group, and sample positions. The hardware's rasterizer setup computes coefficients; the simulator takes them as
//! input (a text file, `agx_run --frag FILE`), like the uniforms of a compute dispatch.
//!
//! File format (one item per line, `#` starts a comment):
//!   size W H                     target size in pixels (sets the default group layout: 8x4-pixel blocks, row-major)
//!   cf I A B C                   coefficient register I: value(x, y) = A*x + B*y + C at pixel position (x, y)
//!   group G X0 Y0                SIMD group G covers the block whose top-left pixel is (X0, Y0) (overrides the default)
//!   samplepos S DX DY            offset inside the pixel of sample S (default 0.5 0.5)
//!   lane L X Y                   explicit pixel of lane L of every group (overrides the quad layout; integer pixel coords)
//!
//! Assumptions not yet verified on hardware (tasks/render-02-fragment-isa.md): the lane -> pixel layout (default: quads
//! of 2x2, four quads across), the f32 evaluation order (fma(A, x, fma(B, y, C))), centroid = center for covered pixels.
use std::sync::OnceLock;

#[derive(Debug, Default, Clone)]
pub struct FragSetup {
    pub width: u32,
    pub height: u32,
    pub cf: Vec<[f32; 3]>,
    pub groups: Vec<(u32, u32, u32)>,
    pub sample_pos: Vec<[f32; 2]>,
    pub lanes: Vec<(u32, u32, u32)>,
}

static SETUP: OnceLock<FragSetup> = OnceLock::new();

pub fn install(s: FragSetup) -> Result<(), String> {
    SETUP.set(s).map_err(|_| "fragment setup installed twice".to_string())
}
pub fn get() -> Option<&'static FragSetup> {
    SETUP.get()
}

impl FragSetup {
    pub fn parse(text: &str) -> Result<FragSetup, String> {
        let mut s = FragSetup::default();
        for (n, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }
            let w: Vec<&str> = line.split_whitespace().collect();
            let u = |i: usize| -> Result<u32, String> {
                w.get(i).ok_or(format!("line {}: missing field", n + 1))?.parse::<u32>().map_err(|e| format!("line {}: {}", n + 1, e))
            };
            let fl = |i: usize| -> Result<f32, String> {
                let t = w.get(i).ok_or(format!("line {}: missing field", n + 1))?;
                if let Some(h) = t.strip_prefix("0x") {
                    return u32::from_str_radix(h, 16).map(f32::from_bits).map_err(|e| format!("line {}: {}", n + 1, e));
                }
                t.parse::<f32>().map_err(|e| format!("line {}: {}", n + 1, e))
            };
            match w[0] {
                "size" => {
                    s.width = u(1)?;
                    s.height = u(2)?;
                }
                "cf" => {
                    let i = u(1)? as usize;
                    if s.cf.len() <= i {
                        s.cf.resize(i + 1, [0.0; 3]);
                    }
                    s.cf[i] = [fl(2)?, fl(3)?, fl(4)?];
                }
                "group" => s.groups.push((u(1)?, u(2)?, u(3)?)),
                "samplepos" => {
                    let i = u(1)? as usize;
                    if s.sample_pos.len() <= i {
                        s.sample_pos.resize(i + 1, [0.5; 2]);
                    }
                    s.sample_pos[i] = [fl(2)?, fl(3)?];
                }
                "lane" => s.lanes.push((u(1)?, u(2)?, u(3)?)),
                other => return Err(format!("line {}: unknown item {}", n + 1, other)),
            }
        }
        Ok(s)
    }

    /// pixel (x, y) of `lane` in SIMD group `group`
    pub fn pixel(&self, group: u32, lane: usize) -> (u32, u32) {
        if let Some(&(_, x, y)) = self.lanes.iter().find(|l| l.0 as usize == lane) {
            return (x, y);
        }
        let (x0, y0) = match self.groups.iter().find(|g| g.0 == group) {
            Some(&(_, x, y)) => (x, y),
            None => {
                let per_row = self.width.div_ceil(8).max(1);
                (8 * (group % per_row), 4 * (group / per_row))
            }
        };
        let q = (lane / 4) as u32;
        (x0 + 2 * (q % 4) + (lane as u32 & 1), y0 + 2 * (q / 4) + ((lane as u32 >> 1) & 1))
    }

    /// value of coefficient register `i` at a position: fma(A, x, fma(B, y, C))
    pub fn plane(&self, i: usize, x: f32, y: f32) -> Result<f32, String> {
        let c = self.cf.get(i).ok_or_else(|| format!("coefficient register {} not set up (--frag)", i))?;
        Ok(c[0].mul_add(x, c[1].mul_add(y, c[2])))
    }
    pub fn sample(&self, s: usize) -> [f32; 2] {
        self.sample_pos.get(s).copied().unwrap_or([0.5; 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_and_eval() {
        let s = FragSetup::parse("size 16 8\ncf 1 0.5 0 1\ncf 0 1 2 3 # c\n").unwrap();
        assert_eq!(s.plane(1, 4.5, 2.5).unwrap(), 3.25);
        assert_eq!(s.plane(0, 1.0, 1.0).unwrap(), 6.0);
        assert!(s.plane(2, 0.0, 0.0).is_err());
        assert_eq!(s.pixel(0, 0), (0, 0));
        assert_eq!(s.pixel(0, 5), (3, 0));
        assert_eq!(s.pixel(1, 0), (8, 0));
        assert_eq!(s.pixel(2, 4), (2, 4 + 0));
    }
}

#[cfg(test)]
mod decode_tests {
    use crate::isa;
    #[test]
    fn captured_iterproj_operands() {
        let code = [0x61u8, 0x91, 0x01, 0, 0, 0, 0, 0];
        let n = isa::opcode_to_number(&code);
        let d = isa::get_instruction_descriptor(n).unwrap();
        let f = d.decode_fields(d.mask_instr(n));
        let ops = d.fields_to_operands(&f).unwrap();
        assert_eq!(d.name, "iterproj");
        // count 0 means four channels; the destination is 32-bit registers, I = cf1, J (the 1/w plane) = cf0
        assert!(matches!(&ops[0], crate::isa::operand::Operand::Tuple(t, _) if t.len() == 4));
        assert!(matches!(ops[1], crate::isa::operand::Operand::CF(1)));
        assert!(matches!(ops[2], crate::isa::operand::Operand::CF(0)));
    }
}
