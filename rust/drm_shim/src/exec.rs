//! A DRM_ASAHI_CMD_COMPUTE on the simulator: the CDM control stream and the USC pipeline words (Mesa's
//! `src/asahi/genxml/cmdbuf.xml`: "CDM Launch Word 0/1", "CDM Unk G14X", "CDM Indirect", "CDM Global/Local
//! size", "CDM Barrier", "CDM Stream Link/Terminate/Return"; "USC Uniform/Uniform High/Texture/Sampler/Shared/
//! Shader/Registers/(No) Preshader") decoded from the VM's memory, each launch run by the functional model
//! (`agx_sim::func`, the preshader first with `run_preamble`) and timed by the dispatch model.

use agx_sim::cost::{Params, SrLaunch};
use agx_sim::dispatch::{self, Launch};
use agx_sim::func::{self, Dispatch, Memory};

/// One decoded launch
#[derive(Debug, Clone)]
pub struct LaunchDesc {
    pub pipeline: u32,
    pub global: [u32; 3],
    pub local: [u32; 3],
    pub code_va: u64,
    pub preshader_va: Option<u64>,
    pub uniforms: Vec<(usize, u32)>,
    pub shared_bytes: u32,
    pub regs: u32,
    pub textures: bool,
}

pub struct Exec<'a> {
    pub mem: &'a mut Memory,
    pub usc_base: u64,
    pub g14x: bool,
    pub vm_slot: u32,
}

/// What a command did: per launch (instructions, predicted ns)
pub struct Outcome {
    pub launches: Vec<(LaunchDesc, u64, Option<f64>)>,
}

fn rd32(m: &Memory, va: u64) -> Result<u32, String> {
    m.read(va, 4).map(|v| v as u32).map_err(|e| format!("read 0x{va:x}: {e}"))
}

fn rd64(m: &Memory, va: u64) -> Result<u64, String> {
    m.read(va, 8).map_err(|e| format!("read 0x{va:x}: {e}"))
}

impl<'a> Exec<'a> {
    /// The launches of the stream [base, end) (a Stream Terminate also ends it), following links and returns
    pub fn decode_stream(&self, base: u64, end: u64) -> Result<Vec<LaunchDesc>, String> {
        let m = &*self.mem;
        let mut out = vec![];
        let mut pc = base;
        let mut ret: Vec<u64> = vec![];
        let mut steps = 0;
        loop {
            steps += 1;
            if steps > 1 << 16 {
                return Err("CDM stream: more than 65536 blocks (a link loop?)".into());
            }
            if ret.is_empty() && pc >= end {
                break;
            }
            let w0 = rd32(m, pc)?;
            match w0 >> 29 {
                0 => {
                    let w1 = rd32(m, pc + 4)?;
                    let mut p = pc + 8;
                    if self.g14x {
                        p += 8;
                    }
                    let mode = (w0 >> 27) & 3;
                    let (global, local) = match mode {
                        0 => {
                            let g = [rd32(m, p)?, rd32(m, p + 4)?, rd32(m, p + 8)?];
                            let l = [rd32(m, p + 12)?, rd32(m, p + 16)?, rd32(m, p + 20)?];
                            p += 24;
                            (g, l)
                        }
                        1 | 2 => {
                            let v = rd64(m, p)?;
                            let addr = ((v & 0xff) << 32) | (((v >> 34) & ((1 << 30) - 1)) << 2);
                            p += 8;
                            let g = [rd32(m, addr)?, rd32(m, addr + 4)?, rd32(m, addr + 8)?];
                            if mode == 1 {
                                let l = [rd32(m, p)?, rd32(m, p + 4)?, rd32(m, p + 8)?];
                                p += 12;
                                (g, l)
                            } else {
                                (g, [rd32(m, addr + 12)?, rd32(m, addr + 16)?, rd32(m, addr + 20)?])
                            }
                        }
                        _ => return Err(format!("CDM launch at 0x{pc:x}: mode 3")),
                    };
                    let pipeline = w1 & !63;
                    let mut d = self.decode_usc(pipeline)?;
                    d.global = global;
                    d.local = local;
                    out.push(d);
                    pc = p;
                }
                1 => {
                    // Stream Link: target hi (bits 0-7), with return (bit 28), target lo (word 1)
                    let lo = rd32(m, pc + 4)? as u64;
                    let target = ((w0 as u64 & 0xff) << 32) | lo;
                    if w0 & (1 << 28) != 0 {
                        ret.push(pc + 8);
                    }
                    pc = target;
                }
                2 => break,
                3 => pc += 4, // Barrier: commands run whole and in order; caches are coherent in the model
                4 => match ret.pop() {
                    Some(r) => pc = r,
                    None => return Err(format!("CDM Stream Return at 0x{pc:x} without a link")),
                },
                t => return Err(format!("CDM block type {t} at 0x{pc:x}")),
            }
        }
        Ok(out)
    }

    /// The USC pipeline words at usc_base + pipeline: every word with a known tag, in any order, up to the first
    /// other byte (the list has no terminator; what follows it in Mesa's layout is zeroed or the next pipeline's
    /// first word, which is also a known tag only for back-to-back pipelines, so it stops at a second Shader word)
    fn decode_usc(&self, pipeline: u32) -> Result<LaunchDesc, String> {
        let m = &*self.mem;
        let mut d = LaunchDesc {
            pipeline,
            global: [0; 3],
            local: [0; 3],
            code_va: 0,
            preshader_va: None,
            uniforms: vec![],
            shared_bytes: 0,
            regs: 256,
            textures: false,
        };
        let mut halves: std::collections::BTreeMap<usize, u16> = Default::default();
        let mut p = self.usc_base + pipeline as u64;
        let mut shader = false;
        let end = p + 4096;
        while p < end {
            let tag = match m.read(p, 1) {
                Ok(t) => t as u8,
                Err(_) if shader => break,
                Err(e) => return Err(format!("USC words at 0x{p:x}: {e}")),
            };
            if tag == 0x0d && shader {
                break;
            }
            match tag {
                0x1d | 0x3d => {
                    let w = rd64(m, p)?;
                    let start = ((w >> 8) & 0xff) as usize + if tag == 0x3d { 256 } else { 0 };
                    let n = match (w >> 20) & 63 {
                        0 => 64,
                        v => v as usize,
                    };
                    let va = (w >> 26) << 2;
                    let bytes = m.read_bytes(va, 2 * n).map_err(|e| format!("USC Uniform buffer 0x{va:x}: {e}"))?;
                    for i in 0..n {
                        halves.insert(start + i, u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]));
                    }
                    p += 8;
                }
                0xdd | 0x9d => {
                    d.textures = true;
                    p += 8;
                }
                0x4d => {
                    let w = rd32(m, p)?;
                    if w & (1 << 8) != 0 {
                        d.shared_bytes = match w >> 24 {
                            0 => 65536,
                            v => v * 256,
                        };
                    }
                    p += 4;
                }
                0x0d => {
                    let lo = rd64(m, p)?;
                    d.code_va = self.usc_base + ((lo >> 16) & 0xffff_ffff);
                    shader = true;
                    p += 6;
                }
                0x8d => {
                    let w = rd32(m, p)?;
                    d.regs = match (w >> 8) & 31 {
                        0 => 256,
                        v => v * 8,
                    };
                    p += 4;
                }
                0x88 => p += 2,
                0x38 => {
                    let w = rd64(m, p)?;
                    d.preshader_va = Some(self.usc_base + (w >> 32));
                    p += 8;
                }
                0x58 => p += 4, // Fragment properties
                _ if shader => break,
                t => return Err(format!("USC pipeline 0x{pipeline:x}: unknown word tag 0x{t:02x} at 0x{p:x} before the Shader word")),
            }
        }
        if !shader {
            return Err(format!("USC pipeline 0x{pipeline:x}: no Shader word"));
        }
        // 16-bit halves -> 32-bit uniform registers
        let mut words: std::collections::BTreeMap<usize, u32> = Default::default();
        for (h, v) in halves {
            let e = words.entry(h / 2).or_insert(0);
            if h % 2 == 0 {
                *e = (*e & 0xffff_0000) | v as u32;
            } else {
                *e = (*e & 0xffff) | ((v as u32) << 16);
            }
        }
        d.uniforms = words.into_iter().collect();
        Ok(d)
    }

    /// The code at va: up to the end of its region (at most 1 MiB), trailing zero halves dropped
    fn code(&self, va: u64) -> Result<Vec<u8>, String> {
        let (base, len) = self
            .mem
            .regions
            .iter()
            .find(|(b, v)| va >= *b && va < *b + v.len() as u64)
            .map(|(b, v)| (*b, v.len() as u64))
            .ok_or_else(|| format!("shader code at 0x{va:x} is not mapped"))?;
        let n = (base + len - va).min(1 << 20) as usize;
        let mut c = self.mem.read_bytes(va, n)?;
        while c.len() > 2 && c[c.len() - 2..] == [0, 0] {
            c.truncate(c.len() - 2);
        }
        Ok(c)
    }

    /// Runs one launch: the preshader (its stored uniforms become the launch's), the timing model on the memory
    /// before the run (if `params`), the functional model. (instructions, predicted ns)
    pub fn run(&mut self, d: &LaunchDesc, params: Option<&Params>, max_steps: u64) -> Result<(u64, Option<f64>), String> {
        if d.local.contains(&0) || (0..3).any(|i| d.global[i] % d.local[i] != 0) {
            return Err(format!("launch: global {:?} is not a multiple of local {:?}", d.global, d.local));
        }
        let mut uniforms = d.uniforms.clone();
        if let Some(pva) = d.preshader_va {
            let pre = self.code(pva)?;
            uniforms = func::run_preamble(&pre, &uniforms, self.mem, 256).map_err(|e| format!("preshader: {e}"))?;
        }
        let code = self.code(d.code_va)?;
        let disp = Dispatch {
            code,
            grid: d.global,
            wg: d.local,
            shared_bytes: d.shared_bytes,
            uniforms: uniforms.clone(),
            num_registers: 256,
            max_steps,
            placement: 0,
            launch: SrLaunch {
                tg_bytes: d.shared_bytes,
                regs: d.regs,
                pipeline: Some(d.pipeline),
                context: Some(self.vm_slot),
                preamble: false,
            },
        };
        let mut ns = None;
        if let Some(p) = params {
            let mut l = Launch::default();
            l.grid = d.global;
            l.wg = d.local;
            l.regs = d.regs.max(8);
            l.tg_bytes = d.shared_bytes;
            l.uniforms = uniforms;
            l.regions = self.mem.regions.iter().map(|(va, b)| (*va, b.len() as u64)).collect();
            if l.validate().is_ok() {
                let prepared = func::PreparedDispatch::new(&disp)?;
                let mem = &*self.mem;
                let tracer = |wg: [u32; 3]| prepared.trace_workgroup(mem, wg);
                ns = dispatch::simulate_with(&disp.code, &l, p, Some(&tracer)).ok().map(|r| r.time_ns);
            }
        }
        let st = func::run(&disp, self.mem)?;
        Ok((st.instructions, ns))
    }
}
