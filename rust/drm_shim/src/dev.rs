//! The state of one open render node (DRM objects are per file: GEM handles, VMs, queues, syncobjs) and the
//! ioctls on it.
//!
//! * A GEM object is a memfd of its size: the client's mmap (at the fake offset GEM_MMAP_OFFSET returns) and the
//!   shim's own mapping are the same pages, so no copies are needed between the CPU and the shim.
//! * A VM is its list of binds (VA range -> object range, read / write). A command runs on a simulator `Memory`
//!   built from the VM's binds (each bind its own region), and the regions bound writable are copied back.
//! * Commands run synchronously in SUBMIT once their in-syncs are signaled; a submit whose in-syncs are not yet
//!   signaled waits (in order, per queue) until a later SIGNAL / SUBMIT signals them.
//! * Time: the GPU clock (GET_TIME, firmware timestamps) runs at `command_timestamp_frequency_hz`; it is the host
//!   monotonic clock, and a queue's commands are placed back to back from max(now, the queue's last end), each as
//!   long as the dispatch model predicts (AGX_SHIM_TIMING=0: 1 us per launch, no timing model).

use crate::exec::Exec;
use crate::uapi::*;
use agx_sim::cost::Params;
use agx_sim::func::Memory;
use std::collections::{BTreeMap, HashMap, VecDeque};

pub const PAGE: u64 = 16384;
pub const TS_HZ: u64 = 24_000_000;

pub type Res<T> = Result<T, i32>;
pub const EINVAL: i32 = 22;
pub const ENOENT: i32 = 2;
pub const ENOMEM: i32 = 12;
pub const ETIME: i32 = 62;
pub const EOPNOTSUPP: i32 = 95;
pub const ENOTTY: i32 = 25;
pub const EFAULT: i32 = 14;

pub fn log_on() -> bool {
    std::env::var_os("AGX_SHIM_LOG").is_some()
}

macro_rules! slog {
    ($($t:tt)*) => { if crate::dev::log_on() { eprintln!("[agx-drm-shim] {}", format!($($t)*)); } };
}
pub(crate) use slog;

pub struct Bo {
    pub size: u64,
    pub memfd: i32,
    pub map: *mut u8,
    pub mmap_offset: u64,
}

// the mapping is the shim's own, used under the files lock
unsafe impl Send for Bo {}

#[derive(Clone, Copy, Debug)]
pub struct Bind {
    pub addr: u64,
    pub range: u64,
    pub handle: u32,
    pub offset: u64,
    pub flags: u32,
}

pub struct Vm {
    pub binds: Vec<Bind>,
    pub kernel: (u64, u64),
    pub slot: u32,
}

pub struct Queue {
    pub vm: u32,
    pub usc_base: u64,
    pub end_ns: u64,
    pub pending: VecDeque<Pending>,
}

pub struct Pending {
    pub cmds: Vec<u8>,
    pub in_syncs: Vec<drm_asahi_sync>,
    pub out_syncs: Vec<drm_asahi_sync>,
}

#[derive(Default)]
pub struct File {
    pub bos: HashMap<u32, Bo>,
    pub next_handle: u32,
    pub vms: HashMap<u32, Vm>,
    pub next_vm: u32,
    pub queues: BTreeMap<u32, Queue>,
    pub next_queue: u32,
    /// timeline value; a binary syncobj is signaled at >= 1
    pub syncobjs: HashMap<u32, u64>,
    pub next_syncobj: u32,
    /// GEM_BIND_OBJECT objects: id -> (handle, offset, range)
    pub objects: HashMap<u32, (u32, u64, u64)>,
    pub next_object: u32,
    pub faults: u64,
}

extern "C" {
    fn memfd_create(name: *const u8, flags: u32) -> i32;
    fn ftruncate(fd: i32, len: i64) -> i32;
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
    fn close(fd: i32) -> i32;
}

pub fn now_ns() -> u64 {
    #[repr(C)]
    struct Ts {
        s: i64,
        ns: i64,
    }
    extern "C" {
        fn clock_gettime(c: i32, t: *mut Ts) -> i32;
    }
    let mut t = Ts { s: 0, ns: 0 };
    unsafe { clock_gettime(1, &mut t) };
    t.s as u64 * 1_000_000_000 + t.ns as u64
}

fn ticks(ns: u64) -> u64 {
    (ns as u128 * TS_HZ as u128 / 1_000_000_000) as u64
}

/// The GPU's identity (GET_PARAMS): the M2 Max the simulator models (driver-reported G14C B1, 38 cores, 1398
/// MHz, the measured 64 commands per submit); the cluster split, chip id and VM range are nominal. AGX_SHIM_PARAMS
/// may override fields: `name=value,...` (gpu_generation, gpu_variant, gpu_revision, chip_id, num_clusters_total,
/// num_cores_per_cluster, max_frequency_khz, vm_start, vm_end, vm_kernel_min_size, features).
pub fn params() -> drm_asahi_params_global {
    let mut p = drm_asahi_params_global {
        features: 0,
        gpu_generation: 14,
        gpu_variant: b'C' as u32,
        gpu_revision: 0x11,
        chip_id: 0x6021,
        num_dies: 1,
        num_clusters_total: 4,
        num_cores_per_cluster: 10,
        max_frequency_khz: 1_398_000,
        core_masks: [0; DRM_ASAHI_MAX_CLUSTERS],
        vm_start: 0x4000,
        vm_end: 1 << 39,
        vm_kernel_min_size: 1 << 32,
        max_commands_per_submission: 64,
        max_attachments: 16,
        command_timestamp_frequency_hz: TS_HZ,
    };
    p.core_masks[..4].copy_from_slice(&[0x3ff, 0x3ff, 0x1ff, 0x1ff]);
    if let Ok(s) = std::env::var("AGX_SHIM_PARAMS") {
        for kv in s.split(',').filter(|x| !x.is_empty()) {
            let (k, v) = kv.split_once('=').unwrap_or((kv, "0"));
            let n = if let Some(h) = v.strip_prefix("0x") { u64::from_str_radix(h, 16).unwrap_or(0) } else { v.parse().unwrap_or(0) };
            match k {
                "gpu_generation" => p.gpu_generation = n as u32,
                "gpu_variant" => p.gpu_variant = v.bytes().next().map(|c| c as u32).unwrap_or(0),
                "gpu_revision" => p.gpu_revision = n as u32,
                "chip_id" => p.chip_id = n as u32,
                "num_clusters_total" => p.num_clusters_total = n as u32,
                "num_cores_per_cluster" => p.num_cores_per_cluster = n as u32,
                "max_frequency_khz" => p.max_frequency_khz = n as u32,
                "vm_start" => p.vm_start = n,
                "vm_end" => p.vm_end = n,
                "vm_kernel_min_size" => p.vm_kernel_min_size = n,
                "features" => p.features = n,
                _ => slog!("AGX_SHIM_PARAMS: unknown field {k}"),
            }
        }
    }
    p
}

/// Multi-cluster G14 has the extra "CDM Unk G14X" launch word (Mesa: gen >= 14 and more than one cluster)
fn g14x() -> bool {
    let p = params();
    p.gpu_generation >= 14 && p.num_clusters_total > 1
}

unsafe fn rd<T: Copy>(p: u64) -> Res<T> {
    if p == 0 {
        return Err(EFAULT);
    }
    Ok(std::ptr::read_unaligned(p as *const T))
}

unsafe fn rd_slice<T: Copy>(p: u64, n: usize) -> Res<Vec<T>> {
    if n == 0 {
        return Ok(vec![]);
    }
    if p == 0 {
        return Err(EFAULT);
    }
    Ok((0..n).map(|i| std::ptr::read_unaligned((p as *const T).add(i))).collect())
}

impl File {
    pub fn gem_create(&mut self, a: &mut drm_asahi_gem_create) -> Res<()> {
        if a.size == 0 || a.size > 64 << 30 {
            return Err(EINVAL);
        }
        let size = a.size.div_ceil(PAGE) * PAGE;
        let fd = unsafe { memfd_create(b"agx-drm-shim-bo\0".as_ptr(), 1) }; // MFD_CLOEXEC
        if fd < 0 || unsafe { ftruncate(fd, size as i64) } != 0 {
            return Err(ENOMEM);
        }
        let map = unsafe { mmap(std::ptr::null_mut(), size as usize, 3, 1, fd, 0) }; // RW, MAP_SHARED
        if map as isize == -1 {
            unsafe { close(fd) };
            return Err(ENOMEM);
        }
        self.next_handle += 1;
        let h = self.next_handle;
        self.bos.insert(h, Bo { size, memfd: fd, map, mmap_offset: (h as u64) << 36 });
        a.handle = h;
        slog!("GEM_CREATE handle {h} size {size:#x} flags {:#x}", a.flags);
        Ok(())
    }

    pub fn gem_close(&mut self, h: u32) -> Res<()> {
        let bo = self.bos.remove(&h).ok_or(ENOENT)?;
        for vm in self.vms.values_mut() {
            vm.binds.retain(|b| b.handle != h);
        }
        unsafe {
            munmap(bo.map, bo.size as usize);
            close(bo.memfd);
        }
        Ok(())
    }

    /// (memfd, offset into it) of an mmap offset of this file
    pub fn mmap_target(&self, off: u64, len: u64) -> Option<(i32, u64)> {
        self.bos.values().find_map(|b| {
            (off >= b.mmap_offset && off + len <= b.mmap_offset + b.size).then(|| (b.memfd, off - b.mmap_offset))
        })
    }

    pub fn vm_bind(&mut self, a: &drm_asahi_vm_bind) -> Res<()> {
        let stride = if a.stride == 0 { std::mem::size_of::<drm_asahi_gem_bind_op>() } else { a.stride as usize };
        let start = params().vm_start;
        for i in 0..a.num_binds as u64 {
            let op: drm_asahi_gem_bind_op = unsafe { rd(a.userptr + i * stride as u64)? };
            let vm = self.vms.get_mut(&a.vm_id).ok_or(ENOENT)?;
            if op.addr % PAGE != 0 || op.range % PAGE != 0 || op.range == 0 {
                return Err(EINVAL);
            }
            // a bind replaces what was there (the kernel's GPUVM semantics), an unbind removes it
            let (lo, hi) = (op.addr, op.addr + op.range);
            let mut kept = vec![];
            for b in vm.binds.drain(..) {
                let (bl, bh) = (b.addr, b.addr + b.range);
                if bh <= lo || bl >= hi {
                    kept.push(b);
                    continue;
                }
                if bl < lo {
                    kept.push(Bind { range: lo - bl, ..b });
                }
                if bh > hi {
                    kept.push(Bind { addr: hi, range: bh - hi, offset: b.offset + (hi - bl), ..b });
                }
            }
            vm.binds = kept;
            if op.flags & DRM_ASAHI_BIND_UNBIND != 0 {
                slog!("VM_BIND unbind {:#x}+{:#x}", op.addr, op.range);
                continue;
            }
            let bo = self.bos.get(&op.handle).ok_or(ENOENT)?;
            let single = op.flags & DRM_ASAHI_BIND_SINGLE_PAGE != 0;
            if op.addr < start || (!single && op.offset + op.range > bo.size) || op.offset % PAGE != 0 {
                return Err(EINVAL);
            }
            let vm = self.vms.get_mut(&a.vm_id).unwrap();
            if single {
                // one page of the object repeated over the range
                for k in 0..op.range / PAGE {
                    vm.binds.push(Bind { addr: op.addr + k * PAGE, range: PAGE, handle: op.handle, offset: op.offset, flags: op.flags });
                }
            } else {
                vm.binds.push(Bind { addr: op.addr, range: op.range, handle: op.handle, offset: op.offset, flags: op.flags });
            }
            slog!("VM_BIND vm {} {:#x}+{:#x} -> handle {} @{:#x} flags {:#x}", a.vm_id, op.addr, op.range, op.handle, op.offset, op.flags);
        }
        Ok(())
    }

    fn sync_ready(&self, s: &drm_asahi_sync) -> Res<bool> {
        let v = *self.syncobjs.get(&s.handle).ok_or(ENOENT)?;
        Ok(match s.sync_type {
            DRM_ASAHI_SYNC_SYNCOBJ => v >= 1,
            DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ => v >= s.timeline_value,
            _ => return Err(EINVAL),
        })
    }

    fn signal(&mut self, s: &drm_asahi_sync) {
        let v = self.syncobjs.entry(s.handle).or_insert(0);
        *v = match s.sync_type {
            DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ => (*v).max(s.timeline_value),
            _ => (*v).max(1),
        };
    }

    pub fn submit(&mut self, a: &drm_asahi_submit) -> Res<()> {
        if !self.queues.contains_key(&a.queue_id) {
            return Err(ENOENT);
        }
        let syncs: Vec<drm_asahi_sync> = unsafe { rd_slice(a.syncs, (a.in_sync_count + a.out_sync_count) as usize)? };
        for s in &syncs {
            if !self.syncobjs.contains_key(&s.handle) {
                return Err(ENOENT);
            }
        }
        let cmds: Vec<u8> = unsafe { rd_slice(a.cmdbuf, a.cmdbuf_size as usize)? };
        let p = Pending {
            cmds,
            in_syncs: syncs[..a.in_sync_count as usize].to_vec(),
            out_syncs: syncs[a.in_sync_count as usize..].to_vec(),
        };
        self.queues.get_mut(&a.queue_id).unwrap().pending.push_back(p);
        self.progress()
    }

    /// Runs every pending submit whose in-syncs are signaled (each queue in order), until none can run
    pub fn progress(&mut self) -> Res<()> {
        loop {
            let mut ran = false;
            let qids: Vec<u32> = self.queues.keys().copied().collect();
            for q in qids {
                let ready = match self.queues[&q].pending.front() {
                    Some(p) => p.in_syncs.iter().map(|s| self.sync_ready(s)).collect::<Res<Vec<_>>>()?.into_iter().all(|x| x),
                    None => false,
                };
                if !ready {
                    continue;
                }
                let p = self.queues.get_mut(&q).unwrap().pending.pop_front().unwrap();
                self.run_cmdbuf(q, &p.cmds)?;
                for s in &p.out_syncs {
                    self.signal(s);
                }
                ran = true;
            }
            if !ran {
                return Ok(());
            }
        }
    }

    /// The simulator memory of a VM: each bind a region with the object's bytes
    fn memory(&self, vm: &Vm) -> Memory {
        let mut m = Memory::default();
        for b in &vm.binds {
            if let Some(bo) = self.bos.get(&b.handle) {
                let s = unsafe { std::slice::from_raw_parts(bo.map.add(b.offset as usize), b.range as usize) };
                m.map(b.addr, s.to_vec());
            }
        }
        m
    }

    fn write_back(&self, vm: &Vm, m: &Memory) {
        for b in &vm.binds {
            if b.flags & DRM_ASAHI_BIND_WRITE == 0 {
                continue;
            }
            if let (Some(bo), Ok(bytes)) = (self.bos.get(&b.handle), m.read_bytes(b.addr, b.range as usize)) {
                unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), bo.map.add(b.offset as usize), bytes.len()) };
            }
        }
    }

    fn timestamp(&self, t: &drm_asahi_timestamp, v: u64) {
        if let Some(&(h, off, range)) = self.objects.get(&t.handle) {
            if (t.offset as u64) + 8 <= range {
                if let Some(bo) = self.bos.get(&h) {
                    unsafe { std::ptr::write_unaligned(bo.map.add((off + t.offset as u64) as usize) as *mut u64, v) };
                }
            }
        }
    }

    fn run_cmdbuf(&mut self, q: u32, buf: &[u8]) -> Res<()> {
        let mut p = 0usize;
        let hs = std::mem::size_of::<drm_asahi_cmd_header>();
        while p + hs <= buf.len() {
            let h: drm_asahi_cmd_header = unsafe { std::ptr::read_unaligned(buf.as_ptr().add(p) as *const _) };
            let body = &buf[p + hs..(p + hs + h.size as usize).min(buf.len())];
            p += hs + h.size as usize;
            match h.cmd_type {
                DRM_ASAHI_CMD_COMPUTE => {
                    let mut c = drm_asahi_cmd_compute::default();
                    let n = body.len().min(std::mem::size_of::<drm_asahi_cmd_compute>());
                    unsafe { std::ptr::copy_nonoverlapping(body.as_ptr(), &mut c as *mut _ as *mut u8, n) };
                    self.compute(q, &c);
                }
                DRM_ASAHI_SET_VERTEX_ATTACHMENTS | DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS | DRM_ASAHI_SET_COMPUTE_ATTACHMENTS => {}
                DRM_ASAHI_CMD_RENDER => {
                    eprintln!("[agx-drm-shim] DRM_ASAHI_CMD_RENDER is not simulated (compute only)");
                    return Err(EOPNOTSUPP);
                }
                _ => return Err(EINVAL),
            }
        }
        Ok(())
    }

    /// One compute command. A fault (an unmapped access, a step limit, an undecodable stream) is reported as the
    /// firmware does: the command completes, both timestamps 0, and only this VM's memory is touched.
    fn compute(&mut self, q: u32, c: &drm_asahi_cmd_compute) {
        let (vm_id, usc_base, qend) = {
            let qu = &self.queues[&q];
            (qu.vm, qu.usc_base, qu.end_ns)
        };
        let vm = match self.vms.get(&vm_id) {
            Some(v) => v,
            None => return,
        };
        let mut mem = self.memory(vm);
        let timing = std::env::var("AGX_SHIM_TIMING").map(|v| v != "0").unwrap_or(true);
        static PARAMS: std::sync::OnceLock<Option<Params>> = std::sync::OnceLock::new();
        let params = if timing { PARAMS.get_or_init(|| Params::new().ok()).as_ref() } else { None };
        let max_steps = std::env::var("AGX_SHIM_MAX_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(2_000_000_000);
        let res = (|| -> Result<Vec<(u64, Option<f64>)>, String> {
            let mut ex = Exec { mem: &mut mem, usc_base, g14x: g14x(), vm_slot: vm.slot };
            let launches = ex.decode_stream(c.cdm_ctrl_stream_base, c.cdm_ctrl_stream_end)?;
            if c.helper_binary != 0 {
                slog!("helper program {:#x} ignored (the functional model gives every thread its stack)", c.helper_binary);
            }
            let mut out = vec![];
            for d in &launches {
                if d.textures {
                    slog!("USC texture / sampler state words ignored (textures are not simulated through the shim)");
                }
                slog!("launch pipeline {:#x} code {:#x} global {:?} local {:?} shared {} regs {} uniforms {}{}",
                      d.pipeline, d.code_va, d.global, d.local, d.shared_bytes, d.regs, d.uniforms.len(),
                      if d.preshader_va.is_some() { " +preshader" } else { "" });
                out.push(ex.run(d, params, max_steps)?);
            }
            Ok(out)
        })();
        let start = now_ns().max(qend);
        match res {
            Ok(runs) => {
                let dur: f64 = runs.iter().map(|r| r.1.unwrap_or(1000.0)).sum();
                let end = start + dur.max(1.0) as u64;
                self.write_back(self.vms.get(&vm_id).unwrap(), &mem);
                self.timestamp(&c.ts_start, ticks(start));
                self.timestamp(&c.ts_end, ticks(end));
                self.queues.get_mut(&q).unwrap().end_ns = end;
                slog!("compute done: {} launches, {} instructions, {:.0} ns", runs.len(), runs.iter().map(|r| r.0).sum::<u64>(), dur);
            }
            Err(e) => {
                self.faults += 1;
                eprintln!("[agx-drm-shim] GPU fault (command reported complete, timestamps 0): {e}");
                self.timestamp(&c.ts_start, 0);
                self.timestamp(&c.ts_end, 0);
            }
        }
    }

    pub fn wait(&mut self, handles: &[u32], points: Option<&[u64]>, timeout: i64, all: bool, first: &mut u32) -> Res<()> {
        loop {
            self.progress()?;
            let mut done = vec![];
            for (i, h) in handles.iter().enumerate() {
                let v = *self.syncobjs.get(h).ok_or(ENOENT)?;
                let want = points.map(|p| p[i].max(1)).unwrap_or(1);
                done.push(v >= want);
            }
            let ok = if all { done.iter().all(|x| *x) } else { done.iter().any(|x| *x) };
            if ok || handles.is_empty() {
                *first = done.iter().position(|x| *x).unwrap_or(0) as u32;
                return Ok(());
            }
            if timeout <= 0 || (now_ns() as i64) >= timeout {
                return Err(ETIME);
            }
            return Err(-1); // the caller releases the lock, sleeps and retries
        }
    }
}

pub fn ts_now() -> u64 {
    ticks(now_ns())
}
