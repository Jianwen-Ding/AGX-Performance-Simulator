//! `libagx_drm_shim.so`: an optional LD_PRELOAD shim that serves the Asahi DRM uAPI (the render node's ioctls
//! and mmaps, `include/drm/asahi_drm.h`) with the AGX Performance Simulator, so an unmodified direct-DRM client
//! (`tests/drm_compute.c`, a runtime built on the uAPI, a driver) runs its compute commands on the functional
//! model and gets firmware timestamps from the dispatch model, on any Linux machine.
//!
//! ```text
//! AGX_SHIM_DRM=1 LD_PRELOAD=rust/drm_shim/target/release/libagx_drm_shim.so ./client /dev/dri/renderD128 ...
//! ```
//!
//! * `AGX_SHIM_DRM` turns it on (unset: every call passes through untouched): `1` serves `/dev/dri/renderD128`,
//!   any other value is the path to serve (e.g. `/dev/agx-sim`, which need not exist).
//! * `AGX_SHIM_TIMING=0`: no dispatch model (functional only; 1 us per launch). `AGX_SHIM_LOG=1`: log every
//!   object and launch to stderr. `AGX_SHIM_MAX_STEPS`: the functional model's step limit per SIMD group (a
//!   kernel that runs longer faults). `AGX_SHIM_PARAMS`: GET_PARAMS overrides (src/dev.rs `params`).
//!
//! Served: DRM_IOCTL_VERSION ("asahi"), GET_CAP (syncobj, timeline), SET_CLIENT_CAP, GEM_CLOSE, the SYNCOBJ
//! family (create, destroy, wait, timeline wait, reset, signal, timeline signal, query, transfer); ASAHI
//! GET_PARAMS, GET_TIME, VM_CREATE / DESTROY / BIND, GEM_CREATE / MMAP_OFFSET / BIND_OBJECT (timestamps),
//! QUEUE_CREATE / DESTROY, SUBMIT (compute; attachments accepted). Not served: render commands (EOPNOTSUPP),
//! PRIME and syncobj fd export / import (EOPNOTSUPP), libdrm device enumeration through sysfs (a client must open
//! the node by path), a dup of the shim's fd. Textures, samplers and the helper program are ignored.
//!
//! Interposed symbols: open, open64, openat, openat64, __open_2, __open64_2, ioctl, mmap, mmap64, close. Calls
//! that are not the shim's go to the kernel through raw syscalls (no dlsym, no allocation: the allocator itself
//! calls mmap).
#![allow(clippy::missing_safety_doc)]

pub mod dev;
pub mod exec;
pub mod uapi;

use dev::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use uapi::*;

extern "C" {
    fn syscall(n: i64, ...) -> i64;
    fn __errno_location() -> *mut i32;
    fn memfd_create(name: *const u8, flags: u32) -> i32;
    fn usleep(us: u32) -> i32;
}

#[cfg(target_arch = "x86_64")]
mod sys {
    pub const OPENAT: i64 = 257;
    pub const IOCTL: i64 = 16;
    pub const MMAP: i64 = 9;
    pub const CLOSE: i64 = 3;
}
#[cfg(target_arch = "aarch64")]
mod sys {
    pub const OPENAT: i64 = 56;
    pub const IOCTL: i64 = 29;
    pub const MMAP: i64 = 222;
    pub const CLOSE: i64 = 57;
}
const AT_FDCWD: i64 = -100;

/// fds the shim serves (fd < 4096), readable without a lock
static FDS: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];

fn is_ours(fd: i32) -> bool {
    fd >= 0 && fd < 4096 && FDS[fd as usize / 64].load(Ordering::Acquire) & (1 << (fd % 64)) != 0
}

fn mark(fd: i32, on: bool) {
    if fd >= 0 && fd < 4096 {
        let b = 1u64 << (fd % 64);
        if on {
            FDS[fd as usize / 64].fetch_or(b, Ordering::AcqRel);
        } else {
            FDS[fd as usize / 64].fetch_and(!b, Ordering::AcqRel);
        }
    }
}

fn files() -> &'static Mutex<HashMap<i32, File>> {
    static F: OnceLock<Mutex<HashMap<i32, File>>> = OnceLock::new();
    F.get_or_init(|| Mutex::new(HashMap::new()))
}

/// the path served, if the shim is on
fn served_path() -> Option<&'static str> {
    static P: OnceLock<Option<String>> = OnceLock::new();
    P.get_or_init(|| match std::env::var("AGX_SHIM_DRM") {
        Ok(v) if v == "1" => Some("/dev/dri/renderD128".into()),
        Ok(v) if !v.is_empty() && v != "0" => Some(v),
        _ => None,
    })
    .as_deref()
}

fn set_errno(e: i32) {
    unsafe { *__errno_location() = e };
}

unsafe fn sys_ret(r: i64) -> i64 {
    if r < 0 && r > -4096 {
        set_errno(-r as i32);
        -1
    } else {
        r
    }
}

unsafe fn open_shim(path: *const u8) -> Option<i32> {
    let want = served_path()?;
    if path.is_null() {
        return None;
    }
    let p = std::ffi::CStr::from_ptr(path as *const std::ffi::c_char).to_str().ok()?;
    if p != want {
        return None;
    }
    let fd = memfd_create(b"agx-drm-shim\0".as_ptr(), 1);
    if fd < 0 {
        return Some(-1);
    }
    files().lock().unwrap().insert(fd, File::default());
    mark(fd, true);
    slog!("open {p} -> fd {fd} (simulated M2 Max)");
    Some(fd)
}

#[no_mangle]
pub unsafe extern "C" fn openat(dirfd: i32, path: *const u8, flags: i32, mode: u32) -> i32 {
    if let Some(fd) = open_shim(path) {
        return fd;
    }
    sys_ret(syscall(sys::OPENAT, dirfd as i64, path, flags as i64, mode as i64)) as i32
}
#[no_mangle]
pub unsafe extern "C" fn openat64(dirfd: i32, path: *const u8, flags: i32, mode: u32) -> i32 {
    openat(dirfd, path, flags, mode)
}
#[no_mangle]
pub unsafe extern "C" fn open(path: *const u8, flags: i32, mode: u32) -> i32 {
    openat(AT_FDCWD as i32, path, flags, mode)
}
#[no_mangle]
pub unsafe extern "C" fn open64(path: *const u8, flags: i32, mode: u32) -> i32 {
    openat(AT_FDCWD as i32, path, flags, mode)
}
#[no_mangle]
pub unsafe extern "C" fn __open_2(path: *const u8, flags: i32) -> i32 {
    openat(AT_FDCWD as i32, path, flags, 0)
}
#[no_mangle]
pub unsafe extern "C" fn __open64_2(path: *const u8, flags: i32) -> i32 {
    openat(AT_FDCWD as i32, path, flags, 0)
}

#[no_mangle]
pub unsafe extern "C" fn close(fd: i32) -> i32 {
    if is_ours(fd) {
        mark(fd, false);
        if let Some(mut f) = files().lock().unwrap().remove(&fd) {
            let hs: Vec<u32> = f.bos.keys().copied().collect();
            for h in hs {
                let _ = f.gem_close(h);
            }
            if f.faults > 0 {
                eprintln!("[agx-drm-shim] fd {fd} closed: {} GPU faults", f.faults);
            }
        }
    }
    sys_ret(syscall(sys::CLOSE, fd as i64)) as i32
}

#[no_mangle]
pub unsafe extern "C" fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8 {
    if is_ours(fd) {
        let target = files().lock().unwrap().get(&fd).and_then(|f| f.mmap_target(off as u64, len as u64));
        return match target {
            Some((memfd, o)) => sys_ret(syscall(sys::MMAP, addr, len, prot as i64, flags as i64, memfd as i64, o as i64)) as *mut u8,
            None => {
                set_errno(EINVAL);
                usize::MAX as *mut u8
            }
        };
    }
    sys_ret(syscall(sys::MMAP, addr, len, prot as i64, flags as i64, fd as i64, off)) as *mut u8
}
#[no_mangle]
pub unsafe extern "C" fn mmap64(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8 {
    mmap(addr, len, prot, flags, fd, off)
}

#[no_mangle]
pub unsafe extern "C" fn ioctl(fd: i32, req: u64, arg: *mut u8) -> i32 {
    if !is_ours(fd) {
        return sys_ret(syscall(sys::IOCTL, fd as i64, req, arg)) as i32;
    }
    let ty = ((req >> 8) & 0xff) as u32;
    let nr = (req & 0xff) as u32;
    if ty != DRM_IOCTL_TYPE {
        set_errno(ENOTTY);
        return -1;
    }
    loop {
        let r = {
            let mut g = files().lock().unwrap();
            let f = match g.get_mut(&fd) {
                Some(f) => f,
                None => {
                    set_errno(9);
                    return -1;
                }
            };
            handle(f, nr, arg)
        };
        match r {
            Ok(()) => return 0,
            Err(-1) => {
                // a syncobj wait that is not satisfied yet: let other threads signal
                usleep(200);
                continue;
            }
            Err(e) => {
                slog!("ioctl nr {nr:#x}: errno {e}");
                set_errno(e);
                return -1;
            }
        }
    }
}

fn copy_str(dst: *mut u8, cap: usize, s: &str) -> usize {
    if !dst.is_null() {
        let n = cap.min(s.len());
        unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), dst, n) };
    }
    s.len()
}

unsafe fn arr<T: Copy>(p: u64, n: u32) -> Res<Vec<T>> {
    if n == 0 {
        return Ok(vec![]);
    }
    if p == 0 {
        return Err(EFAULT);
    }
    Ok((0..n as usize).map(|i| std::ptr::read_unaligned((p as *const T).add(i))).collect())
}

unsafe fn handle(f: &mut File, nr: u32, arg: *mut u8) -> Res<()> {
    if arg.is_null() {
        return Err(EFAULT);
    }
    macro_rules! a {
        ($t:ty) => {
            &mut *(arg as *mut $t)
        };
    }
    match nr {
        NR_VERSION => {
            let v = a!(drm_version);
            v.version_major = 1;
            v.version_minor = 0;
            v.version_patchlevel = 0;
            v.name_len = copy_str(v.name, v.name_len, "asahi");
            v.date_len = copy_str(v.date, v.date_len, "20251126");
            v.desc_len = copy_str(v.desc, v.desc_len, "Apple AGX (AGX Performance Simulator)");
        }
        NR_GET_CAP => {
            let c = a!(drm_get_cap);
            c.value = match c.capability {
                DRM_CAP_SYNCOBJ | DRM_CAP_SYNCOBJ_TIMELINE => 1,
                0x1 => 0,  // DUMB_BUFFER
                0x5 => 3,  // PRIME: import | export flags (not served: the ioctls return EOPNOTSUPP)
                _ => return Err(EINVAL),
            };
        }
        NR_SET_CLIENT_CAP => {}
        NR_GEM_CLOSE => f.gem_close(a!(drm_gem_close).handle)?,
        NR_PRIME_HANDLE_TO_FD | NR_PRIME_FD_TO_HANDLE | NR_SYNCOBJ_HANDLE_TO_FD | NR_SYNCOBJ_FD_TO_HANDLE => return Err(EOPNOTSUPP),
        NR_SYNCOBJ_CREATE => {
            let c = a!(drm_syncobj_create);
            f.next_syncobj += 1;
            c.handle = f.next_syncobj;
            f.syncobjs.insert(c.handle, (c.flags & DRM_SYNCOBJ_CREATE_SIGNALED != 0) as u64);
        }
        NR_SYNCOBJ_DESTROY => {
            f.syncobjs.remove(&a!(drm_syncobj_destroy).handle).ok_or(ENOENT)?;
        }
        NR_SYNCOBJ_RESET | NR_SYNCOBJ_SIGNAL => {
            let s = a!(drm_syncobj_array);
            for h in arr::<u32>(s.handles, s.count_handles)? {
                let v = f.syncobjs.get_mut(&h).ok_or(ENOENT)?;
                *v = if nr == NR_SYNCOBJ_SIGNAL { (*v).max(1) } else { 0 };
            }
            f.progress()?;
        }
        NR_SYNCOBJ_TIMELINE_SIGNAL => {
            let s = a!(drm_syncobj_timeline_array);
            let hs = arr::<u32>(s.handles, s.count_handles)?;
            let ps = arr::<u64>(s.points, s.count_handles)?;
            for (h, p) in hs.iter().zip(ps) {
                let v = f.syncobjs.get_mut(h).ok_or(ENOENT)?;
                *v = (*v).max(p.max(1));
            }
            f.progress()?;
        }
        NR_SYNCOBJ_QUERY => {
            let s = a!(drm_syncobj_timeline_array);
            let hs = arr::<u32>(s.handles, s.count_handles)?;
            for (i, h) in hs.iter().enumerate() {
                let v = *f.syncobjs.get(h).ok_or(ENOENT)?;
                std::ptr::write_unaligned((s.points as *mut u64).add(i), v);
            }
        }
        NR_SYNCOBJ_TRANSFER => {
            let t = a!(drm_syncobj_transfer);
            let src = *f.syncobjs.get(&t.src_handle).ok_or(ENOENT)?;
            let ok = if t.src_point == 0 { src >= 1 } else { src >= t.src_point };
            let d = f.syncobjs.get_mut(&t.dst_handle).ok_or(ENOENT)?;
            if ok {
                *d = (*d).max(if t.dst_point == 0 { 1 } else { t.dst_point });
            }
        }
        NR_SYNCOBJ_WAIT => {
            let w = a!(drm_syncobj_wait);
            let hs = arr::<u32>(w.handles, w.count_handles)?;
            let mut first = 0;
            f.wait(&hs, None, w.timeout_nsec, w.flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL != 0, &mut first)?;
            w.first_signaled = first;
        }
        NR_SYNCOBJ_TIMELINE_WAIT => {
            let w = a!(drm_syncobj_timeline_wait);
            let hs = arr::<u32>(w.handles, w.count_handles)?;
            let ps = arr::<u64>(w.points, w.count_handles)?;
            let mut first = 0;
            f.wait(&hs, Some(&ps), w.timeout_nsec, w.flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL != 0, &mut first)?;
            w.first_signaled = first;
        }
        n if n >= DRM_COMMAND_BASE => asahi(f, n - DRM_COMMAND_BASE, arg)?,
        _ => return Err(EINVAL),
    }
    Ok(())
}

unsafe fn asahi(f: &mut File, id: u32, arg: *mut u8) -> Res<()> {
    macro_rules! a {
        ($t:ty) => {
            &mut *(arg as *mut $t)
        };
    }
    match id {
        ASAHI_GET_PARAMS => {
            let g = a!(drm_asahi_get_params);
            if g.param_group != 0 || g.pointer == 0 {
                return Err(EINVAL);
            }
            let p = params();
            let n = (g.size as usize).min(std::mem::size_of::<drm_asahi_params_global>());
            std::ptr::copy_nonoverlapping(&p as *const _ as *const u8, g.pointer as *mut u8, n);
            g.size = n as u64;
        }
        ASAHI_GET_TIME => a!(drm_asahi_get_time).gpu_timestamp = ts_now(),
        ASAHI_VM_CREATE => {
            let c = a!(drm_asahi_vm_create);
            let p = params();
            if c.kernel_end > p.vm_end || c.kernel_start >= c.kernel_end || c.kernel_end - c.kernel_start < p.vm_kernel_min_size {
                return Err(EINVAL);
            }
            f.next_vm += 1;
            c.vm_id = f.next_vm;
            f.vms.insert(c.vm_id, Vm { binds: vec![], kernel: (c.kernel_start, c.kernel_end), slot: c.vm_id % 64 });
        }
        ASAHI_VM_DESTROY => {
            f.vms.remove(&a!(drm_asahi_vm_destroy).vm_id).ok_or(ENOENT)?;
        }
        ASAHI_VM_BIND => {
            let b = a!(drm_asahi_vm_bind);
            f.vm_bind(b)?;
        }
        ASAHI_GEM_CREATE => f.gem_create(a!(drm_asahi_gem_create))?,
        ASAHI_GEM_MMAP_OFFSET => {
            let m = a!(drm_asahi_gem_mmap_offset);
            m.offset = f.bos.get(&m.handle).ok_or(ENOENT)?.mmap_offset;
        }
        ASAHI_GEM_BIND_OBJECT => {
            let o = a!(drm_asahi_gem_bind_object);
            match o.op {
                DRM_ASAHI_BIND_OBJECT_OP_BIND => {
                    let bo = f.bos.get(&o.handle).ok_or(ENOENT)?;
                    if o.flags & DRM_ASAHI_BIND_OBJECT_USAGE_TIMESTAMPS == 0 || o.offset + o.range > bo.size {
                        return Err(EINVAL);
                    }
                    f.next_object += 1;
                    o.object_handle = f.next_object;
                    f.objects.insert(o.object_handle, (o.handle, o.offset, o.range));
                }
                DRM_ASAHI_BIND_OBJECT_OP_UNBIND => {
                    f.objects.remove(&o.object_handle).ok_or(ENOENT)?;
                }
                _ => return Err(EINVAL),
            }
        }
        ASAHI_QUEUE_CREATE => {
            let q = a!(drm_asahi_queue_create);
            if !f.vms.contains_key(&q.vm_id) || q.priority > 1 {
                // priorities 2 and 3 need CAP_SYS_NICE on the real driver
                return Err(if q.priority > 1 { 1 } else { ENOENT });
            }
            f.next_queue += 1;
            q.queue_id = f.next_queue;
            f.queues.insert(q.queue_id, Queue { vm: q.vm_id, usc_base: q.usc_exec_base, end_ns: 0, pending: Default::default() });
        }
        ASAHI_QUEUE_DESTROY => {
            f.queues.remove(&a!(drm_asahi_queue_destroy).queue_id).ok_or(ENOENT)?;
        }
        ASAHI_SUBMIT => f.submit(a!(drm_asahi_submit))?,
        _ => return Err(EINVAL),
    }
    Ok(())
}
