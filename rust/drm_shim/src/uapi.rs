//! The uAPI structures the shim reads and writes: `include/drm/asahi_drm.h` (Linux, MIT; vendored in
//! `include/drm/`) and the core DRM ones of `drm/drm.h`. `tests/drm_shim_layout.c` checks the sizes and offsets
//! against the C headers.
#![allow(non_camel_case_types, dead_code)]

// ioctl numbers: _IOC_NR (the low 8 bits) of the request, for type 'd'
pub const DRM_IOCTL_TYPE: u32 = b'd' as u32;
pub const NR_VERSION: u32 = 0x00;
pub const NR_GEM_CLOSE: u32 = 0x09;
pub const NR_GET_CAP: u32 = 0x0c;
pub const NR_SET_CLIENT_CAP: u32 = 0x0d;
pub const NR_PRIME_HANDLE_TO_FD: u32 = 0x2d;
pub const NR_PRIME_FD_TO_HANDLE: u32 = 0x2e;
pub const NR_SYNCOBJ_CREATE: u32 = 0xBF;
pub const NR_SYNCOBJ_DESTROY: u32 = 0xC0;
pub const NR_SYNCOBJ_HANDLE_TO_FD: u32 = 0xC1;
pub const NR_SYNCOBJ_FD_TO_HANDLE: u32 = 0xC2;
pub const NR_SYNCOBJ_WAIT: u32 = 0xC3;
pub const NR_SYNCOBJ_RESET: u32 = 0xC4;
pub const NR_SYNCOBJ_SIGNAL: u32 = 0xC5;
pub const NR_SYNCOBJ_TIMELINE_WAIT: u32 = 0xCA;
pub const NR_SYNCOBJ_QUERY: u32 = 0xCB;
pub const NR_SYNCOBJ_TRANSFER: u32 = 0xCC;
pub const NR_SYNCOBJ_TIMELINE_SIGNAL: u32 = 0xCD;

pub const DRM_COMMAND_BASE: u32 = 0x40;
pub const ASAHI_GET_PARAMS: u32 = 0;
pub const ASAHI_GET_TIME: u32 = 1;
pub const ASAHI_VM_CREATE: u32 = 2;
pub const ASAHI_VM_DESTROY: u32 = 3;
pub const ASAHI_VM_BIND: u32 = 4;
pub const ASAHI_GEM_CREATE: u32 = 5;
pub const ASAHI_GEM_MMAP_OFFSET: u32 = 6;
pub const ASAHI_GEM_BIND_OBJECT: u32 = 7;
pub const ASAHI_QUEUE_CREATE: u32 = 8;
pub const ASAHI_QUEUE_DESTROY: u32 = 9;
pub const ASAHI_SUBMIT: u32 = 10;

pub const DRM_CAP_SYNCOBJ: u64 = 0x13;
pub const DRM_CAP_SYNCOBJ_TIMELINE: u64 = 0x14;
pub const DRM_SYNCOBJ_CREATE_SIGNALED: u32 = 1;
pub const DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL: u32 = 1;
pub const DRM_SYNCOBJ_QUERY_FLAGS_LAST_SUBMITTED: u32 = 1;

pub const DRM_ASAHI_MAX_CLUSTERS: usize = 64;
pub const DRM_ASAHI_GEM_WRITEBACK: u32 = 1;
pub const DRM_ASAHI_GEM_VM_PRIVATE: u32 = 2;
pub const DRM_ASAHI_BIND_UNBIND: u32 = 1;
pub const DRM_ASAHI_BIND_READ: u32 = 2;
pub const DRM_ASAHI_BIND_WRITE: u32 = 4;
pub const DRM_ASAHI_BIND_SINGLE_PAGE: u32 = 8;
pub const DRM_ASAHI_BIND_OBJECT_OP_BIND: u32 = 0;
pub const DRM_ASAHI_BIND_OBJECT_OP_UNBIND: u32 = 1;
pub const DRM_ASAHI_BIND_OBJECT_USAGE_TIMESTAMPS: u32 = 1;
pub const DRM_ASAHI_CMD_RENDER: u16 = 0;
pub const DRM_ASAHI_CMD_COMPUTE: u16 = 1;
pub const DRM_ASAHI_SET_VERTEX_ATTACHMENTS: u16 = 2;
pub const DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS: u16 = 3;
pub const DRM_ASAHI_SET_COMPUTE_ATTACHMENTS: u16 = 4;
pub const DRM_ASAHI_SYNC_SYNCOBJ: u32 = 0;
pub const DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ: u32 = 1;
pub const DRM_ASAHI_BARRIER_NONE: u16 = 0xFFFF;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct drm_version {
    pub version_major: i32,
    pub version_minor: i32,
    pub version_patchlevel: i32,
    pub name_len: usize,
    pub name: *mut u8,
    pub date_len: usize,
    pub date: *mut u8,
    pub desc_len: usize,
    pub desc: *mut u8,
}

#[repr(C)]
pub struct drm_gem_close {
    pub handle: u32,
    pub pad: u32,
}

#[repr(C)]
pub struct drm_get_cap {
    pub capability: u64,
    pub value: u64,
}

#[repr(C)]
pub struct drm_syncobj_create {
    pub handle: u32,
    pub flags: u32,
}

#[repr(C)]
pub struct drm_syncobj_destroy {
    pub handle: u32,
    pub pad: u32,
}

#[repr(C)]
pub struct drm_syncobj_wait {
    pub handles: u64,
    pub timeout_nsec: i64,
    pub count_handles: u32,
    pub flags: u32,
    pub first_signaled: u32,
    pub pad: u32,
    // deadline_nsec (newer kernels) follows; not read
}

#[repr(C)]
pub struct drm_syncobj_timeline_wait {
    pub handles: u64,
    pub points: u64,
    pub timeout_nsec: i64,
    pub count_handles: u32,
    pub flags: u32,
    pub first_signaled: u32,
    pub pad: u32,
}

#[repr(C)]
pub struct drm_syncobj_array {
    pub handles: u64,
    pub count_handles: u32,
    pub pad: u32,
}

#[repr(C)]
pub struct drm_syncobj_timeline_array {
    pub handles: u64,
    pub points: u64,
    pub count_handles: u32,
    pub flags: u32,
}

#[repr(C)]
pub struct drm_syncobj_transfer {
    pub src_handle: u32,
    pub dst_handle: u32,
    pub src_point: u64,
    pub dst_point: u64,
    pub flags: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct drm_asahi_params_global {
    pub features: u64,
    pub gpu_generation: u32,
    pub gpu_variant: u32,
    pub gpu_revision: u32,
    pub chip_id: u32,
    pub num_dies: u32,
    pub num_clusters_total: u32,
    pub num_cores_per_cluster: u32,
    pub max_frequency_khz: u32,
    pub core_masks: [u64; DRM_ASAHI_MAX_CLUSTERS],
    pub vm_start: u64,
    pub vm_end: u64,
    pub vm_kernel_min_size: u64,
    pub max_commands_per_submission: u32,
    pub max_attachments: u32,
    pub command_timestamp_frequency_hz: u64,
}

#[repr(C)]
pub struct drm_asahi_get_params {
    pub param_group: u32,
    pub pad: u32,
    pub pointer: u64,
    pub size: u64,
}

#[repr(C)]
pub struct drm_asahi_vm_create {
    pub kernel_start: u64,
    pub kernel_end: u64,
    pub vm_id: u32,
    pub pad: u32,
}

#[repr(C)]
pub struct drm_asahi_vm_destroy {
    pub vm_id: u32,
    pub pad: u32,
}

#[repr(C)]
pub struct drm_asahi_gem_create {
    pub size: u64,
    pub flags: u32,
    pub vm_id: u32,
    pub handle: u32,
    pub pad: u32,
}

#[repr(C)]
pub struct drm_asahi_gem_mmap_offset {
    pub handle: u32,
    pub flags: u32,
    pub offset: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct drm_asahi_gem_bind_op {
    pub flags: u32,
    pub handle: u32,
    pub offset: u64,
    pub range: u64,
    pub addr: u64,
}

#[repr(C)]
pub struct drm_asahi_vm_bind {
    pub vm_id: u32,
    pub num_binds: u32,
    pub stride: u32,
    pub pad: u32,
    pub userptr: u64,
}

#[repr(C)]
pub struct drm_asahi_gem_bind_object {
    pub op: u32,
    pub flags: u32,
    pub handle: u32,
    pub vm_id: u32,
    pub offset: u64,
    pub range: u64,
    pub object_handle: u32,
    pub pad: u32,
}

#[repr(C)]
pub struct drm_asahi_queue_create {
    pub flags: u32,
    pub vm_id: u32,
    pub priority: u32,
    pub queue_id: u32,
    pub usc_exec_base: u64,
}

#[repr(C)]
pub struct drm_asahi_queue_destroy {
    pub queue_id: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct drm_asahi_sync {
    pub sync_type: u32,
    pub handle: u32,
    pub timeline_value: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct drm_asahi_cmd_header {
    pub cmd_type: u16,
    pub size: u16,
    pub vdm_barrier: u16,
    pub cdm_barrier: u16,
}

#[repr(C)]
pub struct drm_asahi_submit {
    pub syncs: u64,
    pub cmdbuf: u64,
    pub flags: u32,
    pub queue_id: u32,
    pub in_sync_count: u32,
    pub out_sync_count: u32,
    pub cmdbuf_size: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_asahi_timestamp {
    pub handle: u32,
    pub offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_asahi_cmd_compute {
    pub flags: u32,
    pub sampler_count: u32,
    pub cdm_ctrl_stream_base: u64,
    pub cdm_ctrl_stream_end: u64,
    pub sampler_heap: u64,
    pub helper_binary: u32,
    pub helper_cfg: u32,
    pub helper_data: u64,
    pub ts_start: drm_asahi_timestamp,
    pub ts_end: drm_asahi_timestamp,
}

#[repr(C)]
pub struct drm_asahi_get_time {
    pub flags: u64,
    pub gpu_timestamp: u64,
}
