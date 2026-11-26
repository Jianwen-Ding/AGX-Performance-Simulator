//! src/uapi.rs against the C headers: tests/layout.c (compiled here with cc) prints each structure's size and
//! the offsets of selected fields.
use agx_drm_shim::uapi::*;
use std::mem::{offset_of, size_of};

#[test]
fn layout_matches_c_headers() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let exe = std::env::temp_dir().join(format!("agx_drm_shim_layout_{}", std::process::id()));
    let st = std::process::Command::new("cc")
        .args(["-I", &format!("{dir}/include"), "-I", "/usr/include/drm", "-o"])
        .arg(&exe)
        .arg(format!("{dir}/tests/layout.c"))
        .status()
        .expect("cc");
    assert!(st.success(), "cc tests/layout.c");
    let out = String::from_utf8(std::process::Command::new(&exe).output().unwrap().stdout).unwrap();
    let _ = std::fs::remove_file(&exe);
    let c: std::collections::HashMap<&str, usize> =
        out.lines().filter_map(|l| l.split_once(' ')).map(|(k, v)| (k, v.parse().unwrap())).collect();
    let rust: &[(&str, usize)] = &[
        ("drm_version", size_of::<drm_version>()),
        ("drm_version.name", offset_of!(drm_version, name)),
        ("drm_version.desc", offset_of!(drm_version, desc)),
        ("drm_gem_close", size_of::<drm_gem_close>()),
        ("drm_get_cap", size_of::<drm_get_cap>()),
        ("drm_syncobj_create", size_of::<drm_syncobj_create>()),
        ("drm_syncobj_destroy", size_of::<drm_syncobj_destroy>()),
        ("drm_syncobj_array", size_of::<drm_syncobj_array>()),
        ("drm_syncobj_timeline_array", size_of::<drm_syncobj_timeline_array>()),
        ("drm_syncobj_transfer", size_of::<drm_syncobj_transfer>()),
        ("drm_syncobj_transfer.src_point", offset_of!(drm_syncobj_transfer, src_point)),
        ("drm_syncobj_wait.timeout_nsec", offset_of!(drm_syncobj_wait, timeout_nsec)),
        ("drm_syncobj_wait.first_signaled", offset_of!(drm_syncobj_wait, first_signaled)),
        ("drm_syncobj_timeline_wait.timeout_nsec", offset_of!(drm_syncobj_timeline_wait, timeout_nsec)),
        ("drm_syncobj_timeline_wait.first_signaled", offset_of!(drm_syncobj_timeline_wait, first_signaled)),
        ("drm_asahi_params_global", size_of::<drm_asahi_params_global>()),
        ("drm_asahi_params_global.core_masks", offset_of!(drm_asahi_params_global, core_masks)),
        ("drm_asahi_params_global.vm_start", offset_of!(drm_asahi_params_global, vm_start)),
        ("drm_asahi_params_global.max_commands_per_submission", offset_of!(drm_asahi_params_global, max_commands_per_submission)),
        ("drm_asahi_params_global.command_timestamp_frequency_hz", offset_of!(drm_asahi_params_global, command_timestamp_frequency_hz)),
        ("drm_asahi_get_params", size_of::<drm_asahi_get_params>()),
        ("drm_asahi_vm_create", size_of::<drm_asahi_vm_create>()),
        ("drm_asahi_vm_destroy", size_of::<drm_asahi_vm_destroy>()),
        ("drm_asahi_gem_create", size_of::<drm_asahi_gem_create>()),
        ("drm_asahi_gem_create.handle", offset_of!(drm_asahi_gem_create, handle)),
        ("drm_asahi_gem_mmap_offset", size_of::<drm_asahi_gem_mmap_offset>()),
        ("drm_asahi_gem_bind_op", size_of::<drm_asahi_gem_bind_op>()),
        ("drm_asahi_gem_bind_op.addr", offset_of!(drm_asahi_gem_bind_op, addr)),
        ("drm_asahi_vm_bind", size_of::<drm_asahi_vm_bind>()),
        ("drm_asahi_vm_bind.userptr", offset_of!(drm_asahi_vm_bind, userptr)),
        ("drm_asahi_gem_bind_object", size_of::<drm_asahi_gem_bind_object>()),
        ("drm_asahi_gem_bind_object.object_handle", offset_of!(drm_asahi_gem_bind_object, object_handle)),
        ("drm_asahi_queue_create", size_of::<drm_asahi_queue_create>()),
        ("drm_asahi_queue_create.usc_exec_base", offset_of!(drm_asahi_queue_create, usc_exec_base)),
        ("drm_asahi_queue_destroy", size_of::<drm_asahi_queue_destroy>()),
        ("drm_asahi_sync", size_of::<drm_asahi_sync>()),
        ("drm_asahi_cmd_header", size_of::<drm_asahi_cmd_header>()),
        ("drm_asahi_submit", size_of::<drm_asahi_submit>()),
        ("drm_asahi_submit.cmdbuf_size", offset_of!(drm_asahi_submit, cmdbuf_size)),
        ("drm_asahi_cmd_compute", size_of::<drm_asahi_cmd_compute>()),
        ("drm_asahi_cmd_compute.cdm_ctrl_stream_base", offset_of!(drm_asahi_cmd_compute, cdm_ctrl_stream_base)),
        ("drm_asahi_cmd_compute.helper", offset_of!(drm_asahi_cmd_compute, helper_binary)),
        ("drm_asahi_cmd_compute.ts", offset_of!(drm_asahi_cmd_compute, ts_start)),
        ("drm_asahi_get_time", size_of::<drm_asahi_get_time>()),
    ];
    let mut bad = vec![];
    for (k, v) in rust {
        match c.get(k) {
            Some(cv) if cv == v => {}
            other => bad.push(format!("{k}: rust {v}, C {other:?}")),
        }
    }
    assert!(bad.is_empty(), "layout mismatches:\n{}", bad.join("\n"));
    assert_eq!(c.len(), rust.len(), "tests/layout.c and tests/layout.rs list different fields");
}
