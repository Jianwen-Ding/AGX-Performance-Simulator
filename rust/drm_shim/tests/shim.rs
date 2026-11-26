//! tests/sync.c under the shim (LD_PRELOAD the built libagx_drm_shim.so: `cargo build --release` first), and
//! with the shim off (AGX_SHIM_DRM unset): every call passes through, so the node (absent here) does not open.
use std::process::Command;

fn shim() -> String {
    let dir = env!("CARGO_MANIFEST_DIR");
    for p in ["target/release/libagx_drm_shim.so", "target/debug/libagx_drm_shim.so"] {
        let f = format!("{dir}/{p}");
        if std::path::Path::new(&f).exists() {
            return f;
        }
    }
    panic!("build the shim first: cargo build --release");
}

fn client() -> std::path::PathBuf {
    let dir = env!("CARGO_MANIFEST_DIR");
    let exe = std::env::temp_dir().join(format!("agx_drm_shim_sync_{}", std::process::id()));
    let st = Command::new("cc")
        .args(["-O1", "-I", &format!("{dir}/include"), "-I", "/usr/include/drm", "-o"])
        .arg(&exe)
        .arg(format!("{dir}/tests/sync.c"))
        .status()
        .expect("cc");
    assert!(st.success());
    exe
}

#[test]
fn sync_pending_timeline_fault() {
    let exe = client();
    let out = Command::new(&exe)
        .env("LD_PRELOAD", shim())
        .env("AGX_SHIM_DRM", "1")
        .env("AGX_SHIM_TIMING", "0")
        .output()
        .unwrap();
    let so = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success() && so.contains("ok"), "{so}{}", String::from_utf8_lossy(&out.stderr));
    // the fault is reported on stderr, as dmesg would
    assert!(String::from_utf8_lossy(&out.stderr).contains("GPU fault"));
    if std::path::Path::new("/dev/dri/renderD128").exists() {
        return; // a real node: the pass-through check would open it
    }
    let off = Command::new(&exe).env("LD_PRELOAD", shim()).env_remove("AGX_SHIM_DRM").output().unwrap();
    let so = String::from_utf8_lossy(&off.stdout);
    // the real open (no such node here: ENOENT, or EPERM in a sandbox), not the shim
    assert!(!off.status.success() && so.contains("fd >= 0"), "{so}");
    let _ = std::fs::remove_file(&exe);
}
