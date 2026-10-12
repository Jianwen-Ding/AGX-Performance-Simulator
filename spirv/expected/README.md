# GPU recordings of the SPIR-V workloads (expected values)

Empty until `make asahi-verify` (or `scripts/spirv_task.py record`) runs on the M2. Every workload is **PENDING** in
`make spirv-verify` until then.

One directory per workload:

| File | Contents |
|---|---|
| `hardware.json` | Device and driver (vk_run's device line), UTC time, numpy version, input SHA-256s, and the GPU's verdict against the numpy reference. Small size: GPU time per dispatch. Large size: GPU time per dispatch, output SHA-256s and throughput. |
| `out<binding>.bin.gz` | Every read-back buffer of the small size, exactly as the GPU left it: these are the expected values. |
| `code/` | The driver's compiled binary from that machine (`shader.bin`, VK_EXT_shader_object) and its extract: `main.bin`, `preamble.bin`, listings, `info.json`. The simulator runs this code. |

How `make spirv-verify` uses a recording:

- It regenerates the inputs from the workload's seed and checks them against the SHA-256s. A mismatch is STALE.
- It runs the functional simulator on `code/`, and compares every output word with `out<binding>.bin.gz`.
- With `--timing`, it also predicts the large size and prints it next to the recorded time.

`record` refuses anything that did not execute on the M2's GPU: a non-AGX device, the drm-shim (`LD_PRELOAD`), or a
non-Apple machine. `--any-device` exists only to test the script. Its recordings are marked
`test_recording_not_hardware`, and `verify` keeps reporting them as PENDING.
