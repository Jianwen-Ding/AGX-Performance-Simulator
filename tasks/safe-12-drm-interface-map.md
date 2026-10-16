# Task safe-12: map the direct-DRM interface (UAPI, command buffer, firmware fields)

Read [safe-probes.md](safe-probes.md) first (safety rules).

**Goal:** a written reference for everything between "AGX machine code" and "the GPU runs it". That covers every
ioctl, structure and field `tests/drm_compute.c` uses, what each one does, and which fields are still unexplained.
Stage 3's CUDA runtime (memory allocation, launches, streams) will be built on this, so it is the direct-DRM
counterpart of the ISA documentation.

**Sources:**
- `/usr/include/drm/asahi_drm.h`, the installed UAPI header (record its kernel version);
- `tests/drm_compute.c`, which uses GET_PARAMS, VM_CREATE / VM_BIND / VM_DESTROY, GEM_CREATE / GEM_MMAP_OFFSET,
  GEM_BIND_OBJECT (timestamps), QUEUE_CREATE / DESTROY, SUBMIT, GET_TIME;
- Mesa's `src/asahi/lib/agx_device.c`, `src/asahi/genxml/*.xml` (USC words, CDM launch words, the Shared word) in
  `AGX_Probe/mesa`;
- the playground's `experiments/baseline`, `results/2026-10-10-command-timing-direct-drm`, and the `sr-direct-drm`
  report.

## Safety

- **Most of this task is reading and host-side calls.** The kernel validates UAPI structures, so a rejected ioctl
  returns an errno and never reaches the GPU.
- **GPU-side fields** are the CDM launch words, USC words, and `AGX_WORD0_OR` / `AGX_SHADER_UNK2` / `AGX_USC_ORDER`
  values.
  - Use only values Mesa emits or that the playground already ran.
  - **Do not flip unknown GPU-side bits.** List them as unexplained instead.
- Helper programs (`AGX_HELPER_*`) stalled twice: document their fields from source only, and do not run them.

## Steps

1. **UAPI table.** For every ioctl and structure in `asahi_drm.h`, record:
   - fields;
   - whether `drm_compute.c` uses it;
   - what the playground has measured about it;
   - open questions.
   Unused ones include: other `drm_asahi_cmd_*` types (render), sync objects / timelines, `GEM_BIND_OBJECT` usages
   other than timestamps, `params_global` fields nobody has read yet, and VM bind flags.
2. **Host-side probes** (no GPU work submitted):
   - Print every `params_global` field and explain each (cross-check the P1 identity numbers).
   - Find the limits by asking for too much and recording the errno:
     - largest single BO;
     - largest VM bind;
     - number of VMs per process;
     - number of queues per VM;
     - VA ranges accepted by VM_BIND;
     - GEM_CREATE flags (writeback vs. not).
   - Queue priorities: 0 and 1 work. Record the errno for 2 and 3 without root.
3. **Command buffer layout.** Annotate the CDM stream and USC words that `drm_compute.c` builds, word by word, with
   each field's name from Mesa genxml and its measured effect (the Shared word, SR18/19, imageblock, preshader,
   barrier word `78 00`). Mark the fields that are unexplained.
4. **Sync primitives.** List how a CUDA runtime would build events and `cudaStreamWaitEvent`: syncobjs, timeline
   points, in/out fences on SUBMIT. Then test one host-side path:
   - two submissions on one queue, the second waiting on the first's syncobj;
   - the control kernel only;
   - firmware timestamps confirm the order.
5. **Unified memory notes:**
   - Can a BO be imported from a CPU allocation (userptr)? Find out from the header and errno only.
   - What does mmap caching mode do to CPU writes the GPU sees? Cross-reference [safe-03](safe-03-memory-system.md) §4.

## Deliverables

- Playground `docs/drm-interface.md` (the reference).
- A `results/<date>-drm-interface/` with the host-side probe outputs and a small `scripts/drm_limits.c` (host-side
  only).
- A `make verify-drm-interface` that re-checks the recorded limits offline. Push both repositories.
