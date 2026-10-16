# Task safe-04: driver and submission behaviour (direct DRM)

Read [safe-probes.md](safe-probes.md) first.

**Goal:** the numbers a CUDA runtime on Asahi needs for streams, events and launch overhead. Everything here runs
known-good kernels (the control kernel, or the delay loop from `run_sr113_reps.py`) and varies only host-side or UAPI
parameters that the harness already exposes, or simple new host code.

## 1. Queue priorities

- `drm_compute` takes `AGX_PRIORITY` (0-3).
- Run two processes concurrently (the `run_sr_vm_slot.py` pattern), one long-running low-priority queue and one
  high-priority queue with short dispatches.
- **Measure:** the firmware start / end timestamps of each submission, and whether high priority preempts, jumps the
  queue, or only reorders.

## 2. Concurrent queues and contexts

- Extend `run_sr113_reps.py --timeline` runs to measure overlap. Today's data shows 64-lane dispatches from different
  processes running at the same time.
- Find how many queues execute simultaneously, and whether one process with several queues behaves like several
  processes. This needs a small `drm_compute` change: an `AGX_QUEUES=N` option that creates N queues on one VM and
  submits round-robin.

## 3. Submission latency

- **Host submit -> firmware start**, and **firmware end -> fence signalled**, from host CLOCK_MONOTONIC against
  `DRM_IOCTL_ASAHI_GET_TIME` (already printed per submission).
- Measure across idle gaps of 0-100 ms, so GPU power-up is separated from driver cost.
- This is the floor for CUDA kernel-launch latency.

## 4. Command-stream limits

- **Several dispatches in one command**: the CDM stream with N launch records, each followed by the Mesa barrier
  word. The words already used are in `tests/drm_compute.c`.
- Find the per-dispatch overhead inside one command, against one command per dispatch.
- Start with N = 2 and check both oracles. Grow N only after each step passes.
- The largest grid is already known (2^30 threads, `make grid-coverage`). Record the largest command size accepted
  and the error returned beyond it (host-side rejection only, never a GPU hang).

## Deliverables

- A playground `novel-results/<date>-submission` with a verify target.
- A latency / overhead table for the CUDA launch path.
