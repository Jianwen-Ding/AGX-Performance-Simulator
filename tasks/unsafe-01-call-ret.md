# Task unsafe-01: `call` and `ret` on G14C (owner-approved, staged, simulator-gated)

Read [safe-probes.md](safe-probes.md) first for the general rules. This task is **not** a safe probe: it runs
control-flow instructions that Mesa never emits. The owner approved it on 2026-10-17 on the condition that every
program around the instruction under test is first checked in `agx_run`, so any failure can only come from the ISA.

**Status when this plan was written (2026-10-17):** `call` is confirmed on hardware; `ret` is unresolved. Nothing in
this file has been executed beyond what the "Established" section lists.

## Why it matters

PTX device functions can be inlined, as Mesa does for every shader. Real calls are needed only for recursion, function
pointers (`call.uni` through a register) and code-size limits. So this task is worth doing for completeness of the
backend and the documentation, but a CUDA translator does not depend on it.

## Known issue: read this before starting

Several assistant responses about the `ret` investigation were stopped mid-way by a safety classifier. It always
happened while analysing the `ret` results, never while running commands. The cause is unknown.
- If your own output is stopped, **stop and tell the owner**. Do not reword the same analysis or split it up to get
  around the block.
- Keep reports factual and tied to recorded words: what was run, what was stored, what the simulator predicted.

## Established (hardware, 6-byte `call`, no `ret`)

| Fact | Evidence |
|---|---|
| The 6-byte `call off` (`0xC010 | off << 16`, pc-relative from the instruction's own address) jumps to its target | probe `C1b_skip`: a marker write between the call and its target never ran |
| It writes r1 (32-bit) = `0x2000 + byte offset of the next instruction`, in every active lane | `C1a_next`, `C1b_skip`, `C1c_pad` |
| `0x2000` is the code's offset in the harness's USC buffer (`AGX_CODE_OFF`, default 0x2000), so the link is **buffer-relative**, not code-relative | `tests/drm_compute.c` lines 159-201 |
| The link moves exactly with the call's position (+12 bytes of padding gives +12) | `C1c_pad` |
| `call` clobbers r1, which the harness uses as the global thread id; programs must save and restore it | all probes use r20 as the copy |
| Execution continues normally after a `call` to the next instruction | `C1a_next` |

The simulator's `call` model (`rust/src/func.rs`, `OpKind::Call`) matches these three runs word for word (whole table
and output buffer).

## Observed for `ret` (unexplained)

Program shape for all three: `call +12` into a function F; F writes a marker and runs `ret rX`; the instruction after
the call is a `jmp_exec_any` over F; code after that stores markers. Byte offsets in the whole program: call at 98,
`jmp` at 104, F at 110.

| Probe | `ret` register | Result |
|---|---|---|
| `C2a_roundtrip` | r1 = the raw link, 0x2068 | **GPU fault**, recovered by firmware. Kernel log: unmapped read at `0x1000004100`, unit `UL1C`, `reason: Unmapped`, vm_slot 0x22. The USC buffer starts at `0x1000000000` and the code at `+0x2000`. 1 hang-budget event. |
| `C2b_ret_code_relative` | link - 0x2000 = 0x68 (the offset of the `jmp`) | **No fault.** Dispatch completed with firmware timestamps, but no store after the `ret` ran. Firmware time ~60 µs against ~9 µs for similar programs. |
| `C2c_ret_code_relative_traced` | same, with stores inside F | Same outcome. The stores inside F landed: marker, link `0x2068`, and the `ret` register `0x68`. Nothing after the `ret` ran. |

Raw records: `/scratch/apple-m2-playground/call-probes/*.json` (scratch is not in git; **bundle them first**, step 0).

## What is not known

1. Where execution goes after `ret` in C2b / C2c. The only fact is that it was not the instruction after the call,
   and that the dispatch did not fault.
2. What the register operand of `ret` means. The fault in C2a is consistent with `code_start + register`, but C2b
   does not return to the expected address under that reading, so that reading is at best incomplete.
3. Whether `ret` needs a hardware call context (a return stack, an exec-mask stack entry, or state set by a 2-byte
   `call reg`), and what it does without one.
4. Whether the 60 µs is a firmware or helper-program path (the earlier helper-program attempts also took ~0.5 s and
   were stopped; see the playground's `notes/failed-approaches.md`, "Custom helper program").
5. The 2-byte `call reg` form, lane-divergent targets, nested calls, and the cost of a call / ret pair.

## Hypotheses to separate (do not assume any)

- **H-A:** `ret` target = code start + a differently scaled or offset register value (for example halves, or minus the
  size of something). Predicts: some register value returns correctly; the fault address scales with the value.
- **H-B:** `ret` outside a hardware call context ends the SIMD group or the lane, as the 16 "ends" opcodes do
  (playground `novel-results/2026-10-11-opcode-descriptions`). Predicts: a marker stored before the `ret` lands,
  nothing after it runs for any register value, timing close to a normal end.
- **H-C:** `ret` with an unrecognised target is handed to firmware (a helper or trap path), which ends the group
  after a delay. Predicts: the elongated time of C2b / C2c, and no dependence on the register value.
- **H-D:** `call reg` (2 bytes) sets up the context that `ret` needs, and the 6-byte `call` does not. Predicts: a
  `call reg` / `ret` pair returns where the 6-byte form does not.
- **H-E:** `ret` returns to the address in a *different* register or an implicit one (for example r0-r1 as a pair).
  Predicts: no change when the named register changes.

## Rules (in addition to safe-probes.md)

1. **Targets stay inside the program.** A `call` or `ret` whose target the simulator cannot place inside the program
   is never run. Out-of-program targets are exactly what `agx_run` refuses.
2. **Fix the simulator before relying on it.** `func.rs` currently accepts `ret` with an assumed meaning
   (target = register value, code-relative) that **contradicts C2b / C2c**. Until a model fits all recorded runs, make
   `agx_run` refuse `ret` unless an explicit environment switch is set, so no probe looks gated by a wrong model.
   `call` may stay as it is.
3. **Expected-to-return programs only after the model fits.** Any program that is meant to come back from a `ret` is
   run only when the simulator predicts it from a model that reproduces every earlier recorded run.
4. **One dispatch at a time, stop at the first fault.** After each: diff `/save/kernel.log`, run
   `gpu_safety.py status`, check shmem in `/sys/fs/cgroup/memory.stat`, run a control dispatch. A completed dispatch
   with no stores is also a result to record, not a success.
5. **Observe, do not steer.** Prefer experiments that record where execution went (markers at candidate landing
   sites, timing) over experiments that depend on landing in one place.
6. **Do not combine** `call` / `ret` with a barrier in another SIMD group, a loop, or anything else that can wait
   forever if the group ends early.
7. **Budget:** 1 of 30 recovered events already used this boot (check `gpu_safety.py status`; it resets on reboot).
   Plan for at most 3 more.

## Plan

### Step 0: offline housekeeping (no GPU)
- Environment: `export PATH=/scratch/rust/cargo/bin:$PATH RUSTUP_HOME=/scratch/rust/rustup CARGO_HOME=/scratch/rust/cargo`,
  then `make rust` in this repo. Git identity: commit with
  `-c user.name="Jianwen Ding" -c user.email="62767966+Jianwen-Ding@users.noreply.github.com"`.
- Bundle the 7 raw records (C1a, C1b, C1c, C2a fault, C2b, C2c, and the C2a fault text with the kernel-log block above)
  into the playground as `results/2026-10-17-call-ret/records.json.gz`, the way `results/2026-10-17-atomic-variants`
  does it. Record the fault block verbatim as text.
- Apply rule 2 in `func.rs` and add a regression check that `agx_run` refuses `ret` without the switch.
- Re-read the committed work: `scripts/probe_calls.py`, `scripts/probe_variants.py` (variant-in-simulator check and the
  per-probe `allow_ops`), `scripts/coverage.py` (`gate(allow=)`), commit `3fb851e`.

### Step 1: what does `ret` do without a call? (cheapest, lowest information risk)
Family R1, no `call` and no function: a marker store, then `ret rX`, then more marker stores, in a straight-line
program. Vary X over registers holding values chosen so each hypothesis predicts a different outcome (a code-relative
offset of the next instruction, the same offset plus 0x2000, zero). Also a run where the `ret` is inside an
`if_icmp` / `pop_exec` block with only some lanes active.
- Records: which markers landed, per lane; firmware time; fault or not.
- Decides H-B / H-C against H-A / H-D.
- Gate: surrogate = the same program with a 2-byte `wait 0` in place of the `ret`; it must pass `agx_run`. The `ret`
  variants are not simulator-gated (the model is unfit), so keep every candidate target an instruction boundary
  inside the program and expect the group may end.

### Step 2: locate the landing site
If step 1 shows `ret` does transfer control, build a landing map: markers at several instruction-aligned candidate
offsets, each block storing a distinct marker and then reaching a final `stop`, selected by the register value. Run one
register value per dispatch. The result is a table: register value → landing marker (or none).

### Step 3: `call reg` (2-byte) and the `call` / `ret` pairing
Only after steps 1-2 give a model. Test H-D: `call r15` with r15 = a code-relative target, then F, then `ret`, with
the same markers. Learn whether the register form sets r1 as well and how lanes with different targets behave (applegpu's
note: it may mask execution to the lanes sharing a value, repeating until all are done; test that with two target
values across lanes, the only divergent test, and only if every target is inside the program).

### Step 4: round trip, nesting, cost
Once one pairing returns correctly and the simulator reproduces every recorded run:
- a round trip with the exact marker layout of C2c;
- two nested calls (the link must be saved in a second register by software);
- a call inside a divergent `if`;
- cost: dependent chains of N call / ret pairs, counted loop with an emulator-gated trip count; report cycles per
  pair at the fast clock cluster, power source recorded.

### Step 5: deliverables
- Simulator: `call` / `ret` / `call reg` models that reproduce every record; `make suite-check` and
  `make coverage-check` unchanged or better.
- Coverage cases (catalog item `call_ret`) for each recorded behaviour.
- Playground: `novel-results/<date>-call-ret/README.md` (claim, evidence, limits), a `make verify-call-ret` target
  with an independent model, `docs/isa-coverage.md` status for `call` / `ret`, and the PTX mapping (`call.uni`,
  `ret`, indirect calls) with the inlining recommendation.
- Update `notes/failed-approaches.md` with the C2a fault and the two quiet runs, whatever the outcome.
- Push both repositories.

## Acceptance

- Every recorded run is reproduced by the simulator, or is listed as unexplained.
- No dispatch was submitted with a target the simulator could not place inside the program, except steps 1-2, where
  that limitation is stated in the report.
- Hang budget use is reported.
