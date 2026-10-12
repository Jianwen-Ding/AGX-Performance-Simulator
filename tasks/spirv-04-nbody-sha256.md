# Task SPIR-V 4: transcendental and integer pipes, N-body and SHA-256

**Workloads:** `nbody` (16 Ki bodies: 268 M interactions, large; 512 bodies, small), `sha256` (256 Ki single-block messages, large; 256, small). The procedure is in `tasks/spirv-workloads.md`.

## Why these

They are two compute-bound kernels on different pipes:

- `nbody`'s inner loop is FMA plus one `inversesqrt` per interaction, so the rsqrt / transcendental pipe (`novel-results/2026-10-11-transcendental-pipe`) sets the pace;
- `sha256` is pure 32-bit integer work, about 3,000 ALU ops per thread: rotates (shift/or, or `bfi` / `extr`), adds, and the logical functions as `bitop` truth tables. It also keeps a 64-entry message schedule live (register pressure, possible spilling).

## Checks

1. **Hardware vs reference.**
   - **nbody:** velocity-update relative error < 1e-4, position relative error < 1e-6.
   - **sha256:** every digest exact.
2. **Simulator vs hardware.**
   - **sha256:** must match word for word.
   - **nbody:** only needs to be close (within the reference tolerance). The simulator's correctly rounded fp32 `rsqrt` differs from the hardware's in about 10–15% of lanes by 1–3 ulp, which is accepted; report the max relative difference.
3. **Compiled code.**
   - **nbody:** which rsqrt form is used: `rsqrt` or `rsqrt_special`, and any Newton step around it?
   - **sha256:** how are rotates compiled? Count the instructions per round. Are there `stack_store` / `stack_load` (spilling of `w[64]`)? If so, `sim` will stop: stack instructions are on the coverage `MANUAL` list. Report it.
4. **Performance.**
   - **nbody:** interactions/s. Compare with the rsqrt throughput per core from the instruction bench.
   - **sha256:** hashes/s, and integer ops/s against the integer ALU peak.
   - Run `predict` for both. The transcendental pipe's issue cost in the model is a direct check.

## Deliverable

`results/<date>-spirv-nbody-sha256/README.md`, plus the artefacts.
