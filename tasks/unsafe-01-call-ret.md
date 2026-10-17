# Task unsafe-01: remaining G14C call/return cases

Updated 2026-10-17. The durable, current instructions are in the adjacent
playground: [call-return agent handoff](../../apple-documentation-playground/notes/call-return-handoff.md).
Read that file and its linked evidence before preparing a new probe. It replaces
the earlier plan that described all returns as unresolved.

Uniform full-32-lane direct, register-indirect and software-saved-link nested
direct calls are measured. Return targets are shader-buffer-relative byte
addresses; the measured continuation uses `pop_exec r0l, 2`.
The simulator opt-in is `AGX_EXPERIMENTAL_RET=buffer-pop2`. The historical value
`1` is a contradicted hypothesis and must not gate hardware.

Remaining work, with detailed construction and prerequisites in the handoff:

1. Isolate explicit return-register selection using two valid landing targets.
2. Partial-mask direct call, isolated return, then combined round trip.
3. Empty-mask issued versus skipped direct call, register call and return.
4. Two divergent register-call targets, after partial-mask behavior is measured.
5. Divergent return targets, isolated from call scheduling.
6. Mixed direct/indirect nesting, greater bounded depth and code placement.
7. Call/return cost after functional scope is stable.
8. Offline analysis of historical quiet out-of-program returns.

Only one agent submits GPU work at a time. Other agents can prepare simulator
regressions, independent oracles and retained-record analysis concurrently.
Every new surrounding program and candidate landing path must pass the simulator
and an independent oracle. Keep all transfers inside decoded instruction
boundaries. A surrogate gate does not establish unknown transfer semantics.

The owner has authorized higher-risk probes and clarified recovery policy:
log faults, validate recovery with an exact known-good control, then continue
if it passes. Persistent unresponsiveness or failed recovery stops GPU
submissions. Preserve and push reports, including hangs, before stopping.
The handoff links the bounded watchdog and unified-memory diagnostic procedure.

Documentation and evidence belong in the playground, not solely in this task
folder. Commit and push every affected repository before handing work off.
