# Rejected first layout

These are raw measurements of a trial, not the shipped layout. The trial used
a 2 KiB dense prefix, halfword-indexed cells in both tracers, inline lazy-tail
lookup and an explicit functional alignment check on every instruction.

It improved padding-heavy cases, but controls exposed regressions: full ALU
dispatch +3.4%, one-lane device atomic execution +8.7%, four-worker coalesced
cost memory tracing +10.1%, and four-worker functional tracing of a 512-add
active program +7.6%. The active-program control here used 512 adds, not the
2048-add control of the final layout. No accuracy check or calibration changed.

The final layout retains the functional cache's original byte indexing, uses
an 8 KiB dense prefix, and separates the tail lookup into a cold path. See the
parent directory for its independent paired measurements and remaining limits.
