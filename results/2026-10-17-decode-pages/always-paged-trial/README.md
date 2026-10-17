# Rejected always-paged functional lookup

This second trial used an 8 KiB dense prefix and cold tail-page lookup, with
byte-indexed functional cells and halfword-indexed cost cells. Every functional
program still selected the paged container, even when short enough to fit in
the prefix. These are raw trial data, not final measurements.

Padding-heavy cost tracing improved 3.535-4.393x at 1 MiB. Long active-code
controls (2048 adds, over 16 KiB, executing the tail) were near baseline.
However, the one-lane device-atomic control was 8.7% slower and four-worker
coalesced functional memory tracing was 13.8% slower. Compact single-worker
functional tracing was 4.8% slower; padded 64 KiB four-worker functional
tracing was 7.4% slower. These regressions were not hidden or rebaselined.

The selected implementation keeps the original dense functional instruction
loop for programs up to 8 KiB, choosing a specialized paged loop only for
larger programs. Its independent measurements are in the parent directory.
