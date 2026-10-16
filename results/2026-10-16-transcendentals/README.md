# fp32 transcendentals: exact hardware models (2026-10-16)

Every fp32 transcendental in the functional simulator now reproduces the M2's result bits (`rust/src/transtab.rs`).
Recorded with `scripts/record_trans_ranges.py` (all 2^23 mantissas of a binade per run, straight-line kernel, each
chunk run twice). The raw recordings (~700 MB) live in `/scratch/agxsim-data/transcendental` and can be regenerated.
`hw_samples.json.gz` keeps 1,026 inputs per binade plus the sin / cos sequence samples.
`make verify-transcendentals` replays them through `agx_run` with the recording kernels (`kernels/`).

| Op | Model | Verified on |
|---|---|---|
| exp2 | `t = trunc(abs(x) * 2^23)`; `q = (x < 0 && t != 0) ? ~t : t` (ones' complement); `2^(q >> 23) * G[q & (2^23 - 1)]`; +inf at or above 2^128, +0 below the normal range | all 637,534,208 inputs of 76 binades, -2^127 .. 2^127 |
| sin_pt_2 | `H[trunc(abs(x) * 2^23)]` for abs(x) < 1 (sign dropped); 1.0 at abs(x) = 1; NaN beyond | all inputs of 60 binades (2^-26 .. 2^127 and 2^-126, both signs) |
| log2 | fixed-point e + F(m), truncated to fp32 (the existing model) | all inputs of 12 binades, 2^-126 .. 2^127 |
| rcp, rsqrt | per-mantissa tables (existing) | exhaustive (earlier) |

G is the [1,2) sample (`testsuite/transcendental/samples/exp2_1.bin.gz`). H is `testsuite/transcendental/sin_pt_2.bin.gz`:
H[0], then int8 first differences.

Effect: test suite 720/721 -> 721/721 exact (prog_071 fixed). Coverage 1,984 exact, 0 approximate. SPIR-V verify PASS
155 -> 164; the 8 remaining CLOSE workloads all differ only by atomic or race order.

Accuracy against PTX, and Mesa's cos sequence (2^-20.29 on hardware, PTX allows 2^-20.4) with a compliant fma
variant: playground `novel-results/2026-10-16-transcendentals`.

## fp16 forms (safe-01 part 2)

`scripts/record_f16_trans.py` recorded all 65,536 fp16 inputs of `<op> r14l, r6h` (fp16 -> fp16) and `<op> r14, r6l`
(fp16 source -> fp32), for rcp, rsqrt, rsqrt_special, log2, exp2, sin_pt_1 and sin_pt_2 (the coverage suite's safe
forms; 0 faults).

- **The fp16 result is the fp32 hardware result rounded to fp16** (round to nearest even), on 100 % of the inputs of
  all seven ops. The fp16 forms are therefore not independently correctly rounded: log2 and exp2 differ from the
  correctly rounded fp16 value on 2 and 4 inputs (double rounding), and the maximum error is 0.51 ulp.
- **fp16 sources keep their denormals.** The fp32 unit sees the exact fp32 value of the fp16 input, so the fp32 tables
  apply directly (h->f32 results equal the fp32 path on every input).
- **`sin_pt_1(-4)` and `sin_pt_1(-0)` are +0, while `sin_pt_1(-2)` is -0.** The simulator returned -0 for -4 before.

The simulator now reproduces every one of the 917,504 results (`f16/*.bin.gz`, checked by `make verify-transcendentals`).
