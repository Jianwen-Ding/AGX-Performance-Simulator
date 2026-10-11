Cycle values are in reference units (dependent 8-byte `iadd x, x, 1` step = 2); real cycles = value x 1.049
(clocks inferred this way are the DVFS levels / 1.049). `mov_imm` has no source register: its latency column is its issue interval.

| Instruction | Class | Latency (cycles) | Issue interval, 1 SIMD group (cycles) | Lanes / cycle / core | SIMD instr / cycle / core |
| --- | --- | --- | --- | --- | --- |
| `iadd rx, rx, c0` | Integer | 2.0 | 1.06 | 64 | 1.99 |
| `isub rx, rx, c0` | Integer | 2.0 | 1.06 | 63 | 1.98 |
| `iadd rx, rx, c0, lsl 2` | Integer | 2.0 | 1.06 | 63 | 1.98 |
| `imadd rx, rx, c1, c0` | Integer | 2.9 | 2.00 | 35 | 1.08 |
| `imadd rx, rx, c1, 0` | Integer | 2.9 | 1.99 | 34 | 1.05 |
| `iadd rx_rx1, rx_rx1, c0_c1` | 64-bit integer | 4.8 | 3.89 | 17 | 0.54 |
| `imadd rx_rx1, rx, c1, c0_c1` | 64-bit integer | 3.9 | 3.90 | 15 | 0.47 |
| `and rx, rx, c0` | Integer | 2.0 | 1.04 | 65 | 2.02 |
| `xor rx, rx, c0` | Integer | 2.0 | 1.05 | 64 | 1.99 |
| `bfi rx, rx, c0, c2` | Bit/shift | 2.9 | 1.99 | 33 | 1.02 |
| `bfeil rx, rx, c0, c2` | Bit/shift | 2.9 | 2.00 | 33 | 1.03 |
| `extr rx, rx, c0, c2` | Bit/shift | 2.9 | 1.99 | 33 | 1.03 |
| `shlhi rx, rx, c0, c2` | Bit/shift | 2.9 | 2.00 | 33 | 1.03 |
| `shrhi rx, rx, c0, c2` | Bit/shift | 2.9 | 1.99 | 33 | 1.03 |
| `asr rx, rx, c2` | Bit/shift | 2.9 | 1.99 | 33 | 1.03 |
| `asrh rx, rx, c2` | Bit/shift | 2.9 | 2.00 | 33 | 1.03 |
| `bitrev rx, rx` | Bit/shift | 2.9 | 1.99 | 30 | 0.93 |
| `popcount rx, rx` | Bit/shift | 2.9 | 1.99 | 30 | 0.93 |
| `ffs rx, rx` | Bit/shift | 2.9 | 2.00 | 30 | 0.93 |
| `intl rx, rx, c0` | Bit/shift | 2.9 | 2.00 | 32 | 0.99 |
| `icmpsel ult, rx, rx, c0, rx, c1` | Compare-select | 2.0 | 2.13 | 38 | 1.19 |
| `fcmpsel lt, rx, rx, c0, rx, c1` | Compare-select | 2.0 | 2.13 | 38 | 1.19 |
| `mov_imm rx, 305419896, 0b0` | Move | n/a | 1.04 | 123 | 3.84 |
| `iadd rx, rx, 1` | Integer | 2.0 | 1.06 | 121 | 3.77 |
| `iadd rx, rx, c0.cache` | Integer | 2.0 | 2.04 | 66 | 2.05 |
| `imadd rx, rx, c1.cache, c0.cache` | Integer | 2.9 | 1.99 | 34 | 1.06 |
| `fadd32 rx, rx, c0` | FP32 | 2.0 | 1.04 | 64 | 2.01 |
| `fmul32 rx, rx, c1` | FP32 | 2.0 | 1.04 | 63 | 1.98 |
| `fmadd32 rx, rx, c1, c0` | FP32 | 2.0 | 2.00 | 44 | 1.38 |
| `fadd32 rx, rx, 0.125` | FP32 | 1.9 | 0.95 | 133 | 4.16 |
| `fadd32 rx, rx, c0.cache` | FP32 | 2.0 | 2.03 | 61 | 1.92 |
| `fadd32 rx, rx.cache, c0.cache` | FP32 | 2.0 | 2.03 | 61 | 1.92 |
| `fmul32 rx, rx, c1.cache` | FP32 | 2.0 | 2.04 | 58 | 1.80 |
| `fmadd32 rx, rx, 0.5, 0.125` | FP32 | 2.0 | 1.04 | 123 | 3.83 |
| `fmadd32 rx, rx, c1.cache, c0.cache` | FP32 | 2.0 | 1.82 | 63 | 1.96 |
| `fadd16 rxl, rxl, c0l` | FP16 | 2.0 | 1.04 | 123 | 3.85 |
| `fmul16 rxl, rxl, c1l` | FP16 | 2.0 | 1.04 | 124 | 3.87 |
| `fmadd16 rxl, rxl, c1l, c0l` | FP16 | 2.0 | 1.04 | 59 | 1.86 |
| `convert f_to_u32, rx, rx, rtz` | Convert | 2.9 | 1.99 | 30 | 0.93 |
| `convert u32_to_f, rx, rx, rte` | Convert | 2.9 | 1.99 | 30 | 0.93 |
| `floor rx, rx` | Transcendental / rounding | 2.9 | 1.90 | 36 | 1.12 |
| `rint rx, rx` | Transcendental / rounding | 2.9 | 1.99 | 30 | 0.93 |
| `rcp rx, rx` | Transcendental / rounding | 4.8 | 2.86 | 21 | 0.65 |
| `rsqrt rx, rx` | Transcendental / rounding | 5.7 | 3.81 | 33 | 1.03 |
| `rsqrt_special rx, rx` | Transcendental / rounding | 5.7 | 3.81 | 33 | 1.03 |
| `log2 rx, rx` | Transcendental / rounding | 4.8 | 1.91 | 37 | 1.14 |
| `exp2 rx, rx` | Transcendental / rounding | 4.8 | 1.90 | 37 | 1.14 |
| `sin_pt_1 rx, rx` | Transcendental / rounding | 2.9 | 1.90 | 36 | 1.13 |
| `sin_pt_2 rx, rx` | Transcendental / rounding | 4.8 | 1.90 | 37 | 1.14 |
| `simd_shuffle rx, rx, c2l` | SIMD-group / quad | 4.8 | 1.99 | 72 | 2.26 |
| `simd_shuffle_down rx, rx, c2l` | SIMD-group / quad | 4.8 | 1.99 | 72 | 2.24 |
| `simd_shuffle_xor rx, rx, c2l` | SIMD-group / quad | 4.8 | 1.99 | 72 | 2.24 |
| `quad_shuffle rx, rx, c2l` | SIMD-group / quad | 3.9 | 1.04 | 120 | 3.77 |
| `simd_iadd rx, rx` | SIMD-group / quad | 22.9 | 22.93 | 6 | 0.17 |
| `simd_prefix_iadd rx, rx` | SIMD-group / quad | 17.3 | 17.24 | 7 | 0.23 |
| `simd_fadd rx, rx` | SIMD-group / quad | 22.9 | 22.93 | 6 | 0.17 |
| `simd_min.u rx, rx` | SIMD-group / quad | 22.0 | 21.98 | 6 | 0.18 |
| `quad_iadd rx, rx` | SIMD-group / quad | 11.5 | 8.65 | 15 | 0.46 |
| `icmp_ballot rx, ult, rx, c0` | SIMD-group / quad | 4.8 | 2.00 | 46 | 1.44 |
| `fcmp_ballot rx, lt, rx, c0` | SIMD-group / quad | 4.9 | 1.99 | 46 | 1.45 |
