# 2026-11-10 round 2: workgroup placement, early-exit occupancy, streaming code refetch

Hardware findings (apple-documentation-playground novel-results/2026-11-10-workgroup-placement,
2026-11-10-early-exit-occupancy, 2026-11-10-code-refetch-stream), probes `scripts/probe_resident_order.py`,
`probe_partial_wg.py`, `probe_fetch_groups.py` (BODY_KB up to 384, WGSTAGGER), `probe_hot_lines.py`; raw hw logs in
`hw/` (job scripts alongside). Evals: `evalsets.sh OUTDIR [env]` (no GPU).

Model changes (dispatch.rs, cost.rs):
1. `placement`: per-core workgroup lists as the hardware deals them (chunks of 8 to clusters 0,1,2,3,3,0,1,2,...;
   round robin within a cluster) for one-round and one-resident grids, simulated by `simulate_placed`
   (AGX_PLACEMENT=0 off). At 38 one-resident workgroups 34-37 run after 24-27 (was: after 0-3).
2. AGX_CHAIN_ALL: a core whose wave-0 chain runs all its waves takes that chain's cost (two unequal workgroups
   came out (L + S) / 2 + S).
3. AGX_EARLY_EXIT: groups that end early free registers, not group slots: workgroups up to 96 / groups per core.
4. AGX_FETCH_STREAM (default on): streamed refetch 31 ns per line (AGX_STREAM_LINE_NS), isolated misses 64,
   first fetches not streamed (cold 70 / jumped 300), the stream stops during waits.

Held-out Rodinia + Halide (159 launches, no Cycles): before median 5, p90 18, mean 7.5, w10 119;
placement + early exit 6 / 18 / 7.6 / 117; final 6 / 18 / 7.7 / 118 (bucketcount +5 -> -21%, myocyte -8 -> 0,
particlefilter likelihood +12 -> +6). Per-launch: `eval-placement/heldout`, `eval-final/heldout`.

Blender Cycles (hwref): shade_surface hw 307.9 us, before 235.3 (-24%), final 280.0 (-9%); intersect_closest hw
554.5, before 355.8 (-36%), final 356.3 (-36%) (`eval-final/cycles.txt`).

### llama.cpp pp512 -fa 0 (hw b2b 5000)

| kernel | hw us | before (2413039) | placement + early exit | final |
|---|---|---|---|---|
| _ZL12rms_norm_f32ILi256ELi1ELb1ELb1ELb0ELb0E | 18.2 | 13.2 (-27%) | 13.2 (-27%) | 13.2 (-27%) |
| _ZL23convert_unary_cont_vec4I6__halffEvPKvPT | 4.4 | 4.1 (-7%) | 4.1 (-8%) | 4.1 (-8%) |
| agx_sgeam | 4.8 | 4.5 (-6%) | 4.5 (-6%) | 4.5 (-6%) |
| agx_sgemm_tile64 | 18.8 | 15.4 (-18%) | 14.9 (-21%) | 14.9 (-21%) |
| agx_hgemm_tile64_cf16_tb | 72.5 | 64.6 (-11%) | 64.6 (-11%) | 64.6 (-11%) |
| agx_hgemm_tile64_cf16_u_tb | 75.8 | 75.8 (+0%) | 75.8 (+0%) | 75.8 (+0%) |
| _ZL9rope_neoxILb1ELb0EffEvPKT1_PT2_iiiiiiiii | 33.5 | 72.6 (+117%) | 33.5 (+0%) | 33.5 (+0%) |
| _ZL17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout | 13.0 | 12.2 (-6%) | 12.2 (-6%) | 12.2 (-6%) |
| _ZL9rope_neoxILb1ELb0Ef6__halfEvPKT1_PT2_iii | 10.4 | 15.9 (+53%) | 9.5 (-8%) | 9.5 (-8%) |
| _ZL10k_set_rowsIfl6__halfEvPKT_PKT0_PT1_llll | 7.9 | 7.2 (-9%) | 7.2 (-9%) | 7.2 (-9%) |
| _ZL12soft_max_f32ILb1ELi512ELi512EfEvPKfPKT2 | 425.2 | 497.3 (+17%) | 497.3 (+17%) | 497.3 (+17%) |
| _ZL10cpy_scalarIXadL_ZL12cpy_1_scalarIffEvPK | 78.8 | 70.9 (-10%) | 70.9 (-10%) | 70.9 (-10%) |
| _ZL21unary_gated_op_kernelIXadL_ZL7op_silufE | 92.5 | 97.2 (+5%) | 97.2 (+5%) | 97.2 (+5%) |
| _ZL17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout | 13.6 | 12.0 (-12%) | 12.0 (-12%) | 12.0 (-12%) |
| _ZL9mul_mat_qIL9ggml_type8ELi128ELb0EL9ggml_ | 380.7 | 442.4 (+16%) | 442.4 (+16%) | 391.8 (+3%) |
| _ZL24mul_mat_q_stream_k_fixupIL9ggml_type8EL | 38.5 | 36.3 (-6%) | 36.1 (-6%) | 36.1 (-6%) |
| _ZL9mul_mat_qIL9ggml_type14ELi128ELb0EL9ggml | 5526.2 | 6129.7 (+11%) | 6173.8 (+12%) | 4155.6 (-25%) |
| _ZL9mul_mat_qIL9ggml_type7ELi128ELb0EL9ggml_ | 946.5 | 1356.9 (+43%) | 1188.9 (+26%) | 876.2 (-7%) |
| _ZL11k_bin_bcastIXadL_ZL6op_addffEEfffJPKfEE | 33.9 | 17.6 (-48%) | 17.6 (-48%) | 17.6 (-48%) |
| _ZL9mul_mat_qIL9ggml_type13ELi128ELb0EL9ggml | 3702.6 | 4512.2 (+22%) | 4507.0 (+22%) | 2876.2 (-22%) |

before: median 12%, mean 22.2%, within 10% 8/20; placement: median 10%, mean 13.5%, within 10% 10/20; final: median 8%, mean 12.6%, within 10% 12/20

### llama.cpp flash attention

| kernel | hw us | before (2413039) | placement + early exit | final |
|---|---|---|---|---|
| _ZL18flash_attn_ext_f16ILi64ELi64ELi1ELi8ELb | 79.7 | 77.6 (-3%) | 77.6 (-3%) | 83.6 (+5%) |
| _ZL33flash_attn_stream_k_fixup_uniformILi64E | 5.0 | 4.9 (-2%) | 4.9 (-3%) | 4.9 (-3%) |
| _ZL18flash_attn_ext_f16ILi64ELi64ELi8ELi8ELb | 1636.0 | 923.6 (-44%) | 1801.7 (+10%) | 1402.5 (-14%) |
| _ZL33flash_attn_stream_k_fixup_generalILi64E | 37.1 | 32.1 (-13%) | 32.1 (-14%) | 32.1 (-14%) |

before: median 8%, mean 15.5%, within 10% 2/4; placement: median 6%, mean 7.5%, within 10% 3/4; final: median 10%, mean 9.0%, within 10% 2/4

### tinygrad

| kernel | hw us | before (2413039) | placement + early exit | final |
|---|---|---|---|---|
| E | 2.9 | 3.1 (+7%) | 3.1 (+9%) | 3.1 (+9%) |
| r_64_32_8_16_4_4_512_4 | 6866.6 | 5722.8 (-17%) | 5722.8 (-17%) | 5722.8 (-17%) |
| r_64_16_32 | 602.4 | 1266.1 (+110%) | 1266.1 (+110%) | 731.5 (+21%) |
| E_8_8_8_16_4 | 188.8 | 183.0 (-3%) | 183.0 (-3%) | 205.4 (+9%) |
| r_32_16_4_32_2_2_2_4_4_128 | 2347.4 | 2050.8 (-13%) | 2050.8 (-13%) | 2050.8 (-13%) |

before: median 13%, mean 30.0%, within 10% 2/5; placement: median 13%, mean 30.4%, within 10% 2/5; final: median 13%, mean 13.8%, within 10% 2/5

### llama.cpp tg

| kernel | hw us | before (2413039) | placement + early exit | final |
|---|---|---|---|---|
| _ZL13quantize_q8_1PKfPvlllllj5uint3 | 4.0 | 4.4 (+10%) | 4.4 (+9%) | 4.4 (+9%) |
| _ZL13mul_mat_vec_qIL9ggml_type8ELi1ELb1ELb1E | 17.2 | 13.9 (-19%) | 13.9 (-19%) | 13.9 (-19%) |
| _ZL13mul_mat_vec_qIL9ggml_type14ELi1ELb1ELb0 | 30.2 | 29.3 (-3%) | 29.3 (-3%) | 29.3 (-3%) |
| _ZL13mul_mat_vec_qIL9ggml_type13ELi1ELb1ELb0 | 29.6 | 26.1 (-12%) | 25.8 (-13%) | 25.8 (-13%) |
| _ZL13mul_mat_vec_qIL9ggml_type7ELi1ELb1ELb1E | 19.0 | 17.8 (-6%) | 17.8 (-6%) | 17.8 (-6%) |
| _ZL12rms_norm_f32ILi256ELi1ELb1ELb1ELb0ELb0E | 5.5 | 4.9 (-11%) | 4.9 (-11%) | 4.9 (-11%) |
| _ZL9rope_neoxILb1ELb0EffEvPKT1_PT2_iiiiiiiii | 6.9 | 6.0 (-13%) | 6.0 (-13%) | 6.0 (-13%) |
| _ZL9rope_neoxILb1ELb0Ef6__halfEvPKT1_PT2_iii | 6.7 | 6.1 (-9%) | 6.1 (-9%) | 6.1 (-9%) |
| _ZL10k_set_rowsIfl6__halfEvPKT_PKT0_PT1_llll | 3.7 | 3.9 (+5%) | 3.9 (+7%) | 3.9 (+7%) |
| _ZL33flash_attn_stream_k_fixup_uniformILi64E | 5.0 | 4.9 (-2%) | 4.9 (-3%) | 4.9 (-3%) |

before: median 10%, mean 9.0%, within 10% 6/10; placement: median 9%, mean 9.3%, within 10% 6/10; final: median 9%, mean 9.3%, within 10% 6/10
