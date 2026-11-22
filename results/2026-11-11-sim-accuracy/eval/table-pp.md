| kernel | hw us | sim before | ratio | sim after | ratio |
|---|---|---|---|---|---|
| rms_norm_f32 | 18.2 | 13.2 | 0.73 | 13.2 | 0.73 |
| convert_unary_cont_vec4I6__halff | 4.4 | 4.1 | 0.93 | 4.1 | 0.93 |
| agx_sgeam | 4.8 | 4.5 | 0.94 | 4.5 | 0.94 |
| agx_sgemm_tile64 | 18.8 | 14.9 | 0.79 | 14.5 | 0.77 |
| agx_hgemm_tile64_cf16_tb | 72.5 | 64.6 | 0.89 | 64.6 | 0.89 |
| agx_hgemm_tile64_cf16_u_tb | 75.8 | 75.8 | 1.00 | 75.8 | 1.00 |
| rope_neoxILb1ELb0Eff | 33.5 | 33.5 | 1.00 | 33.5 | 1.00 |
| quantize_mmq_q8_1 | 13.0 | 12.2 | 0.94 | 12.2 | 0.94 |
| rope_neoxILb1ELb0Ef6__half | 10.4 | 9.5 | 0.91 | 9.5 | 0.91 |
| k_set_rowsIfl6__half | 7.9 | 7.2 | 0.91 | 7.2 | 0.91 |
| soft_max_f32ILb1ELi512ELi512Ef | 425.2 | 497.3 | 1.17 | 463.6 | 1.09 |
| cpy_scalar | 78.8 | 70.9 | 0.90 | 70.9 | 0.90 |
| unary_gated_op_kernel | 92.5 | 97.2 | 1.05 | 97.2 | 1.05 |
| quantize_mmq_q8_1 | 13.6 | 12.0 | 0.88 | 12.0 | 0.88 |
| mul_mat_q t8 | 380.7 | 391.8 | 1.03 | 391.1 | 1.03 |
| mul_mat_q_stream_k_fixup t8 | 38.5 | 36.1 | 0.94 | 36.1 | 0.94 |
| mul_mat_q t14 | 5526.2 | 5224.8 | 0.95 | 5640.0 | 1.02 |
| mul_mat_q t7 | 946.5 | 876.2 | 0.93 | 959.3 | 1.01 |
| k_bin_bcast | 33.9 | 17.6 | 0.52 | 17.6 | 0.52 |
| mul_mat_q t13 | 3702.6 | 2876.2 | 0.78 | 3246.4 | 0.88 |

before: median 8%, p90 27%, mean 11.6%, within 10% 12/20
after:  median 8%, p90 27%, mean 10.3%, within 10% 13/20
