| kernel | hw us | sim before | ratio | sim after | ratio |
|---|---|---|---|---|---|
| quantize_q8_1PKfPvlllllj5uint3 | 4.0 | 4.4 | 1.10 | 4.4 | 1.10 |
| mul_mat_vec_q t8 | 17.2 | 13.9 | 0.81 | 13.9 | 0.81 |
| mul_mat_vec_q t14 | 30.2 | 29.3 | 0.97 | 29.2 | 0.97 |
| mul_mat_vec_q t13 | 29.6 | 25.8 | 0.87 | 25.8 | 0.87 |
| mul_mat_vec_q t7 | 19.0 | 17.8 | 0.94 | 17.8 | 0.94 |
| rms_norm_f32 | 5.5 | 4.9 | 0.89 | 4.9 | 0.89 |
| rope_neoxILb1ELb0Eff | 6.9 | 6.0 | 0.87 | 6.0 | 0.87 |
| rope_neoxILb1ELb0Ef6__half | 6.7 | 6.1 | 0.91 | 6.1 | 0.91 |
| k_set_rowsIfl6__half | 3.7 | 3.9 | 1.05 | 3.9 | 1.05 |
| flash_attn_stream_k_fixup_uniform | 5.0 | 4.9 | 0.98 | 4.9 | 0.98 |

before: median 9%, p90 19%, mean 9.2%, within 10% 5/10
after:  median 9%, p90 19%, mean 9.2%, within 10% 5/10
