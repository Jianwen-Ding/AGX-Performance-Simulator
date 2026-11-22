| kernel | hw us | sim before | ratio | sim after | ratio |
|---|---|---|---|---|---|
| flash_attn_ext_f16 | 79.7 | 83.6 | 1.05 | 83.6 | 1.05 |
| flash_attn_stream_k_fixup_uniform | 5.0 | 4.9 | 0.98 | 4.9 | 0.98 |
| flash_attn_ext_f16 | 1636.0 | 1402.5 | 0.86 | 1402.4 | 0.86 |
| flash_attn_stream_k_fixup_general | 37.1 | 32.1 | 0.87 | 32.1 | 0.87 |

before: median 9%, p90 14%, mean 8.7%, within 10% 2/4
after:  median 9%, p90 14%, mean 8.7%, within 10% 2/4
