# G14C B1 move-immediate field observation — 2026-10-04

Changing immediate bit 0 in `620978563412` → `620979563412` changed exact
active hardware outputs from `0x12345678` to `0x12345679`. Guards, excess-lane
padding and untouched tail were unchanged at counts 33 and 257. Unchanged
controls before/after passed; the changed instruction with a stale wrapper
SHA1 was rejected before submission.

This supports the existing applegpu `mov_imm` interpretation on this G14C B1
shader, extending local hardware coverage. It does not establish a novel opcode,
whole ISA compatibility, special-register behavior or instruction timing.

- [Detailed experiment](../../experiments/arithmetic/first-agx-instruction.md).
- [Exact patch evidence](patch.json): complete control, changed and stale blobs
  as hex, SHA256/SHA1, decoded fields, changed byte offsets and disassembly.
- [Version-matched layout source](source-layout.json): official Mesa 25.3.6
  archive identity, source hashes and relevant serialization excerpts. Upstream
  release/source agreement is distinct from distribution build provenance.
- `before-main.txt`, `after-main.txt`, `before-preamble.txt`,
  `after-preamble.txt`: local applegpu disassembly. Preamble is byte-identical;
  the main differs at the one justified immediate.
- [Raw execution record](../../results/2026-10-04-vulkan-binary-roundtrip/primary/record.json)
  and `mov-{before,patched-33,patched-257,after}-oracle.json` alongside it contain
  all API results and exact uint32 vectors. `stale-integrity/` holds rejection logs.

```sh
python3 -B scripts/run_vulkan_compute.py --probe-mov
python3 -B results/2026-10-04-vulkan-binary-roundtrip/verify_stored.py
```

To reconstruct the exact control blob without compiling or accessing the GPU:

```sh
python3 - <<'PY'
import json
from pathlib import Path
record = json.loads(Path('novel-results/2026-10-04-first-agx-instruction/patch.json').read_text())
Path('/scratch/agx-constant-control.bin').write_bytes(bytes.fromhex(record['input_hex']))
PY
python3 -B scripts/patch_agx_shader.py /scratch/agx-constant-control.bin \
  --output /scratch/agx-one-field-patch
```

The patch output directory must not already exist. Patcher runs no GPU code;
full hardware reproduction uses the bounded combined runner above. Binaries are
regenerable scratch artifacts; durable exact bytes/schema/logs live here and in
the linked results directory. No vendor edit, commit or push was performed.
