# Shader capture: AGX binaries from any Mesa program, with a cycle estimate per instruction

`scripts/agx_capture.py` works like a Metal shader capture for Asahi Mesa. It runs a GLES or Vulkan
program through Mesa's own driver, collects every AGX binary that Mesa compiled, splits each one
into rodata, main shader and preamble, and disassembles them with applegpu. Each instruction is then
annotated with the `scripts/agx_cost.py` cycle model, which predicts issue counts and stall cycles
for 1 and 4 SIMD groups per core.

## Usage

```sh
python3 scripts/agx_capture.py [--ir] [--out DIR] [--json FILE] -- <command ...>   # run + capture
python3 scripts/agx_capture.py --cache DIR                                         # existing cache dir
python3 scripts/agx_capture.py --blob FILE ...                                     # Vulkan shader-object export / raw blob
make capture-demo                                                                  # tests/gles_compute.c mode private
```

| option | meaning |
| --- | --- |
| `--out DIR` | output dir (default `/scratch/apple-m2-playground/capture/<utc>`). It receives `cache/`, `listing.txt`, `capture.json`, one `shader<k>-<region>.bin` per code region (input for `agx_cost.py`), and `run/`, the command's working directory |
| `--ir` | second run with `AGX_MESA_DEBUG=shaders,shaderdb` and the cache disabled (stdout → `ir-dump.txt`, stderr → `ir-stderr.txt`). Dump sections are matched to captured shaders on several properties (see [IR matching](#ir-matching)). A shader is labelled *application shader* only when the match is unambiguous; its NIR plus AGX IR then goes into the JSON. Other labels are *ambiguous* (several sections fit; no IR attached) and *driver-internal?* (no section fits). With `--cache <old out>/cache --ir`, the `ir-dump.txt` next to the cache is used |
| `--ir-dump FILE` | attach an existing dump offline; shaderdb stats are read from `ir-stderr.txt` in the same directory if present |
| `--groups 1,4` | SIMD-group counts to simulate |
| `--path auto\|static\|emulate` | dynamic path for the cost model: `static` runs every instruction once; `emulate` uses applegpu's emulator (`agx_cost.trace`); `auto` picks emulate only if the region has a backward branch |
| `--loop-trips N` | assume every loop runs N times. This is a static path with each backward-branch body repeated, innermost loop first. Without it, every region with a backward branch gets a `!!! WARNING ... loop trip count unknown: costs assume K trip(s)` line (K = how often the back branch executed on the chosen path) and its cycle line is marked `LOOP TRIPS UNKNOWN`; the warning is repeated on stderr and stored as `cost.warning` in the JSON |
| `--timeout 60` | timeout for the command |
| `--mesa auto\|25.1\|25.3\|gl=…,vulkan=…` | struct layout. `auto` reads the installed versions (the numerically highest `libgallium-X.Y.Z.so` for GL, the `Mesa X.Y.Z` string in `libvulkan_asahi.so` for Vulkan) and accepts only the exact versions checked with `tests/agx_cache_layout.c` (`KNOWN_VERSIONS`: 25.1.9 → 25.1 layout, 25.3.6 → 25.3 layout). Any other installed version stops the tool with an error. `--mesa 25.1` (or `25.3`, or an exact known version) forces one layout for both APIs; `--mesa gl=25.1` forces GL and does not try Vulkan layouts at all |

During the capture run, the command gets `MESA_SHADER_CACHE_DIR=<out>/cache` and
`MESA_SHADER_CACHE_DISABLE=false`. Every `*_MESA_DEBUG`, `MESA_SHADER_CACHE*` and `MESA_DISK_CACHE*`
variable is removed, because `agx_disk_cache_init` leaves the cache off whenever compiler debug
flags are set.

## Output (excerpt, `tests/gles_compute.c private`, see `results/2026-10-11-shader-capture/`)

```text
shader 0: compute main  [gl (Mesa 25.1 layout)]  application shader
  source part0/mesa_cache.db@20  binary 420 B  main @128+214  preamble @384  rodata @0 12 B -> u10
  gprs(16-bit) main 10 preamble 2  push uniforms 16  stack 1036/0 words  local 0 B  wg [32, 1, 1]
  Mesa stats: instrs 32  alu 17  fscib 17  ic 16  threads 1024  loops 1  spills 1  fills 2
  Mesa IR: AGX_MESA_DEBUG=shaders section #0 (283 lines, in JSON); matched on stage, workgroup size, preamble present, instruction count, stack size, ...
  !!! WARNING (main): loop trip count unknown: costs assume 1 trip(s) for 1 loop(s) (pc 34..88; path emulate, the emulator sees zero uniforms and memory); pass --loop-trips N
  --- main @128  198 B  32 instructions (+8 padding)
      predicted cycles (LOOP TRIPS UNKNOWN, see warning): 1 group/core = 193  4 groups/core = 197   (path emulate, 32 dynamic instrs, ...)
         pc  hex                      class            n@1  stall@1   n@4  stall@4  disassembly
          0  b500c10520800004         stack_adjust       1        0     4        0  stack_adjust #1036 (words; applegpu: stack_store i8, 1, 2, 0, 1036, 0)
         64  b51a800400c01200         store              1        1     4        4  stack_store i32, 1, 0, x, 4, r3, r4, 0
        152  351a604400c01200         load               1        1     4        4  stack_load r3, i32, 1, 0, x, 4, r3, 1
        160  3800                     wait               1       62     4      248  wait 0
```

`n@G` is how many times the instruction issued, summed over G groups. `stall@G` is the summed gap,
in cycles, between the previous issue and this one in the same group. "Mesa stats" is Mesa's own
`agx2_stats` from the blob, which includes its ALU, FSCIB and IC cycle estimates, so the two models
can be compared. If `agx_cost` cannot cost a region (for example it raises `ValueError` for an
instruction it does not model), the region prints `cost model unavailable: <error text>` and the
listing is still shown without cycle columns. The captured main shader is byte-identical to the one carved by hand in
`results/2026-10-11-stack-via-mesa/record.json`.

## IR matching

`AGX_MESA_DEBUG=shaders` prints, per application shader (driver-internal shaders are suppressed by
`agx_should_dump`), the NIR and then the AGX IR of every function three times (before scheduling,
after scheduling, after RA and lowering; `agx_compile_function_nir`). `shaderdb` adds one
`agx2_stats` line per shader on stderr (`CS shader: 32 instrs, 17 alu, ...`), in the same order.
Those are the same numbers that are serialized in the blob. The stats lines are attached to dump
sections only if their count and stage abbreviations line up.

A captured shader and a dump section are compatible only if every property both sides report is
equal:

* stage, and workgroup size for compute;
* whether a preamble function exists;
* number of instructions in the final AGX IR of the entry point = `agx2_stats.instrs`;
* `stack_adjust #<hex>` in that final IR = `scratch_size` (0 if absent), and the same for the
  preamble and `preamble_scratch_size`;
* all shaderdb stats (instrs, alu, fscib, ic, code size, gprs, uniforms, scratch, threads, loops,
  spills, fills; 25.3 adds preamble instructions) = the blob's `agx2_stats`, when `ir-stderr.txt`
  exists.

When more than one section is compatible, the final-IR instruction counts of the main function and
the preamble are also compared with the disassembled lengths, but only to narrow the choice.
Identical sections (same text and stats, for example a shader compiled twice) count as one
candidate that can be assigned as often as it occurs. A shader is matched only if a single
candidate is left and no other unassigned shader competes for it beyond its multiplicity. Assigned
sections are removed and the process repeats. Anything else is reported as `ambiguous: N ... sections
fit (#a, #b); IR not attached` rather than guessed. The JSON stores the outcome in `ir_match`
(status, section indices, properties compared) and the parsed properties of every section in
`ir_sections`. This was checked against synthetic dumps: a duplicated section gives a match, two
compatible sections that differ only in the source hash are reported as ambiguous, and a section
whose instruction count or stats differ is ruled out.

## How the binaries are located (no pattern search)

Each layer below is parsed exactly as Mesa writes it. The tool accepts a parse only if it consumes
the whole payload, so a payload that is not an AGX shader is skipped rather than misread. Source
references are to the `mesa-25.1.9` and `mesa-25.3.6` tags in `/save/workspace/AGX_Probe/mesa`.

1. **Cache container.** There are two formats:
   - Database (the default for 25.1.9 GL): `mesa_shader_cache_db/part*/mesa_cache.db`, laid out
     as a 20-byte `mesa_db_file_header`, then entries of `{cache_key[20], crc u32, size u32}` and
     `size` bytes each (`src/util/mesa_cache_db.c`).
   - Multi-file (`MESA_DISK_CACHE_MULTI_FILE=1`): one file per item, `mesa_shader_cache/<2 hex>/<38 hex>`.
2. **Cache item** (`src/util/disk_cache_os.c` `create_cache_item_header_and_blob`). The driver-keys
   blob comes first: `cache_version u8`, the build-id string, the GPU name, `ptr_size u8`, and
   `driver_flags u64` (`src/util/disk_cache.c`). Then come the item type (`u32`; 1 = GLSL program
   metadata, which is skipped), `cache_entry_file_data {crc32, uncompressed_size}`, and the zstd
   frame, which is decompressed with `/usr/bin/zstd` and checked against `uncompressed_size`.
   `blob_write_uint32` aligns to 4 bytes (`src/util/blob.c`). That is why the type field starts 2
   bytes after the 74-byte driver keys.
3. **Asahi Gallium payload** (`src/gallium/drivers/asahi/agx_disk_cache.c` `write_shader`). The
   payload holds `struct agx_shader_info` (1176 bytes), then `binary_size` bytes of code, then
   `uvs` (88), `attrib_components_read` (8), `epilog_key` (4), alignment to 4, `push_range_count`,
   and that many 6-byte push ranges. For a geometry shader this is followed by `agx_gs_info` and
   the child shaders `pre_gs`, `gs_copy` and `gs_count`, each recursively the same layout.
4. **Honeykrisp (Vulkan) payload** (`src/asahi/vulkan/hk_shader.c` `hk_api_shader_serialize`). The
   payload starts with a stage byte. Each variant then has `hk_shader_info` (664 bytes in 25.1, 976
   in 25.3), `agx_shader_info`, alignment, `code_size`, `data_size`, code and data. There is 1
   variant per shader, except vertex and tess-eval shaders, which have 2 (hw, sw), and geometry
   shaders, which have 7 in 25.1 and 4 in 25.3. A `VK_EXT_shader_object` export
   (`vkGetShaderBinaryDataEXT`) has a 72-byte `vk_shader_bin_header` first, starting with
   `"MesaVkShaderBin"` (`src/vulkan/runtime/vk_shader.c`).
5. **Regions inside the binary** (`src/asahi/compiler/agx_compile.c`, `agx_compile_function_nir`
   and `agx_compile_shader_nir`). Rodata (`rodata.offset`, `2*size_16` bytes, mapped to uniform
   `base_uniform`) is placed before the main function's code. Every function starts 128-byte
   aligned (`AGX_CODE_ALIGN`). The main shader is at `main_offset` with `main_size` bytes. The
   preamble starts at `preamble_offset` and ends at the next region or at `binary_size`. Each region
   is disassembled up to its last `stop`. The nops after that are Mesa's prefetch padding, which
   is counted but not listed.

The struct sizes and field offsets come from `tests/agx_cache_layout.c`. It contains the
25.1.9/25.3.6 struct definitions copied from the tags (`-DMESA_25_3` selects 25.3) and prints them as
JSON. Between the two versions, `agx_shader_info` stays 1176 bytes, but 25.3 inserts
`uses_sampler_heap` and the texture/sampler state counts. That moves `nr_gprs` from offset 1094
to 1096, and `agx2_stats` gains `preamble_instrs`. `agx_gs_info` grows from 16 to 84 bytes, which
makes `hk_shader_info` grow from 664 to 976. The 976 was confirmed on a real 25.3.6 export: the
`code_size` header appears exactly where predicted. Note that `agx_shader_info.scratch_size` holds
the `stack_adjust` size in 32-bit words, despite the header comment that says bytes.

Mesa's `stack_adjust` (`agx_pack.c` `AGX_OPCODE_STACK_ADJUST`) shares opcode `0xb5` with
`stack_store`. The tool recognises it by Mesa's fixed bits, prints the decoded word count and sets
the class column to `stack_adjust`, whatever class `agx_cost.decode` gives the `stack_store`
encoding.

## Results on this machine (M2 Max G14C, GLES Mesa 25.1.9, Vulkan Mesa 25.3.6)

* **GLES** `tests/gles_compute.c index` and `private`: 3 shaders each. One is the application
  compute shader. In `index` mode its rodata holds exactly the GLSL constants `0x9e3779b9` and
  `0x12345678`, mapped to `u10`. The other two are a fragment shader part and a 1×1×1 compute
  shader (a 4×4 matrix-vector product). Neither appears in the `AGX_MESA_DEBUG=shaders` dump, so
  both are driver-internal. The fragment shader is a non-monolithic part: 4 `mov`s and no `stop`,
  because it gets an epilog linked at draw time.
* **Vulkan**: shaders created through `VK_EXT_shader_object` (`tests/vulkan_compute.c`) are **not**
  written to the disk cache. The cache dir only ever contained an empty multi-file `index`. The
  binary the program exports with `vkGetShaderBinaryDataEXT` parses fully with `--blob`
  (`listing-vulkan-index.txt`). The pipeline path (`vkCreate*Pipelines` → `vk_pipeline_cache` →
  disk cache) stores the same `hk_api_shader_serialize` bytes without the 72-byte header, and the
  parser handles them, but no pipeline-based program was run.

## Limitations

* Only shaders that go through the disk cache are captured. Not captured:
  - libagx precompiled kernels and the scratch helper program, which are built into the driver;
  - VS prologs and FS epilogs and the fast-linked programs built from them at draw time;
  - Vulkan shader objects (use `--blob` on their export).

  Shaders already present in a cache are not recompiled, which is why each run gets a fresh dir.
  Cache writes are asynchronous and are flushed in `disk_cache_destroy`, so a program that exits
  without tearing down its context can lose entries.
* The cost-model path does not see real uniforms. `emulate` runs with zero uniforms, registers
  and memory, so a loop bound by a uniform usually runs once. That is why every region with a
  backward branch carries a warning unless `--loop-trips` is given. In `private` mode the loop
  really runs 1024 times (`listing-gles-private-loop1024.txt`). A `static` path counts both sides of every if/else once. The preamble is costed
  like the main shader, although the hardware runs it once per dispatch or draw, not per group.
* Layouts are hardcoded for exactly Mesa 25.1.9 and 25.3.6 on aarch64, and other versions are
  refused in `auto` mode. For another version, diff the structs listed in `tests/agx_cache_layout.c`,
  regenerate the numbers, and add the version to `KNOWN_VERSIONS`. A forced `--mesa` layout is
  protected only by the exact-length check. That check rejects most wrong layouts, but not fields
  that only move inside `agx_shader_info`.
* Stage labels come from `agx_shader_info.stage`, the hardware stage. A vertex shader that runs as
  compute (Vulkan `variant:sw`, geometry `pre_gs`) shows as compute.
* The GLSL cache item (type 1) is skipped. The single-file Fossilize cache (`MESA_DISK_CACHE_SINGLE_FILE`)
  is not parsed.
