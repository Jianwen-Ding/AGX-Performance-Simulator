/* Print the byte layout of the AGX shader blobs Mesa serializes, as JSON.
 *
 *   cc -std=c11 tests/agx_cache_layout.c -o l && ./l                       # Mesa 25.1.9 (GLES/Gallium driver here)
 *   cc -std=c11 -DMESA_25_3 tests/agx_cache_layout.c -o l && ./l           # Mesa 25.3.6 (libvulkan_asahi.so here)
 *
 * Struct definitions are copied (comments stripped) from the mesa-25.1.9 / mesa-25.3.6 tags:
 *   src/asahi/compiler/agx_compile.h      struct agx_shader_info (+ agx_varyings_fs, agx_cf_binding, agx_rodata)
 *   src/util/shader_stats.xml             struct agx2_stats (isa AGX2; 25.3 adds "Preamble instructions")
 *   src/asahi/lib/agx_uvs.h               struct agx_unlinked_uvs_layout (static_assert == 88)
 *   src/asahi/lib/agx_linker.h            struct agx_fs_epilog_link_info (static_assert == 4)
 *   src/asahi/vulkan/hk_shader.h          struct hk_shader_info (Vulkan only)
 *   src/compiler/nir/nir_xfb_info.h       nir_xfb_info, nir_xfb_output_info
 * Serialization: src/gallium/drivers/asahi/agx_disk_cache.c write_shader() (GL) and
 * src/asahi/vulkan/hk_shader.c hk_api_shader_serialize() (Vulkan).
 * Note blob_write_uint32() aligns the blob to 4 bytes before writing (src/util/blob.c).
 * scripts/agx_capture.py hardcodes the printed numbers and accepts only these exact versions
 * (KNOWN_VERSIONS) unless --mesa forces a layout; add a version there only after checking it here.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define VARYING_SLOT_MAX (32 + 32)
#define AGX_MAX_CF_BINDINGS (VARYING_SLOT_MAX * 4)
typedef enum { GL_VARYING_SLOT_DUMMY = 0 } gl_varying_slot;
typedef enum { STAGE_NONE = -1, STAGE_X = 7 } stage_t;
typedef enum __attribute__((packed)) { NIR_ALU_TYPE_DUMMY = 0x80 } nir_alu_type; /* ENUM_PACKED */
enum gl_frag_depth_layout { FRAG_DEPTH_LAYOUT_NONE };

struct agx_cf_binding {
   uint8_t cf_base;
   gl_varying_slot slot : 8;
   unsigned offset : 2;
   unsigned count : 3;
   bool smooth : 1;
   bool perspective : 1;
   uint8_t pad;
};
struct agx_varyings_fs {
   unsigned nr_cf;
   unsigned nr_bindings;
   bool reads_z;
   struct agx_cf_binding bindings[AGX_MAX_CF_BINDINGS];
};
union agx_varyings { struct agx_varyings_fs fs; };
struct agx_rodata { uint32_t offset; uint16_t base_uniform; uint16_t size_16; };
struct agx2_stats {
   uint32_t instrs, alu, fscib, ic, code_size;
   uint16_t gprs, uniforms;
   uint32_t scratch;
   uint16_t threads;
   uint32_t loops, spills, fills;
#ifdef MESA_25_3
   uint32_t preamble_instrs;
#endif
};
struct agx_shader_info {
   stage_t stage;
   uint32_t binary_size;
   union agx_varyings varyings;
   uint16_t push_count;
   uint16_t local_size;
   uint16_t imageblock_stride;
   unsigned scratch_size, preamble_scratch_size;
   unsigned main_size;
   bool has_preamble;
   unsigned preamble_offset, main_offset;
   bool reads_tib;
   bool early_fragment_tests;
   bool nonzero_viewport;
   bool writes_layer_viewport;
   bool writes_sample_mask;
   enum gl_frag_depth_layout depth_layout;
   bool tag_write_disable;
   bool disable_tri_merging;
   bool uses_draw_id;
   bool uses_base_param;
   bool uses_txf;
#ifdef MESA_25_3
   bool uses_sampler_heap;
   uint8_t texture_state_count, sampler_state_count;
#endif
   uint16_t nr_gprs, nr_preamble_gprs;
   uint64_t outputs;
   uint16_t workgroup_size[3];
   struct agx_rodata rodata;
   struct agx2_stats stats;
};
struct agx_push_range { uint16_t uniform; uint16_t offset; uint8_t table; uint8_t length; };
struct agx_gs_info {
   int mode;
   unsigned count_words;
   unsigned max_indices;
   bool xfb;
   bool prefix_sum;
#ifdef MESA_25_3
   bool multistream;
   int shape;
   uint8_t topology[64];
#endif
};

/* Vulkan (honeykrisp) per-variant header */
struct agx_unlinked_uvs_layout { uint64_t written; uint8_t rest[80]; };
struct agx_interp_info { uint64_t flat, linear; };
struct hk_tess_info { uint32_t packed; };
typedef struct { uint16_t stride; uint16_t varying_count; } nir_xfb_buffer_info;
typedef struct {
   uint8_t buffer;
   uint16_t offset;
   uint8_t location;
#ifdef MESA_25_3
   bool data_is_16bit;
   bool high_16bits;
   bool mediump;
   nir_alu_type mediump_upconvert_type;
#else
   bool high_16bits;
#endif
   uint8_t component_mask;
   uint8_t component_offset;
} nir_xfb_output_info;
typedef struct {
   uint8_t buffers_written;
   uint8_t streams_written;
   nir_xfb_buffer_info buffers[4];
   uint8_t buffer_to_stream[4];
   uint16_t output_count;
   nir_xfb_output_info outputs[];
} nir_xfb_info;
struct hk_shader_info {
   union {
      struct { uint32_t attribs_read; uint32_t attrib_components_read[2]; uint8_t _pad[8]; } vs;
      struct { struct agx_interp_info interp; uint32_t epilog_key; bool b[5]; uint8_t _pad[7]; } fs;
      struct { uint64_t a; uint32_t b; uint8_t c, d; struct hk_tess_info info; } tess;
      struct agx_gs_info gs;
      uint8_t _pad[32];
   };
   struct agx_unlinked_uvs_layout uvs;
   uint8_t xfb_stride[4];
   stage_t stage : 8;
   uint8_t clip_distance_array_size;
   uint8_t cull_distance_array_size;
   uint8_t _pad0[1];
   nir_xfb_info xfb_info;
   nir_xfb_output_info xfb_outputs[64];
};

#define F(s, f, t) printf("   \"%s\": [%zu, \"%s\"],\n", #f, offsetof(struct s, f), t)
int main(void)
{
   _Static_assert(sizeof(struct agx_unlinked_uvs_layout) == 88, "uvs");
   printf("{\n  \"mesa\": \"%s\",\n  \"info_size\": %zu,\n  \"uvs_size\": 88,\n"
          "  \"attrib_components_read_size\": %zu,\n  \"epilog_key_size\": 4,\n"
          "  \"push_range_size\": %zu,\n  \"gs_info_size\": %zu,\n  \"hk_shader_info_size\": %zu,\n"
          "  \"hk_stage_offset\": %d,\n  \"fields\": {\n",
#ifdef MESA_25_3
          "25.3.6",
#else
          "25.1.9",
#endif
          sizeof(struct agx_shader_info), sizeof(unsigned[(16 * 4 + 31) / 32]),
          sizeof(struct agx_push_range), sizeof(struct agx_gs_info), sizeof(struct hk_shader_info),
          (int)offsetof(struct hk_shader_info, xfb_stride) + 4);
   F(agx_shader_info, stage, "i"); F(agx_shader_info, binary_size, "I");
   F(agx_shader_info, push_count, "H"); F(agx_shader_info, local_size, "H");
   F(agx_shader_info, imageblock_stride, "H"); F(agx_shader_info, scratch_size, "I");
   F(agx_shader_info, preamble_scratch_size, "I"); F(agx_shader_info, main_size, "I");
   F(agx_shader_info, has_preamble, "B"); F(agx_shader_info, preamble_offset, "I");
   F(agx_shader_info, main_offset, "I"); F(agx_shader_info, reads_tib, "B");
   F(agx_shader_info, uses_txf, "B");
   F(agx_shader_info, nr_gprs, "H"); F(agx_shader_info, nr_preamble_gprs, "H");
   F(agx_shader_info, outputs, "Q"); F(agx_shader_info, workgroup_size, "3H");
   F(agx_shader_info, rodata, "IHH");
   printf("   \"stats\": [%zu, \"%s\"]\n  }\n}\n", offsetof(struct agx_shader_info, stats),
#ifdef MESA_25_3
          "5I2HIH2x4I"
#else
          "5I2HIH2x3I"
#endif
   );
   return 0;
}
