/* vk_run: run a SPIR-V compute workload described by a spec file; print JSON lines.
 *
 *   vk_run SPEC
 *
 * SPEC is text, one directive per line ('#' comments):
 *   shader PATH.spv               compute shader, entry point "main", descriptor set 0
 *   buffer BINDING SIZE INIT OUT  storage buffer: INIT = file of initial bytes (zero-padded) or '-',
 *                                 OUT = file the buffer is read back to after the functional run, or '-'
 *   dispatch GX GY GZ [PUSHHEX]   one vkCmdDispatch (push constants as hex bytes); dispatches run in order with a
 *                                 compute->compute memory barrier between them
 *   repeat N                      timed runs after the functional run (buffers re-initialised before each)
 *   export PATH                   VK_EXT_shader_object binary of the shader (the driver's compiled code), if supported
 *   props PATH                    VK_KHR_pipeline_executable_properties statistics and internal representations
 *                                 (driver IR / disassembly text), if supported
 *
 * Output: {"device": ...}, {"functional": ...}, one {"rep": k, "total_ns": .., "dispatch_ns": [..]} per timed run
 * (GPU timestamps), then {"summary": ...}. Host-visible coherent memory throughout; select a device with
 * VK_RUN_DEVICE=<index>. Every Vulkan error exits non-zero with {"error": ...}.
 */
#define _POSIX_C_SOURCE 200809L
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXB 16
#define MAXD 256

static void die(const char *what, int code) {
    printf("{\"error\":\"%s\",\"code\":%d}\n", what, code);
    exit(1);
}
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) die(#x, r_); } while (0)

static PFN_vkGetInstanceProcAddr gipa;
static VkInstance inst;
static VkDevice dev;
#define IF(n) PFN_##n n = (PFN_##n)gipa(inst, #n); if (!n) die("missing " #n, 0)
#define DF(n) PFN_##n n = (PFN_##n)vkGetDeviceProcAddr(dev, #n); if (!n) die("missing " #n, 0)

static void *slurp(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb");
    if (!f) die("open input", 0);
    fseek(f, 0, SEEK_END); long len = ftell(f); rewind(f);
    void *b = malloc(len > 0 ? len : 1);
    if (len > 0 && fread(b, 1, len, f) != (size_t)len) die("read input", 0);
    fclose(f); *n = len; return b;
}
static void spit(const char *p, const void *b, size_t n) {
    FILE *f = fopen(p, "wb");
    if (!f || fwrite(b, 1, n, f) != n || fclose(f)) die("write output", 0);
}
static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static void jstr(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') { fputc('\\', f); fputc(*s, f); }
        else if (*s == '\n') fputs("\\n", f);
        else if ((unsigned char)*s < 32) fprintf(f, "\\u%04x", *s);
        else fputc(*s, f);
    }
    fputc('"', f);
}

struct buf { int binding; VkDeviceSize size; char init[512], out[512]; VkBuffer b; VkDeviceMemory m; void *map; void *initdata; size_t initlen; };
struct disp { uint32_t g[3]; uint8_t push[128]; uint32_t npush; };

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 2) { fprintf(stderr, "usage: vk_run SPEC\n"); return 2; }
    char shader[512] = "", exportp[512] = "", propsp[512] = "";
    struct buf B[MAXB]; int nb = 0;
    struct disp D[MAXD]; int nd = 0;
    int repeat = 0;
    uint32_t maxpush = 0;
    FILE *sf = fopen(argv[1], "r");
    if (!sf) die("open spec", 0);
    char line[1024];
    while (fgets(line, sizeof line, sf)) {
        char *h = strchr(line, '#'); if (h) *h = 0;
        char kw[32]; if (sscanf(line, "%31s", kw) != 1) continue;
        if (!strcmp(kw, "shader")) { if (sscanf(line, "%*s %511s", shader) != 1) die("spec shader", 0); }
        else if (!strcmp(kw, "export")) sscanf(line, "%*s %511s", exportp);
        else if (!strcmp(kw, "props")) sscanf(line, "%*s %511s", propsp);
        else if (!strcmp(kw, "repeat")) sscanf(line, "%*s %d", &repeat);
        else if (!strcmp(kw, "buffer")) {
            if (nb == MAXB) die("too many buffers", 0);
            unsigned long long sz; struct buf *b = &B[nb++]; memset(b, 0, sizeof *b);
            if (sscanf(line, "%*s %d %llu %511s %511s", &b->binding, &sz, b->init, b->out) != 4) die("spec buffer", 0);
            b->size = sz;
        } else if (!strcmp(kw, "dispatch")) {
            if (nd == MAXD) die("too many dispatches", 0);
            struct disp *d = &D[nd++]; memset(d, 0, sizeof *d);
            char hex[300] = "";
            if (sscanf(line, "%*s %u %u %u %299s", &d->g[0], &d->g[1], &d->g[2], hex) < 3) die("spec dispatch", 0);
            if (hex[0] && strcmp(hex, "-")) {
                size_t n = strlen(hex) / 2; if (n > 128 || n % 4) die("push constants: up to 128 bytes, multiple of 4", 0);
                for (size_t i = 0; i < n; i++) { unsigned v; sscanf(hex + 2 * i, "%2x", &v); d->push[i] = v; }
                d->npush = n; if (n > maxpush) maxpush = n;
            }
        } else die("unknown spec directive", 0);
    }
    fclose(sf);
    if (!shader[0] || !nd) die("spec needs shader and dispatch", 0);

    void *lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) die("libvulkan.so.1", 0);
    gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
    IF(vkCreateInstance);
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "vk_run", .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    CK(vkCreateInstance(&ici, NULL, &inst));
    IF(vkEnumeratePhysicalDevices); IF(vkGetPhysicalDeviceQueueFamilyProperties);
    IF(vkGetPhysicalDeviceMemoryProperties); IF(vkEnumerateDeviceExtensionProperties); IF(vkCreateDevice);
    IF(vkGetDeviceProcAddr); IF(vkGetPhysicalDeviceFeatures2); IF(vkGetPhysicalDeviceProperties2);
    uint32_t np = 0; CK(vkEnumeratePhysicalDevices(inst, &np, NULL));
    if (!np || np > 8) die("physical devices", np);
    VkPhysicalDevice pds[8]; CK(vkEnumeratePhysicalDevices(inst, &np, pds));
    const char *sel = getenv("VK_RUN_DEVICE"); uint32_t pi = sel ? (uint32_t)atoi(sel) : 0;
    if (pi >= np) die("VK_RUN_DEVICE", pi);
    VkPhysicalDevice pd = pds[pi];
    VkPhysicalDeviceSubgroupProperties sgp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 pp2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sgp};
    vkGetPhysicalDeviceProperties2(pd, &pp2);
    VkPhysicalDeviceProperties props = pp2.properties;
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties qf[16]; if (nq > 16) nq = 16; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
    uint32_t qi = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qi = i; break; }
    if (qi == UINT32_MAX) die("no compute queue", 0);
    int timestamps = qf[qi].timestampValidBits > 0 && props.limits.timestampPeriod > 0;
    uint32_t ne = 0; CK(vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, NULL));
    VkExtensionProperties *ex = calloc(ne, sizeof *ex); CK(vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, ex));
    int has_so = 0, has_pe = 0;
    for (uint32_t i = 0; i < ne; i++) {
        has_so |= !strcmp(ex[i].extensionName, VK_EXT_SHADER_OBJECT_EXTENSION_NAME);
        has_pe |= !strcmp(ex[i].extensionName, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
    }
    VkPhysicalDeviceShaderObjectFeaturesEXT sof = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pef = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR, .pNext = &sof};
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &pef};
    VkPhysicalDeviceVulkan11Features f11 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &f12};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f11};
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    int use_so = has_so && sof.shaderObject && exportp[0];
    int use_pe = has_pe && pef.pipelineExecutableInfo && propsp[0];
    /* enable what the shaders may use (only features the device reports) */
    VkPhysicalDeviceVulkan12Features e12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .shaderFloat16 = f12.shaderFloat16, .shaderInt8 = f12.shaderInt8, .storageBuffer8BitAccess = f12.storageBuffer8BitAccess,
        .uniformAndStorageBuffer8BitAccess = f12.uniformAndStorageBuffer8BitAccess, .shaderBufferInt64Atomics = f12.shaderBufferInt64Atomics,
        .shaderSharedInt64Atomics = f12.shaderSharedInt64Atomics, .scalarBlockLayout = f12.scalarBlockLayout, .vulkanMemoryModel = f12.vulkanMemoryModel};
    VkPhysicalDeviceVulkan11Features e11 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &e12,
        .storageBuffer16BitAccess = f11.storageBuffer16BitAccess};
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR epe = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
        .pipelineExecutableInfo = VK_TRUE};
    VkPhysicalDeviceShaderObjectFeaturesEXT eso = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT, .shaderObject = VK_TRUE};
    void **tail = &e12.pNext;
    if (use_pe) { *tail = &epe; tail = &epe.pNext; }
    if (use_so) { *tail = &eso; tail = &eso.pNext; }
    VkPhysicalDeviceFeatures2 e2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &e11,
        .features = {.shaderInt64 = f2.features.shaderInt64, .shaderInt16 = f2.features.shaderInt16, .shaderFloat64 = f2.features.shaderFloat64}};
    const char *exts[2]; uint32_t nexts = 0;
    if (use_pe) exts[nexts++] = VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME;
    if (use_so) exts[nexts++] = VK_EXT_SHADER_OBJECT_EXTENSION_NAME;
    float prio = 1;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qi, .queueCount = 1, .pQueuePriorities = &prio};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &e2, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
                              .enabledExtensionCount = nexts, .ppEnabledExtensionNames = exts};
    CK(vkCreateDevice(pd, &dci, NULL, &dev));
    printf("{\"device\":"); jstr(stdout, props.deviceName);
    printf(",\"vendor\":%u,\"device_id\":%u,\"driver_version\":%u,\"api\":%u,\"timestamp_period_ns\":%g,\"timestamps\":%d,"
           "\"subgroup_size\":%u,\"shader_object\":%d,\"executable_properties\":%d}\n",
           props.vendorID, props.deviceID, props.driverVersion, props.apiVersion, props.limits.timestampPeriod, timestamps,
           sgp.subgroupSize, use_so, use_pe);

    DF(vkGetDeviceQueue); DF(vkCreateBuffer); DF(vkGetBufferMemoryRequirements); DF(vkAllocateMemory); DF(vkBindBufferMemory);
    DF(vkMapMemory); DF(vkCreateShaderModule); DF(vkCreateDescriptorSetLayout); DF(vkCreatePipelineLayout);
    DF(vkCreateComputePipelines); DF(vkCreateDescriptorPool); DF(vkAllocateDescriptorSets); DF(vkUpdateDescriptorSets);
    DF(vkCreateCommandPool); DF(vkAllocateCommandBuffers); DF(vkBeginCommandBuffer); DF(vkEndCommandBuffer);
    DF(vkCmdBindPipeline); DF(vkCmdBindDescriptorSets); DF(vkCmdPushConstants); DF(vkCmdDispatch); DF(vkCmdPipelineBarrier);
    DF(vkCmdWriteTimestamp); DF(vkCmdResetQueryPool); DF(vkCreateQueryPool); DF(vkGetQueryPoolResults); DF(vkQueueSubmit);
    DF(vkQueueWaitIdle); DF(vkResetCommandBuffer); DF(vkDeviceWaitIdle);
    VkQueue q; vkGetDeviceQueue(dev, qi, 0, &q);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);

    for (int i = 0; i < nb; i++) {
        struct buf *b = &B[i];
        VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = b->size,
                                  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        CK(vkCreateBuffer(dev, &bci, NULL, &b->b));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b->b, &mr);
        uint32_t want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, ti = UINT32_MAX;
        for (uint32_t t = 0; t < mp.memoryTypeCount; t++)
            if ((mr.memoryTypeBits >> t & 1) && (mp.memoryTypes[t].propertyFlags & want) == want) { ti = t; break; }
        if (ti == UINT32_MAX) die("no host-visible coherent memory", 0);
        VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = ti};
        CK(vkAllocateMemory(dev, &mai, NULL, &b->m)); CK(vkBindBufferMemory(dev, b->b, b->m, 0));
        CK(vkMapMemory(dev, b->m, 0, b->size, 0, &b->map));
        if (strcmp(b->init, "-")) {
            b->initdata = slurp(b->init, &b->initlen);
            if (b->initlen > b->size) die("init file larger than buffer", b->binding);
        }
    }
    size_t spvlen; uint32_t *spv = slurp(shader, &spvlen);
    VkShaderModuleCreateInfo smci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = spvlen, .pCode = spv};
    VkShaderModule sm; CK(vkCreateShaderModule(dev, &smci, NULL, &sm));
    VkDescriptorSetLayoutBinding lb[MAXB];
    for (int i = 0; i < nb; i++)
        lb[i] = (VkDescriptorSetLayoutBinding){.binding = B[i].binding, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                               .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dslci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = nb, .pBindings = lb};
    VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
    VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = maxpush ? maxpush : 4};
    VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl,
                                       .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
    VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
    VkComputePipelineCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .flags = use_pe ? (VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR) : 0,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main"},
        .layout = pl};
    double t0 = now_ns();
    VkPipeline pipe; CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
    double compile_ns = now_ns() - t0;

    if (use_pe) {   /* statistics and internal representations as JSON */
        DF(vkGetPipelineExecutablePropertiesKHR); DF(vkGetPipelineExecutableStatisticsKHR); DF(vkGetPipelineExecutableInternalRepresentationsKHR);
        VkPipelineInfoKHR pinfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, .pipeline = pipe};
        uint32_t nx = 0; CK(vkGetPipelineExecutablePropertiesKHR(dev, &pinfo, &nx, NULL));
        VkPipelineExecutablePropertiesKHR *px = calloc(nx, sizeof *px);
        for (uint32_t i = 0; i < nx; i++) px[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
        CK(vkGetPipelineExecutablePropertiesKHR(dev, &pinfo, &nx, px));
        FILE *f = fopen(propsp, "w"); if (!f) die("open props", 0);
        fprintf(f, "[");
        for (uint32_t i = 0; i < nx; i++) {
            fprintf(f, "%s{\"name\":", i ? "," : ""); jstr(f, px[i].name); fprintf(f, ",\"description\":"); jstr(f, px[i].description);
            fprintf(f, ",\"subgroup_size\":%u,\"statistics\":{", px[i].subgroupSize);
            VkPipelineExecutableInfoKHR xi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, .pipeline = pipe, .executableIndex = i};
            uint32_t ns = 0; CK(vkGetPipelineExecutableStatisticsKHR(dev, &xi, &ns, NULL));
            VkPipelineExecutableStatisticKHR *st = calloc(ns, sizeof *st);
            for (uint32_t k = 0; k < ns; k++) st[k].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
            CK(vkGetPipelineExecutableStatisticsKHR(dev, &xi, &ns, st));
            for (uint32_t k = 0; k < ns; k++) {
                fprintf(f, "%s", k ? "," : ""); jstr(f, st[k].name); fputc(':', f);
                switch (st[k].format) {
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: fprintf(f, "%s", st[k].value.b32 ? "true" : "false"); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: fprintf(f, "%lld", (long long)st[k].value.i64); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: fprintf(f, "%llu", (unsigned long long)st[k].value.u64); break;
                default: fprintf(f, "%g", st[k].value.f64);
                }
            }
            fprintf(f, "},\"representations\":{");
            uint32_t nr = 0; CK(vkGetPipelineExecutableInternalRepresentationsKHR(dev, &xi, &nr, NULL));
            VkPipelineExecutableInternalRepresentationKHR *ir = calloc(nr, sizeof *ir);
            for (uint32_t k = 0; k < nr; k++) ir[k].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
            CK(vkGetPipelineExecutableInternalRepresentationsKHR(dev, &xi, &nr, ir));
            for (uint32_t k = 0; k < nr; k++) { ir[k].pData = calloc(1, ir[k].dataSize + 1); }
            VkResult r = vkGetPipelineExecutableInternalRepresentationsKHR(dev, &xi, &nr, ir);
            if (r != VK_SUCCESS && r != VK_INCOMPLETE) die("internal representations", r);
            for (uint32_t k = 0; k < nr; k++) {
                fprintf(f, "%s", k ? "," : ""); jstr(f, ir[k].name); fputc(':', f);
                if (ir[k].isText) jstr(f, (char *)ir[k].pData); else fprintf(f, "\"<binary %zu bytes>\"", ir[k].dataSize);
            }
            fprintf(f, "}}");
        }
        fprintf(f, "]\n"); fclose(f);
    }
    if (use_so) {   /* the driver's compiled binary for the same SPIR-V and layout */
        DF(vkCreateShadersEXT); DF(vkGetShaderBinaryDataEXT);
        VkShaderCreateInfoEXT sci = {.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT, .codeSize = spvlen, .pCode = spv, .pName = "main",
            .setLayoutCount = 1, .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
        VkShaderEXT so; CK(vkCreateShadersEXT(dev, 1, &sci, NULL, &so));
        size_t n = 0; CK(vkGetShaderBinaryDataEXT(dev, so, &n, NULL));
        void *data = malloc(n); CK(vkGetShaderBinaryDataEXT(dev, so, &n, data));
        spit(exportp, data, n);
        printf("{\"exported\":"); jstr(stdout, exportp); printf(",\"bytes\":%zu}\n", n);
    }

    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = nb ? nb : 1};
    VkDescriptorPoolCreateInfo dpci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
    VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
    VkDescriptorSetAllocateInfo dsai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl};
    VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev, &dsai, &ds));
    VkDescriptorBufferInfo dbi[MAXB]; VkWriteDescriptorSet w[MAXB];
    for (int i = 0; i < nb; i++) {
        dbi[i] = (VkDescriptorBufferInfo){.buffer = B[i].b, .offset = 0, .range = VK_WHOLE_SIZE};
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = B[i].binding,
                                      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi[i]};
    }
    vkUpdateDescriptorSets(dev, nb, w, 0, NULL);

    VkQueryPool qp = VK_NULL_HANDLE;
    if (timestamps) {
        VkQueryPoolCreateInfo qpci = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 * nd};
        CK(vkCreateQueryPool(dev, &qpci, NULL, &qp));
    }
    VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = qi};
    VkCommandPool cp; CK(vkCreateCommandPool(dev, &cpi, NULL, &cp));
    VkCommandBufferAllocateInfo cbai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CK(vkBeginCommandBuffer(cb, &bi));
    if (timestamps) vkCmdResetQueryPool(cb, qp, 0, 2 * nd);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    for (int i = 0; i < nd; i++) {
        if (i) vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        if (D[i].npush) vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, D[i].npush, D[i].push);
        if (timestamps) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 2 * i);
        vkCmdDispatch(cb, D[i].g[0], D[i].g[1], D[i].g[2]);
        if (timestamps) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 2 * i + 1);
    }
    CK(vkEndCommandBuffer(cb));
    /* the same command buffer is resubmitted for every run */
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
    uint64_t *ts = calloc(2 * nd, sizeof *ts);
    for (int rep = -1; rep < repeat; rep++) {
        for (int i = 0; i < nb; i++) {
            memset(B[i].map, 0, B[i].size);
            if (B[i].initdata) memcpy(B[i].map, B[i].initdata, B[i].initlen);
        }
        double h0 = now_ns();
        CK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
        CK(vkQueueWaitIdle(q));
        double host = now_ns() - h0;
        double total = 0;
        if (timestamps) {
            CK(vkGetQueryPoolResults(dev, qp, 0, 2 * nd, 2 * nd * sizeof *ts, ts, sizeof *ts, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            total = (ts[2 * nd - 1] - ts[0]) * (double)props.limits.timestampPeriod;
        }
        if (rep < 0) {
            for (int i = 0; i < nb; i++) if (strcmp(B[i].out, "-")) spit(B[i].out, B[i].map, B[i].size);
            printf("{\"functional\":true,\"host_ns\":%.0f,\"gpu_ns\":%.0f,\"compile_ns\":%.0f,\"dispatches\":%d}\n", host, total, compile_ns, nd);
            continue;
        }
        printf("{\"rep\":%d,\"host_ns\":%.0f", rep, host);
        if (timestamps) {
            printf(",\"total_ns\":%.0f,\"dispatch_ns\":[", total);
            for (int i = 0; i < nd; i++) printf("%s%.0f", i ? "," : "", (ts[2 * i + 1] - ts[2 * i]) * (double)props.limits.timestampPeriod);
            printf("]");
        }
        printf("}\n");
    }
    CK(vkDeviceWaitIdle(dev));
    printf("{\"summary\":{\"repeat\":%d,\"buffers\":%d,\"dispatches\":%d}}\n", repeat, nb, nd);
    return 0;
}
