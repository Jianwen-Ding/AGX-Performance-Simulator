/* Bounded shader-object delivery/readback harness; outer timeout required. */
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>

static void require(int ok, const char *stage) {
    if (!ok) { fprintf(stderr, "failure: %s\n", stage); exit(1); }
}
static void check(VkResult r, const char *stage) {
    printf("{\"api\":\"%s\",\"result\":%d}\n", stage, r);
    require(r == VK_SUCCESS, stage);
}
#define I(name) PFN_##name name = (PFN_##name)get(instance, #name); require(name != NULL, #name)
#define D(name) PFN_##name name = (PFN_##name)vkGetDeviceProcAddr(device, #name); require(name != NULL, #name)
#define CALL(name, ...) check(name(__VA_ARGS__), #name)
static void *readfile(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb"); require(f != NULL, "input-open");
    require(fseek(f, 0, SEEK_END) == 0, "input-seek");
    long n = ftell(f); require(n > 0 && n <= 1048576, "input-size");
    rewind(f); *size = (size_t)n;
    void *p = aligned_alloc(16, (*size + 15) & ~(size_t)15); require(p != NULL, "input-allocate");
    require(fread(p, 1, *size, f) == *size && fclose(f) == 0, "input-read"); return p;
}
static void writefile(const char *path, const void *p, size_t n) {
    FILE *f = fopen(path, "wb"); require(f != NULL, "output-open");
    require(fwrite(p, 1, n, f) == n && fclose(f) == 0, "output-write");
}
int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    require(argc == 7 || argc == 10, "usage: export|import|control input count node readback binary-output [groups-x groups-y groups-z]");
    /* follow: import argv[2] and submit, then import argv[6] and submit again on the
       same device/queue; second readback goes to argv[5] with ".2" appended. */
    int follow = strcmp(argv[1], "follow") == 0;
    int importing = strcmp(argv[1], "import") == 0 || follow, control = strcmp(argv[1], "control") == 0;
    require(importing || control || strcmp(argv[1], "export") == 0, "mode");
    char *end; unsigned long parsed = strtoul(argv[3], &end, 10);
    require(*end == 0 && parsed >= 1 && parsed <= 65536, "count"); uint32_t count = (uint32_t)parsed;
    struct stat node; require(stat(argv[4], &node) == 0 && S_ISCHR(node.st_mode), "node-stat");
    void *lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL); require(lib != NULL, "loader");
    PFN_vkGetInstanceProcAddr get = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr"); require(get != NULL, "get-proc");
    VkInstance instance = VK_NULL_HANDLE; I(vkCreateInstance);
    VkApplicationInfo app = {.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName="agx-instruction-delivery", .apiVersion=VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo=&app};
    CALL(vkCreateInstance, &ici, NULL, &instance);
    I(vkDestroyInstance); I(vkEnumeratePhysicalDevices); I(vkGetPhysicalDeviceProperties2);
    I(vkGetPhysicalDeviceFeatures2); I(vkGetPhysicalDeviceQueueFamilyProperties);
    I(vkGetPhysicalDeviceMemoryProperties); I(vkEnumerateDeviceExtensionProperties); I(vkCreateDevice); I(vkGetDeviceProcAddr);
    uint32_t np=0; CALL(vkEnumeratePhysicalDevices, instance, &np, NULL); require(np == 1, "one-device");
    VkPhysicalDevice physical; CALL(vkEnumeratePhysicalDevices, instance, &np, &physical);
    uint32_t ne=0; CALL(vkEnumerateDeviceExtensionProperties, physical, NULL, &ne, NULL); require(ne <= 512, "extension-bound");
    VkExtensionProperties ex[512]; CALL(vkEnumerateDeviceExtensionProperties, physical, NULL, &ne, ex);
    int soext=0, drmext=0; for (uint32_t j=0;j<ne;j++) { soext |= strcmp(ex[j].extensionName, VK_EXT_SHADER_OBJECT_EXTENSION_NAME)==0; drmext |= strcmp(ex[j].extensionName, VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME)==0; }
    require(soext && drmext, "required-extensions");
    VkPhysicalDeviceShaderObjectPropertiesEXT sop={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_PROPERTIES_EXT};
    VkPhysicalDeviceDrmPropertiesEXT drm={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT, .pNext=&sop};
    VkPhysicalDeviceDriverProperties driver={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, .pNext=&drm};
    VkPhysicalDeviceProperties2 props={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext=&driver}; vkGetPhysicalDeviceProperties2(physical,&props);
    printf("{\"identity\":{\"name\":\"%s\",\"driver\":\"%s\",\"info\":\"%s\",\"render_major\":%lld,\"render_minor\":%lld,\"uuid\":\"",props.properties.deviceName,driver.driverName,driver.driverInfo,(long long)drm.renderMajor,(long long)drm.renderMinor);
    for(unsigned j=0;j<VK_UUID_SIZE;j++) printf("%02x",sop.shaderBinaryUUID[j]);
    printf("\",\"binary_version\":%u}}\n",sop.shaderBinaryVersion);
    const uint8_t uuid[16]={0x23,0x13,0xbd,0x7c,0x39,0x09,0xb8,0xad,0x0f,0x31,0x88,0x3e,0xab,0x5c,0x41,0x2e};
    require(strcmp(props.properties.deviceName,"Apple M2 Max (G14C B1)")==0 && driver.driverID==VK_DRIVER_ID_MESA_HONEYKRISP && strcmp(driver.driverInfo,"Mesa 25.3.6")==0 && props.properties.apiVersion>=VK_API_VERSION_1_3 && drm.hasRender && drm.renderMajor==major(node.st_rdev) && drm.renderMinor==minor(node.st_rdev) && memcmp(uuid,sop.shaderBinaryUUID,16)==0 && sop.shaderBinaryVersion==0,"identity-match");
    VkPhysicalDeviceDynamicRenderingFeatures drf={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
    VkPhysicalDeviceShaderObjectFeaturesEXT sf={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT,.pNext=&drf};
    VkPhysicalDeviceFeatures2 features={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,.pNext=&sf}; vkGetPhysicalDeviceFeatures2(physical,&features);
    printf("{\"features\":{\"shaderObject\":%u,\"dynamicRendering\":%u}}\n",sf.shaderObject,drf.dynamicRendering);
    require(sf.shaderObject && drf.dynamicRendering,"features");
    uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,NULL); require(nq>0 && nq<=32,"queue-bound");
    VkQueueFamilyProperties qp[32]; vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,qp); uint32_t family=0;
    while(family<nq && !(qp[family].queueCount && (qp[family].queueFlags & VK_QUEUE_COMPUTE_BIT))) family++;
    require(family<nq && props.properties.limits.maxComputeWorkGroupInvocations>=32 && props.properties.limits.maxComputeWorkGroupSize[0]>=32 && props.properties.limits.maxComputeWorkGroupCount[0]>=9,"compute-limits");
    float priority=1; VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=family,.queueCount=1,.pQueuePriorities=&priority};
    const char *extensions[]={VK_EXT_SHADER_OBJECT_EXTENSION_NAME};
    VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=&sf,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci,.enabledExtensionCount=1,.ppEnabledExtensionNames=extensions};
    VkDevice device; CALL(vkCreateDevice,physical,&dci,NULL,&device);
    D(vkDestroyDevice); D(vkGetDeviceQueue); D(vkCreateBuffer); D(vkDestroyBuffer); D(vkGetBufferMemoryRequirements); D(vkAllocateMemory); D(vkFreeMemory); D(vkBindBufferMemory); D(vkMapMemory); D(vkUnmapMemory);
    D(vkCreateDescriptorSetLayout); D(vkDestroyDescriptorSetLayout); D(vkCreatePipelineLayout); D(vkDestroyPipelineLayout); D(vkCreateDescriptorPool); D(vkDestroyDescriptorPool); D(vkAllocateDescriptorSets); D(vkUpdateDescriptorSets);
    D(vkCreateShadersEXT); D(vkDestroyShaderEXT); D(vkGetShaderBinaryDataEXT); D(vkCmdBindShadersEXT);
    D(vkCreateCommandPool); D(vkDestroyCommandPool); D(vkAllocateCommandBuffers); D(vkBeginCommandBuffer); D(vkEndCommandBuffer); D(vkCmdPipelineBarrier); D(vkCmdBindDescriptorSets); D(vkCmdPushConstants); D(vkCmdDispatch); D(vkCreateFence); D(vkDestroyFence); D(vkQueueSubmit); D(vkWaitForFences);
    VkQueue queue; vkGetDeviceQueue(device,family,0,&queue);
    uint32_t rounded=(count+31)/32*32, words=8+rounded+8+16; size_t bytes=words*4;
    VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=bytes,.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer buffer; CALL(vkCreateBuffer,device,&bci,NULL,&buffer);
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(device,buffer,&req);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(physical,&mp); uint32_t mt=0;
    VkMemoryPropertyFlags wanted=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    while(mt<mp.memoryTypeCount && (!(req.memoryTypeBits & (1u<<mt)) || (mp.memoryTypes[mt].propertyFlags&wanted)!=wanted)) mt++;
    require(mt<mp.memoryTypeCount,"coherent-memory");
    printf("{\"memory\":{\"type\":%u,\"flags\":%u,\"allocation_bytes\":%llu,\"words\":%u}}\n",mt,mp.memoryTypes[mt].propertyFlags,(unsigned long long)req.size,words);
    VkMemoryAllocateInfo mai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=req.size,.memoryTypeIndex=mt}; VkDeviceMemory memory;
    CALL(vkAllocateMemory,device,&mai,NULL,&memory); CALL(vkBindBufferMemory,device,buffer,memory,0);
    uint32_t *mapped; CALL(vkMapMemory,device,memory,0,VK_WHOLE_SIZE,0,(void **)&mapped);
    for(uint32_t j=0;j<words;j++) mapped[j]=j<words-16 ? 0xcdcdcdcdu : 0xa5a50000u+j-(words-16);
    VkDescriptorSetLayoutBinding binding={.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo lci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=1,.pBindings=&binding}; VkDescriptorSetLayout layout;
    CALL(vkCreateDescriptorSetLayout,device,&lci,NULL,&layout);
    VkPushConstantRange push={.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.offset=0,.size=4};
    VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&layout,.pushConstantRangeCount=1,.pPushConstantRanges=&push}; VkPipelineLayout pl;
    CALL(vkCreatePipelineLayout,device,&plci,NULL,&pl);
    VkDescriptorPoolSize ps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1};
    VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&ps}; VkDescriptorPool pool;
    CALL(vkCreateDescriptorPool,device,&dpci,NULL,&pool);
    VkDescriptorSetAllocateInfo dsai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=pool,.descriptorSetCount=1,.pSetLayouts=&layout}; VkDescriptorSet ds;
    CALL(vkAllocateDescriptorSets,device,&dsai,&ds);
    VkDescriptorBufferInfo bi={.buffer=buffer,.offset=0,.range=bytes};
    VkWriteDescriptorSet wd={.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=0,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi}; vkUpdateDescriptorSets(device,1,&wd,0,NULL);
    size_t code_size; void *code=readfile(argv[2],&code_size);
    require(importing || code_size%4==0,"spirv-alignment");
    VkShaderCreateInfoEXT sci={.sType=VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.codeType=importing ? VK_SHADER_CODE_TYPE_BINARY_EXT : VK_SHADER_CODE_TYPE_SPIRV_EXT,.codeSize=code_size,.pCode=code,.pName="main",.setLayoutCount=1,.pSetLayouts=&layout,.pushConstantRangeCount=1,.pPushConstantRanges=&push}; VkShaderEXT shader=VK_NULL_HANDLE;
    CALL(vkCreateShadersEXT,device,1,&sci,NULL,&shader); require(shader!=VK_NULL_HANDLE,"shader-handle"); free(code);
    VkCommandPoolCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=family}; VkCommandPool cp;
    CALL(vkCreateCommandPool,device,&cpci,NULL,&cp);
    VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=cp,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1}; VkCommandBuffer cb;
    CALL(vkAllocateCommandBuffers,device,&cbai,&cb);
    VkCommandBufferBeginInfo cbi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}; CALL(vkBeginCommandBuffer,cb,&cbi);
    VkMemoryBarrier barrier={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,NULL,0,NULL);
    VkShaderStageFlagBits stage=VK_SHADER_STAGE_COMPUTE_BIT; vkCmdBindShadersEXT(cb,1,&stage,&shader);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,NULL); vkCmdPushConstants(cb,pl,stage,0,4,&count);
    uint32_t groups[3]={rounded/32,1,1};
    if(argc==10) for(int j=0;j<3;j++){ unsigned long g=strtoul(argv[7+j],&end,10); require(*end==0 && g>=1 && g<=65536,"groups"); groups[j]=(uint32_t)g; }
    require((uint64_t)groups[0]*groups[1]*groups[2]<=65536,"groups-total");
    if(!control) vkCmdDispatch(cb,groups[0],groups[1],groups[2]);
    barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,NULL,0,NULL);
    CALL(vkEndCommandBuffer,cb);
    VkFenceCreateInfo fci={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; CALL(vkCreateFence,device,&fci,NULL,&fence);
    VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb}; CALL(vkQueueSubmit,queue,1,&submit,fence);
    CALL(vkWaitForFences,device,1,&fence,VK_TRUE,3000000000ull);
    writefile(argv[5],mapped,bytes);
    if(follow) {
        /* Second submission: results are logged, not required, so a lost device is observable. */
        for(uint32_t j=0;j<words;j++) mapped[j]=j<words-16 ? 0xcdcdcdcdu : 0xa5a50000u+j-(words-16);
        size_t code2_size; void *code2=readfile(argv[6],&code2_size);
        sci.codeSize=code2_size; sci.pCode=code2; VkShaderEXT shader2=VK_NULL_HANDLE;
        VkResult r=vkCreateShadersEXT(device,1,&sci,NULL,&shader2); free(code2);
        printf("{\"api\":\"follow:vkCreateShadersEXT\",\"result\":%d}\n",r); require(r==VK_SUCCESS,"follow-shader");
        VkCommandBuffer cb2; CALL(vkAllocateCommandBuffers,device,&cbai,&cb2); CALL(vkBeginCommandBuffer,cb2,&cbi);
        barrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb2,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,NULL,0,NULL);
        vkCmdBindShadersEXT(cb2,1,&stage,&shader2);
        vkCmdBindDescriptorSets(cb2,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,NULL); vkCmdPushConstants(cb2,pl,stage,0,4,&count);
        vkCmdDispatch(cb2,groups[0],groups[1],groups[2]);
        barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cb2,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,NULL,0,NULL);
        CALL(vkEndCommandBuffer,cb2);
        VkFence fence2; CALL(vkCreateFence,device,&fci,NULL,&fence2);
        VkSubmitInfo submit2={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb2};
        r=vkQueueSubmit(queue,1,&submit2,fence2); printf("{\"api\":\"follow:vkQueueSubmit\",\"result\":%d}\n",r);
        if(r==VK_SUCCESS){ r=vkWaitForFences(device,1,&fence2,VK_TRUE,3000000000ull); printf("{\"api\":\"follow:vkWaitForFences\",\"result\":%d}\n",r); }
        size_t n=strlen(argv[5]); char *second=malloc(n+3); require(second!=NULL,"follow-path");
        memcpy(second,argv[5],n); memcpy(second+n,".2",3); writefile(second,mapped,bytes); free(second);
        vkDestroyFence(device,fence2,NULL); vkDestroyShaderEXT(device,shader2,NULL);
    }
    if(!importing && !control && !follow) {
        size_t size=0; CALL(vkGetShaderBinaryDataEXT,device,shader,&size,NULL); require(size>0 && size<=1048576,"binary-size");
        size_t capacity=size; void *binary=aligned_alloc(16,(size+15)&~(size_t)15); require(binary!=NULL,"binary-allocate");
        CALL(vkGetShaderBinaryDataEXT,device,shader,&size,binary); require(size==capacity,"binary-size-stable"); writefile(argv[6],binary,size); free(binary);
        printf("{\"export_bytes\":%zu}\n",size);
    }
    printf("{\"completed\":true,\"dispatch\":%s,\"count\":%u,\"words\":%u}\n",control ? "false":"true",count,words);
    vkDestroyFence(device,fence,NULL); vkDestroyCommandPool(device,cp,NULL); vkDestroyShaderEXT(device,shader,NULL);
    vkDestroyDescriptorPool(device,pool,NULL); vkDestroyPipelineLayout(device,pl,NULL); vkDestroyDescriptorSetLayout(device,layout,NULL);
    vkUnmapMemory(device,memory); vkDestroyBuffer(device,buffer,NULL); vkFreeMemory(device,memory,NULL); vkDestroyDevice(device,NULL); vkDestroyInstance(instance,NULL); dlclose(lib); return 0;
}
