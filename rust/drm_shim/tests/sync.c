/* The shim's synchronization and fault paths (run by tests/shim.rs under LD_PRELOAD, AGX_SHIM_DRM=1):
 *   1. a submit whose in-sync is not signaled stays pending: its out-sync is not signaled until the host signals
 *      the in-sync, then the command runs (its out-sync and its timestamps are set);
 *   2. timeline syncobjs: TIMELINE_SIGNAL / QUERY / TIMELINE_WAIT, a submit signaling a timeline point;
 *   3. a command whose CDM stream is not mapped faults as the firmware reports it: completed, timestamps 0.
 * The shader is a single `stop`. Prints "ok" and exits 0 when every check holds. */
#include <drm.h>
#include "drm/asahi_drm.h"
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

static int fd;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s (errno %d)\n", __LINE__, #c, errno); exit(1); } } while (0)
static int io(unsigned long r, void *a) { return ioctl(fd, r, a); }

struct bo { uint32_t h; uint8_t *map; };
static struct bo bo(uint32_t vm, uint64_t size, uint64_t va)
{
    struct drm_asahi_gem_create gc = {.size = size};
    CHECK(io(DRM_IOCTL_ASAHI_GEM_CREATE, &gc) == 0);
    struct drm_asahi_gem_mmap_offset mo = {.handle = gc.handle};
    CHECK(io(DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET, &mo) == 0);
    struct bo b = {gc.handle, mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)mo.offset)};
    CHECK(b.map != MAP_FAILED);
    if (va) {
        struct drm_asahi_gem_bind_op op = {.flags = DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE, .handle = gc.handle, .range = size, .addr = va};
        struct drm_asahi_vm_bind vb = {.vm_id = vm, .num_binds = 1, .stride = sizeof op, .userptr = (uint64_t)(uintptr_t)&op};
        CHECK(io(DRM_IOCTL_ASAHI_VM_BIND, &vb) == 0);
    }
    return b;
}
static void put32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static uint32_t syncobj(void) { struct drm_syncobj_create c = {0}; CHECK(io(DRM_IOCTL_SYNCOBJ_CREATE, &c) == 0); return c.handle; }
static int signaled(uint32_t h)
{
    struct drm_syncobj_wait w = {.handles = (uint64_t)(uintptr_t)&h, .count_handles = 1, .timeout_nsec = 0};
    return io(DRM_IOCTL_SYNCOBJ_WAIT, &w) == 0;
}

int main(void)
{
    fd = open("/dev/dri/renderD128", O_RDWR);
    CHECK(fd >= 0);
    struct drm_version v = {0};
    char name[16] = {0};
    v.name = name; v.name_len = sizeof name - 1;
    CHECK(io(DRM_IOCTL_VERSION, &v) == 0 && !strcmp(name, "asahi"));
    struct drm_asahi_params_global p = {0};
    struct drm_asahi_get_params gp = {.pointer = (uint64_t)(uintptr_t)&p, .size = sizeof p};
    CHECK(io(DRM_IOCTL_ASAHI_GET_PARAMS, &gp) == 0 && p.gpu_generation == 14);
    struct drm_asahi_vm_create vc = {.kernel_start = p.vm_end - (32ull << 30), .kernel_end = p.vm_end};
    CHECK(io(DRM_IOCTL_ASAHI_VM_CREATE, &vc) == 0);
    const uint64_t usc_va = 1ull << 36, data_va = usc_va + (1ull << 32);
    struct bo usc = bo(vc.vm_id, 16384, usc_va), data = bo(vc.vm_id, 16384, data_va), ts = bo(vc.vm_id, 16384, 0);
    struct drm_asahi_gem_bind_object bt = {.op = DRM_ASAHI_BIND_OBJECT_OP_BIND, .flags = DRM_ASAHI_BIND_OBJECT_USAGE_TIMESTAMPS, .handle = ts.h, .range = 16384};
    CHECK(io(DRM_IOCTL_ASAHI_GEM_BIND_OBJECT, &bt) == 0);

    /* pipeline at usc+0x1000 (Shared none, Shader code at 0x2000, Registers 8, No preshader), code = stop */
    uint8_t *w = usc.map + 0x1000;
    put32(w, 0x4du | (0x24u << 10)); w += 4;
    uint64_t sh = 0x0dull | (3ull << 10) | (0x2000ull << 16); memcpy(w, &sh, 6); w += 6;
    put32(w, 0x8du | (1u << 8) | (1u << 24)); w += 4;
    w[0] = 0x88; w[1] = 0;
    usc.map[0x2000] = 0x88; usc.map[0x2001] = 0;
    uint8_t *c = data.map + 0x100;
    int g14x = p.gpu_generation >= 14 && p.num_clusters_total > 1;
    put32(c, 1u << 1); put32(c + 4, 0x1000); c += 8;
    if (g14x) { put32(c, 0); put32(c + 4, 1u << 30); c += 8; }
    put32(c, 32); put32(c + 4, 1); put32(c + 8, 1); put32(c + 12, 32); put32(c + 16, 1); put32(c + 20, 1); c += 24;
    put32(c, 2u << 29); put32(c + 4, 0); c += 8;
    uint64_t cdm_end = data_va + 0x100 + (uint64_t)(c - (data.map + 0x100));

    struct drm_asahi_queue_create qc = {.vm_id = vc.vm_id, .priority = 1, .usc_exec_base = usc_va};
    CHECK(io(DRM_IOCTL_ASAHI_QUEUE_CREATE, &qc) == 0);
    struct { struct drm_asahi_cmd_header h; struct drm_asahi_cmd_compute c; } cmd = {
        .h = {.cmd_type = DRM_ASAHI_CMD_COMPUTE, .size = sizeof(struct drm_asahi_cmd_compute), .vdm_barrier = DRM_ASAHI_BARRIER_NONE},
        .c = {.cdm_ctrl_stream_base = data_va + 0x100, .cdm_ctrl_stream_end = cdm_end,
              .ts = {.start = {.handle = bt.object_handle, .offset = 0}, .end = {.handle = bt.object_handle, .offset = 8}}}};

    /* 1. pending on an unsignaled in-sync */
    uint32_t in = syncobj(), out = syncobj();
    struct drm_asahi_sync s1[2] = {{.sync_type = DRM_ASAHI_SYNC_SYNCOBJ, .handle = in}, {.sync_type = DRM_ASAHI_SYNC_SYNCOBJ, .handle = out}};
    struct drm_asahi_submit sub = {.syncs = (uint64_t)(uintptr_t)s1, .cmdbuf = (uint64_t)(uintptr_t)&cmd, .queue_id = qc.queue_id,
                                   .in_sync_count = 1, .out_sync_count = 1, .cmdbuf_size = sizeof cmd};
    CHECK(io(DRM_IOCTL_ASAHI_SUBMIT, &sub) == 0);
    uint64_t t[2]; memcpy(t, ts.map, 16);
    CHECK(!signaled(out) && errno == ETIME && t[0] == 0 && t[1] == 0);
    struct drm_syncobj_array sa = {.handles = (uint64_t)(uintptr_t)&in, .count_handles = 1};
    CHECK(io(DRM_IOCTL_SYNCOBJ_SIGNAL, &sa) == 0);
    memcpy(t, ts.map, 16);
    CHECK(signaled(out) && t[0] != 0 && t[1] > t[0]);

    /* 2. timelines */
    uint32_t tl = syncobj();
    uint64_t pt = 5;
    struct drm_syncobj_timeline_array ta = {.handles = (uint64_t)(uintptr_t)&tl, .points = (uint64_t)(uintptr_t)&pt, .count_handles = 1};
    CHECK(io(DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL, &ta) == 0);
    struct drm_asahi_sync s2[2] = {{.sync_type = DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ, .handle = tl, .timeline_value = 5},
                                   {.sync_type = DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ, .handle = tl, .timeline_value = 9}};
    sub.syncs = (uint64_t)(uintptr_t)s2;
    CHECK(io(DRM_IOCTL_ASAHI_SUBMIT, &sub) == 0);
    pt = 0;
    CHECK(io(DRM_IOCTL_SYNCOBJ_QUERY, &ta) == 0 && pt == 9);
    pt = 9;
    struct drm_syncobj_timeline_wait tw = {.handles = (uint64_t)(uintptr_t)&tl, .points = (uint64_t)(uintptr_t)&pt, .count_handles = 1, .timeout_nsec = 0};
    CHECK(io(DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &tw) == 0);
    pt = 10;
    CHECK(io(DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &tw) != 0 && errno == ETIME);

    /* 3. a fault: the CDM stream outside every bind */
    memset(ts.map, 0xff, 16);
    cmd.c.cdm_ctrl_stream_base = data_va + (1ull << 30); cmd.c.cdm_ctrl_stream_end = cmd.c.cdm_ctrl_stream_base + 64;
    uint32_t out3 = syncobj();
    struct drm_asahi_sync s3 = {.sync_type = DRM_ASAHI_SYNC_SYNCOBJ, .handle = out3};
    sub.syncs = (uint64_t)(uintptr_t)&s3; sub.in_sync_count = 0;
    CHECK(io(DRM_IOCTL_ASAHI_SUBMIT, &sub) == 0);
    memcpy(t, ts.map, 16);
    CHECK(signaled(out3) && t[0] == 0 && t[1] == 0);

    struct drm_asahi_queue_destroy qd = {.queue_id = qc.queue_id};
    CHECK(io(DRM_IOCTL_ASAHI_QUEUE_DESTROY, &qd) == 0);
    close(fd);
    printf("ok\n");
    return 0;
}
