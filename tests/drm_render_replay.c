/* Direct-ioctl replay of a render submission: no Mesa, no GL.
 *
 * Executes a plan written by scripts/render_replay.py (from a tests/render_trace.c capture, possibly mutated):
 *   usc_base HEX                     queue usc_exec_base
 *   bo ID SIZE                       create a buffer object (ID = plan-local id)
 *   fill ID OFFSET FILE              copy FILE into BO ID at OFFSET
 *   bind ID VA RANGE OFFSET FLAGS    VM_BIND (flags as captured: 2 = read, 4 = write)
 *   cmdbuf FILE                      the SUBMIT command buffer ({header, payload} records, as captured)
 *   dump VA SIZE FILE                after the last repetition, write SIZE bytes at VA to FILE
 *   reps N                           submit the command buffer N times (each waits; firmware timestamps per rep)
 *   wait_ms N                        syncobj wait limit per submit (default 2000)
 * Render commands get ts_vtx / ts_frag patched to a timestamp BO (offsets 0/8 and 16/24).
 * Output: one JSON line per rep {"rep","wait","vtx_ns","frag_ns","host_ns"} and {"completed":true}.
 * A wait failure prints {"failure":"syncobj-wait"} and exits 2 (the Python driver records it in the hang budget).
 *
 * usage: drm_render_replay /dev/dri/renderD128 plan.txt
 * build: gcc -std=gnu11 -O2 -o drm_render_replay tests/drm_render_replay.c
 */
#define _POSIX_C_SOURCE 200809L
#include <drm/drm.h>
#include <drm/asahi_drm.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static int fd = -1;
static void require(int ok, const char *stage)
{
    if (!ok) { printf("{\"failure\":\"%s\",\"errno\":%d}\n", stage, errno); exit(1); }
}
static void io(unsigned long request, void *arg, const char *stage)
{
    int r; do r = ioctl(fd, request, arg); while (r < 0 && (errno == EINTR || errno == EAGAIN));
    require(r == 0, stage);
}
static uint64_t now_ns(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

#define MAXBO 512
static struct { int used; uint32_t handle; uint64_t size; uint8_t *map; } bos[MAXBO];
#define MAXBIND 1024
static struct { int id; uint64_t va, range, off; } binds[MAXBIND]; static int nbinds;
#define MAXDUMP 64
static struct { uint64_t va, size; char path[512]; } dumps[MAXDUMP]; static int ndumps;

static uint8_t *va_ptr(uint64_t va, uint64_t n)
{
    for (int i = 0; i < nbinds; i++)
        if (binds[i].va <= va && va + n <= binds[i].va + binds[i].range)
            return bos[binds[i].id].map + binds[i].off + (va - binds[i].va);
    return NULL;
}

static uint8_t *read_file(const char *p, size_t *n)
{
    FILE *f = fopen(p, "rb"); require(f != NULL, "file-open");
    require(fseek(f, 0, SEEK_END) == 0, "file-seek"); long sz = ftell(f); rewind(f);
    require(sz >= 0 && sz < (1l << 30), "file-size");
    uint8_t *b = malloc((size_t)sz + 1); require(b != NULL, "malloc");
    require(fread(b, 1, (size_t)sz, f) == (size_t)sz, "file-read"); fclose(f);
    *n = (size_t)sz; return b;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    require(argc == 3, "usage: node plan.txt");
    fd = open(argv[1], O_RDWR | O_CLOEXEC); require(fd >= 0, "open");
    struct drm_asahi_params_global p = {0};
    struct drm_asahi_get_params gp = {.param_group = 0, .pointer = (uint64_t)(uintptr_t)&p, .size = sizeof p};
    io(DRM_IOCTL_ASAHI_GET_PARAMS, &gp, "get-params");
    uint64_t ksize = p.vm_kernel_min_size > (32ull << 30) ? p.vm_kernel_min_size : (32ull << 30);
    struct drm_asahi_vm_create vc = {.kernel_start = p.vm_end - ksize, .kernel_end = p.vm_end};
    io(DRM_IOCTL_ASAHI_VM_CREATE, &vc, "vm-create");

    FILE *pl = fopen(argv[2], "r"); require(pl != NULL, "plan-open");
    char line[1024], a[512]; uint64_t usc_base = 0; uint8_t *cmdbuf = NULL; size_t cmdbuf_size = 0;
    unsigned reps = 1; uint64_t wait_ms = 2000;
    while (fgets(line, sizeof line, pl)) {
        int id; uint64_t x, y, z, w;
        if (sscanf(line, "usc_base %" SCNx64, &x) == 1) usc_base = x;
        else if (sscanf(line, "bo %d %" SCNu64, &id, &x) == 2) {
            require(id >= 0 && id < MAXBO && !bos[id].used && x > 0 && x <= (4ull << 30), "bo-line");
            struct drm_asahi_gem_create gc = {.size = x, .flags = DRM_ASAHI_GEM_WRITEBACK};
            io(DRM_IOCTL_ASAHI_GEM_CREATE, &gc, "gem-create");
            struct drm_asahi_gem_mmap_offset mo = {.handle = gc.handle};
            io(DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET, &mo, "gem-mmap-offset");
            bos[id].map = mmap(NULL, x, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)mo.offset);
            require(bos[id].map != MAP_FAILED, "mmap");
            memset(bos[id].map, 0, x);
            bos[id].used = 1; bos[id].handle = gc.handle; bos[id].size = x;
        } else if (sscanf(line, "fill %d %" SCNu64 " %511s", &id, &x, a) == 3) {
            require(id >= 0 && id < MAXBO && bos[id].used, "fill-id");
            size_t n; uint8_t *b = read_file(a, &n);
            require(x + n <= bos[id].size, "fill-range"); memcpy(bos[id].map + x, b, n); free(b);
        } else if (sscanf(line, "bind %d %" SCNx64 " %" SCNu64 " %" SCNu64 " %" SCNu64, &id, &x, &y, &z, &w) == 5) {
            require(id >= 0 && id < MAXBO && bos[id].used && nbinds < MAXBIND && z + y <= bos[id].size, "bind-line");
            struct drm_asahi_gem_bind_op op = {.flags = (uint32_t)w & (DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE),
                                               .handle = bos[id].handle, .offset = z, .range = y, .addr = x};
            struct drm_asahi_vm_bind vb = {.vm_id = vc.vm_id, .num_binds = 1, .stride = sizeof op, .userptr = (uint64_t)(uintptr_t)&op};
            io(DRM_IOCTL_ASAHI_VM_BIND, &vb, "vm-bind");
            binds[nbinds].id = id; binds[nbinds].va = x; binds[nbinds].range = y; binds[nbinds++].off = z;
        } else if (sscanf(line, "cmdbuf %511s", a) == 1) cmdbuf = read_file(a, &cmdbuf_size);
        else if (sscanf(line, "dump %" SCNx64 " %" SCNu64 " %511s", &x, &y, a) == 3) {
            require(ndumps < MAXDUMP, "dumps"); dumps[ndumps].va = x; dumps[ndumps].size = y;
            strcpy(dumps[ndumps++].path, a);
        } else if (sscanf(line, "reps %" SCNu64, &x) == 1) { require(x >= 1 && x <= 1000, "reps"); reps = (unsigned)x; }
        else if (sscanf(line, "wait_ms %" SCNu64, &x) == 1) { require(x >= 10 && x <= 10000, "wait_ms"); wait_ms = x; }
        else if (line[0] != '#' && line[0] != '\n') { printf("{\"failure\":\"plan-line\",\"line\":\"%.60s\"}\n", line); return 1; }
    }
    fclose(pl);
    require(cmdbuf && cmdbuf_size >= 8 && usc_base, "plan-incomplete");

    struct drm_asahi_gem_create tgc = {.size = 16384, .flags = DRM_ASAHI_GEM_WRITEBACK};
    io(DRM_IOCTL_ASAHI_GEM_CREATE, &tgc, "ts-create");
    struct drm_asahi_gem_mmap_offset tmo = {.handle = tgc.handle};
    io(DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET, &tmo, "ts-mmap-offset");
    uint8_t *tsmap = mmap(NULL, 16384, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)tmo.offset);
    require(tsmap != MAP_FAILED, "ts-mmap");
    struct drm_asahi_gem_bind_object bo_ts = {.op = DRM_ASAHI_BIND_OBJECT_OP_BIND, .flags = DRM_ASAHI_BIND_OBJECT_USAGE_TIMESTAMPS,
                                              .handle = tgc.handle, .range = 16384};
    io(DRM_IOCTL_ASAHI_GEM_BIND_OBJECT, &bo_ts, "bind-timestamps");

    /* Patch timestamps into every render record. */
    for (size_t off = 0; off + sizeof(struct drm_asahi_cmd_header) <= cmdbuf_size;) {
        struct drm_asahi_cmd_header *h = (void *)(cmdbuf + off);
        require(off + sizeof *h + h->size <= cmdbuf_size, "cmdbuf-record");
        if (h->cmd_type == DRM_ASAHI_CMD_RENDER) {
            require(h->size == sizeof(struct drm_asahi_cmd_render), "render-size");
            struct drm_asahi_cmd_render *r = (void *)(h + 1);
            r->ts_vtx = (struct drm_asahi_timestamps){{bo_ts.object_handle, 0}, {bo_ts.object_handle, 8}};
            r->ts_frag = (struct drm_asahi_timestamps){{bo_ts.object_handle, 16}, {bo_ts.object_handle, 24}};
        }
        off += sizeof *h + h->size;
    }

    struct drm_asahi_queue_create qc = {.vm_id = vc.vm_id, .priority = DRM_ASAHI_PRIORITY_MEDIUM, .usc_exec_base = usc_base};
    io(DRM_IOCTL_ASAHI_QUEUE_CREATE, &qc, "queue-create");
    struct drm_syncobj_create sc = {0};
    io(DRM_IOCTL_SYNCOBJ_CREATE, &sc, "syncobj-create");
    int status = 0;
    for (unsigned rep = 0; rep < reps; rep++) {
        memset(tsmap, 0, 32);
        struct drm_asahi_sync out_sync = {.sync_type = DRM_ASAHI_SYNC_SYNCOBJ, .handle = sc.handle};
        struct drm_asahi_submit sub = {.syncs = (uint64_t)(uintptr_t)&out_sync, .cmdbuf = (uint64_t)(uintptr_t)cmdbuf,
                                       .queue_id = qc.queue_id, .out_sync_count = 1, .cmdbuf_size = (uint32_t)cmdbuf_size};
        uint64_t h0 = now_ns();
        io(DRM_IOCTL_ASAHI_SUBMIT, &sub, "submit");
        struct drm_syncobj_wait sw = {.handles = (uint64_t)(uintptr_t)&sc.handle, .count_handles = 1,
                                      .timeout_nsec = (int64_t)(h0 + wait_ms * 1000000ull)};
        int r = ioctl(fd, DRM_IOCTL_SYNCOBJ_WAIT, &sw);
        uint64_t h1 = now_ns();
        uint64_t ts[4]; memcpy(ts, tsmap, 32);
        printf("{\"rep\":%u,\"wait\":%d,\"errno\":%d,\"host_ns\":%" PRIu64 ",\"ts\":[%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "],"
               "\"vtx_ns\":%" PRId64 ",\"frag_ns\":%" PRId64 "}\n", rep, r, r ? errno : 0, h1 - h0, ts[0], ts[1], ts[2], ts[3],
               (int64_t)(ts[1] - ts[0]), (int64_t)(ts[3] - ts[2]));
        if (r) { printf("{\"failure\":\"syncobj-wait\"}\n"); status = 2; break; }
        struct drm_syncobj_array reset = {.handles = (uint64_t)(uintptr_t)&sc.handle, .count_handles = 1};
        io(DRM_IOCTL_SYNCOBJ_RESET, &reset, "syncobj-reset");
    }
    for (int i = 0; i < ndumps; i++) {
        uint8_t *src = va_ptr(dumps[i].va, dumps[i].size); require(src != NULL, "dump-va");
        FILE *o = fopen(dumps[i].path, "wb"); require(o && fwrite(src, 1, dumps[i].size, o) == dumps[i].size && fclose(o) == 0, "dump-write");
    }
    if (!status) printf("{\"completed\":true,\"reps\":%u}\n", reps);

    struct drm_asahi_queue_destroy qd = {.queue_id = qc.queue_id};
    io(DRM_IOCTL_ASAHI_QUEUE_DESTROY, &qd, "queue-destroy");
    for (int i = 0; i < MAXBO; i++) if (bos[i].used) {
        munmap(bos[i].map, bos[i].size);
        struct drm_gem_close gcl = {.handle = bos[i].handle}; io(DRM_IOCTL_GEM_CLOSE, &gcl, "gem-close");
    }
    munmap(tsmap, 16384);
    struct drm_gem_close tcl = {.handle = tgc.handle}; io(DRM_IOCTL_GEM_CLOSE, &tcl, "gem-close");
    struct drm_syncobj_destroy sd = {.handle = sc.handle}; io(DRM_IOCTL_SYNCOBJ_DESTROY, &sd, "syncobj-destroy");
    struct drm_asahi_vm_destroy vd = {.vm_id = vc.vm_id}; io(DRM_IOCTL_ASAHI_VM_DESTROY, &vd, "vm-destroy");
    close(fd);
    return status;
}
