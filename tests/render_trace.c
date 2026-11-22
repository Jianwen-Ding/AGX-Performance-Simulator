/* Render trace: LD_PRELOAD shim that records Mesa's (open-source) asahi uAPI traffic and, on every SUBMIT that
 * contains a DRM_ASAHI_CMD_RENDER, dumps the whole command buffer (all records raw) and the contents of every bound
 * GPU buffer as they are at submit time. Nothing is modified or submitted by the shim; the submit proceeds unchanged.
 *
 * Output: $RENDER_TRACE_DIR/submit-N/{manifest.json, cmdbuf.bin, bo-<va>.bin}
 * Env: RENDER_TRACE_DIR (default /scratch/render/trace), RENDER_TRACE_MAX (max submits dumped, default 4),
 *      RENDER_TRACE_SKIP (render submits to skip first, default 0), RENDER_TRACE_CAP_MB (bytes per bind, default 64),
 *      RENDER_TRACE_AFTER=1 also dumps every bind again after the submit's fence completes (bo-<va>.after.bin):
 *      the shim waits on the submit's out syncobjs (5 s), so the frame's output is in the trace.
 * build: gcc -O2 -fPIC -shared -o render_trace.so tests/render_trace.c -ldl
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <drm/drm.h>
#include <drm/asahi_drm.h>

#define MAXB 8192
static struct { uint32_t handle; uint64_t size; } bos[MAXB]; static int nbos;
static struct { uint64_t addr, range, offset; uint32_t handle, flags; } binds[MAXB]; static int nbinds;
static uint64_t usc_base; static int nrender, ndumped; static int drm_fd = -1;
static int (*real_ioctl)(int, unsigned long, ...);

static uint64_t envu(const char *n, uint64_t d) { const char *v = getenv(n); return v && *v ? strtoull(v, 0, 0) : d; }
static uint64_t bo_size(uint32_t h) { for (int i = 0; i < nbos; i++) if (bos[i].handle == h) return bos[i].size; return 0; }

static void dump_bind(FILE *m, const char *dir, int i, int first, const char *suffix) {
  uint64_t cap = envu("RENDER_TRACE_CAP_MB", 64) << 20, n = binds[i].range < cap ? binds[i].range : cap;
  struct drm_asahi_gem_mmap_offset mo = { .handle = binds[i].handle };
  int ok = real_ioctl(drm_fd, DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET, &mo) == 0;
  uint64_t bs = bo_size(binds[i].handle);
  void *p = ok && bs ? mmap(NULL, bs, PROT_READ, MAP_SHARED, drm_fd, mo.offset) : MAP_FAILED;
  if (m)
    fprintf(m, "%s{\"va\":\"0x%llx\",\"range\":%llu,\"bo_offset\":%llu,\"handle\":%u,\"flags\":%u,\"bo_size\":%llu,\"dumped\":%llu}",
            first ? "" : ",", (unsigned long long)binds[i].addr, (unsigned long long)binds[i].range,
            (unsigned long long)binds[i].offset, binds[i].handle, binds[i].flags, (unsigned long long)bs,
            (unsigned long long)(p == MAP_FAILED ? 0 : n));
  if (p == MAP_FAILED) return;
  char path[512]; snprintf(path, sizeof path, "%s/bo-%llx%s.bin", dir, (unsigned long long)binds[i].addr, suffix);
  FILE *f = fopen(path, "wb"); if (f) { fwrite((char *)p + binds[i].offset, 1, n, f); fclose(f); }
  munmap(p, bs);
}

static char cur_dir[400]; static int cur_valid;

static void on_submit(struct drm_asahi_submit *s) {
  cur_valid = 0;
  char *cb = (char *)(uintptr_t)s->cmdbuf; uint32_t off = 0; int has_render = 0;
  while (off + sizeof(struct drm_asahi_cmd_header) <= s->cmdbuf_size) {
    struct drm_asahi_cmd_header *h = (void *)(cb + off);
    if (h->cmd_type == DRM_ASAHI_CMD_RENDER) has_render = 1;
    off += sizeof *h + h->size;
  }
  if (!has_render) return;
  if (nrender++ < (int)envu("RENDER_TRACE_SKIP", 0)) return;
  if (ndumped >= (int)envu("RENDER_TRACE_MAX", 4)) return;
  const char *root = getenv("RENDER_TRACE_DIR"); if (!root) root = "/scratch/render/trace";
  snprintf(cur_dir, sizeof cur_dir, "%s/submit-%d", root, ndumped++); mkdir(root, 0755); mkdir(cur_dir, 0755);
  char path[512]; snprintf(path, sizeof path, "%s/cmdbuf.bin", cur_dir);
  FILE *c = fopen(path, "wb"); if (c) { fwrite(cb, 1, s->cmdbuf_size, c); fclose(c); }
  snprintf(path, sizeof path, "%s/manifest.json", cur_dir);
  FILE *m = fopen(path, "w"); if (!m) return;
  fprintf(m, "{\"usc_exec_base\":\"0x%llx\",\"queue_id\":%u,\"flags\":%u,\"in_syncs\":%u,\"out_syncs\":%u,\"cmdbuf_size\":%u,"
             "\"sizeof_render\":%zu,\"commands\":[",
          (unsigned long long)usc_base, s->queue_id, s->flags, s->in_sync_count, s->out_sync_count, s->cmdbuf_size,
          sizeof(struct drm_asahi_cmd_render));
  off = 0; int first = 1;
  while (off + sizeof(struct drm_asahi_cmd_header) <= s->cmdbuf_size) {
    struct drm_asahi_cmd_header *h = (void *)(cb + off);
    fprintf(m, "%s{\"type\":%u,\"size\":%u,\"vdm_barrier\":%u,\"cdm_barrier\":%u,\"offset\":%u}", first ? "" : ",",
            h->cmd_type, h->size, h->vdm_barrier, h->cdm_barrier, off);
    first = 0; off += sizeof *h + h->size;
  }
  fprintf(m, "],\"binds\":[");
  for (int i = 0; i < nbinds; i++) dump_bind(m, cur_dir, i, i == 0, "");
  fprintf(m, "]}\n"); fclose(m);
  cur_valid = 1;
}

static void after_submit(struct drm_asahi_submit *s) {
  if (!cur_valid || !envu("RENDER_TRACE_AFTER", 0) || !s->out_sync_count) return;
  struct drm_asahi_sync *sy = (void *)(uintptr_t)s->syncs;
  uint32_t h[16]; unsigned n = 0;
  for (unsigned i = 0; i < s->out_sync_count && n < 16; i++)
    if (sy[s->in_sync_count + i].sync_type == DRM_ASAHI_SYNC_SYNCOBJ) h[n++] = sy[s->in_sync_count + i].handle;
  if (!n) return;
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  struct drm_syncobj_wait w = {.handles = (uint64_t)(uintptr_t)h, .count_handles = n,
                               .timeout_nsec = (int64_t)t.tv_sec * 1000000000ll + t.tv_nsec + 5000000000ll,
                               .flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL | DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT};
  int r = real_ioctl(drm_fd, DRM_IOCTL_SYNCOBJ_WAIT, &w);
  for (int i = 0; i < nbinds; i++) dump_bind(NULL, cur_dir, i, 0, ".after");
  char path[512]; snprintf(path, sizeof path, "%s/after.json", cur_dir);
  FILE *m = fopen(path, "w"); if (m) { fprintf(m, "{\"wait\":%d}\n", r); fclose(m); }
}

int ioctl(int fd, unsigned long req, ...) {
  va_list ap; va_start(ap, req); void *arg = va_arg(ap, void *); va_end(ap);
  if (!real_ioctl) real_ioctl = dlsym(RTLD_NEXT, "ioctl");
  if (req == DRM_IOCTL_ASAHI_SUBMIT) { drm_fd = fd; on_submit(arg); }
  int r = real_ioctl(fd, req, arg);
  if (r) return r;
  if (req == DRM_IOCTL_ASAHI_SUBMIT) after_submit(arg);
  else if (req == DRM_IOCTL_ASAHI_GEM_CREATE && nbos < MAXB) {
    struct drm_asahi_gem_create *g = arg; bos[nbos].handle = g->handle; bos[nbos++].size = g->size; drm_fd = fd;
  } else if (req == DRM_IOCTL_GEM_CLOSE) {
    struct drm_gem_close *g = arg;
    for (int k = 0; k < nbos; k++) if (bos[k].handle == g->handle) { bos[k] = bos[--nbos]; break; }
  } else if (req == DRM_IOCTL_ASAHI_VM_BIND) {
    struct drm_asahi_vm_bind *b = arg;
    for (unsigned i = 0; i < b->num_binds; i++) {
      struct drm_asahi_gem_bind_op *op = (void *)((char *)(uintptr_t)b->userptr + i * b->stride);
      for (int k = 0; k < nbinds; k++)
        if (binds[k].addr < op->addr + op->range && op->addr < binds[k].addr + binds[k].range) binds[k--] = binds[--nbinds];
      if (!(op->flags & DRM_ASAHI_BIND_UNBIND) && nbinds < MAXB) {
        binds[nbinds].addr = op->addr; binds[nbinds].range = op->range; binds[nbinds].offset = op->offset;
        binds[nbinds].handle = op->handle; binds[nbinds++].flags = op->flags;
      }
    }
  } else if (req == DRM_IOCTL_ASAHI_QUEUE_CREATE) {
    struct drm_asahi_queue_create *q = arg; usc_base = q->usc_exec_base;
  }
  return r;
}
