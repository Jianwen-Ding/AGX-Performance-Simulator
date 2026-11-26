/* Sizes and field offsets of the uAPI structures src/uapi.rs mirrors (tests/layout.rs compares them). */
#include <drm.h>
#include <stddef.h>
#include <stdio.h>
#include "drm/asahi_drm.h"
#define S(t) printf("%s %zu\n", #t, sizeof(struct t))
#define O(t, f) printf("%s.%s %zu\n", #t, #f, offsetof(struct t, f))
int main(void)
{
    S(drm_version); O(drm_version, name); O(drm_version, desc);
    S(drm_gem_close); S(drm_get_cap);
    S(drm_syncobj_create); S(drm_syncobj_destroy); S(drm_syncobj_array); S(drm_syncobj_timeline_array);
    S(drm_syncobj_transfer); O(drm_syncobj_transfer, src_point);
    O(drm_syncobj_wait, timeout_nsec); O(drm_syncobj_wait, first_signaled);
    O(drm_syncobj_timeline_wait, timeout_nsec); O(drm_syncobj_timeline_wait, first_signaled);
    S(drm_asahi_params_global); O(drm_asahi_params_global, core_masks); O(drm_asahi_params_global, vm_start);
    O(drm_asahi_params_global, max_commands_per_submission); O(drm_asahi_params_global, command_timestamp_frequency_hz);
    S(drm_asahi_get_params); S(drm_asahi_vm_create); S(drm_asahi_vm_destroy); S(drm_asahi_gem_create);
    O(drm_asahi_gem_create, handle); S(drm_asahi_gem_mmap_offset); S(drm_asahi_gem_bind_op); O(drm_asahi_gem_bind_op, addr);
    S(drm_asahi_vm_bind); O(drm_asahi_vm_bind, userptr); S(drm_asahi_gem_bind_object); O(drm_asahi_gem_bind_object, object_handle);
    S(drm_asahi_queue_create); O(drm_asahi_queue_create, usc_exec_base); S(drm_asahi_queue_destroy); S(drm_asahi_sync);
    S(drm_asahi_cmd_header); S(drm_asahi_submit); O(drm_asahi_submit, cmdbuf_size);
    S(drm_asahi_cmd_compute); O(drm_asahi_cmd_compute, cdm_ctrl_stream_base); O(drm_asahi_cmd_compute, helper); O(drm_asahi_cmd_compute, ts);
    S(drm_asahi_get_time);
    return 0;
}
