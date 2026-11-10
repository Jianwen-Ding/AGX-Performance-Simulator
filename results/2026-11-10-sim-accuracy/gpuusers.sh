#!/bin/bash
# gpuusers.sh: PIDs (and commands) that have the GPU render node open, excluding the given PIDs ($@)
for p in /proc/[0-9]*; do
  pid=${p#/proc/}
  case " $* " in *" $pid "*) continue;; esac
  ls -l $p/fd 2>/dev/null | grep -q renderD128 && echo "$pid $(tr '\0' ' ' < $p/cmdline | cut -c1-100)"
done
