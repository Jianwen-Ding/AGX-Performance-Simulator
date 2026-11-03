#!/bin/bash
# simerr_capture.sh CAPDIR [extra agx_dispatch args]: as simone.sh, stderr too (AGX_ISSUE_PROFILE output)
d=$1; shift; cd $d
python3 - "$@" <<'PY'
import shlex, subprocess, sys
line = [l for l in open('replay.sh') if 'agx_run}"' in l][0]
args = shlex.split(line.split('agx_run}"', 1)[1])[:-1]
out, i = [], 0
while i < len(args):
    if args[i] == '--dump': i += 2; continue
    if args[i] == '--shared': out += ['--tg-bytes', args[i + 1]]; i += 2; continue
    out.append(args[i]); i += 1
cmd = ['/scratch/agxsim-target/release/agx_dispatch'] + out + ['--regs', open('regs.txt').read().strip()] + sys.argv[1:]
r = subprocess.run(cmd, capture_output=True, text=True); sys.stdout.write(r.stdout + r.stderr)
PY
