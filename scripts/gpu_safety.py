#!/usr/bin/env python3
"""Per-boot GPU timeout budget shared by the hardware runners.

Recovered GPU timeouts accumulate: after about 44 in one boot the device wedged
(notes/failed-approaches.md, 2026-10-11) and needed a reboot. Runners record
every timeout or fault here, keyed by the kernel boot id, and refuse to submit
untested encodings once the boot has used LIMIT recoveries.

    import gpu_safety
    gpu_safety.require_budget()          # before a batch that may hang
    gpu_safety.record('funop op=2')      # after each timeout/fault

CLI: python3 scripts/gpu_safety.py [status|reset-check]
"""
import json, os, sys, time
from pathlib import Path

LIMIT = int(os.environ.get('AGX_HANG_BUDGET', '30'))
DIR = Path(os.environ.get('AGX_HANG_DIR', '/scratch/apple-m2-playground/hang-budget'))


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def _path():
    return DIR/f'{boot_id()}.json'


def used():
    p = _path()
    return json.loads(p.read_text())['events'] if p.exists() else []


def record(what):
    DIR.mkdir(parents=True, exist_ok=True)
    ev = used() + [{'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'what': what}]
    _path().write_text(json.dumps({'boot_id': boot_id(), 'limit': LIMIT, 'events': ev}, indent=1)+'\n')
    return len(ev)


def require_budget(needed=1):
    n = len(used())
    if n + needed > LIMIT:
        sys.exit(f'refusing: this boot already used {n} of {LIMIT} GPU timeout recoveries '
                 f'(the device wedged after ~44). Reboot first, or raise AGX_HANG_BUDGET knowingly.')
    return LIMIT - n


if __name__ == '__main__':
    ev = used()
    print(json.dumps({'boot_id': boot_id(), 'used': len(ev), 'limit': LIMIT, 'last': ev[-3:]}, indent=1))
