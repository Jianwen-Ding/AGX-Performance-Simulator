import os,sys,json,subprocess
sys.path.insert(0,'scripts'); os.environ.setdefault('K','64')
import importlib.util
spec=importlib.util.spec_from_file_location('pfo','scripts/probe_fetch_overlap_groups.py')
src=open('scripts/probe_fetch_overlap_groups.py').read().split('\nfor f in')[0]
ns={'__file__':os.path.abspath('scripts/probe_fetch_overlap_groups.py')}; exec(compile(src,'pfo','exec'),ns)
hw={(r['F'],r['G']):r['hw_us'] for r in map(json.loads,open('/scratch/agx-r7/fov/sweep1.jsonl'))}
envs=[dict(x.split('=') for x in a.split(',')) if a!='-' else {} for a in sys.argv[1:]]
for (f,g),h in sorted(hw.items()):
    if f<48: continue
    p='/scratch/agx-r7/fov/k%d.bin'%f; open(p,'wb').write(ns['program'](f))
    c=[ns['DISP'],p,'--grid',f'{32*g},1,1','--wg',f'{32*g},1,1','--regs','64','--alloc','0x100000=1048576','--uniform','0=0x100000','--uniform','1=0','--launch-profile','runtime']
    out=[]
    for e in envs:
        r=subprocess.run(c,capture_output=True,text=True,env=dict(os.environ,**e)); t=json.loads(r.stdout.strip().splitlines()[-1])['time_us']; out.append(f'{t:6.1f} ({100*(t-h)/h:+.0f}%)')
    print(f'F {f:3d} G {g} hw {h:6.1f}  '+'  '.join(out))
