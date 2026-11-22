import sys,shlex,subprocess,os,re,json,statistics,time
# gsweep.py CAPDIR GRID... [--hw]: hw (agx_run_hw b2b, idle-guarded) and sim times of one capture at other grid sizes
args=sys.argv[1:]; hw='--hw' in args; args=[a for a in args if a!='--hw']
p=args[0]; grids=args[1:]
sh=open(p+'/replay.sh').read()
line=[l for l in sh.splitlines() if 'exec' in l][0]
a=shlex.split(line.split('agx_run}"',1)[1])[:-1]; base=[];i=0
while i<len(a):
    if a[i]=='--dump': i+=2; continue
    base.append(a[i]); i+=1
regs=os.environ.get('REGS') or open(p+'/regs.txt').read().strip()
st=re.search(r'the kernel uses (\d+) bytes of stack per thread',sh)
class _S:
    def __init__(s,v): s.v=v
    def group(s,i): return s.v
if os.environ.get('STACK'): st=_S(os.environ['STACK'])
def with_grid(g):
    o=list(base); o[o.index('--grid')+1]=g; return o
B2B=os.environ.get('B2B','1000'); REP=int(os.environ.get('REP','3'))
HWBIN=os.environ.get('HWBIN','/scratch/agx-r5/zcopy/target/release/agx_run_hw')
def idle(): return not subprocess.run('/scratch/agx-r5/gpuusers.sh',shell=True,capture_output=True,text=True).stdout.strip()
for g in grids:
    o=with_grid(g)
    sim=[x if x!='--shared' else '--tg-bytes' for x in o]
    cmd=['/scratch/agx-sim-build/release/agx_dispatch']+sim+['--regs',regs,'--launch-profile','runtime']+(['--stack',st.group(1)] if st else [])
    r=subprocess.run(['true'] if os.environ.get('NOSIM') else cmd,cwd=p,text=True,capture_output=True,env={'AGX_EXPERIMENTAL_RET':'buffer-pop2',**os.environ})
    try: s=json.loads(r.stdout.strip().splitlines()[-1]); sim_us=s['time_us']
    except Exception: sim_us=float('nan'); s={}
    if os.environ.get("PRINTJ"): print(json.dumps(s)[:900]); sys.stderr.write(r.stderr if os.environ.get("PRINTJ")=="2" else r.stderr[-3000:])
    h=float('nan'); runs=[]
    if hw:
        hc=[HWBIN]+o+['--regs',regs,'--back-to-back',B2B,'--timeout-ms','10000']+(['--stack',st.group(1)] if st else [])
        for _ in range(REP):
            for _t in range(50):
                while not idle(): time.sleep(5)
                hr=subprocess.run(hc,cwd=p,text=True,capture_output=True)
                if idle(): break
            try: runs.append(json.loads(hr.stdout.strip().splitlines()[-1])['b2b_median_fw_ns']/1e3)
            except Exception: print('HW FAIL',hr.stdout[-200:],hr.stderr[-200:]); break
        if runs: h=statistics.median(runs)
    print(f'grid {g:>14s} hw {h:9.1f} sim {sim_us:9.1f} ratio {sim_us/h if h==h else float("nan"):.2f} waves {s.get("waves")} wgs/core {s.get("busiest_core_wgs")} runs {runs}',flush=True)
