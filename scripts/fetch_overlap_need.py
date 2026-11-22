import re,subprocess,shlex,os,json,glob,sys
D='/save/workspace/AGX-Performance-Simulator/rust/target/release/agx_dispatch'
capdir,hwf=sys.argv[1],sys.argv[2]
hw={}
for l in open(hwf):
    m=re.match(r'(\S+)\s+hw\s+([\d.]+) us',l)
    if m: hw.setdefault(m.group(1),float(m.group(2)))
for cap in sorted(glob.glob(capdir+'/*/')):
    if not os.path.exists(cap+'replay.sh'): continue
    sh=open(cap+'replay.sh').read(); k=re.search(r'# kernel (\S+)',sh).group(1)
    if k not in hw: continue
    args=shlex.split([l for l in sh.splitlines() if 'exec' in l][0].split('agx_run}"',1)[1])[:-1]
    out=[];i=0
    while i<len(args):
        if args[i]=='--dump': i+=2; continue
        out.append(args[i] if args[i]!='--shared' else '--tg-bytes'); i+=1
    st=re.search(r'uses (\d+) bytes of stack',sh); out+= ['--stack',st.group(1)] if st else []
    regs=open(cap+'regs.txt').read().strip()
    r=subprocess.run([D]+out+['--regs',regs,'--launch-profile','runtime'],cwd=cap,capture_output=True,text=True,env=dict(os.environ,AGX_DEBUG_TIME='1',AGX_THREADS='3'))
    t=[l for l in r.stderr.splitlines() if l.startswith('time:')]
    if not t: print('ERR',k[:40],r.stdout[-100:]); continue
    j=json.loads(r.stdout.strip().splitlines()[-1])
    core,fetch=[float(x) for x in re.search(r'core (\d+) fetch (\d+)',t[-1]).groups()]
    rest=j['time_us']*1000-max(core,fetch)-0.3*min(core,fetch)
    a=(hw[k]*1000-rest-max(core,fetch))/max(min(core,fetch),1)
    print(f"{k[:40]:40s} hw {hw[k]:7.1f} sim {j['time_us']:7.1f} core {core/1e3:6.1f} fetch {fetch/1e3:5.1f} lines {j['code_lines']:4d} gpw {j['groups_per_wg']:2d} wgs/core {j['busiest_core_wgs']:4d} waves {j['waves']:3d} need a={a:+.2f}",flush=True)
