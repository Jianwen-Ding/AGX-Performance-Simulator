import sys,subprocess
# dis.py CODE.bin [stores]: pc size text per instruction
c=open(sys.argv[1],'rb').read().hex()
out=subprocess.run(['/save/workspace/AGX-Performance-Simulator/rust/target/release/agx_asm'],input='!'+c+'\n',text=True,capture_output=True).stdout
pc=0
for r in out.split('|'):
    r=r.strip()
    if not r: continue
    p=r.split(None,2)
    try: sz=int(p[1])
    except: break
    t=p[2] if len(p)>2 else p[0]
    if len(sys.argv)<3 or ('store' in p[0] and 'threadgroup' not in p[0]): print(pc, sz, t)
    pc+=sz
