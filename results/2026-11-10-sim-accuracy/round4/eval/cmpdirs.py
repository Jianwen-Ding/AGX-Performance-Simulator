import sys,re,glob,os
# cmpdirs.py A B [thr]: launches whose err differs by >= thr points (default 2) between two eval dirs
a,b=sys.argv[1],sys.argv[2]; thr=int(sys.argv[3]) if len(sys.argv)>3 else 2
def load(d):
    r={}
    fs=glob.glob(d+'/heldout/*.txt')+[d+f'/{s}.txt' for s in ['tg','tiny','fa','pp']]
    for f in fs:
        if not os.path.exists(f): continue
        for i,l in enumerate(open(f)):
            m=re.search(r'^(\S+)\s+hw\s+([\d.]+).*sim\s+([\d.]+) us\s+err\s+([+-]\d+)%',l)
            if m: r[(os.path.basename(f),m.group(1)[:60],i)]=(float(m.group(2)),float(m.group(3)),int(m.group(4)))
    return r
A,B=load(a),load(b)
for k in sorted(set(A)|set(B)):
    x,y=A.get(k),B.get(k)
    if x is None or y is None: print('MISSING',k,x,y); continue
    if abs(x[2]-y[2])>=thr: print(f'{k[0]:22s} {k[1][:50]:50s} hw {x[0]:9.1f} {x[1]:9.1f} ({x[2]:+d}%) -> {y[1]:9.1f} ({y[2]:+d}%)')
