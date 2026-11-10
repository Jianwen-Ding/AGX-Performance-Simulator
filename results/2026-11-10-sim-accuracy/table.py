import sys,re,os,glob,statistics
# table.py BEFORE AFTER [--md]: per-launch hw / sim before / sim after / ratio (dirs of capcheck outputs or single files)
def load(p):
    fs=sorted(glob.glob(p+'/*.txt')) if os.path.isdir(p) else [p]
    r={}
    for f in fs:
        seen={}
        for l in open(f):
            m=re.match(r'(\S+)\s+hw\s+([\d.]+) us.*sim\s+([\d.]+) us',l)
            if not m: continue
            k=(os.path.basename(f)[:-4] if os.path.isdir(p) else '', m.group(1)); seen[k]=seen.get(k,0)+1
            r[k+(seen[k],)]=(float(m.group(2)),float(m.group(3)))
    return r
a,b=load(sys.argv[1]),load(sys.argv[2])
ks=[k for k in b if k in a]
def name(k):
    t=re.search(r'ggml_type(\d+)',k[1]); n=re.sub(r'^_Z[LN]?\d+','',k[1]); n=re.sub(r'(ILi|IL9|IL18|EvPK|Pf|Ev|ii|IXad).*','',n)
    n=n+(f' t{t.group(1)}' if t else '')
    return (k[0][2:]+':' if k[0] else '')+n[:34]
print('| kernel | hw us | sim before | ratio | sim after | ratio |'); print('|---|---|---|---|---|---|')
for k in ks:
    hw,sa=a[k]; _,sb=b[k]
    print(f'| {name(k)} | {hw:.1f} | {sa:.1f} | {sa/hw:.2f} | {sb:.1f} | {sb/hw:.2f} |')
def st(d,ks): 
    e=sorted(abs(d[k][1]/d[k][0]-1)*100 for k in ks); return f'median {statistics.median(e):.0f}%, p90 {e[int(.9*len(e))]:.0f}%, mean {sum(e)/len(e):.1f}%, within 10% {sum(x<=10 for x in e)}/{len(e)}'
print(f'\nbefore: {st(a,ks)}\nafter:  {st(b,ks)}')
