import sys,re,os,glob
# cmp.py DIR1 DIR2 ...: per-launch errors side by side
dirs=sys.argv[1:]
rows={}
for i,d in enumerate(dirs):
    for f in sorted(glob.glob(d+'/*.txt')):
        seen={}
        for l in open(f):
            m=re.match(r'(\S+)\s+hw\s+([\d.]+) us.*sim\s+([\d.]+) us\s+err\s+([+-]\d+)%',l)
            if not m: continue
            k=(os.path.basename(f)[:-4],m.group(1)); seen[k]=seen.get(k,0)+1
            k=k+(seen[k],)
            rows.setdefault(k,[m.group(2)]+['']*len(dirs))[i+1]=m.group(4)
for k,v in rows.items():
    print(f'{k[0][:16]:16s} {k[1][:38]:38s} {float(v[0]):9.1f} '+' '.join(f'{x:>5s}' for x in v[1:]))
