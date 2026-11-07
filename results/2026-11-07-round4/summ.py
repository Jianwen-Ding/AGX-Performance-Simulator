import sys,re,os,glob,statistics
# summ.py DIR...: median/p90/mean |err| of each DIR over the launches present in all of them
def load(d):
    r={}
    for f in sorted(glob.glob(d+'/*.txt')):
        seen={}
        for l in open(f):
            m=re.match(r'(\S+)\s+hw\s+([\d.]+) us.*err\s+([+-]\d+)%',l)
            if not m: continue
            k=(os.path.basename(f),m.group(1)); seen[k]=seen.get(k,0)+1
            r[k+(seen[k],)]=int(m.group(3))
    return r
ds=[load(d) for d in sys.argv[1:]]
keys=set(ds[0]).intersection(*ds[1:])
for d,r in zip(sys.argv[1:],ds):
    e=sorted(abs(r[k]) for k in keys)
    print(f'{d:30s} n {len(e)} median {statistics.median(e)} p90 {e[int(.9*len(e))]} mean {sum(e)/len(e):.1f} w10 {sum(x<=10 for x in e)} w25 {sum(x<=25 for x in e)}')
