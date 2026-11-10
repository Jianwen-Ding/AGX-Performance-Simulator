import sys,re,glob,statistics,os
# summ.py DIR...: per set (tg tiny fa pp heldout) median / mean / p90 / within10 of |err|
for d in sys.argv[1:]:
    out=[]
    for s in ['tg','tiny','fa','pp','heldout']:
        fs=glob.glob(f'{d}/{s}/*.txt') if s=='heldout' else [f'{d}/{s}.txt']
        e=[]
        for f in fs:
            if not os.path.exists(f): continue
            for l in open(f):
                m=re.search(r' hw .* err\s+([+-]\d+)%',l)
                if m: e.append(abs(int(m.group(1))))
        if not e: out.append(f'{s} -'); continue
        e.sort()
        out.append(f'{s} n{len(e)} med {statistics.median(e):g} mean {sum(e)/len(e):.1f} p90 {e[int(0.9*len(e))]} w10 {sum(x<=10 for x in e)}')
    print(os.path.basename(d.rstrip('/'))+': '+' | '.join(out))
