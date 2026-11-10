import sys,os,shlex,collections
# trunc.py CAPDIR TRACE OUTROOT IDX...: copies of a capture whose code stops at the pc group 0 reaches at trace index IDX
# (a pc executed once only; the next such one after IDX). Memory files are symlinked.
cap,trace,root=sys.argv[1:4]; idxs=[int(x) for x in sys.argv[4:]]
rows=[l.split() for l in open(trace)]
pcs=[int(r[0]) for r in rows]; cnt=collections.Counter(pcs)
code=bytearray(open(os.path.join(cap,'code.bin'),'rb').read())
for idx in idxs:
    j=idx
    first={}
    for k,p in enumerate(pcs): first.setdefault(p,k)
    # (a pc first reached here, by falling through from the previous instruction, an ALU instruction: a stop at a
    # pop_exec reached by jmp_exec_none faulted on hardware, Bad address)
    bad=('stop','pop_exec','else_fcmp','else_icmp','if_icmp','if_fcmp','jmp_exec_none','jmp_exec_any','while_icmp','while_fcmp','wait')
    while first[pcs[j]]!=j or rows[j][1] in bad or not (0 < pcs[j]-pcs[j-1] <= 16) or rows[j-1][1] in bad: j+=1
    pc=pcs[j]
    out=os.path.join(root,f'{os.path.basename(cap)[:40]}-cut{j}')
    os.makedirs(out,exist_ok=True)
    for f in os.listdir(cap):
        if f.startswith('m') and f.endswith('.bin') and not os.path.exists(os.path.join(out,f)):
            os.symlink(os.path.join(cap,f),os.path.join(out,f))
    for f in ('replay.sh','regs.txt','command.txt'):
        open(os.path.join(out,f),'w').write(open(os.path.join(cap,f)).read())
    c=bytearray(code); c[pc]=0x88; c[pc+1]=0x00
    open(os.path.join(out,'code.bin'),'wb').write(c)
    print(out, 'pc', pc, 'trace index', j, 'of', len(pcs), rows[j][1])
