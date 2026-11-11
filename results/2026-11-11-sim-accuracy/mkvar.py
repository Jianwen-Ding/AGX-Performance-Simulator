import os,sys
# mkvar.py CAPDIR OUTDIR PC:HEX...: copy of a capture with the code patched (memory symlinked)
cap,out=sys.argv[1:3]; os.makedirs(out,exist_ok=True)
for f in os.listdir(cap):
    if f.startswith('m') and f.endswith('.bin') and not os.path.exists(f'{out}/{f}'): os.symlink(f'{cap}/{f}',f'{out}/{f}')
for f in ('replay.sh','regs.txt','command.txt','regions.txt'):
    if os.path.exists(f'{cap}/{f}'): open(f'{out}/{f}','w').write(open(f'{cap}/{f}').read())
c=bytearray(open(cap+'/code.bin','rb').read())
for a in sys.argv[3:]:
    pc,h=a.split(':'); b=bytes.fromhex(h); c[int(pc):int(pc)+len(b)]=b
open(out+'/code.bin','wb').write(c); print(out)
