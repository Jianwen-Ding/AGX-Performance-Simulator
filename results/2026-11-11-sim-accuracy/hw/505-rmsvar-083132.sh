# r9: rms_norm pp with the device fence after the store (pc 1088) / the tg fence (698) / the store replaced (hw only)
export B2B=5000 REP=2 NOSIM=1
for d in nofence nostore nofence698 nofenceboth nostore-nofence; do
  echo "== $d"; cd /scratch/agx-r5/r9/rmsvar/$d && python3 /scratch/agx-r5/gsweep.py . 9728,1,1 38912,1,1 131072,1,1 --hw
done
