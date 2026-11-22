cd /scratch/sim-captures
run() { (cd $1 && G=$(grep -o -- "--grid [0-9,]*" replay.sh | head -1 | cut -d' ' -f2) && python3 /scratch/agx-r5/gsweep.py . $G | awk '{print $6}'); }
for m in $MODES; do
 echo "$m: q6k $(export $m; run r5/llama-pp-fa0/311445-cap4582-*) (5526) q5k $(export $m; run r5/llama-pp-fa0/311445-cap5293-*) (3703) q51 $(export $m; run r5/llama-pp-fa0/311445-cap462-*) (946.5) q80 $(export $m; run r5/llama-pp-fa0/311445-cap4513-*) (380.7) fa $(export $m; run r5/llama-fa/*flash_attn_ext_f16ILi64ELi64ELi8ELi8*) (1636)"
done
