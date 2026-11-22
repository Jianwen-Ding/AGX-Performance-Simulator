#!/bin/bash
# qrun.sh WAITPID: after WAITPID exits, run the GPU jobs /scratch/agx-r5/q/*.sh one at a time (oldest name first),
# moving each to q/done with its log; polls for new jobs forever
while kill -0 $1 2>/dev/null; do sleep 10; done
while true; do
  j=$(ls /scratch/agx-r5/q/*.sh 2>/dev/null | head -1)
  if [ -z "$j" ]; then sleep 15; continue; fi
  b=$(basename $j .sh)-$(date +%H%M%S); mv $j /scratch/agx-r5/q/done/$b.sh
  bash /scratch/agx-r5/q/done/$b.sh > /scratch/agx-r5/q/done/$b.log 2>&1
  echo "$(date +%H:%M) $b exit $?" >> /scratch/agx-r5/q/done/qrun.log
done
