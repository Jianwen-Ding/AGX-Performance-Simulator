#!/bin/bash
# worst.sh DIR [N]: largest |err| launches
grep -H " hw " $1/*.txt | awk '{e=$0; sub(/.*err +/,"",e); sub(/%.*/,"",e); e=e+0; x=e<0?-e:e; print x, $0}' | sort -rn | head -${2:-25} | sed 's|^\([0-9]*\) [^:]*/|\1 |' | cut -c1-210
