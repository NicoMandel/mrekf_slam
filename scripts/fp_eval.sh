#!/bin/bash

CONFFILES=$(find ../config/*.yaml -type f)
csvf="false_positive"

for cf in $CONFFILES
do
    rp=$(realpath $cf)
    for s in {2..30..1}
    do
        for sd in {5..105}    # 105
        do
            python arguments.py -d 1 -s $s --seed $sd --config $rp -c $csvf --dynamicfilter --incfilter
        done        
    done
done