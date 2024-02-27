#!/bin/bash
csvf="debug_true_20"
configf="../config/all.yaml"
k=5
db=true

for s in {2..4}              # 30
do
    for d in {1..2}          # 5                          
    do    
        for sd in {5..11}       # 25
        do
            if $db; then
                if ((sd % k == 0)); then
                debug="--debug"
                else
                debug=""
                fi
            fi
            python arguments.py -d $d -s $s --seed $sd --config $configf -c $csvf $debug --true
        done        
    done
done
