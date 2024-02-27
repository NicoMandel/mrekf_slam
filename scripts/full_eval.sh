#!/bin/bash

csvf="all_ate_short"
rp="../config/all.yaml"
# debug setting
k=7
debug=true

for s in {2..30..1}
do
    for d in {1..5..1}                          
    do 
        for sd in {5..25}    # 105
        do
        if $debug; then
            if ((sd % k == 0)); then
                db="--debug"
                else
                db=""
            fi
        fi
            python arguments.py -d $d -s $s --seed $sd --config $rp -c $csvf $db --true
        done        
    done
done