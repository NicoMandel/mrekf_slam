#!/bin/bash

csvf="false_negative"

for s in {2..30..1}
do
    for d in {1..5..1}                          
    do    
        for sd in {5..105}    # 105
        do
            python arguments.py -d $d -s $s --seed $sd -c $csvf --fpfilter --dynamicfilter
        done        
    done
done
