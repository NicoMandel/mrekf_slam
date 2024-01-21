#!/bin/bash
csvf="all_models_20"
configf="../config/all.yaml"

for s in {2..30..1}
do
    for d in {1..5..1}                          
    do    
        for sd in {5..25}
        do
            python arguments.py -d $d -s $s --seed $sd --config $configf -c $csvf
        done        
    done
done