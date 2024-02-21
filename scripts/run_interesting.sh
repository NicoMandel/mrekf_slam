#!/bin/bash
csvf="debug_2_true_vals"
configf="../config/all.yaml"
s=2
for d in {1..5..1}                          
do    
    for sd in {5..25}
    do
        python arguments.py -d $d -s $s --seed $sd --config $configf -c $csvf --debug --true #--datmo
    done        
done
