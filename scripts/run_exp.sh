#!/bin/bash

for s in {2..30..5}
do
    for d in {1..20..5}
    do    
        for sd in {1..5}
        do
            python arguments.py -d $d -s $s --seed $sd
        done        
    done
done