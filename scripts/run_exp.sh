#!/bin/bash
csvf="datmo_baseline_interesting"
configf="../config/all.yaml"

# General Hypothesis - DATMO is ALWAYS worse on other robots ATE!

# case A - 5 dynamic, 2 static and 3 static
# H1 - hypothesis:
#   * DATMo worse than KIS on self localisations -> additional information for localisation
for s in {2..3}              
do    
    for sd in {1..2}       
    do
        python arguments.py -d 5 -s $s --seed $sd --config $configf -c $csvf --debug --true --incfilter --fpfilter
    done        
done

# case B - 1 / 2 d, 20 static
# H1 - hypothesis:
#   * DATMOs own ATE ist sehr gut
#   * own ATE is LESS bad than in the case before   -> this might be dependent on the motion model
for d in {1..2}              
do    
    for sd in {1..2}       
    do
        python arguments.py -d $d -s 20 --seed $sd --config $configf -c $csvf --debug --true --incfilter --fpfilter
    done        
done

# case C - 10 d, 20 s 
# H1 - hypothesis
for sd in {1..2}       
do
    python arguments.py -d $d -s 20 --seed $sd --config $configf -c $csvf --debug --true --incfilter --fpfilter
done        
