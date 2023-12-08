#!/bin/bash

# for s in {2..30..5}
# do
#     for sd in {5..25}
#     do
#         python arguments.py -d 0 -s $s --seed $sd
#     done        
# done

# for s in {2..20..2}
# do
#     for sd in {5..25}
#     do
#         python arguments.py -d 0 -s $s --seed $sd
#     done        
# done

for s in {2..20..1}
do
    for sd in {5..105}
    do
        python arguments.py -d 0 -s $s --seed $sd
    done        
done