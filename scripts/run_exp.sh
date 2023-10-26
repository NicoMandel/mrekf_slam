#!/bin/bash

for i in {1..20..5}
do
    for j in a b c d
        do
        dt=$(date '+%Y%m%d_%H:%M:%S');
        echo "${dt},${i},${j}" >> ../results/experiments.csv
    done
done