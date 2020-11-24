#!/bin/bash
for seed in 123456 12345 1234 123 12;
do
    for pert_k in 0 2 4 6
    do
            outfile=./res/seed$seed\_k$pert_k.txt
            echo "python main.py --seed $seed --pert_k $pert_k --fout $outfile"
            python toy_main.py --seed $seed --pert_k $pert_k --fout $outfile
    done
done