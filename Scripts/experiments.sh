#!/bin/bash

for ftype in bruteforce disttrunc latefuse
do
    for rseed in 1209384752 42 3515
    do
    python main.py \
        --dataset=kim \
        --data_dir=../../../Data/ \
        --workspace=../Workspaces/ \
        --fuse_type=${ftype} \
        --seed=${rseed} \
        --da &> kim_da_${ftype}_seed${rseed}_log.txt
    done
done