#!/bin/bash

for SEED in 42 3515 4520
do
    for dataset in "acl" "kim" "scicite"
    do
        for readout in "mean" "cls" "ch" 
        do
            python ../new_main.py \
                --dataset=${dataset} \
                --data_dir=../../../Data/ \
                --workspace=../Workspaces/${dataset}_scibert_${readout}_seed${SEED} \
                --readout=${readout} \
                --lm=scibert \
                --lr=2e-5 \
                --seed=${SEED} &> ${dataset}_scibert_${readout}_seed${SEED}_lr2e-5_log.txt

            python ../new_main.py \
                --dataset=${dataset} \
                --data_dir=../../../Data/ \
                --workspace=../Workspaces/${dataset}_bert_${readout}_seed${SEED} \
                --readout=${readout} \
                --lm=bert \
                --lr=2e-5 \
                --seed=${SEED} &> ${dataset}_bert_${readout}_seed${SEED}_lr2e-5_log.txt
        done
    done
done