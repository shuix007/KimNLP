#!/bin/bash

for SEED in 3515 4520 42
do
    for readout in "mean" # "cls" 
    do
        for lm in "scibert"
        do
            for d in "kim" "acl_arc" "scicite"
            do
                python ../new_main.py \
                    --dataset=${d} \
                    --data_dir=../../../NewData \
                    --workspace=../Workspaces/without_abstract \
                    --readout=${readout} \
                    --lm=${lm} \
                    --seed=${SEED} &> wo_abstract_${d}_${lm}_${readout}_seed${SEED}_log.txt

                python ../new_main.py \
                    --dataset=${d} \
                    --data_dir=../../../NewData \
                    --workspace=../Workspaces/with_abstract \
                    --readout=${readout} \
                    --lm=${lm} \
                    --use_abstract \
                    --seed=${SEED} &> w_abstract_${d}_${lm}_${readout}_seed${SEED}_log.txt
            done
        done
    done
done