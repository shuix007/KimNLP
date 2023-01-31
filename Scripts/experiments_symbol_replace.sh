#!/bin/bash

declare -a seed_list=("42" "3515" "4520")
declare -a readout_list=("cls" "ch")
# declare -a readout_list=("mean")
declare -a data_list=("kim" "kim_symbol_replaced")
declare -a lm_list=("allenai/scibert_scivocab_uncased" "bert-base-uncased")

for SEED in ${seed_list[@]}; do
for readout in ${readout_list[@]}; do
for lm in ${lm_list[@]}; do
for data in ${data_list[@]}; do
    if [ "${lm}" == "allenai/scibert_scivocab_uncased" ]; then
        lm_name=scibert
    else
        lm_name=bert
    fi

    python ../main_fpretrain.py \
        --dataset=${data} \
        --data_dir=../../../Data \
        --workspace=../Workspaces/kim_finetune \
        --scheduler=slanted \
        --context_readout=${readout} \
        --lm=${lm} \
        --one_step \
        --seed=${SEED} &> finetune_${data}_${lm_name}_${readout}_seed${SEED}_log.txt
done
done
done
done