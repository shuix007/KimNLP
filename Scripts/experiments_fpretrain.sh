#!/bin/bash

base_lm='allenai/scibert_scivocab_uncased'
# base_lm=bert-base-uncased

# for SEED in 42 3515 4520
# do
#     for readout in "cls" "ch" "mean" 
#     do
#         for lm in "127872" "170496" "21312" "42624" "85248" "106560" "149184" "191808" "213120" "63936"
#         do
#             python ../main_fpretrain.py \
#             --dataset=kimonesentence \
#             --data_dir=../../../Data \
#             --workspace=../Workspaces/kim_fpretrain \
#             --scheduler=slanted \
#             --context_readout=${readout} \
#             --lm=/export/scratch/zeren/KimNLP/Pre-Trained-BERT/checkpoint-${lm} \
#             --one_step \
#             --seed=${SEED} &> fpretrain_kimonesentence_bert_${lm}_${readout}_seed${SEED}_log.txt
#         done

#         python ../main_fpretrain.py \
#             --dataset=kimonesentence \
#             --data_dir=../../../Data \
#             --workspace=../Workspaces/kim_fpretrain \
#             --scheduler=slanted \
#             --context_readout=${readout} \
#             --lm=${base_lm} \
#             --one_step \
#             --seed=${SEED} &> fpretrain_kimonesentence_bert_${readout}_seed${SEED}_log.txt
#     done
# done

for SEED in 3515
do
    for readout in "mean" 
    do
        for lm in "213120"
        do
            python ../main_fpretrain.py \
            --dataset=kim \
            --data_dir=../../../Data \
            --workspace=../Workspaces/kim_fpretrain \
            --scheduler=slanted \
            --context_readout=${readout} \
            --lm=/export/scratch/zeren/KimNLP/Pre-Trained-SciBERT/checkpoint-${lm} \
            --one_step \
            --seed=${SEED} &> fpretrain_scibert_${lm}_${readout}_seed${SEED}_log.txt
        done
    done
done