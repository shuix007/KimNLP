#!/bin/bash

declare -a lm_list=("/export/scratch/zeren/KimNLP/Pre-Trained-SciBERT/checkpoint-213120/")
declare -a seed_list=("42" "3515" "4520")
declare -a mode_list=("context" "abstract" "all" "mix-abstract-all" "mix-abstract-context" "mix")

for lm in ${lm_list[@]}; do
for mode in ${mode_list[@]}; do
for seed in ${seed_list[@]}; do
    python ../new_main.py \
        --dataset=kim \
        --data_dir=../../../NewData/ \
        --workspace=../Workspaces/test_context_eval_on_two \
        --lm=${lm} \
        --mode=${mode} \
        --seed=${seed} &> release_experiment_fpscibert_${mode}_seed${seed}_log.txt
done
done
done