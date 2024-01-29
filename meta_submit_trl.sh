#!/bin/bash -l

declare -a seed_list=("4" "5")
declare -a double_data_list=("kim-acl" "kim-scicite" "acl-scicite" "acl-kim" "scicite-kim" "scicite-acl")
declare -a triple_data_list=("acl-kim-scicite" "kim-acl-scicite" "scicite-acl-kim")

for seed in ${seed_list[@]}; do
for data in ${double_data_list[@]}; do
    sbatch slurm_trl.sh \
        -s ${seed} \
        -d ${data}
done
done

for seed in ${seed_list[@]}; do
for data in ${triple_data_list[@]}; do
    sbatch slurm_trl.sh \
        -s ${seed} \
        -d ${data}
done
done