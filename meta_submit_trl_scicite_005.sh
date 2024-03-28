#!/bin/bash -l

declare -a seed_list=("1" "2" "3" "4" "5")
declare -a double_data_list=("scicite_005-kim" "scicite_005-acl")
declare -a triple_data_list=("scicite_005-acl-kim")

for seed in ${seed_list[@]}; do
for data in ${double_data_list[@]}; do
    sbatch slurm_trl_scicite_005.sh \
        -s ${seed} \
        -d ${data}
done
done

for seed in ${seed_list[@]}; do
for data in ${triple_data_list[@]}; do
    sbatch slurm_trl_scicite_005.sh \
        -s ${seed} \
        -d ${data}
done
done