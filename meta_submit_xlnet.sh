#!/bin/bash -l

declare -a seed_list=("1" "2" "3" "4" "5")
declare -a data_list=("scicite" "kim" "acl")

for seed in ${seed_list[@]}; do
for data in ${data_list[@]}; do
    sbatch slurm_xlnet.sh \
        -d ${data} \
        -s ${seed}
done
done