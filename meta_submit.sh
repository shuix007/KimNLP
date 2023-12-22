#!/bin/bash -l

declare -a seed_list=("1" "2" "3")
declare -a data_list=("kim" "acl" "scicite" "kim-acl" "kim-scicite" "acl-scicite" "acl-kim" "scicite-kim" "scicite-acl" "acl-kim-scicite" "kim-acl-scicite" "scicite-acl-kim")

for seed in ${seed_list[@]}; do
for data in ${data_list[@]}; do
    sbatch slurm.sh -d ${data} -s ${seed} -p CL
done
done