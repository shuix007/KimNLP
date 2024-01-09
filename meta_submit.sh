#!/bin/bash -l

declare -a seed_list=("1" "2" "3")
declare -a double_data_list=("kim-acl" "kim-scicite" "acl-scicite" "acl-kim" "scicite-kim" "scicite-acl")
declare -a triple_data_list=("acl-kim-scicite" "kim-acl-scicite" "scicite-acl-kim")
declare -a lambda_list=("0.2" "0.4" "0.6" "0.8" "1.0")

for seed in ${seed_list[@]}; do
for data in ${double_data_list[@]}; do
for lmbdas in ${lambda_list[@]}; do
    sbatch slurm.sh \
        -d ${data} \
        -s ${seed} \
        -p "1"-${lmbdas}
done
done
done

for seed in ${seed_list[@]}; do
for data in ${triple_data_list[@]}; do
for lmbdas1 in ${lambda_list[@]}; do
for lmbdas2 in ${lambda_list[@]}; do
    sbatch slurm.sh \
        -d ${data} \
        -s ${seed} \
        -p "1"-${lmbdas1}-${lmbdas2}
done
done
done
done