#!/bin/bash -l

declare -a seed_list=("1" "2" "3" "4" "5")
declare -a double_data_list=("scicite_005-kim" "scicite_005-acl")
declare -a triple_data_list=("scicite_005-acl-kim")
declare -a lambda_list=("0.2" "0.4" "0.6" "0.8" "1.0" "0.1" "0.3" "0.5" "0.7" "0.9")
declare -a even_lambda_list=("0.2" "0.4" "0.6" "0.8" "1.0")
declare -a odd_lambda_list=("0.1" "0.3" "0.5" "0.7" "0.9")

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
for lmbdas1 in ${even_lambda_list[@]}; do
for lmbdas2 in ${even_lambda_list[@]}; do
    sbatch slurm.sh \
        -d ${data} \
        -s ${seed} \
        -p "1"-${lmbdas1}-${lmbdas2}
done
done

for lmbdas1 in ${odd_lambda_list[@]}; do
for lmbdas2 in ${odd_lambda_list[@]}; do
    sbatch slurm.sh \
        -d ${data} \
        -s ${seed} \
        -p "1"-${lmbdas1}-${lmbdas2}
done
done

done
done