#!/bin/bash -l

declare -a seed_list=("1" "2" "3" "4" "5")
declare -a data_list=("scicite_010")
declare -a readout_list=("cls" "ch")

for seed in ${seed_list[@]}; do
for data in ${data_list[@]}; do
for readout in ${readout_list[@]}; do
    sbatch slurm_readout.sh \
        -d ${data} \
        -s ${seed} \
        -p ${readout}
done
done
done