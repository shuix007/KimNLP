#!/bin/bash

for SEED in 42
do
    for main_dataset in "kim" "scicite" "acl" 
    do
        for lm in "bert" "scibert" 
        do
            python ../main.py \
            --dataset=${main_dataset} \
            --data_dir=../../../Data \
            --workspace=../Workspaces/baseline${main_dataset}_${lm}_seed${SEED} \
            --scheduler=slanted \
            --context_readout=cls \
            --lm=${lm} \
            --one_step \
            --seed=${SEED} \
            --inference_only
        done
    done
done

# SEED=42

# lr=2e-5

# for SEED in 42 3515 4520
# do
#     for main_dataset in "kim" "acl" "scicite"
#     do
#         for readout in "cls" "mean" "ch"
#         do
#             python ../main.py \
#                 --dataset=${main_dataset} \
#                 --data_dir=../../../Data/ \
#                 --workspace=../Workspaces/main${main_dataset}_bruteforce_${readout}_seed${SEED}_l20_10epochs \
#                 --lr_finetune=${lr} \
#                 --num_epochs_finetune=10 \
#                 --scheduler=slanted \
#                 --fuse_type=bruteforce \
#                 --context_readout=${readout} \
#                 --seed=${SEED} &> main${main_dataset}_bruteforce_${readout}_seed${SEED}_l20_10epochs_log.txt
#         done
#     done
# done



# python ../main.py \
#     --dataset=${MAIN_DATANAME} \
#     --data_dir=../../../Data/ \
#     --workspace=../Workspaces/${MAIN_DATANAME}_twostep_bruteforce_seed42 \
#     --fuse_type=bruteforce \
#     --two_step \
#     --seed=42 &> ${MAIN_DATANAME}_twostep_bruteforce_seed42_log.txt

# MAIN_DATANAME=kim

# for main_dataset in acl scicite
# do
#     python ../main.py \
#         --dataset=${MAIN_DATANAME} \
#         --data_dir=../../../Data/ \
#         --aux_datasets=${aux_dataset} \
#         --workspace=../Workspaces/main${MAIN_DATANAME}_aux${aux_dataset}_twostep_bruteforce_seed42 \
#         --fuse_type=bruteforce \
#         --two_step \
#         --seed=42 &> main${MAIN_DATANAME}_aux${aux_dataset}_twostep_bruteforce_seed42_log.txt
# done

# python ../main.py \
#     --dataset=${MAIN_DATANAME} \
#     --data_dir=../../../Data/ \
#     --workspace=../Workspaces/${MAIN_DATANAME}_twostep_bruteforce_seed42 \
#     --fuse_type=bruteforce \
#     --two_step \
#     --seed=42 &> ${MAIN_DATANAME}_twostep_bruteforce_seed42_log.txt

# MAIN_DATANAME=acl

# for main_dataset in kim scicite
# do
#     python ../main.py \
#         --dataset=${MAIN_DATANAME} \
#         --data_dir=../../../Data/ \
#         --aux_datasets=${aux_dataset} \
#         --workspace=../Workspaces/main${MAIN_DATANAME}_aux${aux_dataset}_twostep_bruteforce_seed42 \
#         --fuse_type=bruteforce \
#         --two_step \
#         --seed=42 &> main${MAIN_DATANAME}_aux${aux_dataset}_twostep_bruteforce_seed42_log.txt
# done

# python ../main.py \
#     --dataset=${MAIN_DATANAME} \
#     --data_dir=../../../Data/ \
#     --workspace=../Workspaces/${MAIN_DATANAME}_twostep_bruteforce_seed42 \
#     --fuse_type=bruteforce \
#     --two_step \
#     --seed=42 &> ${MAIN_DATANAME}_twostep_bruteforce_seed42_log.txt