#!/bin/bash

# lm=scibert

for SEED in 42 3515 4520
do
    for lm in "bert" "scibert"
    do
        for mtl in "multihead" "singlehead"
        do
            python ../main.py \
                --dataset=acl \
                --data_dir=../../../Data/ \
                --aux_datasets=kim \
                --workspace=../Workspaces/deltaentropy_${mtl}_${lm}_mainacl_auxkim_seed${SEED} \
                --lambdas=1.0,0.0372 \
                --multitask=${mtl} \
                --lm=${lm} \
                --scheduler=slanted \
                --seed=${SEED} &> deltaentropy_${mtl}_${lm}_mainacl_auxkim_seed${SEED}_log.txt

            python ../main.py \
                --dataset=kim \
                --data_dir=../../../Data/ \
                --aux_datasets=acl \
                --workspace=../Workspaces/deltaentropy_${mtl}_${lm}_mainkim_auxacl_seed${SEED} \
                --lambdas=1.0,0.1959 \
                --multitask=${mtl} \
                --lm=${lm} \
                --scheduler=slanted \
                --seed=${SEED} &> deltaentropy_${mtl}_${lm}_mainkim_auxacl_seed${SEED}_log.txt
            
            python ../main.py \
                --dataset=scicite \
                --data_dir=../../../Data/ \
                --aux_datasets=acl \
                --workspace=../Workspaces/deltaentropy_${mtl}_${lm}_mainscicite_auxacl_seed${SEED} \
                --lambdas=1.0,0.2744 \
                --multitask=${mtl} \
                --lm=${lm} \
                --scheduler=slanted \
                --seed=${SEED} &> deltaentropy_${mtl}_${lm}_mainscicite_auxacl_seed${SEED}_log.txt

            python ../main.py \
                --dataset=kim \
                --data_dir=../../../Data/ \
                --aux_datasets=scicite \
                --workspace=../Workspaces/deltaentropy_${mtl}_${lm}_mainkim_auxscicite_seed${SEED} \
                --lambdas=1.0,0.0881 \
                --multitask=${mtl} \
                --lm=${lm} \
                --scheduler=slanted \
                --seed=${SEED} &> deltaentropy_${mtl}_${lm}_mainkim_auxscicite_seed${SEED}_log.txt
        done
    done
done

# for main_dataset in "kim" "acl" "scicite"
# do
#     for aux_dataset in "kim" "acl" "scicite"
#     do
#         if [ "${main_dataset}" == "acl" ]; then
#             lr=3e-5
#         else
#             lr=5e-5
#         fi

#         if [ "${main_dataset}" != "${aux_dataset}" ]; then
#             python ../main.py \
#                 --dataset=${main_dataset} \
#                 --data_dir=../../../Data/ \
#                 --aux_datasets=${aux_dataset} \
#                 --lr_finetune=${lr} \
#                 --workspace=../Workspaces/main${main_dataset}_aux${aux_dataset}_twostep_slanted_bruteforce_seed42 \
#                 --lambdas=1.0,0.1 \
#                 --num_epochs_finetune=15 \
#                 --scheduler=slanted \
#                 --fuse_type=bruteforce \
#                 --two_step \
#                 --seed=42 &> main${main_dataset}_aux${aux_dataset}_twostep_slanted_bruteforce_seed42_log.txt
#         else
#             python ../main.py \
#                 --dataset=${main_dataset} \
#                 --data_dir=../../../Data/ \
#                 --workspace=../Workspaces/main${main_dataset}_twostep_slanted_bruteforce_seed42 \
#                 --lr_finetune=${lr} \
#                 --num_epochs_finetune=15 \
#                 --scheduler=slanted \
#                 --fuse_type=bruteforce \
#                 --two_step \
#                 --seed=42 &> main${main_dataset}_twostep_slanted_bruteforce_seed42_log.txt
#         fi
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