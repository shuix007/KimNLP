#!/bin/bash

# SEED=42

for SEED in 42 3515 4520
do
    for main_dataset in "kim" "acl" "scicite"
    do
        for aux_dataset in "kim" "acl" "scicite"
        do
            if [ "${main_dataset}" == "acl" ]; then
                lr=2e-5
            else
                lr=2e-5
            fi

            if [ "${main_dataset}" != "${aux_dataset}" ]; then
                python ../main.py \
                    --dataset=${main_dataset} \
                    --data_dir=../../../Data/ \
                    --aux_datasets=${aux_dataset} \
                    --lr_finetune=${lr} \
                    --workspace=../Workspaces/main${main_dataset}_aux${aux_dataset}_bruteforce_seed${SEED} \
                    --lambdas=1.0,0.1 \
                    --num_epochs_finetune=15 \
                    --scheduler=slanted \
                    --fuse_type=bruteforce \
                    --seed=${SEED} &> main${main_dataset}_aux${aux_dataset}_bruteforce_seed${SEED}_log.txt
            else
                python ../main.py \
                    --dataset=${main_dataset} \
                    --data_dir=../../../Data/ \
                    --workspace=../Workspaces/main${main_dataset}_bruteforce_seed${SEED} \
                    --lr_finetune=${lr} \
                    --num_epochs_finetune=15 \
                    --scheduler=slanted \
                    --fuse_type=bruteforce \
                    --seed=${SEED} &> main${main_dataset}_bruteforce_seed${SEED}_log.txt
            fi
        done
    done
done
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