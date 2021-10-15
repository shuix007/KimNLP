#!/bin/bash

# for ftype in bruteforce disttrunc latefuse
# do
#     for rseed in 1209384752 42 3515
#     do
#     python ../main.py \
#         --dataset=kim \
#         --data_dir=../../../Data/ \
#         --workspace=../Workspaces/ \
#         --fuse_type=${ftype} \
#         --seed=${rseed} \
#         --da &> kim_da_${ftype}_seed${rseed}_log.txt
#     done
# done

# python ../main.py \
#     --dataset=kim \
#     --data_dir=../../../Data/ \
#     --workspace=../Workspaces/ \
#     --fuse_type=disttrunc \
#     --seed=42 &> kim_disttrunc_seed42_log.txt

# for rseed in 42 3515
# do
# python ../main.py \
#     --dataset=kim \
#     --data_dir=../../../Data/ \
#     --workspace=../Workspaces/ \
#     --fuse_type=disttrunc \
#     --seed=${rseed} \
#     --da &> kim_da_disttrunc_seed${rseed}_log.txt
# done

python ../main.py \
    --dataset=kim \
    --data_dir=../../../Data/ \
    --workspace=../Workspaces/ \
    --fuse_type=disttrunc \
    --seed=1209384752 &> kim_disttrunc_seed1209384752_log.txt

python ../main.py \
    --dataset=kim \
    --data_dir=../../../Data/ \
    --workspace=../Workspaces/ \
    --fuse_type=disttrunc \
    --seed=1209384752 \
    --da &> kim_da_disttrunc_seed1209384752_log.txt