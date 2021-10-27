#!/bin/bash

DATANAME=scicite

for ftype in bruteforce latefuse
do
    for rseed in 1234567890 42 60100
    do
    python ../main.py \
        --dataset=${DATANAME} \
        --data_dir=../../../Data/ \
        --workspace=../Workspaces/${DATANAME}_ctxonly_twostep_${ftype}_seed${rseed} \
        --fuse_type=${ftype} \
        --context_only \
        --two_step \
        --seed=${rseed} &> ${DATANAME}_ctxonly_twostep_${ftype}_seed${rseed}_log.txt
    done
done

for ftype in bruteforce latefuse
do
    for rseed in 1234567890 42 60100
    do
    python ../main.py \
        --dataset=${DATANAME} \
        --data_dir=../../../Data/ \
        --workspace=../Workspaces/${DATANAME}_ctxonly_onestep_${ftype}_seed${rseed} \
        --fuse_type=${ftype} \
        --context_only \
        --seed=${rseed} &> ${DATANAME}_ctxonly_onestep_${ftype}_seed${rseed}_log.txt
    done
done

for ftype in bruteforce disttrunc latefuse
do
    for rseed in 1234567890 42 60100
    do
    python ../main.py \
        --dataset=${DATANAME} \
        --data_dir=../../../Data/ \
        --workspace=../Workspaces/${DATANAME}_onestep_${ftype}_seed${rseed} \
        --fuse_type=${ftype} \
        --seed=${rseed} &> ${DATANAME}_onestep_${ftype}_seed${rseed}_log.txt
    done
done

# for ftype in bruteforce disttrunc latefuse
# do
#     for rseed in 1234567890 42 60100
#     do
#     python ../main.py \
#         --dataset=${DATANAME} \
#         --data_dir=../../../Data/ \
#         --workspace=../Workspaces/${DATANAME}_twostep_${ftype}_seed${rseed} \
#         --fuse_type=${ftype} \
#         --two_step \
#         --seed=${rseed} &> ${DATANAME}_twostep_${ftype}_seed${rseed}_log.txt
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

# python ../main.py \
#     --dataset=kim \
#     --data_dir=../../../Data/ \
#     --workspace=../Workspaces/ \
#     --fuse_type=disttrunc \
#     --seed=1209384752 &> kim_disttrunc_seed1209384752_log.txt

# python ../main.py \
#     --dataset=kim \
#     --data_dir=../../../Data/ \
#     --workspace=../Workspaces/ \
#     --fuse_type=disttrunc \
#     --seed=1209384752 \
#     --da &> kim_da_disttrunc_seed1209384752_log.txt