#!/bin/bash -l

#SBATCH -p gk
#SBATCH --gres=gpu:1

#SBATCH --output=result-%j.out
#SBATCH --error=result-%j.err

#SBATCH --time=1:00:00

#SBATCH --ntasks=2

#SBATCH --mem=40G

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=shuix007@umn.edu

while getopts s:d: flag
do
    case "${flag}" in
        s) seed=${OPTARG};;
        d) data=${OPTARG};;
    esac
done

conda activate KimNLP

cd /home/karypisg/shuix007/KimNLP/KimNLP

# TRL on training set
if [[ ${data} == "scicite_005-kim" ]]; then
    clslmbdas="1-0.0473"
    chlmbdas="1-0.1016"
elif [[ ${data} == "scicite_005-acl" ]]; then
    clslmbdas="1-0.0365"
    chlmbdas="1-0.0964"
elif [[ ${data} == "scicite_020-kim" ]]; then
    clslmbdas="1-0.0678"
    chlmbdas="1-0.0794"
elif [[ ${data} == "scicite_020-acl" ]]; then
    clslmbdas="1-0.0301"
    chlmbdas="1-0.0636"
else
    echo "Warning: Unrecognized value of data: ${data}"
fi

if [[ ${data} == "scicite_005-acl-kim" ]]; then
    clslmbdas="1-0.0365-0.0473"
    chlmbdas="1-0.0964-0.1016"
elif [[ ${data} == "scicite_020-acl-kim" ]]; then
    clslmbdas="1-0.0301-0.0678"
    chlmbdas="1-0.0636-0.0794"
else
    echo "Warning: Unrecognized value of data: ${data}"
fi

CURRENTEPOCTIME=`date +"%Y-%m-%d-%H-%M-%S"`
# Run the PyTorch training script
python main.py \
    --dataset=${data} \
    --lambdas=${clslmbdas} \
    --data_dir=Data \
    --lm=scibert \
    --readout=cls \
    --workspace=Workspaces/${CURRENTEPOCTIME}-${data}-${seed}-${clslmbdas}-scibert-cls \
    --seed=${seed} &> ${CURRENTEPOCTIME}-${data}-${seed}-${clslmbdas}-scibert-cls-log.txt

python main.py \
    --dataset=${data} \
    --lambdas=${chlmbdas} \
    --data_dir=Data \
    --lm=scibert \
    --readout=ch \
    --workspace=Workspaces/${CURRENTEPOCTIME}-${data}-${seed}-${chlmbdas}-scibert-ch \
    --seed=${seed} &> ${CURRENTEPOCTIME}-${data}-${seed}-${chlmbdas}-scibert-ch-log.txt

# python main.py \
#     --dataset=${data} \
#     --lambdas=${bertlmbdas} \
#     --data_dir=Data \
#     --lm=bert \
#     --readout=ch \
#     --workspace=Workspaces/${CURRENTEPOCTIME}-${data}-${seed}-${bertlmbdas}-bert-trl \
#     --seed=${seed} &> ${CURRENTEPOCTIME}-${data}-${seed}-${bertlmbdas}-bert-trl-log.txt