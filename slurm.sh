#!/bin/bash -l

#SBATCH -p gk
#SBATCH --gres=gpu:1

#SBATCH --output=result-%j.out
#SBATCH --error=result-%j.err

#SBATCH --time=1:00:00

#SBATCH --ntasks=2

#SBATCH --mem=40G

#SBATCH --mail-type=ALL
#SBATCH --mail-user=shuix007@umn.edu

while getopts s:d:p: flag
do
    case "${flag}" in
        s) seed=${OPTARG};;
        d) data=${OPTARG};;
        p) lmbdas=${OPTARG};;
    esac
done

conda activate KimNLP

cd /home/karypisg/shuix007/KimNLP/KimNLP

CURRENTEPOCTIME=`date +"%Y-%m-%d-%H-%M-%S"`
# Run the PyTorch training script
python main.py \
    --dataset=${data} \
    --lmbdas=${lmbdas} \
    --data_dir=Data \
    --lm=scibert \
    --workspace=Workspaces/${CURRENTEPOCTIME}-${data}-${seed}-${lmbdas}-scibert \
    --seed=${seed} &> ${CURRENTEPOCTIME}-${data}-${seed}-${lmbdas}-scibert-log.txt

python main.py \
    --dataset=${data} \
    --lmbdas=${lmbdas} \
    --data_dir=Data \
    --lm=bert \
    --workspace=Workspaces/${CURRENTEPOCTIME}-${data}-${seed}-${lmbdas}-bert \
    --seed=${seed} &> ${CURRENTEPOCTIME}-${data}-${seed}-${lmbdas}-bert-log.txt