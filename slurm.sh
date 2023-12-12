#!/bin/bash -l

#SBATCH -p gk
#SBATCH --gres=gpu:1

#SBATCH --output=result-%j.out
#SBATCH --error=result-%j.err

#SBATCH --time=4:00:00

#SBATCH --ntasks=2

#SBATCH --mem=40G

#SBATCH --mail-type=ALL
#SBATCH --mail-user=shuix007@umn.edu

while getopts d: flag
do
    case "${flag}" in
        d) data=${OPTARG};;
    esac
done

conda activate KimNLP

cd /home/karypisg/shuix007/KimNLP/KimNLP

CURRENTEPOCTIME=`date +"%Y-%m-%d-%H-%M-%S"`
# Run the PyTorch training script
python main.py \
    --dataset=${data} \
    --data_dir=Data \
    --workspace=Workspaces/${CURRENTEPOCTIME}-${data}-scibert &> ${CURRENTEPOCTIME}-${data}-scibert-log.txt