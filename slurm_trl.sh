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

if [[ ${data} == "kim-acl" ]]; then
    bertlmbdas="1-0.1491"
    scibertlmbdas="1-0.2059"
elif [[ ${data} == "kim-scicite" ]]; then
    bertlmbdas="1-0.0934"
    scibertlmbdas="1-0.1274"
elif [[ ${data} == "acl-scicite" ]]; then
    bertlmbdas="1-0.0664"
    scibertlmbdas="1-0.0849"
elif [[ ${data} == "acl-kim" ]]; then
    bertlmbdas="1-0.0871"
    scibertlmbdas="1-0.0890"
elif [[ ${data} == "scicite-kim" ]]; then
    bertlmbdas="1-0.1084"
    scibertlmbdas="1-0.1005"
elif [[ ${data} == "scicite-acl" ]]; then
    bertlmbdas="1-0.2273"
    scibertlmbdas="1-0.2760"
else
    echo "Warning: Unrecognized value of data: ${data}"
fi

if [[ ${data} == "acl-kim-scicite" ]]; then
    bertlmbdas="1-0.0871-0.0664"
    scibertlmbdas="1-0.0890-0.0849"
elif [[ ${data} == "kim-acl-scicite" ]]; then
    bertlmbdas="1-0.1491-0.0934"
    scibertlmbdas="1-0.2059-0.1274"
elif [[ ${data} == "scicite-acl-kim" ]]; then
    bertlmbdas="1-0.2273-0.1084"
    scibertlmbdas="1-0.2760-0.1005"
else
    echo "Warning: Unrecognized value of data: ${data}"
fi

CURRENTEPOCTIME=`date +"%Y-%m-%d-%H-%M-%S"`
# Run the PyTorch training script
python main.py \
    --dataset=${data} \
    --lambdas=${scibertlmbdas} \
    --data_dir=Data \
    --lm=scibert \
    --readout=ch \
    --workspace=Workspaces/${CURRENTEPOCTIME}-${data}-${seed}-${scibertlmbdas}-scibert-trl \
    --seed=${seed} &> ${CURRENTEPOCTIME}-${data}-${seed}-${scibertlmbdas}-scibert-trl-log.txt

python main.py \
    --dataset=${data} \
    --lambdas=${bertlmbdas} \
    --data_dir=Data \
    --lm=bert \
    --readout=ch \
    --workspace=Workspaces/${CURRENTEPOCTIME}-${data}-${seed}-${bertlmbdas}-bert-trl \
    --seed=${seed} &> ${CURRENTEPOCTIME}-${data}-${seed}-${bertlmbdas}-bert-trl-log.txt