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

# if [[ ${data} == "kim-acl" ]]; then
#     bertlmbdas="1-0.1491"
#     scibertlmbdas="1-0.2059"
# elif [[ ${data} == "kim-scicite" ]]; then
#     bertlmbdas="1-0.0934"
#     scibertlmbdas="1-0.1274"
# elif [[ ${data} == "acl-scicite" ]]; then
#     bertlmbdas="1-0.0664"
#     scibertlmbdas="1-0.0849"
# elif [[ ${data} == "acl-kim" ]]; then
#     bertlmbdas="1-0.0871"
#     scibertlmbdas="1-0.0890"
# elif [[ ${data} == "scicite-kim" ]]; then
#     bertlmbdas="1-0.1084"
#     scibertlmbdas="1-0.1005"
# elif [[ ${data} == "scicite-acl" ]]; then
#     bertlmbdas="1-0.2273"
#     scibertlmbdas="1-0.2760"
# else
#     echo "Warning: Unrecognized value of data: ${data}"
# fi

# if [[ ${data} == "acl-kim-scicite" ]]; then
#     bertlmbdas="1-0.0871-0.0664"
#     scibertlmbdas="1-0.0890-0.0849"
# elif [[ ${data} == "kim-acl-scicite" ]]; then
#     bertlmbdas="1-0.1491-0.0934"
#     scibertlmbdas="1-0.2059-0.1274"
# elif [[ ${data} == "scicite-acl-kim" ]]; then
#     bertlmbdas="1-0.2273-0.1084"
#     scibertlmbdas="1-0.2760-0.1005"
# else
#     echo "Warning: Unrecognized value of data: ${data}"
# fi


# if [[ ${data} == "kim-acl" ]]; then
#     bertlmbdas="1-0.1369"
#     scibertlmbdas="1-0.1863"
# elif [[ ${data} == "kim-scicite" ]]; then
#     bertlmbdas="1-0.0927"
#     scibertlmbdas="1-0.0782"
# elif [[ ${data} == "acl-scicite" ]]; then
#     bertlmbdas="1-0.0818"
#     scibertlmbdas="1-0.0592"
# elif [[ ${data} == "acl-kim" ]]; then
#     bertlmbdas="1-0.0795"
#     scibertlmbdas="1-0.0935"
# elif [[ ${data} == "scicite-kim" ]]; then
#     bertlmbdas="1-0.0926"
#     scibertlmbdas="1-0.1108"
# elif [[ ${data} == "scicite-acl" ]]; then
#     bertlmbdas="1-0.2242"
#     scibertlmbdas="1-0.2803"
# else
#     echo "Warning: Unrecognized value of data: ${data}"
# fi

# if [[ ${data} == "acl-kim-scicite" ]]; then
#     bertlmbdas="1-0.0795-0.0818"
#     scibertlmbdas="1-0.0935-0.0592"
# elif [[ ${data} == "kim-acl-scicite" ]]; then
#     bertlmbdas="1-0.1369-0.0927"
#     scibertlmbdas="1-0.1863-0.0782"
# elif [[ ${data} == "scicite-acl-kim" ]]; then
#     bertlmbdas="1-0.2242-0.0926"
#     scibertlmbdas="1-0.2803-0.1108"
# else
#     echo "Warning: Unrecognized value of data: ${data}"
# fi

# TRL on validation set
# if [[ ${data} == "kim-acl" ]]; then
#     bertlmbdas="1-0.2415"
#     scibertlmbdas="1-0.2954"
# elif [[ ${data} == "kim-scicite" ]]; then
#     bertlmbdas="1-0.1536"
#     scibertlmbdas="1-0.1473"
# elif [[ ${data} == "acl-scicite" ]]; then
#     bertlmbdas="1-0.0821"
#     scibertlmbdas="1-0.0718"
# elif [[ ${data} == "acl-kim" ]]; then
#     bertlmbdas="1-0.0960"
#     scibertlmbdas="1-0.1191"
# elif [[ ${data} == "scicite-kim" ]]; then
#     bertlmbdas="1-0.1030"
#     scibertlmbdas="1-0.0931"
# elif [[ ${data} == "scicite-acl" ]]; then
#     bertlmbdas="1-0.2336"
#     scibertlmbdas="1-0.2940"
# else
#     echo "Warning: Unrecognized value of data: ${data}"
# fi

# if [[ ${data} == "acl-kim-scicite" ]]; then
#     bertlmbdas="1-0.0960-0.0821"
#     scibertlmbdas="1-0.1191-0.0718"
# elif [[ ${data} == "kim-acl-scicite" ]]; then
#     bertlmbdas="1-0.2415-0.1536"
#     scibertlmbdas="1-0.2954-0.1473"
# elif [[ ${data} == "scicite-acl-kim" ]]; then
#     bertlmbdas="1-0.2336-0.1030"
#     scibertlmbdas="1-0.2940-0.0931"
# else
#     echo "Warning: Unrecognized value of data: ${data}"
# fi

# TRL on training set
if [[ ${data} == "kim-acl" ]]; then
    bertlmbdas="1-0.1488"
    scibertlmbdas="1-0.1889"
elif [[ ${data} == "kim-scicite" ]]; then
    bertlmbdas="1-0.0800"
    scibertlmbdas="1-0.0805"
elif [[ ${data} == "acl-scicite" ]]; then
    bertlmbdas="1-0.0746"
    scibertlmbdas="1-0.0652"
elif [[ ${data} == "acl-kim" ]]; then
    bertlmbdas="1-0.0922"
    scibertlmbdas="1-0.0951"
elif [[ ${data} == "scicite-kim" ]]; then
    bertlmbdas="1-0.1095"
    scibertlmbdas="1-0.1059"
elif [[ ${data} == "scicite-acl" ]]; then
    bertlmbdas="1-0.2339"
    scibertlmbdas="1-0.2785"
else
    echo "Warning: Unrecognized value of data: ${data}"
fi

if [[ ${data} == "acl-kim-scicite" ]]; then
    bertlmbdas="1-0.0922-0.0746"
    scibertlmbdas="1-0.0951-0.0652"
elif [[ ${data} == "kim-acl-scicite" ]]; then
    bertlmbdas="1-0.1488-0.0800"
    scibertlmbdas="1-0.1889-0.0805"
elif [[ ${data} == "scicite-acl-kim" ]]; then
    bertlmbdas="1-0.2339-0.1095"
    scibertlmbdas="1-0.2785-0.1059"
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