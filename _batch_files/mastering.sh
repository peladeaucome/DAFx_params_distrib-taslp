#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=A40
#SBATCH --exclude=node47,node43,node42,node34
#SBATCH --output=slurm/mastering/slurm-%j.out
#SBATCH --array=1
# set -x

t=$SLURM_ARRAY_TASK_ID

if [ $t == 1 ]; then
    python train_mastering.py hydra=deter model.name=deter
elif [ $t == 2 ]; then
    python train_mastering.py hydra=infer model.name=infer model.distrib.num_mixtures=1 model.flow.length=1 model.distrib.type=unif
elif [ $t == 3 ]; then
    python train_mastering.py hydra=infer model.name=infer model.distrib.num_mixtures=1 model.flow.length=2 model.distrib.type=unif
elif [ $t == 4 ]; then
    python train_mastering.py hydra=infer model.name=infer model.distrib.num_mixtures=6 model.flow.length=1 model.distrib.type=full
elif [ $t == 5 ]; then
    python train_mastering.py hydra=infer model.name=infer model.distrib.num_mixtures=6 model.flow.length=1 model.distrib.type=unif
elif [ $t == 6 ]; then
    python train_mastering.py hydra=infer model.name=infer model.distrib.num_mixtures=24 model.flow.length=1 model.distrib.type=unif
elif [ $t == 7 ]; then
    python train_mastering.py hydra=infer model.name=infer model.distrib.num_mixtures=1 model.flow.length=1 model.distrib.type=unif model.beta.start=0.02
elif [ $t == 8 ]; then
    python train_mastering.py hydra=infer model.name=infer model.distrib.num_mixtures=1 model.flow.length=2 model.distrib.type=unif model.beta.start=0.02
elif [ $t == 9 ]; then
    sleep 9
    python train_mastering.py hydra=infer model.name=infer model.distrib.num_mixtures=6 model.flow.length=1 model.distrib.type=full model.beta.start=0.02
elif [ $t == 10 ]; then
    python train_mastering.py hydra=infer model.name=infer model.distrib.num_mixtures=6 model.flow.length=1 model.distrib.type=unif model.beta.start=0.02
elif [ $t == 11 ]; then
    sleep 10
    python train_mastering.py hydra=infer model.name=infer model.distrib.num_mixtures=24 model.flow.length=1 model.distrib.type=unif model.beta.start=0.02
elif [ $t == 12 ]; then
    python train_mastering.py hydra=infer model.name=infer model.distrib.num_mixtures=1 model.flow.length=0 model.distrib.type=unif
elif [ $t == 13 ]; then
    python train_mastering.py hydra=infer model.name=infer model.distrib.num_mixtures=6 model.flow.length=0 model.distrib.type=full 
elif [ $t == 14 ]; then
    python test_mastering.py
fi

# python test_mastering.py
