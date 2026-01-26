#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=P100
#SBATCH --exclude=node45,node46,node47,node17
#SBATCH --output=slurm/MSM_const/slurm-%j.out
# HYDRA_FULL_ERROR=1
# export HYDRA_FULL_ERROR
set -x

song_key=$SLURM_ARRAY_TASK_ID
python train_MSmastering_1song.py model.name=deter hydra=deter experiment.song_key=$song_key

# python train_MSmastering_1song.py model.name=infer hydra=infer model.distrib.num_mixtures=1 model.flow.length=1
# python train_MSmastering_1song.py model.name=infer hydra=infer model.distrib.num_mixtures=1 model.flow.length=2
# python train_MSmastering_1song.py model.name=infer hydra=infer model.distrib.num_mixtures=1 model.flow.length=3
# python train_MSmastering_1song.py model.name=infer hydra=infer model.distrib.num_mixtures=1 model.flow.length=4

# python train_MSmastering_1song.py model.name=infer hydra=infer model.distrib.num_mixtures=4 model.flow.length=1
# python train_MSmastering_1song.py model.name=infer hydra=infer model.distrib.num_mixtures=4 model.flow.length=2
# python train_MSmastering_1song.py model.name=infer hydra=infer model.distrib.num_mixtures=4 model.flow.length=3
# python train_MSmastering_1song.py model.name=infer hydra=infer model.distrib.num_mixtures=4 model.flow.length=4

# python train_MSmastering_1song.py model.name=infer hydra=infer model.distrib.num_mixtures=4 model.flow.length=4 model.flow.layers=static
