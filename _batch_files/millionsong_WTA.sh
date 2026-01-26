#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=V100
#SBATCH --exclude=node47
#SBATCH --output=slurm/millionsong_wta/slurm-%j.out

set -x


python train_millionsong_wta.py hydra=deter model.name=deter model.num_hyp=1 model/frontend=melnext_small_2 model.frontend.type=melnext_att
# python train_millionsong_wta.py hydra=deter model.name=deter model.num_hyp=1 model.learning_rate=0.00001 model/frontend=melnext_big model.mlp.size=512
# python train_millionsong_wta.py hydra=deter model.name=deter model.num_hyp=4 model.annealing_length=45 experiment.batch_size=64 experiment.max_epochs=50
