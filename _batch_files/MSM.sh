#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=P100
#SBATCH --exclude=node45,node46,node47,node17


set -x

# python train_MSmastering.py -m model.fx_list=[peq5,drc,sc],[peq5,drc],[peq5,sc],[peq5] model.mlp.type=res model.mode=usage
# python train_MSmastering.py -m model.fx_list=[peq5,drc,sc],[peq5,drc],[peq5,sc],[peq5] model.mlp.type=res model.mode=estimation

# python train_MSmastering.py -m model.fx_list=[peq6,drc,sc],[peq6,drc,peq5,drc,sc,peq5] model.mode=estimation
python train_MSmastering.py model.name=ae experiment.dataset_seed=0
# python train_MSmastering.py model.fx_list=[peq6,drc,peq5,drc,sc,peq5] model.mode=estimation model.loss=mel model.name=inference
# python train_MSmastering.py model.fx_list=[peq6,drc,peq5,drc,sc,peq5] model.mode=estimation model.frontend.n_mels=64 model.frontend.channels_list=[64,64,64,128,128,128] model.frontend.convnext_bottleneck_factor=1 model.loss=mel model.mlp.size=128
