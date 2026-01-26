#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=A40
#SBATCH --exclude=node47,node43,node42,node34
#SBATCH --output=slurm/ddx7_56/slurm-%j.out
#SBATCH --array=9
set -x

t=$SLURM_ARRAY_TASK_ID

if [ $t == 1 ]; then
    python train_ddx7_56.py model=1bandeq/deter hydra=deter
elif [ $t == 2 ]; then
    python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=1 model.flow.length=1 
elif [ $t == 3 ]; then
    python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=1 model.flow.length=2 model.distrib.type=gaussian_log
elif [ $t == 4 ]; then
    python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=6 model.flow.length=1 model.distrib.type=full resume_training=logs/ddx7_56/2026-01-12_18-59-16/run/infer/mix_6_full/flow_False_dsf_1/sot/lightning_logs/version_0/checkpoints/last.ckpt
elif [ $t == 5 ]; then
    python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=6 model.flow.length=1 model.distrib.type=unif 
elif [ $t == 6 ]; then
    python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=24 model.flow.length=1 model.distrib.type=unif
elif [ $t == 7 ]; then
    sleep 5
    python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=1 model.flow.length=1 model.beta.start=0.005
elif [ $t == 8 ]; then
    sleep 5
    python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=1 model.flow.length=2 model.beta.start=0.005
elif [ $t == 9 ]; then
    sleep 5
    python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=6 model.flow.length=1 model.distrib.type=full model.beta.start=0.005 resume_training=logs/ddx7_56/2026-01-13_09-57-16/run/infer/mix_6_full/flow_False_dsf_1/sot/lightning_logs/version_0/checkpoints/last.ckpt
elif [ $t == 10 ]; then
    sleep 5
    python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=6 model.flow.length=1 model.distrib.type=unif model.beta.start=0.005
elif [ $t == 11 ]; then
    sleep 5
    python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=24 model.flow.length=1 model.distrib.type=unif model.beta.start=0.005
elif [ $t == 12 ]; then
    python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=1 model.flow.length=0 model.distrib.type=unif model.beta.end=0.005
elif [ $t == 13 ]; then
    python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=6 model.flow.length=0 model.distrib.type=full model.beta.end=0.005
elif [ $t == 14 ]; then
    python train_ddx7_56.py model=1bandeq/deter hydra=deter model.name=synthperm model.learning_rate=0.0001 model/vector_field=ddx7/ffn
elif [ $t == 15 ]; then
    python train_ddx7_56.py model=1bandeq/deter hydra=deter model.name=synthperm model.learning_rate=0.0001 model/vector_field=ddx7/p2t
fi


# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=6 model.distrib.type=unif model.warmup_length=3000 model.flow.length=1 resume_training="logs/ddx7_56/2025-11-11_10-34-54/run/infer/mix_6_unif/flow_False_dsf_1/sot/lightning_logs/version_0/checkpoints/last.ckpt"

# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=1 model.flow.length=1
# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=1 model.flow.length=2
# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=6 model.flow.length=1 model.distrib.type=full
# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=6 model.flow.length=1 model.distrib.type=unif
# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=24 model.flow.length=1 model.distrib.type=unif

# python train_ddx7_56.py model=1bandeq/deter hydra=deter model.name=synthperm model.learning_rate=0.0001 model/vector_field=ddx7/ffn
# python train_ddx7_56.py model=1bandeq/deter hydra=deter model.name=synthperm model.learning_rate=0.0001 model/vector_field=ddx7/p2t

# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=1 model.flow.length=1 model.beta.start=0.005 
# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=1 model.flow.length=2 model.beta.start=0.005 
# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=6 model.flow.length=1 model.distrib.type=full model.beta.start=0.005 
# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=6 model.flow.length=1 model.distrib.type=unif model.beta.start=0.005 
# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=24 model.flow.length=1 model.distrib.type=unif model.beta.start=0.001


# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=1 model.distrib.type=full model.flow.length=1 model.start_from_ae=true
# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=6 model.distrib.type=full model.flow.length=1 model.start_from_ae=true
# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=6 model.distrib.type=unif model.flow.length=1 model.start_from_ae=true
# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=24 model.distrib.type=unif model.flow.length=1 model.start_from_ae=true


# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=1 model.distrib.type=full model.flow.length=1 experiment.max_epochs=4000 model.warmup_length=2000 model.start_from_ae=true
# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=6 model.distrib.type=full model.flow.length=1 experiment.max_epochs=4000 model.warmup_length=2000 model.start_from_ae=true
# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=6 model.distrib.type=unif model.flow.length=1 experiment.max_epochs=4000 model.warmup_length=2000 model.start_from_ae=true
# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=24 model.distrib.type=unif model.flow.length=1 experiment.max_epochs=4000 model.warmup_length=2000 model.start_from_ae=true


# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=6 model.flow.length=1 model.distrib.type=full model.learning_rate=0.00002

# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=6 model.flow.length=1 model.distrib.type=unif resume_training=logs/ddx7_56/2025-04-28_13-56-28/run/infer/mix_6_unif/flow_False_dsf_1/sot/lightning_logs/version_0/checkpoints/last.ckpt
# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=6 model.flow.length=1 model.distrib.type=unif model.learning_rate=0.00002


# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=1 model.flow.length=2 
# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=1 model.flow.length=2 model.beta.end=0.001 
# python train_ddx7_56.py model.name=infer hydra=infer model.distrib.num_mixtures=6 model.flow.length=1 model.distrib.type=unif resume_training=logs/ddx7_56/2025-04-23_11-38-34/run/infer/mix_6_unif/flow_False_dsf_1/sot/lightning_logs/version_0/checkpoints/best.ckpt
