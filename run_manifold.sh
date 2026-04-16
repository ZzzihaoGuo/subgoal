#!/bin/bash
#SBATCH --job-name=manifold
#SBATCH --account=brics.a5l
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=manifold_target_%j.out

# Activate conda environment
source ~/ENTER/etc/profile.d/conda.sh
conda activate dgppo

cd /home/a5l/zihao1996.a5l/project/subgoal

python train_try_manifold.py \
    --env LidarBicycleTarget \
    --algo informarl_subgoal \
    -n 3 \
    --obs 3
