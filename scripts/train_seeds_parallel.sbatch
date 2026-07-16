#!/bin/bash
#SBATCH --job-name=fifth-weight2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate biology-prediction_gpu

cd /path/to/biology_prediction/service
export MPLCONFIGDIR=/tmp/${USER}/matplotlib
export PYTHON_BIN=python

mkdir -p logs
bash scripts/train_seeds_parallel.sh 4 10
