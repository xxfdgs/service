#!/bin/bash
#SBATCH -J biology_prediction
#SBATCH -p gpu4090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err

conda activate biology-prediction_gpu
bash scripts/train_seeds_parallel.sh 4 10
