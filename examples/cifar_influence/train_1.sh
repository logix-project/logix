#!/bin/bash
#SBATCH --job-name=noLora
#SBATCH --output=train_log/noLora.out
#SBATCH --error=train_log/noLora.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=512G

# Your job command for each model
python compute_influences_pca_no_save.py --ekfac
# python compute_brittleness.py --startIdx 30 --endIdx 40