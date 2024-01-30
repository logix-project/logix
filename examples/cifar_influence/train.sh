#!/bin/bash
#SBATCH --job-name=if
#SBATCH --output=train_log/myjob_%a.out
#SBATCH --error=train_log/myjob_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=6-6
#SBATCH --mem=256G

# Your job command for each model
# python train.py --model_id $SLURM_ARRAY_TASK_ID
python compute_brittleness.py --startIdx $(($SLURM_ARRAY_TASK_ID*10)) --endIdx $(($SLURM_ARRAY_TASK_ID*10+10))
# python compute_influences_pca_no_save.py --ekfac --model_id $SLURM_ARRAY_TASK_ID --lora
