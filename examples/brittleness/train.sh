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
#SBATCH --array=0-9
#SBATCH --mem=256G

# Your job command for each model
# python train.py --model_id $SLURM_ARRAY_TASK_ID
python compute_brittleness_ensemble.py --startIdx $(($SLURM_ARRAY_TASK_ID*10)) --endIdx $(($SLURM_ARRAY_TASK_ID*10+10)) --data_name cifar10 --ekfac
# python compute_influence_ensemble.py --ekfac --model_id $SLURM_ARRAY_TASK_ID --data cifar10
