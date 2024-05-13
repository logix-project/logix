#!/usr/bin/bash
#SBATCH --job-name=if
#SBATCH --output /data/tir/projects/tir3/users/hahn2/logix/examples/language_modeling/slurm-out/\%j.out
#SBATCH --err /data/tir/projects/tir3/users/hahn2/logix/examples/language_modeling/slurm-out/\%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --mem=256GB
#SBATCH --time 48:00:00
#SBATCH --mail-user=hahn2@andrew.cmu.edu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=general

source ~/.bashrc
conda init
conda activate if

set -x

# accelerate launch --num_processes 2 --num_machines 1 --multi_gpu --main_process_port 63221 extract_log.py --model_name EleutherAI/pythia-1.4b --lora random --hessian raw --batch_size 1 --mlp_only --data_name openwebtext
python compute_influence.py --model_name EleutherAI/pythia-1.4b --lora random --hessian raw --split generated --mlp_only --data_name openwebtext --mode cosine
