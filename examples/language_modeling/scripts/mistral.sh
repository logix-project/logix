#!/usr/bin/bash
#SBATCH --job-name=pythia
#SBATCH --output ./logs/\%j.out
#SBATCH --err ./logs/\%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --mem=80GB
#SBATCH --time 24:00:00
#SBATCH --mail-user=//delete
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=general

source ~/.bashrc
conda init
conda activate analog

set -x

# accelerate launch --num_processes 2 --num_machines 1 --multi_gpu --main_process_port 63252 extract_log.py --model_name mistralai/Mistral-7B-v0.1 --lora random --hessian raw --batch_size 1
python compute_influence.py --model_name mistralai/Mistral-7B-v0.1 --lora random --hessian raw --split valid
python compute_influence.py --model_name mistralai/Mistral-7B-v0.1 --lora random --hessian raw --split external
# python compute_influence.py --model_name mistralai/Mistral-7B-v0.1 --lora random --hessian raw --split generated
