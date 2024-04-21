#!/usr/bin/bash
#SBATCH --job-name=pythia
#SBATCH --output /home/sangkeuc/logs/\%j.out
#SBATCH --err /home/sangkeuc/logs/\%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --mem=80GB
#SBATCH --time 24:00:00
#SBATCH --mail-user=sangkeuc@andrew.cmu.edu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=general

source ~/.bashrc
conda init
conda activate analog

set -x

# accelerate launch --num_processes 2 --num_machines 1 --multi_gpu --main_process_port 63221 extract_log.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --lora random --hessian raw --batch_size 1 --mlp_only
# python compute_influence.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --lora random --hessian raw --split valid --mlp_only
python compute_influence.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --lora random --hessian raw --split external --mlp_only
# python compute_influence.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --lora random --hessian raw --split generated
