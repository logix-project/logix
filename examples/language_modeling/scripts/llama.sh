#!/usr/bin/bash
#SBATCH --job-name=pythia
#SBATCH --output ./logs/\%j.out
#SBATCH --err ./logs/\%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100_80GB:2
#SBATCH --mem=160GB
#SBATCH --time 24:00:00
#SBATCH --mail-user=//delete
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=general

source ~/.bashrc
conda init
conda activate analog

set -x

accelerate launch --num_processes 2 --num_machines 1 --multi_gpu --main_process_port 63252 extract_log.py --model_name /data/models/huggingface/meta-llama/Llama-2-7b-hf --lora random --hessian raw --batch_size 2
#python extract_log.py --model_name "$model_name" --lora random --hessian raw --batch_size 2
#python compute_influence.py --model_name "$model_name" --lora random --hessian raw --split valid
#CUDA_VISIBLE_DEVICES=0 python compute_influence.py --model_name /data/models/huggingface/meta-llama/Llama-2-7b-hf --lora random --hessian raw --split external
#CUDA_VISIBLE_DEVICES=0 python compute_influence.py --model_name /data/models/huggingface/meta-llama/Llama-2-7b-hf --lora random --hessian raw --split generated
