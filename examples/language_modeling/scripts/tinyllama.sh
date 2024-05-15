#!/usr/bin/bash
#SBATCH --job-name=pythia
#SBATCH --output ./logs/\%j.out
#SBATCH --err ./logs/\%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40:4
#SBATCH --mem=128GB
#SBATCH --time 24:00:00
#SBATCH --mail-user=//delete
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=general

source ~/.bashrc
conda init
conda activate analog

#model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

set -x

#accelerate launch --multi_gpu --num_processes 4 --main_process_port 19523 extract_log.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --lora random --hessian raw --batch_size 2
#python extract_log.py --model_name "$model_name" --lora random --hessian raw --batch_size 2
#python compute_influence.py --model_name "$model_name" --lora random --hessian raw --split valid
python compute_influence.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --lora random --hessian raw --split external
#CUDA_VISIBLE_DEVICES=0 python compute_influence.py --model_name "$model_name" --lora random --hessian raw --split generated
