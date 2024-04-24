#!/bin/bash

if [ $# -eq 0 ]; then
  echo "Please provide the model name as an argument."
  exit 1
fi

model_name=$1

# raw + random lora
#accelerate launch --multi_gpu --num_processes 2 --main_process_port 29505 extract_log.py --model_name "$model_name" --lora random --hessian raw --batch_size 2
#CUDA_VISIBLE_DEVICES=0 python extract_log.py --model_name "$model_name" --lora random --hessian raw --batch_size 2
#python compute_influence.py --model_name "$model_name" --lora random --hessian raw --split valid
python compute_influence.py --model_name "$model_name" --lora random --hessian raw --split external
python compute_influence.py --model_name "$model_name" --lora random --hessian raw --split generated
