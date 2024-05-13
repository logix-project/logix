#!/usr/bin/bash
#SBATCH --job-name=if
#SBATCH --output /data/tir/projects/tir3/users/hahn2/logix/examples/language_modeling/slurm-out/\%j.out
#SBATCH --err /data/tir/projects/tir3/users/hahn2/logix/examples/language_modeling/slurm-out/\%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=128GB
#SBATCH --time 24:00:00
#SBATCH --mail-user=hahn2@andrew.cmu.edu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=general

source ~/.bashrc
conda init
conda activate if

set -x

# accelerate launch --num_processes 2 --num_machines 1 --multi_gpu --main_process_port 63221 extract_log.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --lora random --hessian raw --batch_size 1 --mlp_only --data_name openwebtext
# accelerate launch --num_processes 2 --num_machines 1 --multi_gpu --main_process_port 63221 extract_log.py --model_name meta-llama/Meta-Llama-3-8B --lora random --hessian raw --batch_size 1 --mlp_only
# python compute_influence.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --lora random --hessian raw --split valid --mlp_only
# python compute_influence.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --lora random --hessian raw --split valid --mlp_only --data_name openwebtext
# python compute_influence.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --lora random --hessian raw --split external --mlp_only --data_name openwebtext --mode cosine
python compute_influence.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --lora random --hessian raw --split generated --mlp_only --data_name openwebtext --mode dot
