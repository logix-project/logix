#!/usr/bin/zsh
#SBATCH --job-name=analog_if
#SBATCH --output /data/tir/projects/tir6/general/hahn2/analog/examples/bert_influence/slurm-out/\%j.out
#SBATCH --err /data/tir/projects/tir6/general/hahn2/analog/examples/bert_influence/slurm-out/\%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64GB
#SBATCH --time 48:00:00
#SBATCH --mail-user=hahn2@andrew.cmu.edu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=babel-shared

conda init
conda activate if

# pip install .

python compute_influences_pca.py
