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

# Function to run the command and check its exit status
run_command() {
    local max_attempts=2
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        "$@"
        local status=$?
        if [ $status -eq 0 ]; then
            return 0  # Success, exit the loop
        else
            echo "Error: Command $@ failed with exit code $status, retrying (attempt $attempt of $max_attempts)..."
            sleep 10s  # Adjust delay between retries as needed
            ((attempt++))
        fi
    done
    echo "Error: Command $@ failed after $max_attempts attempts."
    return 1
}
# python train.py --model_id $SLURM_ARRAY_TASK_ID

run_command python compute_brittleness_ensemble.py \
--startIdx $(($SLURM_ARRAY_TASK_ID*10)) \
--endIdx $(($SLURM_ARRAY_TASK_ID*10+10)) \
--data_name cifar10 \
--lora \
--use_full_covariance

# python compute_influence_ensemble.py \
# --ekfac \
# --model_id $SLURM_ARRAY_TASK_ID \
# --data cifar10 \
# --sample
