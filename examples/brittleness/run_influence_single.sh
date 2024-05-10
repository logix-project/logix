#!/bin/bash
#SBATCH --job-name=run_one
#SBATCH --output=train_log/run_one.out
#SBATCH --error=train_log/run_one.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=4G

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

# Your job command for each model
# run_command python compute_influence_cifar_mnist.py --data mnist --damping 1e-10
run_command python compute_influence_glue.py --data rte
