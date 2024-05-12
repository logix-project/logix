#!/bin/bash
read -p "Enter the number of GPUs: " NUM_GPUS
export TOTAL_INDICES=100
# Create a new job script with dynamic SBATCH directives
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=if
#SBATCH --output=train_log/cifar_%a.out
#SBATCH --error=train_log/cifar_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=48:00:00
#SBATCH --mem=4G
#SBATCH --array=0-$(($NUM_GPUS-1))

INDICES_PER_TASK=$(( ($TOTAL_INDICES + $NUM_GPUS - 1) / $NUM_GPUS))
REMAINDER=$(($TOTAL_INDICES % $NUM_GPUS))

# Adjust array range based on the number of GPUs


# Function to run the command and check its exit status
run_command() {
    local max_attempts=2
    local attempt=1
    while [ \$attempt -le \$max_attempts ]; do
        "\$@"
        local status=\$?
        if [ \$status -eq 0 ]; then
            return 0  # Success, exit the loop
        else
            echo "Error: Command \$@ failed with exit code \$status, retrying (attempt \$attempt of \$max_attempts)"
            sleep 10s  # Adjust delay between retries as needed
            ((attempt++))
        fi
    done
    echo "Error: Command \$@ failed after \$max_attempts attempts."
    return 1
}

echo "Running command for indices \$((\$SLURM_ARRAY_TASK_ID * \$INDICES_PER_TASK)) to \$((\$SLURM_ARRAY_TASK_ID == $NUM_GPUS-1 ? $(($TOTAL_INDICES)) : (\$SLURM_ARRAY_TASK_ID + 1) * \$INDICES_PER_TASK))"

run_command python compute_brittleness_cifar_mnist.py \
--startIdx \$((\$SLURM_ARRAY_TASK_ID * \$INDICES_PER_TASK)) \
--endIdx \$((\$SLURM_ARRAY_TASK_ID == $(($NUM_GPUS-1)) ? $(($TOTAL_INDICES)) : (\$SLURM_ARRAY_TASK_ID + 1) * \$INDICES_PER_TASK)) \
--model_id 0 \
--data cifar10 \
# --damping 1e-10 \
# --scoreFileName "TrakRobertaV2"

EOT
