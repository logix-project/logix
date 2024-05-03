#!/bin/bash
#SBATCH --job-name=if
#SBATCH --output=train_log/myjob_%a.out
#SBATCH --error=train_log/myjob_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=36:00:00
#SBATCH --array=0-9
#SBATCH --mem=4G

NUM_GPUS=${NUM_GPUS:-10}  # Defaults to 10 if not specified outside
TOTAL_INDICES=100  # 0 to 100 inclusive

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

# run_command python compute_influence_ensemble.py \
# --model_id $SLURM_ARRAY_TASK_ID \
# --data cifar10 \
# --damping 1e-10
echo "Running command for indices $(($SLURM_ARRAY_TASK_ID * $INDICES_PER_TASK)) to $(($SLURM_ARRAY_TASK_ID == $NUM_GPUS-1 ? $TOTAL_INDICES : ($SLURM_ARRAY_TASK_ID + 1) * $INDICES_PER_TASK))..."

run_command python compute_brittleness_ensemble.py \
--startIdx $(($SLURM_ARRAY_TASK_ID * $INDICES_PER_TASK)) \
--endIdx $(($SLURM_ARRAY_TASK_ID == $NUM_GPUS-1 ? $TOTAL_INDICES - 1 : ($SLURM_ARRAY_TASK_ID + 1) * $INDICES_PER_TASK - 1)) \
--model_id 0 \
--data cifar10 \
--damping 1e-10
# --scoreFileName "TrakRobertaV2"

#they should have same args, if score file name automatically generated.