#!/usr/bin/bash

# Logging configuration
#SBATCH --job-name=cot-copy
#SBATCH --output=/checkpoint/vivc/cot-copy/%a-%t.out
#SBATCH --error=/checkpoint/vivc/cot-copy/%a-%t.err
#SBATCH --mail-type=END
#SBATCH --mail-user=vivc@meta.com

# Job specification
#SBATCH --partition=devlab
#SBATCH --time=5:00:00
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --partition=scavenge
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --array=1-30


python /private/home/vivc/code/llm/cot/scripts/grid_run.py --num-tasks $SLURM_ARRAY_TASK_COUNT --task-id $SLURM_ARRAY_TASK_ID --config_filename binary
