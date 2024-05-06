#!/bin/zsh

name_exp () {
    # Basically any name to track the experiment with important params.
    NAME="${MODEL}_nh${N_HEAD}_nl${N_LAYER}_ep${NB_EPOCHS}"
}

setup_exp () {
    # Setup the folder structure for the experiments
    BASE_FOLDER=/checkpoint/${USER}/Experiments/${PWD##*/}
    
    # The folder structure is as follows:
    # BASE_FOLDER
    # ├── EXP_NAME
    # │   ├── code
    # │   │   ├── NAME
    # │   │   ├── scripts
    # │   │   │   ├── NAME
    # │   ├── log
    # │   │   ├── NAME

    EXP_FOLDER=${BASE_FOLDER}/${EXP_NAME}/
    CODE_FOLDER=${BASE_FOLDER}/${EXP_NAME}/code/${NAME}
    SCRIPTS_FOLDER=${BASE_FOLDER}/${EXP_NAME}/code/scripts/${NAME}
    OUTPUT_FOLDER=${BASE_FOLDER}/${EXP_NAME}/log/${NAME}

    mkdir -p ${EXP_FOLDER}
    mkdir -p ${CODE_FOLDER}
    mkdir -p ${SCRIPTS_FOLDER}
    mkdir -p ${OUTPUT_FOLDER}

    # Just a comment file in case you need to add something
    echo "${COMMENT}" > ${EXP_FOLDER}/${NAME}/README.md

    # Copying the code to launch this version and be able to keep working here.
    cp -r ${PWD}/* ${CODE_FOLDER}
}

write_script () {
    echo """#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --constraint=volta32gb
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${WORLD_SIZE}
#SBATCH --mem=320G
#SBATCH --cpus-per-task=10
#SBATCH --gpus=${WORLD_SIZE}
#SBATCH --gpus-per-node=${WORLD_SIZE}
#SBATCH --partition=scavenge
#SBATCH --array=1-${ARRAY}
#SBATCH --signal=B:SIGUSR1@120

#SBATCH --job-name=${EXP_NAME}_${NAME}
#SBATCH --output=${OUTPUT_FOLDER}/%j/%j.out
#SBATCH --error=${OUTPUT_FOLDER}/%j/%j.err
#SBATCH --open-mode=append

mkdir -p ${OUTPUT_FOLDER}/\${SLURM_JOB_ID}

term_handler()
{
        echo "Caught termination signal. Killing the process"
        exit -1
        rm ${OUTPUT_FOLDER}/\${SLURM_JOB_ID}/running
}

trap 'term_handler' TERM
trap 'term_handler' SIGUSR1


srun --unbuffered \\
    --output=${OUTPUT_FOLDER}/%j/%j_%t.out \\
    --error=${OUTPUT_FOLDER}/%j/%j_%t.err \\
    python ${CODE_FOLDER}/src/cot/train.py \\
    --n_head ${N_HEAD} \\
    --n_layer ${N_LAYER} \\
    --nb_epochs ${NB_EPOCHS} \\
    --load_checkpoint ${LOAD_CHECKPOINT}
    
touch ${OUTPUT_FOLDER}/\${SLURM_JOB_ID}/done
""" > ${SCRIPTS_FOLDER}/launch.sh
}

EXP_NAME="pretrain"
COMMENT="pretrain models"

ARRAY=1
NODES=1
WORLD_SIZE=1

# Define different functions with different parameters
simple_model () {
    MODEL="simple"
    N_HEAD=1
    N_LAYER=1
    NB_EPOCHS=1500
}

# List all the functions to run
MODEL_CONFIGS=(
    simple_model
)

# Iterate through all the functions
for setup_model in $MODEL_CONFIGS
do
$setup_model

# Execute the setup functions
name_exp

setup_exp

write_script

# Launch the job
sbatch ${SCRIPTS_FOLDER}/launch.sh

done