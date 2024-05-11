#!/bin/zsh
###############################################################
######################## DO NOT TOUCH! ########################
###############################################################
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

    EXP_FOLDER=${BASE_FOLDER}/${EXP_NAME}
    CODE_FOLDER=${EXP_FOLDER}/code/${NAME}
    SCRIPTS_FOLDER=${EXP_FOLDER}/code/scripts/${NAME}
    OUTPUT_FOLDER=${EXP_FOLDER}/log/${NAME}

    mkdir -p ${EXP_FOLDER}
    mkdir -p ${CODE_FOLDER}
    mkdir -p ${SCRIPTS_FOLDER}
    mkdir -p ${OUTPUT_FOLDER}

    # Just a comment file in case you need to add something
    echo "${COMMENT}" > ${OUTPUT_FOLDER}/README.md

    # Copying the code to launch this version and be able to keep working here.
    cp -r ${PWD}/* ${CODE_FOLDER}
}

#SBATCH --constraint=volta32gb
write_script () {
    echo """#!/bin/zsh
#SBATCH --time=48:00:00
#SBATCH --time=00:05:00
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${WORLD_SIZE}
#SBATCH --mem=320G
#SBATCH --mem=16G
#SBATCH --cpus-per-task=10
#SBATCH --cpus-per-task=2
#SBATCH --gpus=${WORLD_SIZE}
#SBATCH --gpus-per-node=${WORLD_SIZE}
#SBATCH --partition=scavenge
#SBATCH --array=1-${ARRAY_SIZE}
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

PARAMS=(${PARAMS})
PARAMS_SWEEP=(${PARAMS_SWEEP})
P=\`expr \${SLURM_ARRAY_TASK_ID} '-' 1\`
P=\`expr \${P} '*' \${#PARAMS}\`

# Get the parameters
for i in {1..\${#PARAMS}}; do
    eval \"\${PARAMS[\$i]}=\${PARAMS_SWEEP[\`expr \${P} + \$i\`]}\"
done

srun --unbuffered \\
    --output=${OUTPUT_FOLDER}/%j/%j_%t.out \\
    --error=${OUTPUT_FOLDER}/%j/%j_%t.err \\
    python ${CODE_FOLDER}/src/cot/train.py \\
    --problem ${PROBLEM} \\
    --nb_len ${NB_LEN} \\
    --split_probas ${SPLIT_PROBAS} \\
    --max_nb_data_per_len ${MAX_NB_DATA_PER_LEN} \\
    --zipf_offset ${ZIPF_OFFSET} \\
    --zipf_coef ${ZIPF_COEF} \\
    --emb_dim ${EMB_DIM} \\
    --emb_dropout ${EMB_DROPOUT} \\
    --n_head ${N_HEAD} \\
    --n_layer ${N_LAYER} \\
    --nb_epochs ${NB_EPOCHS} \\
    --learning_rate ${LEARNING_RATE} \\
    --checkpoint_freq ${CHECKPOINT_FREQ} \\
    --overwrite_checkpoint ${OVERWRITE_CHECKPOINT} \\
    --load_checkpoint ${LOAD_CHECKPOINT} \\
    --eval_freq ${EVAL_FREQ} \\
    --flash ${FLASH}
    
touch ${OUTPUT_FOLDER}/\${SLURM_JOB_ID}/done
""" > ${SCRIPTS_FOLDER}/launch.sh
}
###############################################################
###############################################################
###############################################################

### Experiment settings ###
EXP_NAME="binary-copy"
COMMENT="pretrain models"

NODES=1
WORLD_SIZE=1
###########################

### Default config ###
PROBLEM="binary-copy"
NB_LEN=8
SPLIT_PROBAS=0.5
MAX_NB_DATA_PER_LEN=10_000
ZIPF_OFFSET=0
ZIPF_COEF=0
EMB_DIM=128
EMB_DROPOUT=0.1
N_HEAD=2
N_LAYER=2
NB_EPOCHS=1000
LEARNING_RATE=1e-3
CHECKPOINT_FREQ=100
OVERWRITE_CHECKPOINT=True
LOAD_CHECKPOINT=False
EVAL_FREQ=10
FLASH=None
######################

### Define models ###
simple_model () {
    MODEL="simple"
    N_HEAD=1
    N_LAYER=1
    NB_EPOCHS=1500
}
#####################

### List configs to iterate on ###
MODEL_CONFIGS=(
    simple_model
)
###################################

### Parameters sweep ###
# This is where you define the parameters sweep.
# Start with defining which parameters while be swept.
# Then define the values for each parameter.
# The values will be combined in the order they are defined.
# Try to keep them aligned as much as possible for clarity.
PARAMS=(LR      EMB_DIM)
PARAMS_SWEEP=(
        1e-4    128
        3e-4    128
        6e-4    128
        1e-3    128
)
############################
ARRAY_SIZE=`expr ${#PARAMS_SWEEP} / ${#PARAMS}`

name_exp () {
    # Basically any name to track the experiment with important params.
    NAME="${MODEL}_nh${N_HEAD}_nl${N_LAYER}_ep${NB_EPOCHS}"
}

# Iterate through all the models configs
for setup_model in $MODEL_CONFIGS
do
$setup_model

# Execute the setup functions
name_exp

setup_exp

write_script

# ${SCRIPTS_FOLDER}/launch.sh

# Launch the job
sbatch ${SCRIPTS_FOLDER}/launch.sh

done