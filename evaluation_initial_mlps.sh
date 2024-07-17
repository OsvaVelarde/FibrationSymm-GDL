#!/bin/bash

_PATH='./'
TITLE='network_'$1
filename1='cfgfiles/struct_'$TITLE'.cfg'
filename2='cfgfiles/train_'$TITLE'.cfg' 

NUM_EXPS=$2

source $filename1
source $filename2

NET_FOLDER=$_PATH'results/'$TITLE

export PYTHONPATH=$(pwd):$PYTHONPATH

# ===========================================================
# ===================== EVALUATION ==========================
python3.8 src/deepnetworks/evaluation_initial.py \
        --PATH $_PATH \
        --title $TITLE\
        --num-exps $NUM_EXPS \
        --test-batch-size $TEST_BATCH_SIZE \
        --device-type $DEVICE_TYPE \
        --gpu-number $GPU_NUMBER \
        --PATHdata "$PATHDATA" \
        --dataset $DATASET \
        --model $MODEL \
        --layers ${LAYERS[@]}\
        --bias $BIAS 

# ===========================================================
# ================== ACTIVATIONS ============================
python3.8 src/fibrations_in_mlps/activations_to_csv.py \
    --PATH $_PATH \
    --title $TITLE\
    --num-exps $NUM_EXPS \
    --layers ${LAYERS[@]}\
    --dataset $DATASET\
    --epoch 'initial'

# ===========================================================
# =================== PLOT FIBRATIONS =======================

python3.8 src/plots/plot_mlp_fibrations_per_class.py \
    --PATH $_PATH\
    --title $TITLE\
    --num-exps $NUM_EXPS\
    --layers ${LAYERS[@]}\
    --epoch 'initial'

python3.8 src/plots/plot_mlp_fibrations_gral.py \
    --PATH $_PATH\
    --title $TITLE\
    --num-exps $NUM_EXPS\
    --layers ${LAYERS[@]}\
    --epoch 'initial'

python3.8 src/plots/plot_mlp_activity_vs_size.py \
    --PATH $_PATH\
    --title $TITLE\
    --num-exps $NUM_EXPS\
    --layers ${LAYERS[@]}\
    --epoch 'initial'

python3.8 src/plots/plot_mlp_num_fibers.py \
    --PATH $_PATH\
    --title $TITLE\
    --epoch 'initial'