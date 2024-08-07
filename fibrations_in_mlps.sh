#!/bin/bash

_PATH='./'
TITLE='network_'$1

filename1='cfgfiles/struct_'$TITLE'.cfg'
filename2='cfgfiles/train_'$TITLE'.cfg' 
source $filename1
source $filename2

NET_FOLDER=$_PATH'results/'$TITLE

NUM_EXPS=$2

export PYTHONPATH=$(pwd):$PYTHONPATH

# ===========================================================
# ===================== TRAINING ============================
if [ -d "$NET_FOLDER" ];
then
    echo "$TITLE exists. If necessary, remove the old version"
else
    echo 'Training Stage - Classifier' $TITLE

    python3.8 src/deepnetworks/training.py \
        --PATH $_PATH \
        --title $TITLE\
        --num-exps $NUM_EXPS \
        --num-train-epochs $NUM_EPOCHS \
        --train-batch-size $TRAIN_BATCH_SIZE \
        --test-batch-size $TEST_BATCH_SIZE \
        --lr-rate $LR_RATE \
        --device-type $DEVICE_TYPE \
        --gpu-number $GPU_NUMBER \
        --PATHdata "$PATHDATA" \
        --dataset $DATASET \
        --model $MODEL \
        --layers ${LAYERS[@]}\
        --bias $BIAS 
fi

# ===========================================================
# ================== ACTIVATIONS ============================

python3.8 src/fibrations_in_mlps/activations_to_csv.py \
    --PATH $_PATH \
    --title $TITLE\
    --num-exps $NUM_EXPS \
    --layers ${LAYERS[@]}\
    --dataset $DATASET\
    --epoch 'final'

# ===========================================================
# ===================== PLOT TRAINING =======================

python3.8 src/plots/plot_training.py \
    --PATH $_PATH \
    --title $TITLE

# ===========================================================
# =================== PLOT FIBRATIONS =======================

python3.8 src/plots/plot_mlp_fibrations_per_class.py \
    --PATH $_PATH\
    --title $TITLE\
    --num-exps $NUM_EXPS\
    --layers ${LAYERS[@]}\
    --epoch 'final'

python3.8 src/plots/plot_mlp_fibrations_gral.py \
    --PATH $_PATH\
    --title $TITLE\
    --num-exps $NUM_EXPS\
    --layers ${LAYERS[@]}\
    --epoch 'final'

python3.8 src/plots/plot_mlp_activity_vs_size.py \
    --PATH $_PATH\
    --title $TITLE\
    --num-exps $NUM_EXPS\
    --layers ${LAYERS[@]}\
    --epoch 'final'

python3.8 src/plots/plot_mlp_num_fibers.py \
    --PATH $_PATH\
    --title $TITLE\
    --epoch 'final'