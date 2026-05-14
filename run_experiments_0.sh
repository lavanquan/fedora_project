#!/bin/bash

# run_experiments.sh
# Script to run federated learning experiments

# Activate environment (adjust as needed)
# source venv/bin/activate
# conda activate your_env_name

# Define common parameters
NUM_EPOCHS=2
NUM_ROUNDS=10
RANK=4
LR=3e-4
OUTPUT_DIR="results"
CUDA_DEVICE="0"
MODEL_NAME="roberta-base"  # Có thể thay đổi

mkdir -p $OUTPUT_DIR

METHODS=("fedora+kd")
DATASETS=("sst2")

for METHOD in "${METHODS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo "Running $METHOD on $DATASET with $MODEL_NAME..."
        OUTPUT_FILE="${OUTPUT_DIR}/${METHOD}_${DATASET}_${MODEL_NAME}_${RANK}.log"
        
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python main.py \
            --method $METHOD \
            --dataset $DATASET \
            --model_name $MODEL_NAME \
            --num_epochs $NUM_EPOCHS \
            --num_rounds $NUM_ROUNDS \
            --rank $RANK \
            --lr $LR \
            --output_file $OUTPUT_FILE \
            > $OUTPUT_FILE 2>&1
        
        if [ $? -eq 0 ]; then
            echo "Completed $METHOD on $DATASET. Results saved to $OUTPUT_FILE"
        else
            echo "Error running $METHOD on $DATASET. Check $OUTPUT_FILE."
        fi
        sleep 5
    done
done

echo "All experiments completed!"