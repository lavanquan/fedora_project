#!/bin/bash

# ===============================================================
# PARALLEL FEDERATED LEARNING EXPERIMENT RUNNER
# ===============================================================
# This script runs multiple federated learning experiments in parallel
# across available GPUs, with each experiment assigned to a specific GPU.
# 
# FEATURES:
# - Distributes experiments across multiple GPUs automatically
# - Includes memory optimization parameters to prevent OOM errors
# - Handles MUON methods with unique port assignments
# - Saves output to separate log files
#
# USAGE:
# ./parallelly_run.sh
# 
# To customize which methods/datasets to run, modify the METHODS and DATASETS arrays below
# ===============================================================

# ---- EXPERIMENT CONFIGURATION PARAMETERS ----
NUM_EPOCHS=2
NUM_ROUNDS=10
RANK=4
LR=3e-4
OUTPUT_DIR="results"
MODEL_NAME="roberta-large"

# ---- MEMORY OPTIMIZATION PARAMETERS ----
# These settings help prevent out-of-memory errors
BATCH_SIZE=8                    # Reduces memory usage (original default: 32)
GRADIENT_ACCUMULATION_STEPS=4   # Accumulates gradients over multiple steps
USE_FP16="--fp16"               # Enables mixed precision training

# Comment out the line below to disable mixed precision if it causes issues
# USE_FP16=""

# ---- METHODS AND DATASETS TO RUN ----
# Add or remove methods/datasets as needed
METHODS=("fedora+muon" "fedora+kd+muon" "kd+muon" "muon+ns")
# METHODS=("fedora+muon" "base")  # Uncomment to use fewer methods
DATASETS=("sst2")
# DATASETS=("sst2" "qqp")  # Uncomment to add more datasets

# ---- GPU CONFIGURATION ----
# Set the number of GPUs available on your system
total_gpus=4  # Modify this based on your hardware

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# ---- PORT CONFIGURATION FOR MUON METHODS ----
# MUON uses distributed training requiring unique ports
BASE_PORT=12355  # Starting port number

# ---- EXPERIMENT RUNNER FUNCTION ----
run_experiment() {
    local method=$1
    local dataset=$2
    local cuda_device=$3
    
    # Each MUON process needs a unique port to avoid conflicts
    local port=$((BASE_PORT + cuda_device*1000))
    
    echo "Running $method on $dataset with GPU $cuda_device"
    OUTPUT_FILE="${OUTPUT_DIR}/${method}_${dataset}_${MODEL_NAME}_${RANK}_gpu${cuda_device}.log"
    
    # Display extra information for MUON methods
    if [[ "$method" == *"muon"* ]]; then
        echo "  - MUON method detected: Using communication port $port"
        PORT_PARAM="--master_port $port"
    else
        PORT_PARAM=""
    fi
    
    # Run the experiment with specific CUDA device
    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
        --method $method \
        --dataset $dataset \
        --model_name $MODEL_NAME \
        --num_epochs $NUM_EPOCHS \
        --num_rounds $NUM_ROUNDS \
        --rank $RANK \
        --lr $LR \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        $USE_FP16 \
        --output_file $OUTPUT_FILE \
        --cuda_device 0 \
        $PORT_PARAM \
        > $OUTPUT_FILE 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed $method on $dataset using GPU $cuda_device"
        echo "  Results saved to $OUTPUT_FILE"
    else
        echo "✗ Error running $method on $dataset on GPU $cuda_device"
        echo "  Check $OUTPUT_FILE for details"
    fi
}

# ---- MAIN EXECUTION LOOP ----
echo "Starting parallel experiments across $total_gpus GPUs"
echo "--------------------------------------------------------"
echo "Methods: ${METHODS[@]}"
echo "Datasets: ${DATASETS[@]}"
echo "Memory settings: batch_size=$BATCH_SIZE, gradient_accumulation=$GRADIENT_ACCUMULATION_STEPS, fp16=$USE_FP16"
echo "--------------------------------------------------------"

# Distribute experiments across GPUs
gpu_counter=0

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        # Assign GPU using round-robin
        gpu=$((gpu_counter % total_gpus))
        
        # Run experiment in background to allow parallelism
        run_experiment "$method" "$dataset" $gpu &
        
        # Add a small delay to ensure logging clarity
        sleep 2
        
        # Move to next GPU for next job
        gpu_counter=$((gpu_counter + 1))
    done
done

# Wait for all background processes to finish
wait

echo "--------------------------------------------------------"
echo "All experiments completed!"
echo ""
echo "MEMORY OPTIMIZATION GUIDE:"
echo "- Current effective batch size: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "- If encountering OOM errors: Try batch_size=4, gradient_accumulation_steps=8"
echo "- For faster training (if memory allows): Try batch_size=16, gradient_accumulation_steps=2"
echo "--------------------------------------------------------"
