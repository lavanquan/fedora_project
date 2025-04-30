#!/bin/bash

# ===============================================================
# SINGLE-GPU FEDERATED LEARNING EXPERIMENT RUNNER
# ===============================================================
# This script runs a federated learning experiment on a single specified GPU.
# 
# FEATURES:
# - Runs on a specific GPU of your choice
# - Includes memory optimization parameters to prevent OOM errors
# - Handles MUON methods with random port assignment
# - Provides detailed configuration output
#
# USAGE:
# ./run_single_gpu.sh [GPU_ID] [METHOD] [DATASET]
#
# EXAMPLES:
# ./run_single_gpu.sh 0 base sst2           # Run base method on GPU 0 with SST-2 dataset
# ./run_single_gpu.sh 2 fedora+muon qqp     # Run fedora+muon on GPU 2 with QQP dataset
# ./run_single_gpu.sh 1                     # Run default method/dataset on GPU 1
# ===============================================================

# ---- CHECK COMMAND LINE ARGUMENTS ----
if [ -z "$1" ]; then
    echo "ERROR: GPU ID is required"
    echo ""
    echo "USAGE: ./run_single_gpu.sh [GPU_ID] [METHOD] [DATASET]"
    echo "EXAMPLES:"
    echo "  ./run_single_gpu.sh 0 base sst2"
    echo "  ./run_single_gpu.sh 2 fedora+muon qqp"
    exit 1
fi

# ---- PARSE COMMAND LINE ARGUMENTS ----
GPU_ID=$1
METHOD=${2:-"fedora+muon"}  # Default method if not specified
DATASET=${3:-"sst2"}        # Default dataset if not specified

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

# To disable mixed precision, uncomment the line below
# USE_FP16=""

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# ---- PORT CONFIGURATION FOR MUON METHODS ----
# Generate a random port between 12000-15000 to avoid conflicts
RANDOM_PORT=$((12000 + RANDOM % 3000))

# ---- PREPARE OUTPUT FILE ----
OUTPUT_FILE="${OUTPUT_DIR}/${METHOD}_${DATASET}_${MODEL_NAME}_${RANK}_gpu${GPU_ID}.log"

# ---- DISPLAY CONFIGURATION ----
echo "==============================================================="
echo "FEDERATED LEARNING EXPERIMENT CONFIGURATION"
echo "==============================================================="
echo "  - GPU ID: $GPU_ID"
echo "  - Method: $METHOD"
echo "  - Dataset: $DATASET"
echo "  - Model: $MODEL_NAME"
echo ""
echo "TRAINING PARAMETERS:"
echo "  - Epochs per round: $NUM_EPOCHS"
echo "  - Federated learning rounds: $NUM_ROUNDS"
echo "  - LoRA rank: $RANK"
echo "  - Learning rate: $LR"
echo ""
echo "MEMORY OPTIMIZATION:"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  - Effective batch size: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "  - Mixed precision: $(if [[ -n "$USE_FP16" ]]; then echo "Enabled"; else echo "Disabled"; fi)"
echo ""
echo "OUTPUT: $OUTPUT_FILE"
echo "==============================================================="

# ---- HANDLE MUON METHODS ----
# Only add port parameter for MUON methods
PORT_PARAM=""
if [[ "$METHOD" == *"muon"* ]]; then
    PORT_PARAM="--master_port $RANDOM_PORT"
    echo "MUON method detected: Using communication port $RANDOM_PORT"
    echo "Note: If you see ncclSystemError, this is expected and training should still complete"
    echo "==============================================================="
fi

# ---- RUN THE EXPERIMENT ----
echo "Starting experiment..."
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
    --method $METHOD \
    --dataset $DATASET \
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
    $PORT_PARAM

RESULT=$?
echo ""
if [ $RESULT -eq 0 ]; then
    echo "✓ Experiment completed successfully!"
    echo "  Results saved to: $OUTPUT_FILE"
else
    echo "✗ Experiment encountered an error. Check the output above and log file."
fi

# ---- MEMORY OPTIMIZATION GUIDE ----
echo ""
echo "==============================================================="
echo "MEMORY OPTIMIZATION GUIDE"
echo "==============================================================="
echo "Current configuration:"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  - Effective batch size: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "  - Mixed precision: $(if [[ -n "$USE_FP16" ]]; then echo "Enabled"; else echo "Disabled"; fi)"
echo ""
echo "If you encounter out-of-memory errors:"
echo "  - Try batch_size=4, gradient_accumulation_steps=8"
echo "  - For extreme cases: batch_size=2, gradient_accumulation_steps=16"
echo ""
echo "If your GPU has plenty of memory and you want faster training:"
echo "  - Try batch_size=16, gradient_accumulation_steps=2"
echo "  - Original settings: batch_size=32, gradient_accumulation_steps=1"
echo ""
echo "To disable optimizations and run with original settings:"
echo "  - Edit this script to set batch_size=32, gradient_accumulation_steps=1"
echo "  - And comment out the USE_FP16 variable"
echo "==============================================================="
