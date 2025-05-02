#!/bin/bash

# ===============================================================
# FEDERATED LEARNING EXPERIMENT RUNNER
# ===============================================================
# This script provides flexible execution of federated learning experiments:
# - Run experiments on specific GPUs
# - Run a single experiment on one GPU
# - Queue multiple distributed experiments to run sequentially across available GPUs
#
# USAGE MODES:
# 1. Single experiment on specific GPU:
#    ./run_experiment.sh --gpu 0 --method fedora --dataset sst2
#
# 2. Queue experiments across multiple GPUs (distributed):
#    ./run_experiment.sh --queue --gpus "0,1,2,3" --methods "fedora,kd,base" --datasets "sst2,qqp"
#
# 3. Queue experiments on a single GPU:
#    ./run_experiment.sh --queue --gpu 2 --methods "fedora,kd,base" --datasets "sst2,qqp"
# All methods
# --methods "base","fedora","ns","ns_manifold","kd","muon","fedora+kd","fedora+muon","fedora+kd+muon","kd+muon","muon+ns","muon+ns_manifold"

# Note that!!!!!!!!!!!!!!!!!
# Need quotes:
# - Multiple comma-separated values: --methods "fedora,kd,base"
# - Multiple comma-separated GPUs: --gpus "0,1,2,3"
# - Any argument with spaces: --model "roberta base"
# Don't need quotes:
# - Single values without special characters: --gpu 0
# - Numeric values: --batch-size 64
# - Single method/dataset: --method fedora
# - Flags without arguments: --queue
# ===============================================================

# ---- HELPER FUNCTIONS ----
print_error() {
    echo -e "\e[31mERROR: $1\e[0m"  # Print error in red
}

print_success() {
    echo -e "\e[32m$1\e[0m"  # Print success in green
}

print_warning() {
    echo -e "\e[33m$1\e[0m"  # Print warning in yellow
}

# ---- DEFAULT CONFIGURATION PARAMETERS ----
NUM_EPOCHS=2
NUM_ROUNDS=10
RANK=4
LR=3e-4
OUTPUT_DIR="results"
MODEL_NAME="roberta-base"

# ---- MEMORY OPTIMIZATION PARAMETERS ----
BATCH_SIZE=128                   # Reduces memory usage (original default: 32)
GRADIENT_ACCUMULATION_STEPS=1   # Accumulates gradients over multiple steps
USE_FP16=""                     # Mixed precision disabled by default

# ---- DEFAULT METHODS AND DATASETS ----
DEFAULT_METHODS=("fedora+muon" "fedora+kd+muon" "kd+muon" "muon+ns")
DEFAULT_DATASETS=("sst2")
DEFAULT_GPUS=(0)

# ---- EXECUTION MODE ----
QUEUE_MODE=false
SINGLE_MODE=true

# ---- PORT CONFIGURATION FOR MUON METHODS ----
BASE_PORT=12355  # Starting port number

# ---- PARSE COMMAND LINE ARGUMENTS ----
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --gpu)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --gpu"
        exit 1
      fi
      GPU_ID="$2"
      DEFAULT_GPUS=($GPU_ID)
      shift 2
      ;;
    --gpus)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --gpus"
        exit 1
      fi
      IFS=',' read -r -a DEFAULT_GPUS <<< "$2"
      shift 2
      ;;
    --method)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --method"
        exit 1
      fi
      METHOD="$2"
      DEFAULT_METHODS=("$METHOD")
      shift 2
      ;;
    --methods)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --methods"
        exit 1
      fi
      IFS=',' read -r -a DEFAULT_METHODS <<< "$2"
      shift 2
      ;;
    --dataset)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --dataset"
        exit 1
      fi
      DATASET="$2"
      DEFAULT_DATASETS=("$DATASET")
      shift 2
      ;;
    --datasets)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --datasets"
        exit 1
      fi
      IFS=',' read -r -a DEFAULT_DATASETS <<< "$2"
      shift 2
      ;;
    --queue)
      QUEUE_MODE=true
      SINGLE_MODE=false
      shift
      ;;
    --epochs)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --epochs"
        exit 1
      fi
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --rounds)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --rounds"
        exit 1
      fi
      NUM_ROUNDS="$2"
      shift 2
      ;;
    --rank)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --rank"
        exit 1
      fi
      RANK="$2"
      shift 2
      ;;
    --lr)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --lr"
        exit 1
      fi
      LR="$2"
      shift 2
      ;;
    --batch-size)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --batch-size"
        exit 1
      fi
      BATCH_SIZE="$2"
      shift 2
      ;;
    --grad-accum)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --grad-accum"
        exit 1
      fi
      GRADIENT_ACCUMULATION_STEPS="$2"
      shift 2
      ;;
    --model)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --model"
        exit 1
      fi
      MODEL_NAME="$2"
      shift 2
      ;;
    --output-dir)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --output-dir"
        exit 1
      fi
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --fp16)
      USE_FP16="--fp16"
      shift
      ;;
    --no-fp16) # Corrected typo here
      USE_FP16=""
      shift
      ;;
    --help)
      echo "==============================================================="
      echo "FEDERATED LEARNING EXPERIMENT RUNNER - HELP"
      # ... (rest of help message) ...
      echo "==============================================================="
      exit 0
      ;;
    *)
      print_error "Unknown option: $key"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# ---- CREATE OUTPUT DIRECTORY ----
mkdir -p $OUTPUT_DIR 2>/dev/null || { print_error "Failed to create output directory: $OUTPUT_DIR"; exit 1; }

# ---- DISPLAY CONFIGURATION ----
echo "==============================================================="
echo "FEDERATED LEARNING EXPERIMENT CONFIGURATION"
echo "==============================================================="

# ---- EXPERIMENT RUNNER FUNCTION ----
run_experiment() {
    local method=$1
    local dataset=$2
    local cuda_device=$3
    local exp_id=$4

    # Define FINAL log file (written to by main.py and post-processing)
    OUTPUT_FILE="${OUTPUT_DIR}/${method}_${dataset}_${MODEL_NAME}_${RANK}_gpu${cuda_device}.log"
    # Define TEMPORARY verbose log file
    VERBOSE_LOG_FILE="${OUTPUT_DIR}/${method}_${dataset}_${MODEL_NAME}_${RANK}_gpu${cuda_device}.verbose.log"

    # Clear previous verbose log if it exists
    rm -f "$VERBOSE_LOG_FILE"

    # Each MUON process needs a unique port to avoid conflicts
    local port=$((BASE_PORT + exp_id*100 + 10#$cuda_device))

    echo "Running $method on $dataset with GPU $cuda_device (Experiment ID: $exp_id)"
    echo "  - Final Log: $OUTPUT_FILE"
    echo "  - Verbose Log: $VERBOSE_LOG_FILE"


    # Extra configuration for MUON methods
    PORT_PARAM=""
    if [[ "$method" == *"muon"* ]]; then
        PORT_PARAM="--master_port $port"
        echo "  - MUON method detected: Using communication port $port"
    fi

    # Record start time
    START_TIME=$(date +%s)

    # Run the experiment:
    # - Let main.py write its summary to $OUTPUT_FILE
    # - Use tee to show everything on terminal (/dev/tty)
    # - Use tee again to dump *everything* raw into $VERBOSE_LOG_FILE
    {
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
            --output_file "$OUTPUT_FILE" \
            --cuda_device 0 \
            $PORT_PARAM \
            2>&1 | tee /dev/tty | tee -a "$VERBOSE_LOG_FILE" # Log ALL raw output

        # Capture the exit status of the Python script (from the pipe)
        RESULT=${PIPESTATUS[0]}
    } # The curly braces ensure PIPESTATUS is captured correctly

    # Record end time and duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(( (DURATION % 3600) / 60 ))
    SECONDS=$((DURATION % 60))

    # --- Post-Processing Step ---
    echo "Processing verbose log into $OUTPUT_FILE..."
    # 1. Clean control characters from verbose log
    # 2. Filter for desired lines (milestones, key info, errors) using grep
    # 3. Append the filtered lines to the main OUTPUT_FILE
    if [ -f "$VERBOSE_LOG_FILE" ]; then
        sed -E 's/\r//g; s/\x1b\[[0-9;]*[a-zA-Z]//g' "$VERBOSE_LOG_FILE" | \
        grep -E '100%[ |].*it/s|avg loss:|Aggregating|Global Evaluation Result:|Completed|Error|Warning|Traceback|--- Round|--- Training Start|--- Training End' >> "$OUTPUT_FILE"
        # Optional: Remove the verbose log after processing if desired
        # rm -f "$VERBOSE_LOG_FILE"
        echo "Log processing complete."
    else
        print_warning "Verbose log file not found: $VERBOSE_LOG_FILE"
    fi
    # --- End Post-Processing ---


    # Final status reporting (print to terminal, append final status to $OUTPUT_FILE)
    if [ $RESULT -eq 0 ]; then
        printf "\n%s\n" "$(print_success "✓ Completed $method on $dataset using GPU $cuda_device")" >> "$OUTPUT_FILE"
        printf "  Results summary appended to: %s\n" "$OUTPUT_FILE" >> "$OUTPUT_FILE"
        printf "  Duration: %sh %sm %ss\n" $HOURS $MINUTES $SECONDS >> "$OUTPUT_FILE"
        # Also print to terminal
        print_success "✓ Completed $method on $dataset using GPU $cuda_device"
        echo "  Results summary appended to: $OUTPUT_FILE"
        echo "  Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    else
        printf "\n%s\n" "$(print_error "✗ Error running $method on $dataset on GPU $cuda_device")" >> "$OUTPUT_FILE"
        printf "  Filtered error details appended to: %s\n" "$OUTPUT_FILE" >> "$OUTPUT_FILE" # Point to the main log
        printf "  Refer to full verbose log for complete details: %s\n" "$VERBOSE_LOG_FILE" >> "$OUTPUT_FILE" # Mention verbose log
        printf "  Duration before error: %sh %sm %ss\n" $HOURS $MINUTES $SECONDS >> "$OUTPUT_FILE"
        # Also print to terminal
        print_error "✗ Error running $method on $dataset on GPU $cuda_device"
        echo "  Filtered error details appended to: $OUTPUT_FILE" # Point to the main log
        echo "  Refer to full verbose log for complete details: $VERBOSE_LOG_FILE" # Mention verbose log
        echo "  Duration before error: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    fi

    return $RESULT
}

# ---- MAIN EXECUTION ----
# Count total experiments
total_experiments=${#DEFAULT_METHODS[@]}
total_experiments=$((total_experiments * ${#DEFAULT_DATASETS[@]}))
echo "Total experiments to run: $total_experiments"

# Track experiment ID
exp_id=0

# Choose execution mode
if $QUEUE_MODE; then
    # Queue mode: run experiments in parallel across available GPUs (one per GPU)
    echo "Starting queued experiments (parallel execution across GPUs)..."
    echo "--------------------------------------------------------"

    # Create an array to track running processes on each GPU
    declare -A running_processes
    for gpu in "${DEFAULT_GPUS[@]}"; do
        running_processes[$gpu]=""
    done

    # Set up counters
    total_completed=0
    exp_waiting=()  # Store waiting experiments as "method dataset exp_id"

    # Queue up all experiments
    for dataset in "${DEFAULT_DATASETS[@]}"; do
        for method in "${DEFAULT_METHODS[@]}"; do
            exp_waiting+=("$method $dataset $exp_id")
            exp_id=$((exp_id + 1))
        done
    done

    # Process the queue, utilizing all available GPUs in parallel
    processed_count=0
    while [[ $processed_count -lt $total_experiments ]]; do
        # Check for finished processes and free up GPUs
        for gpu in "${!running_processes[@]}"; do # Iterate over keys (GPU IDs)
             pid=${running_processes[$gpu]}
             if [[ -n "$pid" ]] && ! kill -0 $pid 2>/dev/null; then
                 # Process finished
                 running_processes[$gpu]="" # Mark GPU as free
                 processed_count=$((processed_count + 1))
                 echo "GPU $gpu finished process $pid. Total completed: $processed_count/$total_experiments"
            fi
        done

        # Check for free GPUs and start new jobs from the waiting queue
        for gpu in "${!running_processes[@]}"; do
            if [[ -z "${running_processes[$gpu]}" ]] && [[ ${#exp_waiting[@]} -gt 0 ]]; then
                 # Get the next experiment
                 next_exp="${exp_waiting[0]}"
                 exp_waiting=("${exp_waiting[@]:1}")  # Remove the first element

                 # Parse experiment details
                 read -r next_method next_dataset next_exp_id <<< "$next_exp"

                 # Run in background
                 echo "Starting experiment $next_exp_id ($next_method/$next_dataset) on GPU $gpu..."
                 run_experiment "$next_method" "$next_dataset" "$gpu" "$next_exp_id" &
                 running_processes[$gpu]=$! # Store the PID

                 # Small delay to avoid race conditions/port conflicts
                 sleep 1
             fi
        done

        # Sleep briefly before checking statuses again if queue is not empty or processes are running
        if [[ ${#exp_waiting[@]} -gt 0 ]] || [[ $(jobs -p | wc -l) -gt 0 ]]; then
             sleep 2
        fi
    done

    # Final wait just in case (should be redundant if logic above is correct)
    echo "Waiting for any final background processes to complete..."
    wait

    echo "--------------------------------------------------------"
    print_success "All queued experiments completed!"

else
    # Single mode: run just one experiment
    echo "Starting single experiment..."
    echo "--------------------------------------------------------"

    # Use first method, dataset, and GPU
    method=${DEFAULT_METHODS[0]}
    dataset=${DEFAULT_DATASETS[0]}
    gpu=${DEFAULT_GPUS[0]}

    run_experiment "$method" "$dataset" $gpu $exp_id

    echo "--------------------------------------------------------"
    print_success "Single experiment completed!"
fi
