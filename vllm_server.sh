#!/bin/bash

# Configuration parameters
# CUDA_DEVICES=${1:-"0"}
# export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
TENSOR_PARALLEL_SIZE=$(echo $CUDA_DEVICES | tr ',' '\n' | wc -l)
HOST="0.0.0.0"
PORT=8002
MODEL="openai/gpt-oss-20b"


# Display system info and configuration
cat << EOF
CUDA Devices: $CUDA_VISIBLE_DEVICES
Current IP: $(hostname -I | cut -d' ' -f1)

Starting VLLM server:
  Model: $MODEL
  CUDA Devices: $CUDA_DEVICES
  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE
  Host: $HOST
  Port: $PORT

EOF

# Start VLLM server
vllm serve $MODEL \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --host $HOST \
    --port $PORT \
    --disable-log-stats \
