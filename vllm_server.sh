#!/bin/bash

# Configuration parameters
CUDA_DEVICES=${1:-"3"}
PORT=8002
HOST="localhost"
TENSOR_PARALLEL_SIZE=$(echo $CUDA_DEVICES | tr ',' '\n' | wc -l)
MODEL="openai/gpt-oss-20b"

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

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
    --disable-log-stats

# Display server info
cat << EOF

VLLM server started with PID: $!
Server available at: http://$HOST:$PORT
EOF