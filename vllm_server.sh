CUDA_DEVICES=${1:-"6,7"}
PORT=${2:-8002}
HOST=${3:-"localhost"}
TENSOR_PARALLEL_SIZE=${4:-2}
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

echo "Current IP address: $(hostname -I | cut -d' ' -f1)"
echo "Starting VLLM server with the following configuration:"
echo "CUDA Devices: $CUDA_DEVICES"
echo "Port: $PORT"
echo "Host: $HOST"
echo "Model: Qwen/Qwen2.5-72B-Instruct"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo ""

vllm serve Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --host $HOST \
    --port $PORT \
    --disable-log-stats \

echo "VLLM server started in background with PID: $!"
echo "Server will be available at: http://$HOST:$PORT"