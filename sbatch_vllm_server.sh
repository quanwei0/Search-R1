#!/bin/bash

#SBATCH --partition=hpc-mid

#SBATCH --gres=gpu:4

#SBATCH --time=96:00:00

#SBATCH --job-name=h100_vllm_server

#SBATCH --output=my_job_%j.out

#SBATCH --error=my_job_%j.err


source /mnt/home/siliang/miniconda3/bin/activate
conda init


conda activate vllm

bash vllm_serve/vllm_server.sh $CUDA_VISIBLE_DEVICES "0.0.0.0" 8002 "openai/gpt-oss-120b"