conda create -n searchr1 python=3.9 -y
source activate searchr1

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install vllm==0.6.3


pip install -e .


pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install wandb
pip install transformers datasets pyserini

pip install numpy==1.26.4
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y
pip install uvicorn fastapi
