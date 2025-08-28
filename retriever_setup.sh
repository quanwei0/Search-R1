conda create -n retriever python=3.10 -y
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install transformers==4.53.3 datasets==4.0.0 pyserini==1.2.0

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y

## API function
pip install uvicorn fastapi