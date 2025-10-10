Install search-r1 env
```bash
conda create -n searchr1 python=3.9 -y
conda activate searchr1
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
pip install wandb
```

Install retriever env
```bash
conda create -n retriever python=3.10 -y
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y

## API function
pip install uvicorn fastapi
```

Download the indexing and corpus
```bash
save_path=./data
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

Process the NQ dataset
```bash
python scripts/data_process/nq_search.py --data_source nq --local_dir ./data/nq_search
```

Download the dataset in the paper
```bash
huggingface-cli download --repo-type dataset --local-dir ./data/nq_hotpotqa_train  PeterJinGo/nq_hotpotqa_train
```