# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Search-R1 is a reinforcement learning framework for training reasoning and searching interleaved LLMs - language models that learn to reason and make tool calls (e.g., to search engines). Built on top of veRL (Volcano Engine Reinforcement Learning), it extends DeepSeek-R1 concepts with interleaved search engine access and provides an open-source RL training pipeline.

## Key Commands

### Environment Setup
```bash
# Main environment
conda create -n searchr1 python=3.9
conda activate searchr1
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3
pip install -e .
pip3 install flash-attn --no-build-isolation
pip install wandb

# Retriever environment (optional, separate environment)
conda create -n retriever python=3.10
conda activate retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```

### Training Commands
```bash
# PPO training
bash train_ppo.sh

# GRPO training  
bash train_grpo.sh

# Different model sizes and advantage estimators
bash train_ppo_7b_gae.sh
bash train_ppo_7b_weighted_gae.sh
bash train_ppo_7b_turn_level_gae.sh
```

### Data Processing
```bash
# Process NQ dataset
python scripts/data_process/nq_search.py

# Download corpus and indexing
python scripts/download.py --save_path ./data
cat ./data/part_* > ./data/e5_Flat.index
gzip -d ./data/wiki-18.jsonl.gz
```

### Retrieval Server
```bash
# Launch local retrieval server
conda activate retriever
bash retrieval_launch.sh
```

### Inference
```bash
conda activate searchr1
python infer.py
```

### Building Search Indices
```bash
bash search_r1/search/build_index.sh
```

## Architecture Overview

### Core Components

**1. Training Pipeline (verl/trainer/)**
- `main_ppo.py`: Main PPO training entry point with reward management
- `ppo/ray_trainer.py`: Core Ray-based PPO trainer with multi-agent support
- `config/ppo_trainer.yaml`: Default PPO configuration template

**2. Search Integration (search_r1/)**
- `llm_agent/generation.py`: LLM generation manager for multi-turn interactions
- `search/retrieval_server.py`: Local retrieval server implementation
- Supports multiple search engines: BM25, dense retrievers, online APIs

**3. Models and Workers (verl/workers/)**
- `fsdp_workers.py`: FSDP-based model workers
- `megatron_workers.py`: Megatron-LM based workers  
- `rollout/vllm_rollout/`: vLLM-based rollout workers for generation

**4. Reward Functions (verl/utils/reward_score/)**
- `qa_em_new.py`: Question-answering exact match rewards
- Multiple reward types: answer correctness, format correctness, retrieval correctness

### Key Design Patterns

**Multi-Turn Generation**: The system supports complex multi-turn interactions where LLMs can:
- Generate reasoning steps
- Make search queries
- Process retrieved information
- Continue reasoning with new context

**Hybrid Engine**: Uses Ray for distributed coordination with:
- Actor-rollout workers for generation
- Critic workers for value estimation  
- Reference policy workers for KL penalty
- Reward model workers (optional)

**Flexible Reward Design**: Supports multiple reward components:
- Answer correctness (exact match)
- Format correctness (proper tool call formatting)
- Retrieval correctness (search quality)
- Mixed outcome rewards combining multiple signals

## Configuration System

The system uses Hydra for configuration management with:
- Base configs in `verl/trainer/config/`
- Override parameters via command line
- Support for different model sizes, RL algorithms, and search engines

Key config sections:
- `data`: Dataset paths and preprocessing settings
- `actor_rollout_ref`: Model paths, FSDP settings, rollout parameters
- `critic`: Critic model configuration
- `algorithm`: RL algorithm settings (PPO/GRPO, advantage estimation)
- `trainer`: Training loop settings, logging, checkpointing
- `retriever`: Search engine URL and parameters

## Development Notes

**Multi-Node Training**: Supports distributed training across multiple nodes for 30B+ models (see `docs/multinode.md`)

**Search Engine Integration**: Pluggable search engine architecture supporting:
- Local sparse retrievers (BM25)
- Local dense retrievers (flat + ANN indexing)
- Online search APIs (Google, Bing, etc.)

**Advantage Estimation**: Multiple methods supported:
- GAE (Generalized Advantage Estimation)
- Masked GAE
- Turn-level GAE
- Weighted GAE
- GRPO (Group Relative Policy Optimization)

**State Masking**: Supports masking of intermediate reasoning tokens vs. action tokens for more targeted learning

## Testing and Validation

The system includes validation loops that:
- Run inference on validation sets
- Compute multiple reward metrics
- Track performance across different data sources
- Save trajectory examples for inspection

Key validation files are saved to `outputs/log_val_traj/` with decoded trajectories showing the full reasoning and search process.

## Important File Locations

- Training scripts: `train_*.sh` (root directory)
- Main training code: `verl/trainer/main_ppo.py`
- Core trainer: `verl/trainer/ppo/ray_trainer.py`
- Search integration: `search_r1/llm_agent/generation.py`
- Data processing: `scripts/data_process/`
- Configuration: `verl/trainer/config/ppo_trainer.yaml`