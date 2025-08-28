conda create -n searchr1 python=3.9 -y
conda activate searchr1

# git clone https://github.com/dottxt-ai/outlines.git
# cd outlines
# git checkout 0.0.46

# Remove all code related to pyairports, including:
# - modifications to outlines/types/__init__.py
# - deletion of outlines/types/airports.py
# - changes to pyproject.toml
# - updates to tests/test_types.py

pip install .

pip3 install vllm==0.6.3

# just install the dependendcy
pip install verl==0.1.0
pip uninstall verl

# flash attention 2
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip install wandb