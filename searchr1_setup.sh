conda init
conda create -n searchr1-test python=3.9 -y
conda activate searchr1-test

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
cd outlines
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OUTLINES=0.0.46
pip install -e . 

pip3 install vllm==0.6.3

# just install the dependendcy
pip install verl==0.1.0
pip uninstall verl -y

# flash attention 2
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip install wandb