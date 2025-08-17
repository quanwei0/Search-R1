source /code/hongpaul-sandbox/search/miniconda/bin/activate
conda init
conda activate search
pip install verl==0.1.0
python scripts/data_process/nq_search.py --data_source hotpotqa --local_dir ./data/hotpotqa_search
pip uninstall verl -y