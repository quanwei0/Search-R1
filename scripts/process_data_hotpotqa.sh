# eval "$(/mnt/home/siliang/miniconda3/bin/conda shell.bash hook)"

# conda activate searchr1

# save_path=./data
# python scripts/download.py --save_path $save_path
# cat $save_path/part_* > $save_path/e5_Flat.index
# gzip -d $save_path/wiki-18.jsonl.gz
source /code/hongpaul-sandbox/search/miniconda/bin/activate
conda init
conda activate search
pip install verl==0.1.0
python scripts/data_process/nq_search.py --local_dir ./data/nq_hotpotqa_train
pip uninstall verl -y