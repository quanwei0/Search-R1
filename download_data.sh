source activate /opt/conda/envs/searchr1-test

pip install verl==0.1.0 --index-url https://pypi.org/simple
save_path=./data
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz

# python scripts/data_process/nq_search.py
huggingface-cli download --repo-type dataset --local-dir ./data/nq_hotpotqa_train  PeterJinGo/nq_hotpotqa_train
pip uninstall verl -y
