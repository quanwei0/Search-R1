eval "$(/mnt/home/siliang/miniconda3/bin/conda shell.bash hook)"

conda activate searchr1

save_path=./data
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz

python scripts/data_process/nq_search.py --local_dir ./data/nq_search