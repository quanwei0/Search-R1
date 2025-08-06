# Accept parameters from calling script, use defaults if not provided
CUDA_DEVICES=${1:-"0,1,2,3"}
RETRIEVAL_PORT=${2:-"8001"}

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
file_path=./data
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

echo "Starting retrieval server with GPU: $CUDA_DEVICES, Port: $RETRIEVAL_PORT"

python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu \
                                            --port $RETRIEVAL_PORT &