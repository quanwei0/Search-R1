from huggingface_hub import HfApi

api = HfApi()

name = "nq-search-r1-ppo-qwen2.5-7b-em-gae-mixed-reward-new7"
folder_path = "/mnt/home/siliang/code/Search-R1/verl_checkpoints/" + name
repo_id = "quanwei0/" + name

api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    private=False,
    exist_ok=True
)


api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model",
)
