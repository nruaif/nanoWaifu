from huggingface_hub import hf_hub_download

repo_id = "animetimm/danbooru-wdtagger-v4-w640-ws-full"

# By default download first 60 shards (00000.tar to 00059.tar)
files = [f"train/{i:05d}.tar" for i in range(60)]

for f in files:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=f,
        repo_type="dataset",  # important!
        local_dir="train",
        local_dir_use_symlinks=False
    )
    print(f"Downloaded: {path}")
