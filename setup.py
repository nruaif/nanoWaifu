import os
import subprocess

def run_command(command):
    print(f"Executing: {command}")
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

def main():
    # 1. Install requirements and helper libraries
    print("--- Installing Dependencies ---")
    run_command("uv pip install -r requirements.txt")
    # Install additional libraries used in the new implementation
    run_command("uv pip install huggingface_hub matplotlib")

    # 2. Download dataset from Hugging Face
    print("\n--- Downloading Dataset ---")
    repo_id = "animetimm/danbooru-wdtagger-v4-w640-ws-150k"
    
    # We use huggingface-cli via uv run to ensure it uses the local environment
    # We include only the train shards to save space/time as requested
    download_cmd = (
        f"uv run huggingface-cli download {repo_id} "
        f"--repo-type dataset "
        f"--local-dir ./ "
        f"--include 'train/0000*.tar' "
        f"--quiet"
    )
    run_command(download_cmd)

    print("\n--- Setup Complete ---")
    print("Implementation: EPG with Unified SigLIP (Stage 1) and Flow Matching (Stage 2)")
    print("Architecture: ViT with Registers, GLU, and Spatial DW Conv")
    print(f"Data available at: {os.path.abspath('train/')}")
    print("Shards: 00001.tar through 00006.tar")

if __name__ == "__main__":
    main()
