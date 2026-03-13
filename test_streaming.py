import torch
from dataset import WDSLoader
import time

def test_remote_streaming():
    url = "https://huggingface.co/datasets/Shio-Koube/150k-anime/resolve/main/train/00001.tar"
    print(f"🌐 Testing remote streaming from: {url}")
    
    # 1. Initialize Loader
    # Using small batch size for testing
    loader_obj = WDSLoader(
        url=url,
        image_size=512,
        batch_size=2,
        num_workers=0 # Single process for easier debugging
    )
    
    dataloader = loader_obj.make_loader()
    
    # 2. Iterate and print samples
    print("⏳ Fetching first batch...")
    start_time = time.time()
    
    try:
        # Get first batch
        batch = next(iter(dataloader))
        elapsed = time.time() - start_time
        
        print(f"✅ Successfully fetched batch in {elapsed:.2f}s")
        
        # Batch is a list of (image, prompt, coords) because of our custom collate_fn
        for i, (img, prompt, coords) in enumerate(batch):
            print(f"\n🖼️ Sample {i+1}:")
            print(f"   Image shape: {img.shape}")
            print(f"   Prompt: {prompt[:200]}...")
            print(f"   Coords: {coords}")
            
    except Exception as e:
        print(f"❌ Error during streaming: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_remote_streaming()
