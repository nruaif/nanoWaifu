import torch
from dataset import WDSLoader
import time

def test_streaming():
    url = "https://huggingface.co/datasets/Shio-Koube/150k-anime/resolve/main/train/00001.tar"
    batch_size = 8
    num_batches = 10
    image_size = 256
    
    print(f"Testing streaming from: {url}")
    print(f"Batch size: {batch_size}, Number of batches: {num_batches}")
    
    # Initialize loader
    # use_advanced_captions=True by default which handles the json/tags logic
    loader_obj = WDSLoader(
        url=url,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=0,
        use_advanced_captions=True
    )
    
    dataloader = loader_obj.make_loader()
    
    start_time = time.time()
    
    try:
        data_iter = iter(dataloader)
        for i in range(num_batches):
            batch_start = time.time()
            batch = next(data_iter)
            # batch is (images, prompts, coords)
            images, prompts, coords = batch
            
            elapsed = time.time() - batch_start
            print(f"Batch {i+1}/{num_batches} loaded in {elapsed:.2f}s")
            print(f"  Images shape: {images.shape}")
            print(f"  First prompt snippet: {prompts[0][:100]}...")
            
    except Exception as e:
        print(f"Error during streaming: {e}")
    
    total_time = time.time() - start_time
    print(f"\nStreaming test complete in {total_time:.2f}s")

if __name__ == "__main__":
    test_streaming()
