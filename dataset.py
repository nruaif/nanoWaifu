import webdataset as wds
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import io
from PIL import Image
import json

def warn_and_continue(exn):
    print(f"Warning: {exn}")
    return True

class WDSLoader:
    def __init__(self, url, tags_path, image_size=64, batch_size=16, num_workers=4):
        self.url = url
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tag_vocab = self.load_tag_vocab(tags_path)
        self.num_tags = len(self.tag_vocab)

        # Transforms parameters
        self.scale = (0.5, 1.0)
        self.ratio = (3. / 4., 4. / 3.)

    def load_tag_vocab(self, tags_path):
        """Load tag vocabulary from a text file (one tag per line). Returns {tag: index} dict."""
        tag_vocab = {}
        with open(tags_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                tag = line.strip()
                if tag and tag not in tag_vocab:
                    tag_vocab[tag] = len(tag_vocab)
        return tag_vocab

    def preprocess(self, sample):
        # Find image key
        image_key = None
        for key in ["image", "jpg", "jpeg", "png", "webp"]:
            if key in sample:
                image_key = key
                break
        
        if image_key is None:
            if not hasattr(self, "_log_missing_key_count"): self._log_missing_key_count = 0
            if self._log_missing_key_count < 5:
                print(f"Skipping sample: No image key found. Available keys: {list(sample.keys())}")
                self._log_missing_key_count += 1
            return None

        # Decode image
        try:
            image_bytes = sample[image_key]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None

        # Decode JSON for tags
        try:
            if "json" in sample:
                meta = json.loads(sample["json"])
            else:
                meta = {}
            
            tags_str = meta.get("tags", "")
            # Build multi-hot tag vector
            tag_vec = torch.zeros(self.num_tags, dtype=torch.float32)
            if tags_str:
                for tag in tags_str.split(" "):
                    tag = tag.strip()
                    if tag in self.tag_vocab:
                        tag_vec[self.tag_vocab[tag]] = 1.0
        except Exception as e:
            print(f"Error parsing metadata: {e}")
            return None

        # Random Resized Crop logic for View 1
        i1, j1, h1, w1 = transforms.RandomResizedCrop.get_params(image, scale=self.scale, ratio=self.ratio)
        W, H = image.size
        rel_coords1 = torch.tensor([i1 / H, j1 / W, h1 / H, w1 / W], dtype=torch.float32)

        # Apply crop and resize for View 1
        image1 = F.resized_crop(image, i1, j1, h1, w1, size=(self.image_size, self.image_size))
        image1 = F.to_tensor(image1)
        image1 = (image1 - 0.5) * 2.0 

        # Random Resized Crop logic for View 2
        i2, j2, h2, w2 = transforms.RandomResizedCrop.get_params(image, scale=self.scale, ratio=self.ratio)
        
        # Apply crop and resize for View 2
        image2 = F.resized_crop(image, i2, j2, h2, w2, size=(self.image_size, self.image_size))
        image2 = F.to_tensor(image2)
        image2 = (image2 - 0.5) * 2.0 

        return {
            "image1": image1,
            "coords1": rel_coords1,
            "image2": image2,
            "tags": tag_vec,
        }

    def make_loader(self):
        dataset = (
            wds.WebDataset(self.url, nodesplitter=wds.split_by_node, handler=warn_and_continue,)
            .shuffle(1000)
            .map(self.preprocess, handler=warn_and_continue,)
            .select(lambda x: x is not None)
            .to_tuple("image1", "coords1", "image2", "tags", handler=warn_and_continue,)
            .batched(self.batch_size, partial=False)
        )
        
        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader
