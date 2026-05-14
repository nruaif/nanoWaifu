import webdataset as wds
import torch
from torch.utils.data import DataLoader
import pandas as pd
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import random
import io
import math
from PIL import Image
import json
import numpy as np


def warn_and_continue(exn):
    print(f"Warning: {exn}")
    return True


# --- 1. Preprocessing Function ---
def transform_sample(sample):
    """
    Transform a webdataset sample with advanced caption processing.
    Supports Pixiv metadata, tagger format, and aggressive tag dropping.
    """
    # Adapting to common WDS formats (jpg/png/webp)
    # Using 'pil' decoding in WDS pipeline usually yields standard keys or 'jpg', 'png' etc.
    # We check for common image keys

    image = None
    for key in ["jpg", "png", "webp", "jpeg"]:
        if key in sample:
            image = sample[key]
            break

    if image is None:
        raise ValueError("No image found in sample")

    # Handle prompts
    # Assuming 'json' or 'txt' or 'caption'
    prompt = ""
    full_prompt = ""

    if "json" in sample:
        json_data = sample["json"]
        if isinstance(json_data, bytes):
            json_data = json.loads(json_data.decode("utf-8"))
        # User specific tag logic from previous file
        if isinstance(json_data, dict):
            parts = []
            full_parts = []

            # --- Pixiv Extraction ---
            pixiv_tags_str = ""
            pixiv_title_str = ""

            if "pixiv" in json_data:
                p_data = json_data["pixiv"]

                # Extract Title
                if "work" in p_data and "titl" in p_data["work"]:
                    pixiv_title_str = str(p_data["work"]["titl"])

                # Extract Tags (Eng > Romaji > Orig)
                if "tags" in p_data and "tags" in p_data["tags"]:
                    p_tags_list = p_data["tags"]["tags"]
                    extracted_p_tags = []
                    if isinstance(p_tags_list, list):
                        for t_obj in p_tags_list:
                            t_val = t_obj.get("eng")
                            if not t_val:
                                t_val = t_obj.get("romaji")
                            if not t_val:
                                t_val = t_obj.get("orig")

                            if t_val:
                                extracted_p_tags.append(str(t_val))
                    pixiv_tags_str = " ".join(extracted_p_tags)

            # Check for new structure (Pixiv/Tagger format)
            if "tags" in json_data and isinstance(json_data["tags"], list):
                all_general = []
                all_character = []
                all_rating = []

                for tag_entry in json_data["tags"]:
                    if "tags" in tag_entry and isinstance(tag_entry["tags"], dict):
                        t = tag_entry["tags"]

                        def extract_names(tag_list):
                            if isinstance(tag_list, list):
                                return [str(item["name"]) for item in tag_list if
                                        isinstance(item, dict) and "name" in item]
                            return []

                        all_general.extend(extract_names(t.get("general", [])))
                        all_character.extend(extract_names(t.get("character", [])))
                        all_rating.extend(extract_names(t.get("rating", [])))

                # Full Prompt Construction
                full_parts.extend(all_rating)
                full_content_tags = all_character + all_general
                np.random.shuffle(full_content_tags)
                full_parts.extend(full_content_tags)
                full_prompt = " ".join(full_parts)[:512]

                # Dropped Prompt Construction
                parts.extend(all_rating)

                # 1. Aggressive: 40% chance to drop ALL character tags
                if all_character and np.random.random() < 0.4:
                    all_character = []

                parts.extend(all_character)

                # 2. Aggressive: General tags processing
                np.random.shuffle(all_general)
                # Keep fewer tags: random between 1 and len
                if len(all_general) > 1:
                    keep_count = np.random.randint(1, len(all_general) + 1)
                    all_general = all_general[:keep_count]

                parts.extend(all_general)

            # Fallback to old structure
            else:
                rating = json_data.get("rating", [])
                character_tags = json_data.get("character_tags", [])
                general_tags = json_data.get("general_tags", [])

                # Helper to process tag lists
                def process_tags(tags):
                    if isinstance(tags, list):
                        return [str(t) for t in tags]
                    return [str(tags)]

                # Add rating (usually kept at start)
                parts.extend(process_tags(rating))
                full_parts.extend(process_tags(rating))

                char_parts = process_tags(character_tags)
                gen_parts = process_tags(general_tags)

                # Full Prompt
                all_tags_full = char_parts + gen_parts
                np.random.shuffle(all_tags_full)
                full_parts.extend(all_tags_full)
                full_prompt = " ".join(full_parts)[:512]

                # Dropped Prompt
                # Aggressive drop logic
                if char_parts and np.random.random() < 0.4:
                    char_parts = []

                parts.extend(char_parts)

                np.random.shuffle(gen_parts)
                if len(gen_parts) > 1:
                    keep_count = np.random.randint(1, len(gen_parts) + 1)
                    gen_parts = gen_parts[:keep_count]

                parts.extend(gen_parts)

            prompt = " ".join(parts)[:512]

            # --- Replacement Logic ---
            # 20% chance to replace with Title
            # 20% chance to replace with Pixiv Tags
            # Independent events. If both occur, we combine them.

            replace_with_title = (pixiv_title_str != "") and (np.random.random() < 0.2)
            replace_with_ptags = (pixiv_tags_str != "") and (np.random.random() < 0.2)

            if replace_with_title and replace_with_ptags:
                prompt = f"{pixiv_title_str} {pixiv_tags_str}"[:512]
            elif replace_with_title:
                prompt = pixiv_title_str[:512]
            elif replace_with_ptags:
                prompt = pixiv_tags_str[:512]

        else:
            prompt = str(json_data)
            full_prompt = prompt
    elif "txt" in sample:
        txt_data = sample["txt"]
        if isinstance(txt_data, bytes):
            txt_data = txt_data.decode("utf-8")
        prompt = txt_data
        full_prompt = prompt
    elif "caption" in sample:
        cap_data = sample["caption"]
        if isinstance(cap_data, bytes):
            cap_data = cap_data.decode("utf-8")
        prompt = cap_data
        full_prompt = prompt

    return {
        "image": image,
        "prompts": prompt,
        "full_prompts": full_prompt,
        "key": sample.get("__key__", "unknown")
    }


def bucket_batch(data, batch_size):
    """
    Groups samples into buckets by resolution and yields stacked batches.
    """
    buckets = {}
    for sample in data:
        # sample is {image, prompt, full_prompt, coords}
        if sample is None: continue
        img = sample["image"]
        h, w = img.shape[1:]
        key = (h, w)

        if key not in buckets:
            buckets[key] = []

        buckets[key].append(sample)

        if len(buckets[key]) == batch_size:
            batch = buckets[key]
            images = torch.stack([s["image"] for s in batch])
            prompts = [s["prompt"] for s in batch]
            coords = torch.stack([s["coords"] for s in batch])
            yield images, prompts, coords
            buckets[key] = []


class WDSLoader:
    def __init__(self, url, csv_path=None, image_size=64, batch_size=16, num_workers=4, use_advanced_captions=True):
        """
        WebDataset Loader for nanoWaifu.

        Args:
            url: WebDataset URL or path
            csv_path: Path to CSV for class mapping (optional, for backward compatibility)
            image_size: Target image size
            batch_size: Batch size
            num_workers: Number of workers for DataLoader
            use_advanced_captions: Use advanced caption processing with tag dropping
        """
        self.url = url
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_advanced_captions = use_advanced_captions

        # Optional class map for backward compatibility
        self.class_map = None
        self.num_classes = 0
        if csv_path:
            self.class_map = self.load_class_map(csv_path)
            self.num_classes = len(self.class_map)

        # Precalculate buckets
        base_size = self.image_size
        target_area = base_size ** 2

        # Common aspect ratios for anime/manga
        # 1:1, 3:4, 4:3, 9:16, 16:9
        aspect_ratios = [1.0, 0.75, 1.33, 0.56, 1.78]
        self.buckets = []
        for ar in aspect_ratios:
            # w * h = area, w / h = ar  =>  h^2 * ar = area => h = sqrt(area / ar)
            h = int(math.sqrt(target_area / ar))
            w = int(h * ar)
            # Round to nearest multiple of 16 for FCDM/MAR/Patching compatibility
            h = (h // 32) * 32
            w = (w // 32) * 32
            if h > 0 and w > 0:
                self.buckets.append((h, w))

        self.bucket_ars = [bw / bh for bh, bw in self.buckets]

    def load_class_map(self, csv_path):
        if csv_path.endswith('.txt'):
            with open(csv_path, 'r', encoding='utf-8') as f:
                tags = [line.strip() for line in f if line.strip()]
            return {tag: i for i, tag in enumerate(tags)}
        # Using latin-1 encoding to avoid UnicodeDecodeError on special characters
        df = pd.read_csv(csv_path, encoding='latin-1')
        return dict(zip(df['character'], df['id']))

    def preprocess(self, sample):
        """
        Preprocess a webdataset sample with image augmentation and caption processing.
        """
        try:
            # Use advanced caption processing if enabled
            if self.use_advanced_captions:
                transformed = transform_sample(sample)
                image = transformed["image"]
                prompt = transformed["prompts"]
                full_prompt = transformed["full_prompts"]
            else:
                # Fallback to simple processing
                # Find image key
                image = None
                for key in ["jpg", "png", "webp", "jpeg", "image"]:
                    if key in sample:
                        image = sample[key]
                        break

                if image is None:
                    return None

                # Simple prompt extraction
                if "json" in sample:
                    meta = sample["json"] if isinstance(sample["json"], dict) else json.loads(sample["json"])
                    prompt = meta.get("character", "unknown")
                    full_prompt = prompt
                elif "txt" in sample:
                    txt_data = sample["txt"]
                    if isinstance(txt_data, bytes):
                        txt_data = txt_data.decode("utf-8")
                    prompt = txt_data
                    full_prompt = prompt
                elif "caption" in sample:
                    cap_data = sample["caption"]
                    if isinstance(cap_data, bytes):
                        cap_data = cap_data.decode("utf-8")
                    prompt = cap_data
                    full_prompt = prompt
                else:
                    prompt = ""
                    full_prompt = ""

            # Decode image if it's bytes
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image)).convert("RGB")
            elif not isinstance(image, Image.Image):
                return None

            w, h = image.size
            img_ar = w / h

            # Find closest bucket by aspect ratio
            best_bucket_idx = min(range(len(self.bucket_ars)), key=lambda i: abs(self.bucket_ars[i] - img_ar))
            target_h, target_w = self.buckets[best_bucket_idx]
            target_ar = self.bucket_ars[best_bucket_idx]

            # Resize to cover
            if img_ar > target_ar:
                # Image is wider than bucket, match height
                new_h = target_h
                new_w = int(target_h * img_ar)
            else:
                # Image is taller than bucket, match width
                new_w = target_w
                new_h = int(target_w / img_ar)

            image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

            # Random crop to bucket size
            left = random.randint(0, max(0, new_w - target_w))
            top = random.randint(0, max(0, new_h - target_h))
            image = image.crop((left, top, left + target_w, top + target_h))

            # Random Horizontal Flip
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # To Tensor and Normalize [-1, 1]
            image = F.to_tensor(image)
            #image = (image - 0.5) * 2.0

            # Dummy coords for backward compatibility
            rel_coords = torch.tensor([0.0, 0.0, 1.0, 1.0])

            return {
                "image": image,
                "prompt": full_prompt,
                "full_prompt": full_prompt,
                "coords": rel_coords
            }

        except Exception as e:
            print(f"Error preprocessing sample: {e}")
            return None

    def make_loader(self):
        dataset = (
            wds.WebDataset(
                self.url,
                nodesplitter=wds.split_by_node,
                handler=warn_and_continue,
                workersplitter=wds.split_by_worker
            )
            .shuffle(10_000, initial=1_000)
            .map(self.preprocess, handler=warn_and_continue)
            .select(lambda x: x is not None)
            .compose(lambda data: bucket_batch(data, self.batch_size))
        )

        loader = wds.WebLoader(
            dataset,
            batch_size=None,  # Already batched by bucket_batch
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=4 if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0
        )

        return loader
