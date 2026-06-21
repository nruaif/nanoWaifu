import torch
import torch.nn as nn
import random

class TagProcessor:
    def __init__(self, tags_file):
        with open(tags_file, 'r', encoding='utf-8') as f:
            self.tags = [line.strip() for line in f if line.strip()]
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.tags)}
        self.num_classes = len(self.tags)

    def process_prompts(self, prompts, device, dropout_prob=0.0):
        indices = []
        offsets = [0]
        for p in prompts:
            if random.random() < dropout_prob:
                indices.append(self.num_classes)
            else:
                tags = p.split()
                count = 0
                for t in tags:
                    if t in self.tag_to_idx:
                        indices.append(self.tag_to_idx[t])
                        count += 1
                if count == 0:
                    indices.append(self.num_classes)
            offsets.append(len(indices))

        indices = torch.tensor(indices, dtype=torch.long, device=device)
        offsets = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
        return indices, offsets


class MiniT2IWrapper(nn.Module):
    def __init__(self, num_classes, model_id="MiniT2I/MiniT2I", seq_len=32, transformer=None):
        super().__init__()
        
        if transformer is not None:
            self.transformer = transformer
        else:
            import sys
            import os
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            
            diffusers_dir = os.path.join(os.path.dirname(__file__), 'minit2i-torch', 'diffusers')
            if diffusers_dir not in sys.path:
                sys.path.append(diffusers_dir)
            from mmdit import DiffusionModel, MMJiTConfig
            
            print(f">>> Loading {model_id} via huggingface_hub...")
            ckpt_path = hf_hub_download(repo_id=model_id, filename="minit2i-b-16/transformer/diffusion_pytorch_model.safetensors")
            
            cfg = MMJiTConfig()
            self.transformer = DiffusionModel(cfg)
            
            state_dict = load_file(ckpt_path)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            self.transformer.load_state_dict(new_state_dict, strict=False)
        
        if hasattr(self.transformer, "mmjit_config"):
            t5_hidden_size = self.transformer.mmjit_config.txt_input_size
        else:
            t5_hidden_size = self.transformer.cfg.txt_input_size
            
        self.tag_embedding = nn.Embedding(num_classes + 1, t5_hidden_size)
        nn.init.normal_(self.tag_embedding.weight, std=0.02)
        
        self.seq_len = seq_len
        self.num_classes = num_classes

    def forward(self, img, t, y_indices, y_offsets, return_layer_match=False, **kwargs):
        B = len(y_offsets)
        device = y_indices.device
        
        padded_indices = torch.full((B, self.seq_len), self.num_classes, dtype=torch.long, device=device)
        attn_mask = torch.zeros((B, self.seq_len), dtype=torch.float32, device=device)
        
        offsets_list = y_offsets.tolist()
        total_len = len(y_indices)
        for i in range(B):
            start = offsets_list[i]
            end = offsets_list[i+1] if i + 1 < B else total_len
            item_len = end - start
            copy_len = min(item_len, self.seq_len)
            if copy_len > 0:
                padded_indices[i, :copy_len] = y_indices[start:start+copy_len]
                attn_mask[i, :copy_len] = 1.0

        context = self.tag_embedding(padded_indices)
        
        v_pred = self.transformer.pred_velocity(img, t, context, attn_mask)
        
        match_loss = torch.tensor(0.0, device=device)
        
        if return_layer_match:
            return v_pred, match_loss
        return v_pred

    def sample(self, y_indices, y_offsets, image_height=512, image_width=512, cfg_scale=6.0, generator=None, num_inference_steps=100):
        B = len(y_offsets)
        device = y_indices.device
        dtype = next(self.transformer.parameters()).dtype
        
        padded_indices = torch.full((B, self.seq_len), self.num_classes, dtype=torch.long, device=device)
        attn_mask = torch.zeros((B, self.seq_len), dtype=torch.float32, device=device)
        
        offsets_list = y_offsets.tolist()
        total_len = len(y_indices)
        for i in range(B):
            start = offsets_list[i]
            end = offsets_list[i+1] if i + 1 < B else total_len
            item_len = end - start
            copy_len = min(item_len, self.seq_len)
            if copy_len > 0:
                padded_indices[i, :copy_len] = y_indices[start:start+copy_len]
                attn_mask[i, :copy_len] = 1.0

        context = self.tag_embedding(padded_indices).to(dtype)
        attn_mask = attn_mask.to(dtype)
        
        if hasattr(self.transformer, "mmjit_config"):
            old_steps = self.transformer.mmjit_config.n_T
            self.transformer.model.cfg.n_T = num_inference_steps
        else:
            old_steps = self.transformer.cfg.n_T
            self.transformer.cfg.n_T = num_inference_steps
        
        try:
            images = self.transformer.sample(
                context, attn_mask, image_height=image_height, image_width=image_width, cfg_scale=cfg_scale, generator=generator, progress=False
            )
        finally:
            if hasattr(self.transformer, "mmjit_config"):
                self.transformer.model.cfg.n_T = old_steps
            else:
                self.transformer.cfg.n_T = old_steps
            
        return images