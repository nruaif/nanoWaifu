import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DriftingAndPerceptualLoss(nn.Module):
    def __init__(self, model_name='convnext_tiny.dinov3_lvd1689m', drift_weight=1.0, lpips_weight=1.0, device='cuda', dtype=torch.bfloat16):
        super().__init__()
        self.drift_weight = drift_weight
        self.lpips_weight = lpips_weight
        
        # 1. Load Pretrained Feature Extractor
        print(f"Loading {model_name}...")
        self.feature_extractor = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
        ).eval()
        
        # Move to device and dtype
        self.feature_extractor.to(device=device, dtype=dtype)
        self.dtype = dtype

        # Freeze weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # 2. Standard ImageNet Normalization (for input tensors in 0-1 range)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

        # 3. Drifting Parameters (from Paper Appendix A.6)
        self.temperatures = [0.02, 0.05, 0.2] 

    def normalize_input(self, x):
        """Expects x in [0, 1]. Normalizes to ImageNet stats."""
        return (x - self.mean) / self.std

    def compute_pairwise_dist(self, x, y):
        """Computes Euclidean distance matrix between shape [B, D] and [B2, D]"""
        return torch.cdist(x, y, p=2)

    def compute_drift_component(self, feat_x, feat_y):
        """
        Calculates the Drifting Loss for a single feature scale.
        Logic follows Algorithm 2 and Appendix A.6 of the paper.
        
        Args:
            feat_x: Generated features [B, N_tokens, C] -> flattened to [B*N, C] inside
            feat_y: Target (Real) features [B, N_tokens, C] -> flattened to [B*N, C] inside
        """
        # Flatten spatial dimensions: [B, C, H, W] -> [B*H*W, C]
        # The paper treats every spatial location as an independent sample
        b, c, h, w = feat_x.shape
        x_flat = feat_x.permute(0, 2, 3, 1).reshape(-1, c) # Generated (Negatives)
        y_flat = feat_y.permute(0, 2, 3, 1).reshape(-1, c) # Real (Positives)
        
        # Ensure float32 for distance calculations to avoid overflow/underflow in half precision
        x_flat = x_flat.float()
        y_flat = y_flat.float()

        # --- 1. Feature Normalization (Eq. 20, 21) ---
        # Compute mean distance to scale features so average dist is sqrt(C)
        with torch.no_grad():
            # Estimate mean distance using a subset or batch mean to save memory
            # Here we calc exact batch mean dist
            raw_dist = torch.cdist(x_flat, y_flat)
            mean_dist = raw_dist.mean()
            # Avoid divide by zero
            scale_S = (mean_dist / (c ** 0.5)).clamp(min=1e-6)
        
        # Normalize features
        x_norm = x_flat / scale_S
        y_norm = y_flat / scale_S

        # --- 2. Perceptual Loss (LPIPS-style) ---
        # Simple MSE on the normalized features
        loss_lpips_scale = F.mse_loss(x_norm, y_norm)

        # --- 3. Drifting Loss ---
        # Calculate pairwise distances on normalized features
        dist_pos = torch.cdist(x_norm, y_norm) # Distance to Real
        dist_neg = torch.cdist(x_norm, x_norm) # Distance to Self (Generated)
        
        # Mask self-distance in neg to infinity (don't repel self)
        eye_mask = torch.eye(dist_neg.shape[0], device=dist_neg.device).bool()
        dist_neg.masked_fill_(eye_mask, 1e6)

        total_drift_loss_scale = 0
        
        # Iterate over temperatures (Appendix A.6 "Multiple temperatures")
        for T in self.temperatures:
            # Softmax normalization (Alg 2)
            # Logits
            logit_pos = -dist_pos / T
            logit_neg = -dist_neg / T
            
            # Double Softmax (Sinkhorn-like) described in Alg 2
            # 1. Softmax over target (col)
            # 2. Softmax over source (row)
            # 3. Geometric mean of both
            
            # Positives
            attn_pos_row = F.softmax(logit_pos, dim=1) # over y
            attn_pos_col = F.softmax(logit_pos, dim=0) # over x
            attn_pos = torch.sqrt(attn_pos_row * attn_pos_col + 1e-8)
            
            # Negatives
            attn_neg_row = F.softmax(logit_neg, dim=1)
            attn_neg_col = F.softmax(logit_neg, dim=0)
            attn_neg = torch.sqrt(attn_neg_row * attn_neg_col + 1e-8)
            
            # Normalize weights (Alg 2)
            # "split" isn't needed if we computed separately, but we need row sums
            w_pos = attn_pos / (attn_pos.sum(dim=1, keepdim=True) + 1e-8) # weight per x
            w_neg = attn_neg / (attn_neg.sum(dim=1, keepdim=True) + 1e-8) # weight per x

            # Calculate Attractors and Repulsors
            # target_pos is the "weighted mean" of real data pulling x
            target_pos = w_pos @ y_norm 
            # target_neg is the "weighted mean" of fake data pushing x
            target_neg = w_neg @ x_norm

            # Drift Vector V (Eq. 10: V = V_pos - V_neg)
            # V_pos = target_pos - x
            # V_neg = target_neg - x
            # V = (target_pos - x) - (target_neg - x) = target_pos - target_neg
            V = target_pos - target_neg

            # --- Drift Normalization (Eq. 23, 24) ---
            with torch.no_grad():
                # Normalize V so E[||V||^2] approx C
                # lambda_j = sqrt( E[ ||V||^2 / C ] )
                v_norm_sq = (V ** 2).sum(dim=1).mean()
                scale_lambda = torch.sqrt(v_norm_sq / c).clamp(min=1e-6)
            
            V_normalized = V / scale_lambda

            # --- Target Calculation ---
            # x_drifted = stopgrad(x + V)
            x_target = (x_norm + V_normalized).detach()

            # MSE Loss (Eq. 26)
            loss_T = F.mse_loss(x_norm, x_target)
            total_drift_loss_scale += loss_T

        return total_drift_loss_scale, loss_lpips_scale

    def forward(self, x, y):
        """
        Args:
            x: Generated images [B, 3, H, W] in range [0, 1]
            y: Target/Real images [B, 3, H, W] in range [0, 1]
        """
        # Ensure batch size > 1 for Drifting (need negatives!)
        if x.shape[0] < 2:
            # Fallback to just LPIPS or return 0 if strictly required
            # raising ValueError might break training if last batch is small
            # return {"loss": torch.tensor(0.0, device=x.device, requires_grad=True)}
            # Better to just return 0 loss but warn
             # raise ValueError("Drifting loss requires batch size > 1 to calculate repulsion.")
             pass

        # Cast input to match feature extractor dtype
        x_in = x.to(dtype=self.dtype)
        y_in = y.to(dtype=self.dtype)

        # Normalize
        x_in = self.normalize_input(x_in)
        y_in = self.normalize_input(y_in)

        # Extract Features
        # Using no_grad is redundant if params are frozen but good for safety
        # However, we need gradients for x, so we can't use torch.no_grad() context manager globally
        # But feature extractor parameters are frozen.
        
        feats_x = self.feature_extractor(x_in)
        
        with torch.no_grad():
             feats_y = self.feature_extractor(y_in)

        total_drift_loss = 0
        total_lpips_loss = 0

        # Loop through multi-scale features
        for fx, fy in zip(feats_x, feats_y):
            drift_loss, lpips_loss = self.compute_drift_component(fx, fy)
            total_drift_loss += drift_loss
            total_lpips_loss += lpips_loss

        combined_loss = (self.drift_weight * total_drift_loss) + (self.lpips_weight * total_lpips_loss)
        
        return {
            "loss": combined_loss,
            "drift_loss": total_drift_loss,
            "lpips_loss": total_lpips_loss
        }
