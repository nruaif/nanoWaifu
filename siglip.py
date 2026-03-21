import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (SupCon)
    Reference: https://arxiv.org/abs/2004.11362
    
    This implementation supports multiple views per sample and handles 
    both self-supervised (no labels) and supervised (with labels) settings.
    """
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, views_online, views_target, labels=None):
        """
        Args:
            views_online: List of Tensors of shape [B, D] (online views)
            views_target: List of Tensors of shape [B, D] (target/EMA views)
            labels: Ground truth labels of shape [B] (optional)
        Returns:
            loss: Scalar loss
            logits: Scaled similarity matrix [K_on*B, K_tgt*B]
            sim_matrix: Raw cosine similarity matrix [K_on*B, K_tgt*B]
        """
        # 1. Normalize and concatenate views
        # Each view is normalized to the unit hypersphere as per SupCon paper
        z_on = torch.cat([F.normalize(v, dim=-1) for v in views_online], dim=0) 
        z_tgt = torch.cat([F.normalize(v, dim=-1) for v in views_target], dim=0) 
        
        B = views_online[0].shape[0]
        K_on = len(views_online)
        K_tgt = len(views_target)
        device = z_on.device
        
        # 2. Compute similarity matrix
        # sim_matrix[i, j] is cosine similarity between online view i and target view j
        sim_matrix = torch.matmul(z_on, z_tgt.T)
        logits = sim_matrix / self.temperature
        
        # 3. Define positive mask
        # If labels are provided, samples with the same label are considered positives
        if labels is None:
            # Self-supervised: only views of the same sample are positive
            indices = torch.arange(B, device=device)
        else:
            # For samples with label -1, give them a unique label to treat as self-supervised
            indices = labels.clone()
            mask_no_label = (indices == -1)
            if mask_no_label.any():
                # Ensure unique labels don't collide with existing labels
                offset = indices.max() + 1
                unique_indices = torch.arange(B, device=device) + offset
                indices[mask_no_label] = unique_indices[mask_no_label]
            
        indices_on = indices.repeat(K_on)
        indices_tgt = indices.repeat(K_tgt)
        
        # mask[i, j] = 1 if view i and view j belong to the same class
        mask = (indices_on[:, None] == indices_tgt[None, :]).float()
        
        # 4. Log-sum-exp for stability and InfoNCE (Eq. 2 in paper)
        # We use the L_out_sup formulation which was identified as superior in the paper.
        
        # Subtract max for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits_stable = logits - logits_max.detach()
        
        exp_logits = torch.exp(logits_stable)
        # Denominator: sum of exp(logits) over all target views
        denom = exp_logits.sum(dim=1, keepdim=True)
        
        # Log-probability: log( exp(z_i*z_p/tau) / sum_a exp(z_i*z_a/tau) )
        log_prob = logits_stable - torch.log(denom + 1e-6)
        
        # 5. Average over positives for each anchor (Eq. 2)
        # P(i) is the set of indices of positives in z_tgt for anchor i in z_on
        num_positives = mask.sum(dim=1)
        # Avoid division by zero (each sample should at least be positive with itself/its EMA)
        num_positives = torch.where(num_positives > 0, num_positives, torch.ones_like(num_positives))
        
        # L_out_sup = sum_{i} (-1/|P(i)|) * sum_{p in P(i)} log_prob
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / num_positives
        
        # 6. Final loss: mean over all anchors
        loss = -mean_log_prob_pos.mean()
        
        return loss, logits, sim_matrix
