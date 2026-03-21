import torch
import torch.nn as nn
import torch.nn.functional as F

class SigLIPLoss(nn.Module):
    """
    Sigmoid Loss for Language-Image Pre-training (SigLIP)
    Generalized for multiple views.
    """
    def __init__(self, init_tau=1.0, init_bias=-10.0):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(init_tau))
        self.bias = nn.Parameter(torch.tensor(init_bias))

    def forward(self, views_online, views_target):
        """
        views_online: List of tensors, each (B, D)
        views_target: List of tensors, each (B, D)
        """
        B = views_online[0].shape[0]
        K_on = len(views_online)
        K_tgt = len(views_target)
        
        # Concatenate all views
        z_on = torch.cat([F.normalize(v, dim=-1) for v in views_online], dim=0) # (K_on*B, D)
        z_tgt = torch.cat([F.normalize(v, dim=-1) for v in views_target], dim=0) # (K_tgt*B, D)
        
        # Pairwise dot products: (K_on*B, K_tgt*B)
        sim_matrix = torch.matmul(z_on, z_tgt.T)
        logits = sim_matrix * self.log_tau.exp() + self.bias
        
        # Construct Labels
        indices_on = torch.arange(B, device=z_on.device).repeat(K_on)
        indices_tgt = torch.arange(B, device=z_tgt.device).repeat(K_tgt)
        
        labels = (indices_on[:, None] == indices_tgt[None, :]).float()
        labels = 2 * labels - 1
        
        # Sigmoid loss
        loss = -F.logsigmoid(labels * logits).sum() / (B * K_on * K_tgt)
        return loss, logits, sim_matrix
