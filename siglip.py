import torch
import torch.nn as nn
import torch.nn.functional as F

class SigLIPLoss(nn.Module):
    """
    Robust Sigmoid Loss for Language-Image Pre-training (SigLIP)
    """
    def __init__(self, init_tau=1.0, init_bias=-10.0):
        super().__init__()
        # Clamp log_tau to prevent exp(log_tau) from exploding
        self.log_tau = nn.Parameter(torch.tensor(init_tau))
        self.bias = nn.Parameter(torch.tensor(init_bias))

    def forward(self, views_online, views_target):
        B = views_online[0].shape[0]
        K_on = len(views_online)
        K_tgt = len(views_target)
        
        z_on = torch.cat([F.normalize(v, dim=-1) for v in views_online], dim=0) 
        z_tgt = torch.cat([F.normalize(v, dim=-1) for v in views_target], dim=0) 
        
        # Limit tau to roughly 100 (exp(4.6)) to prevent float16 overflow/instability
        tau = self.log_tau.clamp(max=4.6).exp()
        
        sim_matrix = torch.matmul(z_on, z_tgt.T)
        logits = sim_matrix * tau + self.bias
        
        indices_on = torch.arange(B, device=z_on.device).repeat(K_on)
        indices_tgt = torch.arange(B, device=z_tgt.device).repeat(K_tgt)
        
        labels = (indices_on[:, None] == indices_tgt[None, :]).float()
        labels = 2 * labels - 1
        
        # Using a more stable logsigmoid implementation
        # loss = log(1 + exp(-x))
        loss = F.softplus(-labels * logits).sum() / B 
        
        return loss, logits, sim_matrix
