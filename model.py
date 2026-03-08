import torch
import torch.nn as nn
import timm

def SIGReg(x, global_step, num_slices=256, chunk_size=32, t_max=3.0, n_points=17):
    """SIGReg with Epps-Pulley statistic. x is (N, K) tensor.
       Matches the official LeJEPA EppsPulley+SlicingUnivariateTest formulation.
       - Integrates over [0, t_max] with doubled weights (symmetric CF trick).
       - Scales by batch size N (as in the official repo).
       - Chunked projection to reduce peak memory."""
    with torch.amp.autocast('cuda', enabled=False): # accumulate in float32
        x = x.float()
        N, K = x.shape
        device = x.device

        # Precompute integration grid [0, t_max] with symmetry-doubled weights
        t = torch.linspace(0, t_max, n_points, device=device)  # (n_points,)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, device=device)  # doubled for symmetry
        weights[0] = dt  # t=0 endpoint only gets single weight
        weights[-1] = dt # t=t_max endpoint only gets single weight
        phi = torch.exp(-0.5 * t ** 2)  # Standard Normal CF: exp(-t^2/2)
        # Precompute weights * phi together for efficiency
        w_phi = weights * phi  # (n_points,)

        # Random projection matrix (seeded deterministically per step)
        g = torch.Generator(device=device).manual_seed(global_step)
        A = torch.randn((K, num_slices), generator=g, device=device)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-10)  # Column-normalize

        T_total = torch.tensor(0.0, device=device)

        if chunk_size < 1:
            chunk_size = num_slices

        for i in range(0, num_slices, chunk_size):
            # Project: (N, chunk)
            x_proj = x @ A[:, i:i+chunk_size]  # (N, chunk)
            # Expand for integration: (N, chunk, n_points)
            x_t = x_proj.unsqueeze(2) * t  # (N, chunk, n_points)

            # ECF via cos and sin (avoids complex tensors)
            cos_mean = x_t.cos().mean(0)  # (chunk, n_points)
            sin_mean = x_t.sin().mean(0)  # (chunk, n_points)

            # Epps-Pulley integrand: |(ECF(t) - phi(t))|^2 * weight
            err = (cos_mean - phi).square() + sin_mean.square()  # (chunk, n_points)

            # Integrate via trapezoidal rule with pre-fused weights
            T_total = T_total + (err @ w_phi).sum()

        # Scale by N as per official implementation
        return T_total * N

class CoAtNeXtEncoder(nn.Module):
    """
    Pure LeJEPA SSL Encoder using CoAtNeXt backbone.
    Processes images, pools features, and projects them to an 8192-dim space.
    Binarizes the representation via Straight-Through Estimator (STE).
    """
    def __init__(self, backbone_model='coatnext_nano_rw_224.sw_in1k', proj_dim=8192, pretrained=True):
        super().__init__()
        
        # Backbone: returns global pooled features (num_classes=0 drops the classifier head)
        self.backbone = timm.create_model(
            backbone_model,
            pretrained=pretrained,
            num_classes=0,
        )
        backbone_dim = self.backbone.num_features
        
        # Projection Head: Maps from backbone_dim to target projection dimension (e.g. 8192)
        # Using a simple 3-layer MLP as standard in many SSL methods
        hidden_dim = 2048
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )

        # Decode: project binary_ste back to the logit space.
        # Creates a discrete bottleneck: encode → binarize (STE) → decode.
        # SIGReg and inv_loss operate on the decoded (reconstructed) logits.
        self.decode = nn.Linear(proj_dim, proj_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) raw image in [-1, 1]
        Returns:
            binary_ste: (B, proj_dim) binary {0, 1} via STE
            decoded_logits: (B, proj_dim) binary_ste projected back into logit space
        """
        # Convert [-1, 1] to roughly [0, 1]
        x = x * 0.5 + 0.5
        
        # Extract features (B, backbone_dim)
        features = self.backbone(x)
        
        # Encode: project features to the discrete bottleneck space
        logits = self.proj(features)
        
        # Binarize via STE: forward = hard threshold, backward = straight-through
        binary = (logits > 0).float()
        binary_ste = logits + (binary - logits).detach()

        # Decode: project binary_ste back into logit space
        # SIGReg and inv_loss operate on these reconstructed logits
        decoded_logits = self.decode(binary_ste)
        
        return binary_ste, decoded_logits