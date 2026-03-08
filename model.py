import torch
import torch.nn as nn
import timm

class CoAtNeXtEncoder(nn.Module):
    """
    Pure SSL Encoder using CoAtNeXt backbone.
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