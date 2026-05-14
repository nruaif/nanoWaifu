import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyAwareFMLoss(nn.Module):
    """
    Frequency-aware Flow Matching loss from DeCo.

    Input:
        pred : [B,C,H,W]
        target : [B,C,H,W]

    Returns:
        scalar loss
    """

    def __init__(
        self,
        block_size: int = 8,
        quality: int = 85,
        mode: str = "inv_gamma",
        gamma: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()

        self.block_size = block_size
        self.reduction = reduction

        self.register_buffer(
            "dct_mat",
            self._create_dct_matrix(block_size),
            persistent=False,
        )

        self.register_buffer(
            "freq_weight",
            self._build_freq_weight(
                quality=quality,
                mode=mode,
                gamma=gamma,
            ),
            persistent=False,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        pred = self.rgb_to_ycbcr(pred)
        target = self.rgb_to_ycbcr(target)

        pred_freq = self.block_dct(pred)
        target_freq = self.block_dct(target)

        loss = self.freq_weight.to(pred_freq) * (pred_freq - target_freq).pow(2)

        if self.reduction == "mean":
            return loss.mean()

        elif self.reduction == "sum":
            return loss.sum()

        elif self.reduction == "none":
            return loss
            
        return loss

    @staticmethod
    def _create_dct_matrix(N: int):
        n = torch.arange(N, dtype=torch.float32)
        k = torch.arange(N, dtype=torch.float32).unsqueeze(1)

        C = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))

        alpha = torch.sqrt(torch.tensor(2.0) / N) * torch.ones(N)
        alpha[0] = math.sqrt(1.0 / N)

        return alpha.unsqueeze(1) * C

    @staticmethod
    def rgb_to_ycbcr(x: torch.Tensor):
        r = x[:, 0:1]
        g = x[:, 1:2]
        b = x[:, 2:3]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b

        return torch.cat([y, cb, cr], dim=1)

    def block_dct(self, x: torch.Tensor):
        bs = self.block_size

        B, C, H, W = x.shape

        pad_h = (-H) % bs
        pad_w = (-W) % bs

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        B, C, H2, W2 = x.shape

        Bh = H2 // bs
        Bw = W2 // bs

        blocks = (
            x.unfold(2, bs, bs)
             .unfold(3, bs, bs)
             .contiguous()
        )

        # [B,C,Bh,Bw,bs,bs]
        blocks = blocks.view(-1, bs, bs)

        Cmat = self.dct_mat.to(device=x.device, dtype=x.dtype)

        dct = torch.matmul(Cmat.unsqueeze(0), blocks)
        dct = torch.matmul(dct, Cmat.t().unsqueeze(0))

        return dct.view(B, C, Bh, Bw, bs, bs)

    @staticmethod
    def _scale_q(base_q, quality):
        quality = max(1, min(100, int(quality)))

        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - 2 * quality

        q = torch.floor((base_q * scale + 50) / 100)

        return q.clamp(1, 255)

    def _build_freq_weight(
        self,
        quality=85,
        mode="inv_gamma",
        gamma=1.0,
    ):
        lum_q = torch.tensor([
            [16,11,10,16,24,40,51,61],
            [12,12,14,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],
            [14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],
            [24,35,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],
            [72,92,95,98,112,100,103,99],
        ], dtype=torch.float32)

        chr_q = torch.tensor([
            [17,18,24,47,99,99,99,99],
            [18,21,26,66,99,99,99,99],
            [24,26,56,99,99,99,99,99],
            [47,66,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],
        ], dtype=torch.float32)

        Qy = self._scale_q(lum_q, quality)
        Qc = self._scale_q(chr_q, quality)

        def q_to_weight(Q):
            if mode == "inv":
                w = 1.0 / Q

            elif mode == "inv_gamma":
                w = (Q.mean() / Q).pow(gamma)

            else:
                raise ValueError(mode)

            return w / w.mean()

        wy = q_to_weight(Qy)
        wc = q_to_weight(Qc)

        w = torch.stack([wy, wc, wc], dim=0)

        # [1,C,1,1,8,8]
        return w.unsqueeze(0).unsqueeze(2).unsqueeze(3)

def pseudo_huber_loss(pred, target, delta=1.0, reduction='mean'):
    """
    Pseudo-Huber Loss (Charbonnier loss)
    L(x) = delta^2 * (sqrt(1 + (x/delta)^2) - 1)
    """
    diff = pred - target
    loss = (delta**2) * (torch.sqrt(1 + (diff / delta)**2) - 1)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss





# ------------------------------------------------------------
# Massive-aware cosine similarity
# ------------------------------------------------------------

def massive_cosine_similarity(
    f1,
    f2,
    percentile=0.90,
    multiplier=5.0,
    lambda_scale=0.5,
    spike_q=0.999,
    clamp=20.0,
    eps=1e-6,
):
    """
    f1, f2: [B,T,D]
    """

    # --------------------------------------------------------
    # detect massive channels
    # --------------------------------------------------------

    feat = torch.cat([f1, f2], dim=1)

    energy = torch.quantile(
        feat.abs().flatten(0,1),
        q=spike_q,
        dim=0
    )

    threshold = multiplier * torch.quantile(
        energy,
        percentile
    )

    massive_idx = torch.where(
        energy > threshold
    )[0]

    if len(massive_idx) == 0:
        massive_idx = torch.topk(energy, 1).indices

    # --------------------------------------------------------
    # modulation
    # --------------------------------------------------------

    def modulate(x):

        x = x.clone()

        mask = torch.zeros(
            x.shape[-1],
            device=x.device,
            dtype=torch.bool
        )

        mask[massive_idx] = True

        normal = x[..., ~mask]
        massive = x[..., mask]

        # stabilize spikes
        massive = torch.tanh(
            massive / clamp
        ) * clamp

        massive = massive / (
            massive.std(
                dim=(0,1),
                keepdim=True
            ) + eps
        )

        # adaptive alpha
        alpha = lambda_scale * (
            normal.pow(2).mean(dim=-1, keepdim=True).sqrt()
            /
            (
                massive.pow(2).mean(dim=-1, keepdim=True).sqrt()
                + eps
            )
        )

        massive = alpha * massive

        out = torch.zeros_like(x)

        out[..., ~mask] = normal
        out[..., mask] = massive

        return F.normalize(out, dim=-1)

    f1 = modulate(f1)
    f2 = modulate(f2)

    # --------------------------------------------------------
    # cosine similarity
    # --------------------------------------------------------

    sim = torch.matmul(
        f1,
        f2.transpose(-1, -2)
    )

    return sim, massive_idx

