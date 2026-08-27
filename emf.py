"""
Euler Mean Flows (EMF): one-step / few-step generative training.

Implementation of "Trajectory Consistency for One-Step Generation on Euler
Mean Flows" (anonymous, under review), x1-prediction variant. EMF replaces
the hard trajectory-consistency constraint of flow maps with a locally
linearized surrogate (Eq. 17), giving direct data supervision for
long-horizon flow maps without any JVP / gradient computation.

Time convention (paper): t = 0 is pure noise, t = 1 is data,
    x_t = (1 - t) * x0 + t * x1,   x0 ~ N(0, I),  x1 ~ data.
This is the mirror image of the flow-matching path in train.py, where
t = 1 is noise; everything in this module follows the paper's convention.

The model is called as model(x, t, y_indices, y_offsets, r=r) and predicts
the flow endpoint x~_{t->r}(x_t) = (1 - t) * u_{t->r}(x_t) + x_t for the
interval [t, r] with t <= r.

Guidance is baked into the training target (paper C.1), so sampling needs a
single forward pass per step with no classifier-free guidance:
    target_inst = w * x1 + (1 - w - k) * model(x_t, t, r=t, C_null)
                       + k * model(x_t, t, r=t, C)
with effective guidance scale w' = w / (1 - k).
"""
import torch


MIN_DENOM = 0.02  # clamp for (1 - t) / (1 - r) denominators (paper Sec. 4.3)


# =============================================================================
# Time sampler (paper C.1, following MeanFlow)
# =============================================================================
def sample_emf_times(B, device, dist="uniform", logit_loc=-0.4, logit_scale=1.0,
                     interval_ratio=0.25):
    """
    Sample per-sample interval endpoints (t, r) with t <= r.

    t and r are drawn independently from T1 (U[0,1] by default, or a
    logit-normal) and swapped so t <= r. A fraction `interval_ratio` of the
    samples keeps r != t (trajectory-consistency pairs); the rest get r = t,
    which reduces the EMF loss to the flow-matching objective and anchors the
    instantaneous field (Theorem 4.3 validity condition).
    """
    if dist == "logit_normal":
        eps = torch.randn(2, B, device=device)
        s = torch.sigmoid(logit_loc + logit_scale * eps)
    else:
        s = torch.rand(2, B, device=device)
    t = torch.minimum(s[0], s[1])
    r = torch.maximum(s[0], s[1])
    interval = torch.rand(B, device=device) < interval_ratio
    r = torch.where(interval, r, t)
    return t, r


# =============================================================================
# Conditioning helpers
# =============================================================================
def _cat_cond(c1, c2):
    """Concatenate two (indices, offsets) conditioning pairs along the batch."""
    i1, o1 = c1
    i2, o2 = c2
    return torch.cat([i1, i2]), torch.cat([o1, o2 + len(i1)])


def _select_cond(y_indices, y_offsets, sel):
    """Index an (indices, offsets) conditioning pair by a sample subset."""
    ends = torch.cat([
        y_offsets[1:],
        torch.tensor([len(y_indices)], device=y_indices.device),
    ])
    chunks = [y_indices[y_offsets[i]:ends[i]] for i in sel]
    lengths = torch.tensor([len(c) for c in chunks], device=y_indices.device)
    new_indices = torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.long, device=y_indices.device)
    new_offsets = torch.zeros(len(chunks), dtype=torch.long, device=y_indices.device)
    if len(chunks) > 1:
        new_offsets[1:] = torch.cumsum(lengths[:-1], dim=0)
    return new_indices, new_offsets


# =============================================================================
# EMF training loss (paper Eq. 18, Algorithm 1, x1-prediction)
# =============================================================================
def emf_loss(model, xt, x1, t, r, cond, cond_null, delta_t=0.05,
             cfg_scale=2.5, cfg_k=0.4, adaptive_c=1e-3, adaptive_p=1.0):
    """
    Euler Mean Flow loss for a batch (x1-prediction).

    Per training step this costs one optimized forward pass plus two
    stop-gradient passes: a single batched forward for both CFG branches of
    the instantaneous field, and a (sub-batch) forward for the t+dt endpoint.

    Args:
        model: callable model(x, t, y_indices, y_offsets, r) -> endpoint.
        xt: (B, C, H, W) noised states, xt = (1-t)*x0 + t*x1 (paper convention).
        x1: (B, C, H, W) clean data targets.
        t, r: (B,) interval endpoints with t <= r (r = t for the
            instantaneous / flow-matching fraction).
        cond, cond_null: (y_indices, y_offsets) conditioning for the real and
            null labels (null unused when cfg_scale == 0).
        delta_t: linearization step size for the local Euler segment.
        cfg_scale: effective guidance scale w' = w / (1 - k) baked into the
            target. Set to 0 to disable CFG (pure conditional training).
        cfg_k: CFG mixing coefficient k.
        adaptive_c, adaptive_p: adaptive loss weight w = 1 / (||d||^2 + c)^p
            applied with stop-gradient (paper C.1, following MeanFlow / ECM).

    Returns:
        (loss_scalar, metrics_dict)
    """
    B = xt.shape[0]
    t_col = t.view(B, 1, 1, 1)
    y_idx, y_off = cond

    # Optimized forward: x~_{t->r}(x_t, C)
    pred = model(xt, t, y_idx, y_off, r=r)

    use_cfg = float(cfg_scale) != 0.0
    with torch.no_grad():
        # dt must fit inside [t, 1] and stay non-zero for the coefficient
        dt_col = torch.clamp(torch.full_like(t_col, delta_t),
                             max=1.0 - t_col).clamp(min=1e-4)

        if use_cfg:
            k = float(cfg_k)
            w = float(cfg_scale) * (1.0 - k)
            # Effective guidance w' = w / (1 - k); values > 1 imply
            # 1 - w - k < 0 (extrapolation past the unconditional field),
            # which is the intended CFG behavior.
            # One batched stop-gradient forward for both CFG branches:
            # inst_c = x~_{t->t}(x_t, C), inst_u = x~_{t->t}(x_t, C_null)
            idx2, off2 = _cat_cond(cond, cond_null)
            t2 = torch.cat([t, t])
            out2 = model(torch.cat([xt, xt]), t2, idx2, off2, r=t2)
            inst_c, inst_u = out2.chunk(2, dim=0)
            target = w * x1 + (1.0 - w - k) * inst_u + k * inst_c
        else:
            inst_c = model(xt, t, y_idx, y_off, r=t)
            target = x1

        # Midpoint via the guided instantaneous field (Algorithm 1, line 11):
        # x_{t+dt} = x_t + dt * (x~_{t->t}(x_t) - x_t) / (1 - t)
        denom_t = (1.0 - t_col).clamp(min=MIN_DENOM)
        x_tdt = xt + (dt_col / denom_t) * (inst_c - xt)

        # Interval correction coefficient (Eq. 17):
        #   coef = (r - t - dt)+ * (1 - t) / (1 - r) / dt
        gap = r - t - dt_col.view(B)
        active = gap > 0

        if active.any():
            coef = (gap.clamp(min=0.0) * (1.0 - t)
                    / (1.0 - r).clamp(min=MIN_DENOM) / dt_col.view(B))
            sel = active.nonzero(as_tuple=True)[0]
            sel_idx, sel_off = _select_cond(y_idx, y_off, sel)
            # x~_{t+dt -> r}(x_{t+dt}, C), stop-gradient (Algorithm 1, line 12)
            next_pred = model(x_tdt[sel], (t + dt_col.view(B))[sel], sel_idx, sel_off, r=r[sel])
            corr = torch.zeros_like(pred)
            corr[sel] = next_pred - pred[sel].detach()
            target = target + coef.view(B, 1, 1, 1) * corr

    # Adaptive loss with the x1 time weight 1/(1-t)^2 (paper Sec. 4.3)
    residual = pred - target
    dims = list(range(1, residual.dim()))
    per_sample_sq = residual.pow(2).sum(dim=dims)
    per_sample_sq = per_sample_sq / (1.0 - t).clamp(min=MIN_DENOM).pow(2)
    weight = 1.0 / (per_sample_sq.detach() + adaptive_c).pow(adaptive_p)
    loss = (weight * per_sample_sq).mean()

    return loss, {"loss": loss.item(), "active_frac": active.float().mean().item()}


# =============================================================================
# EMF sampling (paper Algorithm 2)
# =============================================================================
@torch.no_grad()
def sample_emf(model, tag_processor, latent_size, batch_size, prompts, device,
               steps=1, noise=None):
    """
    One-step / few-step EMF sampling from t = 0 (noise) to t = 1 (data).

    Guidance was baked into the model during training, so no CFG passes are
    needed here. The step update follows from u_{t->r}(x) =
    (x~_{t->r}(x) - x) / (1 - t):  x_r = x + (r - t) * u_{t->r}(x), which
    for a single step (t=0, r=1) reduces to x1 = x~_{0->1}(x0).
    """
    in_channels = model.in_channels if hasattr(model, "in_channels") else (
        model.module.in_channels if hasattr(model, "module") else 32)
    model.eval()
    if isinstance(latent_size, (tuple, list)):
        H, W = latent_size
    else:
        H = W = latent_size

    if noise is not None:
        x = noise.clone().to(device)[:batch_size]
    else:
        x = torch.randn(batch_size, in_channels, H, W, device=device)

    y_indices, y_offsets = tag_processor.process_prompts(prompts[:batch_size], device)

    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    for i in range(steps):
        t_curr, t_next = ts[i], ts[i + 1]
        t_vec = torch.full((x.shape[0],), float(t_curr), device=device)
        r_vec = torch.full((x.shape[0],), float(t_next), device=device)

        out = model(x, t_vec, y_indices, y_offsets, r=r_vec)
        denom = max(1.0 - float(t_curr), MIN_DENOM)
        x = x + float(t_next - t_curr) * (out - x) / denom

    return x


# =============================================================================
# Self-test: overfit a tiny FCDM on CPU
# =============================================================================
if __name__ == "__main__":
    from model_dit import FCDM, TagProcessor

    print("=" * 60)
    print("EMF Test Suite")
    print("=" * 60)
    torch.manual_seed(0)
    device = torch.device("cpu")

    # -- time sampler sanity -------------------------------------------------
    t, r = sample_emf_times(4096, device, interval_ratio=0.25)
    assert (t <= r + 1e-6).all(), "t must be <= r"
    frac = (r > t).float().mean().item()
    assert 0.20 < frac < 0.30, f"interval fraction off: {frac}"
    print(f"  [OK] time sampler: interval fraction {frac:.3f}, t<=r")

    # -- toy data: 8 fixed latent "images" -----------------------------------
    B, C, H, W = 8, 4, 16, 16
    data = torch.randn(B, C, H, W)

    tags = [f"tag{i}" for i in range(B)]
    with open("_emf_tags.txt", "w") as f:
        f.write("\n".join(tags) + "\nnull")
    tp = TagProcessor("_emf_tags.txt")
    prompts = [tags[i % B] for i in range(B)]
    cond = tp.process_prompts(prompts, device)
    cond_null = tp.process_prompts([""] * B, device)

    for label, cfg_scale, steps, use_cfg_label in [
        ("no CFG", 0.0, 500, False), ("baked CFG", 1.5, 800, True)
    ]:
        model = FCDM(in_channels=C, dim=32, depth=[1, 1, 2, 1, 1],
                     num_classes=tp.num_classes).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for step in range(steps):
            # For CFG, label dropout trains the null branch so the fixed
            # point m_c = w'*x1 + (1-w')*m_u converges to the data (C.1).
            c = tp.process_prompts(prompts, device,
                                   dropout_prob=0.1 if use_cfg_label else 0.0)
            t, r = sample_emf_times(B, device, interval_ratio=0.5)
            noise = torch.randn(B, C, H, W)
            xt = (1 - t.view(B, 1, 1, 1)) * noise + t.view(B, 1, 1, 1) * data
            loss, metrics = emf_loss(model, xt, data, t, r, c, cond_null,
                                     delta_t=0.05, cfg_scale=cfg_scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # Direct 1-step endpoint check: model(x0, t=0, r=1) should reproduce
        # the overfit targets. (The adaptive loss itself saturates near 1
        # while residuals are large, so it is not a progress metric.)
        with torch.no_grad():
            noise = torch.randn(B, C, H, W)
            endpoint = model(noise, torch.zeros(B), cond[0], cond[1], r=torch.ones(B))
            mse01 = torch.nn.functional.mse_loss(endpoint, data).item()
        assert mse01 < 0.05, f"{label}: 1-step endpoint mse {mse01:.4f}"

        samples = sample_emf(model, tp, (H, W), B, prompts, device, steps=1,
                             noise=torch.randn(B, C, H, W))
        assert samples.shape == data.shape
        print(f"  [OK] x1-prediction ({label}): 1-step mse {mse01:.4f}")

    # -- few-step sampling smoke test -----------------------------------------
    samples2 = sample_emf(model, tp, (H, W), B, prompts, device, steps=4)
    assert samples2.shape == data.shape
    print(f"  [OK] few-step (4) sampling: {tuple(samples2.shape)}")

    import os
    os.remove("_emf_tags.txt")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
