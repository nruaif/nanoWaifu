# Adversarial Flow Training

This checkout includes a latent-space implementation of ByteDance Seed's
[Adversarial Flow Models](https://github.com/ByteDance-Seed/Adversarial-Flow-Models).
It uses the existing tag-conditioned `TokenformerDiT` as the generator and a
separate time- and tag-conditioned DiT discriminator.

The default `config.yaml` trains the official designated one-step transition
from noise at `t=1` to data at `t=0`. Set `adversarial.steps` to another positive
integer to train a designated few-step model. Validation automatically uses the
matching one/few-step sampler.

The adversarial objective contains:

- Relativistic softplus discriminator and generator losses.
- Finite-difference R1 and R2 penalties.
- The discriminator centering penalty.
- Gradient normalization on discriminator feedback to the generator.
- A cosine-decayed optimal-transport regularizer.

## Checkpoints

`training.init_from` imports generator weights only and starts a fresh AFM run at
step zero. This is the correct way to post-train an existing flow-matching model.

`training.resume_from` resumes a complete checkpoint. AFM resumes require saved
generator, discriminator, both optimizer states, and gradient-normalization
state. A flow-matching checkpoint cannot be resumed as AFM; use `init_from`
instead.

The default configuration reads the newest generator checkpoint from
`outputs_dit/` and writes independent AFM checkpoints to `outputs_afm/`.

## Run

```powershell
.\.venv\Scripts\python.exe train.py --config config.yaml
```

Run the focused CPU checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
