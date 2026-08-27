import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import AutoencoderTiny
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import EncoderOutput, DecoderOutput
from diffusers.configuration_utils import ConfigMixin, register_to_config


def normalize_latent_f8(z, mean, std):
    """
    Normalize an f/8 latent ([B, C, H, W], C = stats_channels // 4) against
    FLUX.2 VAE stats, which are stored in the VAE's internal 2x2-patchified
    layout ([1, 4C, 1, 1]).

    The latent is temporarily pixel-unshuffled into the stats' layout, so the
    channel/position alignment matches exactly, then shuffled back.
    """
    zp = F.pixel_unshuffle(z, 2)
    zp = (zp - mean.to(z)) / std.to(z)
    return F.pixel_shuffle(zp, 2)


class Flux2TinyAutoEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            latent_channels: int = 128,
            encoder_block_out_channels: list[int] = [64, 64, 64, 64],
            decoder_block_out_channels: list[int] = [64, 64, 64, 64],
            act_fn: str = "silu",
            upsampling_scaling_factor: int = 2,
            num_encoder_blocks: list[int] = [1, 3, 3, 3],
            num_decoder_blocks: list[int] = [3, 3, 3, 1],
            latent_magnitude: float = 3.0,
            latent_shift: float = 0.5,
            force_upcast: bool = False,
            scaling_factor: float = 0.13025,
    ) -> None:
        super().__init__()
        self.tiny_vae = AutoencoderTiny(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_block_out_channels=encoder_block_out_channels,
            decoder_block_out_channels=decoder_block_out_channels,
            act_fn=act_fn,
            latent_channels=latent_channels // 4,
            upsampling_scaling_factor=upsampling_scaling_factor,
            num_encoder_blocks=num_encoder_blocks,
            num_decoder_blocks=num_decoder_blocks,
            latent_magnitude=latent_magnitude,
            latent_shift=latent_shift,
            force_upcast=force_upcast,
            scaling_factor=scaling_factor,
        )
        self.extra_encoder = nn.Conv2d(
            latent_channels // 4, latent_channels,
            kernel_size=4, stride=2, padding=1
        )
        self.extra_decoder = nn.ConvTranspose2d(
            latent_channels, latent_channels // 4,
            kernel_size=4, stride=2, padding=1
        )
        self.residual_encoder = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, latent_channels),
            nn.SiLU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
        )
        self.residual_decoder = nn.Sequential(
            nn.Conv2d(latent_channels // 4, latent_channels // 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, latent_channels // 4),
            nn.SiLU(),
            nn.Conv2d(latent_channels // 4, latent_channels // 4, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor, return_dict: bool = True) -> EncoderOutput:
        encoded = self.tiny_vae.encode(x, return_dict=False)[0]
        compressed = self.extra_encoder(encoded)
        enhanced = self.residual_encoder(compressed) + compressed

        if return_dict:
            return EncoderOutput(latent=enhanced)
        return enhanced

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput:
        decompressed = self.extra_decoder(z)
        enhanced = self.residual_decoder(decompressed) + decompressed
        decoded = self.tiny_vae.decode(enhanced, return_dict=False)[0]

        if return_dict:
            return DecoderOutput(sample=decoded)
        return decoded

    def forward(self, sample: torch.Tensor, return_dict: bool = True) -> DecoderOutput:
        encoded = self.encode(sample, return_dict=False)[0]
        decoded = self.decode(encoded, return_dict=False)[0]

        if return_dict:
            return DecoderOutput(sample=decoded)
        return decoded
