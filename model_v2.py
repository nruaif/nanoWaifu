import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from collections import namedtuple

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class Timesteps(nn.Module):
    def __init__(self, num_channels: int = 256):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, in_features, out_features):
        super(TimestepEmbedding, self).__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, out_features, bias=True)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)

        return sample

class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim=1024, conv_shortcut=True):
        super(ResnetBlock2D, self).__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-05, affine=True)
        self.conv1 = SeparableConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.time_emb_proj = nn.Linear(time_embed_dim, out_channels, bias=True)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-05, affine=True)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv2 = SeparableConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = None
        if conv_shortcut:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]
        hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor

class Attention(nn.Module):
    def __init__(self, inner_dim, cross_attention_dim=None, num_heads=None, dropout=0.0):
        super(Attention, self).__init__()
        if num_heads is None:
            self.head_dim = 64
            self.num_heads = inner_dim // self.head_dim
        else:
            self.num_heads = num_heads
            self.head_dim = inner_dim // num_heads

        self.scale = self.head_dim**-0.5
        if cross_attention_dim is None:
            cross_attention_dim = inner_dim
        self.to_q = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.to_out = nn.ModuleList([
            nn.Linear(inner_dim, inner_dim), 
            nn.Dropout(dropout, inplace=False)
        ])

    def forward(self, hidden_states, encoder_hidden_states=None):
        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states) if encoder_hidden_states is not None else self.to_k(hidden_states)
        v = self.to_v(encoder_hidden_states) if encoder_hidden_states is not None else self.to_v(hidden_states)
        
        b, t, _ = q.size()
        
        q = q.view(b, t, self.num_heads, self.head_dim)
        k = k.view(b, k.size(1), self.num_heads, self.head_dim)
        v = v.view(b, v.size(1), self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, -1)

        for layer in self.to_out:
            attn_output = layer(attn_output)

        return attn_output

class GEGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super(GEGLU, self).__init__()
        self.proj = nn.Linear(in_features, out_features * 2, bias=True)

    def forward(self, x):
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return x1 * torch.nn.functional.gelu(x2)

class FeedForward(nn.Module):
    def __init__(self, in_features, out_features):
        super(FeedForward, self).__init__()
        self.net = nn.ModuleList([
            GEGLU(in_features, out_features * 3),
            nn.Dropout(p=0.0, inplace=False),
            nn.Linear(out_features * 3, out_features, bias=True),
        ])

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

class BasicTransformerBlock(nn.Module):
    def __init__(self, hidden_size, use_self_attn=True, cross_attention_dim=1024):
        super(BasicTransformerBlock, self).__init__()
        self.use_self_attn = use_self_attn
        if self.use_self_attn:
            self.norm1 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
            self.attn1 = Attention(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn2 = Attention(hidden_size, cross_attention_dim)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.ff = FeedForward(hidden_size, hidden_size)

    def forward(self, x, encoder_hidden_states=None):
        if self.use_self_attn:
            residual = x
            x = self.norm1(x)
            x = self.attn1(x)
            x = x + residual

        residual = x
        x = self.norm2(x)
        if encoder_hidden_states is not None:
            x = self.attn2(x, encoder_hidden_states)
        else:
            x = self.attn2(x)
        x = x + residual

        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = x + residual
        return x

class Transformer2DModel(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, use_self_attn=True, cross_attention_dim=1024):
        super(Transformer2DModel, self).__init__()
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-06, affine=True)
        self.proj_in = nn.Linear(in_channels, out_channels, bias=True)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(out_channels, use_self_attn=use_self_attn, cross_attention_dim=cross_attention_dim) for _ in range(n_layers)]
        )
        self.proj_out = nn.Linear(out_channels, out_channels, bias=True)

    def forward(self, hidden_states, encoder_hidden_states=None):
        batch, _, height, width = hidden_states.shape
        res = hidden_states
        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states)

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        return hidden_states + res

class Downsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample2D, self).__init__()
        self.conv = SeparableConv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample2D, self).__init__()
        self.conv = SeparableConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class DownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim=1024):
        super(DownBlock2D, self).__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels, out_channels, time_embed_dim=time_embed_dim, conv_shortcut=False),
            ResnetBlock2D(out_channels, out_channels, time_embed_dim=time_embed_dim, conv_shortcut=False),
        ])
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels, out_channels)])

    def forward(self, hidden_states, temb):
        output_states = []
        for module in self.resnets:
            hidden_states = module(hidden_states, temb)
            output_states.append(hidden_states)

        hidden_states = self.downsamplers[0](hidden_states)
        output_states.append(hidden_states)

        return hidden_states, output_states

class CrossAttnDownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, has_downsamplers=True, use_self_attn=True, time_embed_dim=1024, cross_attention_dim=1024):
        super(CrossAttnDownBlock2D, self).__init__()
        self.attentions = nn.ModuleList([
            Transformer2DModel(out_channels, out_channels, n_layers, use_self_attn=use_self_attn, cross_attention_dim=cross_attention_dim),
            Transformer2DModel(out_channels, out_channels, n_layers, use_self_attn=use_self_attn, cross_attention_dim=cross_attention_dim),
        ])
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels, out_channels, time_embed_dim=time_embed_dim),
            ResnetBlock2D(out_channels, out_channels, time_embed_dim=time_embed_dim, conv_shortcut=False),
        ])
        self.downsamplers = None
        if has_downsamplers:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, out_channels)])

    def forward(self, hidden_states, temb, encoder_hidden_states):
        output_states = []
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states

class CrossAttnUpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel, n_layers, use_self_attn=True, time_embed_dim=1024, cross_attention_dim=1024):
        super(CrossAttnUpBlock2D, self).__init__()
        self.attentions = nn.ModuleList([
            Transformer2DModel(out_channels, out_channels, n_layers, use_self_attn=use_self_attn, cross_attention_dim=cross_attention_dim),
            Transformer2DModel(out_channels, out_channels, n_layers, use_self_attn=use_self_attn, cross_attention_dim=cross_attention_dim),
            Transformer2DModel(out_channels, out_channels, n_layers, use_self_attn=use_self_attn, cross_attention_dim=cross_attention_dim),
        ])
        self.resnets = nn.ModuleList([
            ResnetBlock2D(prev_output_channel + out_channels, out_channels, time_embed_dim=time_embed_dim),
            ResnetBlock2D(2 * out_channels, out_channels, time_embed_dim=time_embed_dim),
            ResnetBlock2D(out_channels + in_channels, out_channels, time_embed_dim=time_embed_dim),
        ])
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels)])

    def forward(self, hidden_states, res_hidden_states_tuple, temb, encoder_hidden_states):
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

class UpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel, time_embed_dim=1024):
        super(UpBlock2D, self).__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(out_channels + prev_output_channel, out_channels, time_embed_dim=time_embed_dim),
            ResnetBlock2D(out_channels * 2, out_channels, time_embed_dim=time_embed_dim),
            ResnetBlock2D(out_channels + in_channels, out_channels, time_embed_dim=time_embed_dim),
        ])

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states

class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(self, in_features, n_layers=4, use_self_attn=True, time_embed_dim=1024, cross_attention_dim=1024):
        super(UNetMidBlock2DCrossAttn, self).__init__()
        self.attentions = nn.ModuleList([Transformer2DModel(in_features, in_features, n_layers=n_layers, use_self_attn=use_self_attn, cross_attention_dim=cross_attention_dim)])
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_features, in_features, time_embed_dim=time_embed_dim, conv_shortcut=False),
            ResnetBlock2D(in_features, in_features, time_embed_dim=time_embed_dim, conv_shortcut=False),
        ])

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states

class UNet2DConditionModel(nn.Module):
    def __init__(self, num_classes=12476):
        super(UNet2DConditionModel, self).__init__()
        self.num_classes = num_classes

        self.config = namedtuple("config", "in_channels sample_size")
        self.config.in_channels = 4
        self.config.sample_size = 64

        time_embed_dim = 1024
        
        self.conv_in = nn.Conv2d(4, 256, kernel_size=3, stride=1, padding=1)
        self.time_proj = Timesteps(num_channels=256)
        self.time_embedding = TimestepEmbedding(in_features=256, out_features=time_embed_dim)
        
        self.y_embedder = nn.Embedding(num_classes + 1, time_embed_dim)

        # channels = [256, 512, 896], num_blocks = [0, 2, 4]
        self.down_blocks = nn.ModuleList([
            DownBlock2D(in_channels=256, out_channels=256, time_embed_dim=time_embed_dim),
            CrossAttnDownBlock2D(
                in_channels=256, out_channels=512, n_layers=2, 
                has_downsamplers=True, use_self_attn=False, time_embed_dim=time_embed_dim
            ),
            CrossAttnDownBlock2D(
                in_channels=512, out_channels=896, n_layers=4, 
                has_downsamplers=False, use_self_attn=True, time_embed_dim=time_embed_dim
            ),
        ])
        
        self.up_blocks = nn.ModuleList([
            CrossAttnUpBlock2D(
                in_channels=512, out_channels=896, prev_output_channel=896, n_layers=4, 
                use_self_attn=True, time_embed_dim=time_embed_dim
            ),
            CrossAttnUpBlock2D(
                in_channels=256, out_channels=512, prev_output_channel=896, n_layers=2, 
                use_self_attn=False, time_embed_dim=time_embed_dim
            ),
            UpBlock2D(in_channels=256, out_channels=256, prev_output_channel=512, time_embed_dim=time_embed_dim),
        ])
        
        self.mid_block = UNetMidBlock2DCrossAttn(896, n_layers=4, use_self_attn=True, time_embed_dim=time_embed_dim)
        
        self.conv_norm_out = nn.GroupNorm(32, 256, eps=1e-05, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x_in, t, y_indices, y_offsets=None, return_features=False, encoder_hidden_states=None, **kwargs):
        timesteps = t.expand(x_in.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=x_in.dtype)

        emb = self.time_embedding(t_emb)

        if y_indices is not None:
            if y_indices.dim() == 1 and y_offsets is not None:
                sequences = []
                offsets = y_offsets.tolist()
                offsets.append(y_indices.size(0))
                for i in range(len(offsets) - 1):
                    start = offsets[i]
                    end = offsets[i+1]
                    sequences.append(y_indices[start:end])
                y_indices = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.num_classes)
            encoder_hidden_states = self.y_embedder(y_indices)

        sample = self.conv_in(x_in)

        s0 = sample
        sample, [s1, s2, s3] = self.down_blocks[0](sample, temb=emb)
        sample, [s4, s5, s6] = self.down_blocks[1](sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
        sample, [s7, s8] = self.down_blocks[2](sample, temb=emb, encoder_hidden_states=encoder_hidden_states)

        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        sample = self.up_blocks[0](
            hidden_states=sample, temb=emb, res_hidden_states_tuple=[s6, s7, s8], encoder_hidden_states=encoder_hidden_states
        )
        sample = self.up_blocks[1](
            hidden_states=sample, temb=emb, res_hidden_states_tuple=[s3, s4, s5], encoder_hidden_states=encoder_hidden_states
        )
        sample = self.up_blocks[2](
            hidden_states=sample, temb=emb, res_hidden_states_tuple=[s0, s1, s2]
        )

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return [sample]

if __name__ == "__main__":
    model = UNet2DConditionModel()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {num_params / 1e6:.2f} M")
    
    batch_size = 2
    sample = torch.randn(batch_size, 4, 64, 64)
    timesteps = torch.tensor([1, 2])
    encoder_hidden_states = torch.randn(batch_size, 77, 2048)
    
    # Batch of 2, unpadded sequence
    y_indices = torch.tensor([10, 20, 30, 40, 50, 60])
    y_offsets = torch.tensor([0, 4])  # First seq length 4, second length 2
    
    print("Running forward pass...")
    out = model(sample, timesteps, y_indices, y_offsets)
    print("Forward pass successful!")
    print("Output shape:", out[0].shape)