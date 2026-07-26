"""rf-conditioned modulation blocks shared by the hiding network

The robustness factor ``rf`` controls the imperceptibility/robustness trade-off.
It is injected into the convolutional blocks through FiLM modulation, whose
parameters are predicted from a sinusoidal embedding of ``rf``.
"""

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Maps a scalar rf in [0, 1] to a sinusoidal feature vector"""

    def __init__(self, num_freqs: int = 8):
        super().__init__()
        freqs = (2.0 ** torch.arange(num_freqs)) * torch.pi
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        phases = x * self.freqs
        return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)


class FiLM(nn.Module):
    """Feature-wise linear modulation conditioned on rf"""

    def __init__(self, num_channels: int, hidden: int = 128,
                 scale: float = 1.0, num_freqs: int = 8):
        super().__init__()
        self.scale = float(scale)
        self.embed = SinusoidalEmbedding(num_freqs=num_freqs)

        self.mlp = nn.Sequential(
            nn.Linear(num_freqs * 2, hidden),
            nn.SiLU(inplace=False),
            nn.Linear(hidden, num_channels * 2),
        )

        # Zero-init the last layer so modulation starts as identity.
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, x, rf):
        rf_embedded = self.embed(rf)

        style = self.mlp(rf_embedded) * self.scale

        gamma, beta = style.chunk(2, dim=1)

        gamma = gamma.view(-1, x.size(1), 1, 1)
        beta = beta.view(-1, x.size(1), 1, 1)

        return (1.0 + gamma) * x + beta


class ConvDownBlock(nn.Module):
    """Encoder block: strided conv, InstanceNorm, rf-conditioned FiLM and LeakyReLU"""

    def __init__(self, in_c, out_c, film_scale=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
        self.norm = nn.InstanceNorm2d(out_c, affine=False)
        self.film = FiLM(out_c, hidden=128, scale=film_scale)
        self.act = nn.LeakyReLU(0.1, inplace=False)

    def forward(self, x, rf):
        x = self.conv(x)
        x = self.norm(x)
        x = self.film(x, rf)
        x = self.act(x)
        return x


class ConvUpBlock(nn.Module):
    """Decoder block: transposed conv, InstanceNorm, rf-conditioned FiLM and ReLU"""

    def __init__(self, in_c, out_c, film_scale=1.0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
        self.norm = nn.InstanceNorm2d(out_c, affine=False)
        self.film = FiLM(out_c, hidden=128, scale=film_scale)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x, rf):
        x = self.conv(x)
        x = self.norm(x)
        x = self.film(x, rf)
        x = self.act(x)
        return x
