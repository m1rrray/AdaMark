"""HidingNet: rf-conditioned U-Net that produces an additive watermark

Layer attribute names are part of the checkpoint format and must not be renamed.
"""

import torch
import torch.nn as nn

from .modulation import FiLM, ConvDownBlock, ConvUpBlock


class HidingNet(nn.Module):
    """U-Net that turns a secret payload into an additive watermark pattern

    Every block is conditioned on rf through FiLM, so one trained model spans the
    whole imperceptibility/robustness range. The tanh output lies in [-1, 1] and
    gets its final amplitude from ``enforce_watermark_budget``.
    """

    def __init__(self, in_channels=3, out_channels=3, film_scale=1.0, use_film=True):
        super().__init__()

        self.enc1 = ConvDownBlock(in_channels, 32, film_scale, use_film)   # 256 -> 128
        self.enc2 = ConvDownBlock(32, 64, film_scale, use_film)            # 128 -> 64
        self.enc3 = ConvDownBlock(64, 128, film_scale, use_film)           # 64 -> 32
        self.enc4 = ConvDownBlock(128, 256, film_scale, use_film)          # 32 -> 16
        self.enc5 = ConvDownBlock(256, 512, film_scale, use_film)          # 16 -> 8
        self.enc6 = ConvDownBlock(512, 1024, film_scale, use_film)         # 8 -> 4
        self.enc7 = ConvDownBlock(1024, 1024, film_scale, use_film)        # 4 -> 2

        self.dec6 = ConvUpBlock(1024, 1024, film_scale, use_film)      # 2 -> 4
        self.dec5 = ConvUpBlock(2048, 512, film_scale, use_film)       # 4 -> 8, 1024 dec6 + 1024 enc6
        self.dec4 = ConvUpBlock(1024, 256, film_scale, use_film)       # 8 -> 16, 512 dec5 + 512 enc5
        self.dec3 = ConvUpBlock(512, 128, film_scale, use_film)        # 16 -> 32, 256 dec4 + 256 enc4
        self.dec2 = ConvUpBlock(256, 64, film_scale, use_film)         # 32 -> 64, 128 dec3 + 128 enc3
        self.dec1 = ConvUpBlock(128, 32, film_scale, use_film)         # 64 -> 128, 64 dec2 + 64 enc2

        self.final_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)  # 128 -> 256
        self.film_final = FiLM(out_channels, hidden=128, scale=film_scale, enabled=use_film)
        self.final_act = nn.Tanh()

    def forward(self, secret_img, rf):
        e1 = self.enc1(secret_img, rf)
        e2 = self.enc2(e1, rf)
        e3 = self.enc3(e2, rf)
        e4 = self.enc4(e3, rf)
        e5 = self.enc5(e4, rf)
        e6 = self.enc6(e5, rf)
        e7 = self.enc7(e6, rf)

        d6 = self.dec6(e7, rf)
        d5 = self.dec5(torch.cat([d6, e6], dim=1), rf)
        d4 = self.dec4(torch.cat([d5, e5], dim=1), rf)
        d3 = self.dec3(torch.cat([d4, e4], dim=1), rf)
        d2 = self.dec2(torch.cat([d3, e3], dim=1), rf)
        d1 = self.dec1(torch.cat([d2, e2], dim=1), rf)

        out = self.final_conv(torch.cat([d1, e1], dim=1))
        out = self.film_final(out, rf)
        watermark = self.final_act(out)

        return watermark
