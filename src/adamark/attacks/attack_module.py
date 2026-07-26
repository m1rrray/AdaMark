"""Training-time attack pipeline combining tampering and distortion"""

import random

import torch
import torch.nn as nn

from .distortion import Distortion
from .tampering import Tampering


class AttackModule(nn.Module):
    """Applies tampering and distortion to a watermarked container during training

    Returns the attacked image and the binary tamper mask used as the localization
    target. A fraction of samples become negatives: reverted to cover, with a full mask.
    """

    def __init__(self, p_tamper=0.5, p_distort=0.8, p_hardcore=0.15, p_negative=0.15,
                 jpeg_quality=15.0, max_blur_sigma=1.75):
        super().__init__()
        self.p_tamper = p_tamper
        self.p_distort = p_distort
        self.p_hardcore = p_hardcore
        self.p_negative = p_negative

        self.tampering = Tampering()
        self.distortion = Distortion(jpeg_quality=jpeg_quality, max_blur_sigma=max_blur_sigma)

    def forward(self, container, cover, rf):
        B, _, H, W = container.shape
        device = container.device

        img_out = container.clone()
        mask_out = torch.zeros((B, 1, H, W), device=device)

        is_negative = (torch.rand((B,), device=device) < self.p_negative)
        if is_negative.any():
            img_out[is_negative] = cover[is_negative].clone()
            mask_out[is_negative] = 1.0

        do_tamper = (torch.rand((B,), device=device) < self.p_tamper) & (~is_negative)
        if do_tamper.any():
            t_img, t_mask = self.tampering(img_out[do_tamper], cover[do_tamper])
            img_out[do_tamper] = t_img
            mask_out[do_tamper] = t_mask

        if random.random() < self.p_hardcore:
            img_out = self.distortion(img_out, rf, force_max=True)
        elif random.random() < self.p_distort:
            img_out = self.distortion(img_out, rf, force_max=False)

        return img_out.clamp(0, 1), mask_out.clamp(0, 1)
