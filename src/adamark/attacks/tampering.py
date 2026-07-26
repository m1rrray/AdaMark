"""Tampering operations: generate masks and splice or copy-move forged regions"""

import random

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F


class IrregularMask(nn.Module):
    """Generates irregular tamper masks from brush strokes, noise blobs or boxes"""

    def __init__(
        self,
        min_strokes=2,
        max_strokes=8,
        min_width=5,
        max_width=25,
        noise_down=16,
        noise_thresh=(0.75, 0.92),
        feather_sigma=(1.0, 4.0),
        morph_iters=(0, 2),
    ):
        super().__init__()
        self.min_strokes = min_strokes
        self.max_strokes = max_strokes
        self.min_width = min_width
        self.max_width = max_width
        self.noise_down = noise_down
        self.noise_thresh = noise_thresh
        self.feather_sigma = feather_sigma
        self.morph_iters = morph_iters

    @torch.no_grad()
    def _brush_mask(self, B, H, W, device):
        mask = torch.zeros((B, 1, H, W), device=device)
        for i in range(B):
            n = random.randint(self.min_strokes, self.max_strokes)
            for _ in range(n):
                x0 = random.randint(0, W - 1)
                y0 = random.randint(0, H - 1)
                dx = random.randint(-W // 4, W // 4)
                dy = random.randint(-H // 4, H // 4)
                x1 = max(0, min(W - 1, x0 + dx))
                y1 = max(0, min(H - 1, y0 + dy))

                width = random.randint(self.min_width, self.max_width)
                steps = max(abs(x1 - x0), abs(y1 - y0), 1)
                for s in range(steps + 1):
                    t = s / steps
                    x = int(round(x0 * (1 - t) + x1 * t))
                    y = int(round(y0 * (1 - t) + y1 * t))
                    xL, xR = max(0, x - width // 2), min(W, x + width // 2 + 1)
                    yT, yB = max(0, y - width // 2), min(H, y + width // 2 + 1)
                    mask[i, 0, yT:yB, xL:xR] = 1.0
        return mask

    @torch.no_grad()
    def _noise_blob_mask(self, B, H, W, device):
        h2 = max(4, H // self.noise_down)
        w2 = max(4, W // self.noise_down)
        noise = torch.rand((B, 1, h2, w2), device=device)
        noise = F.interpolate(noise, size=(H, W), mode="bilinear", align_corners=False)
        thr = random.uniform(*self.noise_thresh)
        return (noise > thr).float()

    @torch.no_grad()
    def _box_mask(self, B, H, W, device):
        mask = torch.zeros((B, 1, H, W), device=device)
        for i in range(B):
            box_w = random.randint(int(W * 0.1), int(W * 0.3))
            box_h = random.randint(int(H * 0.1), int(H * 0.3))
            x0 = random.randint(0, W - box_w)
            y0 = random.randint(0, H - box_h)
            mask[i, 0, y0:y0 + box_h, x0:x0 + box_w] = 1.0
        return mask

    def forward(self, B, H, W, device):
        choice = random.random()
        if choice < 0.33:
            mask = self._brush_mask(B, H, W, device)
        elif choice < 0.66:
            mask = self._noise_blob_mask(B, H, W, device)
        else:
            mask = self._box_mask(B, H, W, device)

        iters = random.randint(*self.morph_iters)
        if iters > 0:
            kernel = torch.ones((3, 3), device=device, dtype=mask.dtype)
            for _ in range(iters):
                if random.random() < 0.5:
                    mask = kornia.morphology.dilation(mask, kernel)
                else:
                    mask = kornia.morphology.erosion(mask, kernel)

        if self.feather_sigma[1] > 0:
            sigma = random.uniform(*self.feather_sigma)
            k = int(2 * round(3 * sigma) + 1)
            k = max(3, min(k, 31))
            mask = kornia.filters.gaussian_blur2d(mask, (k, k), (sigma, sigma)).clamp(0, 1)

        return mask


class Tampering(nn.Module):
    """Training-time tampering: irregular mask plus copy-move or cross-paste"""

    def __init__(self):
        super().__init__()
        self.mask_gen = IrregularMask(feather_sigma=(0.0, 0.0))

    def _copy_move(self, x, cover, mask):
        B, _, H, W = x.shape
        out = x.clone()
        for i in range(B):
            dx = random.randint(-W // 4, W // 4)
            dy = random.randint(-H // 4, H // 4)
            src = torch.roll(cover[i:i + 1], shifts=(dy, dx), dims=(2, 3))
            out[i:i + 1] = x[i:i + 1] * (1 - mask[i:i + 1]) + src * mask[i:i + 1]
        return out

    def _cross_paste(self, x, cover, mask):
        src = torch.roll(cover, shifts=-1, dims=0)
        return x * (1 - mask) + src * mask

    def forward(self, x, cover):
        B, _, H, W = x.shape
        mask = self.mask_gen(B, H, W, x.device)
        mask = (mask > 0.5).float()

        if random.random() < 0.5:
            out = self._copy_move(x, cover, mask)
        else:
            out = self._cross_paste(x, cover, mask)

        return out.clamp(0, 1), mask


class BlockTampering(nn.Module):
    """Evaluation-time tampering with rectangular blocks, without IrregularMask"""

    def __init__(self, max_tamper_blocks=2, min_tamper_size=32, max_tamper_size=80):
        super().__init__()
        self.max_tamper_blocks = max_tamper_blocks
        self.min_tamper_size = min_tamper_size
        self.max_tamper_size = max_tamper_size

    def _rand_mask(self, B, H, W, device):
        mask = torch.zeros((B, 1, H, W), device=device)
        for i in range(B):
            num_blocks = random.randint(1, self.max_tamper_blocks)
            for _ in range(num_blocks):
                bh = random.randint(self.min_tamper_size, min(self.max_tamper_size, H))
                bw = random.randint(self.min_tamper_size, min(self.max_tamper_size, W))
                top = random.randint(0, H - bh)
                left = random.randint(0, W - bw)
                mask[i, :, top:top + bh, left:left + bw] = 1.0
        return mask

    def forward(self, x):
        B, _, H, W = x.shape
        device = x.device
        mask = self._rand_mask(B, H, W, device)

        src = torch.roll(x, shifts=-1, dims=0)
        out = x * (1 - mask) + src * mask

        return out.clamp(0, 1), mask
