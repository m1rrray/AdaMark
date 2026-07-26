"""Differentiable JPEG and blur distortions used during training and evaluation"""

import random

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.io as tv_io


class Distortion(nn.Module):
    """rf-scaled stochastic JPEG or blur distortion applied during training

    Severity grows with rf, so high-rf samples see stronger degradation. JPEG is
    straight-through: a real codec in the forward pass, a differentiable
    approximation in the backward pass.
    """

    def __init__(self, jpeg_quality=15.0, max_blur_sigma=1.75):
        super().__init__()
        self.min_jpeg_quality = float(jpeg_quality)
        self.max_blur_sigma = float(max_blur_sigma)

    def forward(self, x, rf, force_max=False):
        x = x.clamp(0, 1)
        B, _, H, W = x.shape

        rf_vec = rf.view(-1) if rf.dim() > 0 else rf.expand(B)
        rf_vec = rf_vec.to(device=x.device).clamp(0, 1)

        if force_max:
            sev = torch.ones_like(rf_vec)
        else:
            sev = rf_vec * torch.empty_like(rf_vec).uniform_(0.0, 1.0)

        sev = sev ** 0.5

        op = random.choices(["jpeg", "blur"], weights=[0.75, 0.25])[0]

        if op == "blur":
            sigma_vec = 0.1 + sev * (self.max_blur_sigma - 0.1)
            sigma_tensor = sigma_vec.unsqueeze(1).repeat(1, 2)
            x = kornia.filters.gaussian_blur2d(x, (11, 11), sigma_tensor)
            return x.clamp(0, 1)

        # JPEG: optionally downscale then compress.
        min_scale = 0.5
        x_rescaled = torch.empty_like(x)
        for i in range(B):
            s_val = sev.view(-1)[i].item()
            current_scale = 1.0 - s_val * (1.0 - min_scale)
            if current_scale < 0.99:
                x_d = F.interpolate(x[i:i + 1], scale_factor=current_scale, mode="bilinear",
                                    align_corners=False, recompute_scale_factor=False)
                x_rescaled[i:i + 1] = F.interpolate(x_d, size=(H, W), mode="bilinear",
                                                    align_corners=False)
            else:
                x_rescaled[i:i + 1] = x[i:i + 1]

        quality_vec = 100.0 - sev * (100.0 - self.min_jpeg_quality)

        x_diff = kornia.enhance.jpeg_codec_differentiable(x_rescaled, quality_vec)

        with torch.no_grad():
            x_uint8 = (x_rescaled.clamp(0, 1) * 255.0).to(torch.uint8).cpu()
            res = []
            for i in range(B):
                q = int(quality_vec[i].item())
                if q >= 100:
                    res.append(x_uint8[i])
                else:
                    img_bytes = tv_io.encode_jpeg(x_uint8[i], quality=q)
                    res.append(tv_io.decode_jpeg(img_bytes))
            x_real = (torch.stack(res).float() / 255.0).to(x.device)

        # Straight-through estimator.
        x_out = x_real.detach() + x_diff - x_diff.detach()
        return x_out.clamp(0, 1)


def distortion_deterministic_oneof(x, rf_attack: float, mode: str,
                                   min_jpeg_quality=50.0, max_blur_sigma=2.0,
                                   blur_kernel=(7, 7)):
    """Deterministic single-mode distortion used in the validation loop"""

    B = x.shape[0]
    device = x.device
    rf = float(rf_attack)

    if rf <= 1e-8:
        return x.clamp(0.0, 1.0)

    if mode == "jpeg":
        q = 100.0 - rf * (100.0 - float(min_jpeg_quality))
        q_tensor = torch.full((B,), q, device=device, dtype=torch.float32)
        x = kornia.enhance.jpeg_codec_differentiable(x, q_tensor)

    elif mode == "blur":
        sigma = 0.1 + rf * (float(max_blur_sigma) - 0.1)
        sigma_tensor = torch.full((B, 2), sigma, device=device, dtype=torch.float32)
        x = kornia.filters.gaussian_blur2d(x, blur_kernel, sigma_tensor)

    else:
        raise ValueError(f"Unknown mode={mode}, expected 'jpeg' or 'blur'")

    return x.clamp(0.0, 1.0)


class JpegDistortion(nn.Module):
    """Real non-differentiable JPEG round-trip with optional pre-blur, for evaluation

    Quality >= 100 combined with sigma <= 0 is a no-op pass-through.
    """

    def __init__(self, jpeg_quality=100.0, blur_sigma=0.0, blur_kernel_size=7):
        super().__init__()
        self.jpeg_quality = int(jpeg_quality)
        self.blur_sigma = float(blur_sigma)

        self.gaussian_blur = None
        if self.blur_sigma and self.blur_sigma > 0:
            self.gaussian_blur = kornia.filters.GaussianBlur2d(
                (blur_kernel_size, blur_kernel_size),
                (self.blur_sigma, self.blur_sigma),
            )

    def forward(self, image_batch):
        B = image_batch.size(0)
        device = image_batch.device

        if self.jpeg_quality >= 100 and self.blur_sigma <= 0:
            return image_batch

        if self.gaussian_blur is not None:
            image_batch = self.gaussian_blur(image_batch)

        images_uint8 = (image_batch.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu()

        jpeg_list = []
        for i in range(B):
            jpeg_bytes = tv_io.encode_jpeg(images_uint8[i], quality=self.jpeg_quality)
            jpeg_list.append(tv_io.decode_jpeg(jpeg_bytes))

        return torch.stack(jpeg_list).to(device).float() / 255.0
