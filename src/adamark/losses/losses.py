import torch
import torch.fft
import torch.nn as nn


class HidingLoss(nn.Module):
    """Imperceptibility loss on the container: 1 - SSIM between container and cover"""

    def __init__(self, ssim_weight=1.0):
        super().__init__()
        self.ssim_weight = float(ssim_weight)

    def forward(self, container, cover, ssim_func):
        B = container.shape[0]

        ssim_val = ssim_func(
            container.clamp(0.0, 1.0),
            cover.clamp(0.0, 1.0),
            window_size=11,
            max_val=1.0,
        )

        if ssim_val.dim() == 0:
            ssim_val = ssim_val.unsqueeze(0).expand(B)
        elif ssim_val.dim() > 1:
            ssim_val = ssim_val.mean(dim=[d for d in range(1, ssim_val.dim())])

        loss_per_image = 1.0 - ssim_val

        return (self.ssim_weight * loss_per_image).mean()


class RevealingLoss(nn.Module):
    """L1 between the retrieved secret and the mask-aware target secret

    Inside tampered regions the target is inverted (1 - secret), which turns the
    revealing net into a tamper localizer: mismatch reveals where the image changed.
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, retrieved_secret, secret_images, mask, lambda_r_base=1.0):
        target = secret_images * (1.0 - mask) + (1.0 - secret_images) * mask
        loss_per_image = self.l1(retrieved_secret, target).mean(dim=(1, 2, 3))

        return (loss_per_image * lambda_r_base).mean()


class FFTLoss(nn.Module):
    """Pushes the watermark's low-frequency energy ratio toward an rf-dependent target

    Higher rf -> more energy allowed in low frequencies, i.e. more robust and less hidden.
    """

    def __init__(self, threshold_freq=0.55, min_lf=0.025, max_lf=0.4):
        super().__init__()
        self.threshold_freq = float(threshold_freq)
        self.min_lf = float(min_lf)
        self.max_lf = float(max_lf)

    def forward(self, watermark, rf):
        w = watermark
        rf_vec = rf.reshape(w.shape[0]).detach()
        device = w.device

        fft = torch.fft.fft2(w, norm="ortho")
        power = fft.real ** 2 + fft.imag ** 2

        fy = torch.fft.fftfreq(w.shape[2], device=device)
        fx = torch.fft.fftfreq(w.shape[3], device=device)
        Y, X = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(X ** 2 + Y ** 2) * 2.0

        lf_mf_mask = (radius <= self.threshold_freq).to(w.dtype).unsqueeze(0).unsqueeze(0)
        hf_mask = (radius > self.threshold_freq).to(w.dtype).unsqueeze(0).unsqueeze(0)

        energy_lf = torch.sum(power * lf_mf_mask, dim=(1, 2, 3))
        energy_hf = torch.sum(power * hf_mask, dim=(1, 2, 3))

        total_energy = energy_lf + energy_hf + 1e-8
        ratio_lf = energy_lf / total_energy

        target_lf = self.min_lf + rf_vec * (self.max_lf - self.min_lf)

        loss_per_batch = torch.abs(ratio_lf - target_lf)
        return loss_per_batch.mean()


def enforce_watermark_budget(w_raw, rf_tensor, rms_min=0.005, rms_max=0.0225, eps=1e-6):
    """Rescale the watermark so its per-image RMS matches an rf-dependent budget

    Higher rf -> larger RMS budget, i.e. a stronger and more robust watermark.
    """

    w_centered = w_raw - w_raw.mean(dim=(1, 2, 3), keepdim=True)

    rms = torch.sqrt(torch.mean(w_centered ** 2, dim=(1, 2, 3), keepdim=True) + eps).detach()

    rf_view = rf_tensor.view(-1, 1, 1, 1).to(w_raw.device)
    target_rms = rms_min + rf_view * (rms_max - rms_min)

    return w_centered * (target_rms / rms)
