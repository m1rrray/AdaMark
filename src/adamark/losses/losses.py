import torch
import torch.nn as nn
import torch.fft


class HidingLoss(nn.Module):
    """Imperceptibility loss on the container: 1 - SSIM(container, cover)."""

    def __init__(self, ssim_weight=1.0):
        super().__init__()
        self.ssim_weight = float(ssim_weight)

    def forward(self, container, cover, rf_vec=None, lambda_h_base=1.0,
                w_fragile=1.0, w_robust=1.0, ssim_func=None):
        B = container.shape[0]

        if ssim_func is not None:
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
        else:
            loss_per_image = torch.zeros(B, device=container.device, dtype=container.dtype)

        return (self.ssim_weight * loss_per_image).mean()


class RevealingLoss(nn.Module):
    """L1 between the retrieved secret and the mask-aware target secret.

    Inside tampered regions the target is inverted (1 - secret), which turns the
    revealing net into a tamper localizer: mismatch reveals where the image changed.
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, retrieved_secret, secret_images, mask, rf_vec, lambda_r_base=1.0):
        target = secret_images * (1.0 - mask) + (1.0 - secret_images) * mask
        loss_per_image = self.l1(retrieved_secret, target).mean(dim=(1, 2, 3))

        dynamic_weight = lambda_r_base * 1.0
        return (loss_per_image * dynamic_weight).mean()


class FFTLoss(nn.Module):
    """Pushes the watermark's low-frequency energy ratio toward an rf-dependent target.

    Higher rf -> more energy allowed in low frequencies (more robust, less hidden).
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

        target_lf = self.min_lf + (rf_vec ** 1) * (self.max_lf - self.min_lf)

        loss_per_batch = torch.abs(ratio_lf - target_lf)
        return loss_per_batch.mean()


def enforce_watermark_budget(w_raw, rf_tensor=None, rms_min=0.005, rms_max=0.0225,
                             eps=1e-6, clip=None):
    """Rescale the watermark so its per-image RMS matches an rf-dependent budget.

    Higher rf -> larger RMS budget (stronger, more robust watermark).
    """
    w = w_raw
    w_centered = w - w.mean(dim=(1, 2, 3), keepdim=True)

    rms = torch.sqrt(torch.mean(w_centered ** 2, dim=(1, 2, 3), keepdim=True) + eps).detach()

    if rf_tensor is None:
        target_rms = torch.full_like(rms, float(rms_min))
    else:
        rf_view = rf_tensor.view(-1, 1, 1, 1).to(w.device)
        target_rms = rms_min + rf_view * (rms_max - rms_min)

    w_final = w_centered * (target_rms / rms)

    if clip is not None:
        w_final = w_final.clamp(-clip, clip)

    return w_final
