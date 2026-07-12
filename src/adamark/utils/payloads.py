import torch
import torch.nn.functional as F


def generate_qr_payloads(B, H, W, block_size=8, device="cuda"):
    """Generate QR-like secret payloads: random binary blocks upsampled to (H, W).

    Bits live on a coarse grid of ``block_size`` pixels, are nearest-upsampled to
    full resolution, broadcast to 3 channels and rescaled to [0.1, 0.9].
    """
    small_h, small_w = H // block_size, W // block_size
    bits = torch.randint(0, 2, (B, 1, small_h, small_w), device=device).float()
    payload = F.interpolate(bits, size=(H, W), mode="nearest")
    payload = payload.repeat(1, 3, 1, 1)
    payload = payload * 0.8 + 0.1
    return payload
