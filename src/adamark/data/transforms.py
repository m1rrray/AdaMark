import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T


def get_adaptive_transforms(image_size=256):
    """Resize to a square and convert to a [0, 1] tensor."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])


class CenterCropResize(nn.Module):
    """Center-crop by ``ratio`` then resize back to the original resolution.

    Used both as a training/eval crop attack and to crop the reference secret so
    retrieval metrics stay aligned after the crop.
    """

    def __init__(self, ratio=0.8, mode="bilinear"):
        super().__init__()
        self.ratio = float(ratio)
        self.mode = mode

    def forward(self, x):
        B, C, H, W = x.shape
        nh, nw = int(H * self.ratio), int(W * self.ratio)
        top = (H - nh) // 2
        left = (W - nw) // 2
        crop = x[:, :, top:top + nh, left:left + nw]

        if self.mode == "nearest":
            return F.interpolate(crop, size=(H, W), mode="nearest")
        return F.interpolate(crop, size=(H, W), mode=self.mode, align_corners=False)
