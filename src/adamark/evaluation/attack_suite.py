"""Builds the evaluation attack suite by name, as image and optional mask transforms"""

import kornia
import torch

from adamark.attacks.distortion import JpegDistortion
from adamark.data.transforms import CenterCropResize


def build_eval_attacks(device):
    """Return a dict mapping attack_name to a pair of image_fn and mask_fn or None

    Names follow the ``<category>_<param>`` convention parsed downstream by reporting.
    ``mask_fn`` is provided only for geometric attacks that also move the GT mask.
    """

    attacks_dict = {}

    jpeg_qualities = [100, 90, 80, 70, 60, 50, 40, 30, 20]
    for q in jpeg_qualities:
        dist = JpegDistortion(jpeg_quality=q).to(device)
        attacks_dict[f"jpeg_{q}"] = (lambda x, d=dist: d(x), None)

    blur_sigmas = [0.5, 1.0, 1.5, 2.0]
    for s in blur_sigmas:
        ks = max(3, int(2 * round(3 * s) + 1))
        blur = kornia.filters.GaussianBlur2d((ks, ks), (s, s)).to(device)
        attacks_dict[f"blur_{s}"] = (lambda x, b=blur: b(x), None)

    crop_ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    for r in crop_ratios:
        crop_img = CenterCropResize(ratio=r, mode="bilinear").to(device)
        crop_mask = CenterCropResize(ratio=r, mode="nearest").to(device)
        attacks_dict[f"crop_{r}"] = (lambda x, c=crop_img: c(x), lambda m, c=crop_mask: c(m))

    noise_stds = [0.05, 0.1, 0.15]
    for std in noise_stds:
        gauss_noise = kornia.augmentation.RandomGaussianNoise(mean=0.0, std=std, p=1.0).to(device)
        attacks_dict[f"noise_{std}"] = (lambda x, n=gauss_noise: n(x).clamp(0, 1), None)

    rot_degrees = [5.0]
    for deg in rot_degrees:
        attacks_dict[f"rot_{deg}"] = (
            lambda x, d=deg: kornia.geometry.transform.rotate(x, torch.tensor([d], device=device)),
            lambda m, d=deg: kornia.geometry.transform.rotate(m, torch.tensor([d], device=device), mode="nearest"),
        )

    jitter_strengths = [0.1, 0.3, 0.5]
    for s in jitter_strengths:
        color_jitter = kornia.augmentation.ColorJitter(
            brightness=s, contrast=s, saturation=s, hue=0.0, p=1.0).to(device)
        attacks_dict[f"color_{s}"] = (lambda x, c=color_jitter: c(x), None)

    grayscale = kornia.augmentation.RandomGrayscale(p=1.0).to(device)
    attacks_dict["grayscale_1.0"] = (lambda x, g=grayscale: g(x), None)

    hflip_img = kornia.augmentation.RandomHorizontalFlip(p=1.0).to(device)
    attacks_dict["hflip_1.0"] = (lambda x, f=hflip_img: f(x), lambda m, f=hflip_img: f(m))

    return attacks_dict
