"""Single-pass evaluation of imperceptibility, retrieval robustness and tamper localization"""

import piq
import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from adamark.attacks.tampering import BlockTampering
from adamark.losses import enforce_watermark_budget
from adamark.utils import generate_qr_payloads


def calculate_ssim(img1, img2):
    return piq.ssim(img1.float(), img2.float(), data_range=1.0)


def calculate_psnr(img1, img2):
    return piq.psnr(img1.float(), img2.float(), data_range=1.0)


def evaluate_model_single_pass(hiding_net, revealing_net, loader, device, attacks_dict,
                               rf_value, verbose=True, tamper_threshold=0.5):
    """Evaluate the adaptive model at a fixed rf across the attack suite

    For each attack the function measures retrieval quality, as PSNR and SSIM of the
    recovered secret, and tamper localization, as pixel AUROC, AP and F1 of the
    mismatch map.
    """

    hiding_net.eval()
    revealing_net.eval()

    tamper_attack = BlockTampering()

    desc = f"Eval RF={rf_value:.2f}"
    iterable = tqdm(loader, desc=desc) if verbose else loader

    metrics = {
        "imperceptibility": {"psnr": 0.0, "ssim": 0.0, "count": 0},
        "retrieval": {atk: {"psnr": 0.0, "ssim": 0.0, "count": 0}
                      for atk in attacks_dict},
        "tamper": {atk: {"auroc": None, "ap": None, "TP": 0, "FP": 0, "TN": 0, "FN": 0}
                   for atk in attacks_dict},
    }

    for atk in attacks_dict:
        metrics["tamper"][atk]["auroc"] = BinaryAUROC(thresholds=1000).to(device)
        metrics["tamper"][atk]["ap"] = BinaryAveragePrecision(thresholds=1000).to(device)

    with torch.no_grad():
        for batch in iterable:
            if len(batch) == 3:
                cover_img, tampered_dataset_img, mask = batch
                cover_img = cover_img.to(device)
                tampered_dataset_img = tampered_dataset_img.to(device)
                mask = mask.to(device)
                using_real_masks = True
            else:
                cover_img, _ = batch
                cover_img = cover_img.to(device)
                using_real_masks = False

            B, _, H, W = cover_img.shape

            with torch.autocast(device_type=device.type):
                secret_img = generate_qr_payloads(B, H, W, block_size=8, device=device)
                rf_tensor = torch.full((B, 1), rf_value, device=device)
                watermark = hiding_net(secret_img, rf_tensor)
                watermark = enforce_watermark_budget(watermark, rf_tensor)

                container_img = (cover_img + watermark).clamp(0, 1)

                if using_real_masks:
                    tampered_img = container_img * (1.0 - mask) + tampered_dataset_img * mask
                else:
                    tampered_img, mask = tamper_attack(container_img)

                half_B = B // 2
                if half_B > 0:
                    tampered_img[:half_B] = container_img[:half_B]
                    mask[:half_B] = 0.0

                mse = torch.mean((cover_img - container_img) ** 2, dim=(1, 2, 3))
                psnrs = 10 * torch.log10(1.0 / (mse + 1e-10))
                metrics["imperceptibility"]["psnr"] += psnrs.sum()
                metrics["imperceptibility"]["ssim"] += calculate_ssim(cover_img, container_img) * B
                metrics["imperceptibility"]["count"] += B

                for atk_name, (atk_img_fn, atk_mask_fn) in attacks_dict.items():
                    atk_container = atk_img_fn(container_img)
                    atk_tampered = atk_img_fn(tampered_img)
                    atk_mask = atk_mask_fn(mask) if atk_mask_fn else mask

                    combined_input = torch.cat([atk_container, atk_tampered], dim=0)
                    combined_output = revealing_net(combined_input).clamp(0, 1)
                    retrieved_clean, retrieved_tamp = combined_output.chunk(2, dim=0)

                    if "crop" in atk_name:
                        reference_secret = atk_img_fn(secret_img)
                    else:
                        reference_secret = secret_img

                    metrics["retrieval"][atk_name]["psnr"] += calculate_psnr(reference_secret, retrieved_clean) * B
                    metrics["retrieval"][atk_name]["ssim"] += calculate_ssim(reference_secret, retrieved_clean) * B
                    metrics["retrieval"][atk_name]["count"] += B

                    raw_score_map = torch.abs(reference_secret - retrieved_tamp).mean(dim=1, keepdim=True)

                    score_map = F.avg_pool2d(raw_score_map, kernel_size=15, stride=1, padding=7)

                    # Subsample every 4th pixel to reduce noise and speed up metrics.
                    stride = 4
                    sub_score = score_map[:, :, ::stride, ::stride].flatten()
                    sub_mask = (atk_mask[:, :, ::stride, ::stride] >= 0.5).int().flatten()

                    if sub_mask.numel() > 0:
                        metrics["tamper"][atk_name]["auroc"].update(sub_score, sub_mask)
                        metrics["tamper"][atk_name]["ap"].update(sub_score, sub_mask)

                        y_pred = (sub_score > tamper_threshold).int()
                        metrics["tamper"][atk_name]["TP"] += (y_pred * sub_mask).sum()
                        metrics["tamper"][atk_name]["FP"] += (y_pred * (1 - sub_mask)).sum()
                        metrics["tamper"][atk_name]["FN"] += ((1 - y_pred) * sub_mask).sum()
                        metrics["tamper"][atk_name]["TN"] += ((1 - y_pred) * (1 - sub_mask)).sum()

    final_results = {
        "imperceptibility": {
            "psnr": float(metrics["imperceptibility"]["psnr"]) / metrics["imperceptibility"]["count"],
            "ssim": float(metrics["imperceptibility"]["ssim"]) / metrics["imperceptibility"]["count"],
        },
        "retrieval_robustness": {},
        "tamper_localization": {},
    }

    for atk in attacks_dict:
        c = metrics["retrieval"][atk]["count"]
        final_results["retrieval_robustness"][atk] = {
            "psnr": float(metrics["retrieval"][atk]["psnr"]) / c,
            "ssim": float(metrics["retrieval"][atk]["ssim"]) / c,
        }

        auroc_metric = metrics["tamper"][atk]["auroc"]
        ap_metric = metrics["tamper"][atk]["ap"]

        TP = float(metrics["tamper"][atk]["TP"])
        FP = float(metrics["tamper"][atk]["FP"])
        TN = float(metrics["tamper"][atk]["TN"])
        FN = float(metrics["tamper"][atk]["FN"])
        eps = 1e-8

        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        fpr = FP / (FP + TN + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)

        final_results["tamper_localization"][atk] = {
            "pixel_auc": auroc_metric.compute().item() if auroc_metric._update_called else float("nan"),
            "pixel_ap": ap_metric.compute().item() if ap_metric._update_called else float("nan"),
            "pixel_precision": precision,
            "pixel_fpr": fpr,
            "pixel_f1": f1,
        }

        auroc_metric.reset()
        ap_metric.reset()

    return final_results
