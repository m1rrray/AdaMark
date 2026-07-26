"""Persist evaluation results to Excel and per-attack metric plots"""

import math
import os

import matplotlib.pyplot as plt
import pandas as pd

ATTACK_LABELS = {
    "jpeg": "JPEG compression",
    "blur": "Gaussian blur",
    "crop": "Cropping",
    "noise": "Gaussian noise",
    "rot": "Rotation, deg",
    "color": "Color jitter",
    "grayscale": "Grayscale",
    "hflip": "Horizontal flip",
}

ATTACKS = ["jpeg", "blur", "crop", "noise", "rot", "color", "grayscale", "hflip"]

# Attacks whose parameter decreases as the distortion gets stronger.
DESCENDING_ATTACKS = ["jpeg", "crop"]


def save_results_to_excel(results_rf, dataset_name="dataset", output_dir="results"):
    """Flatten per-rf results into a tidy table and write an .xlsx file"""

    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for res in results_rf:
        model_name = f"RF={res['rf']}"

        rows.append({
            "Dataset": dataset_name, "Model": model_name, "Attack_Category": "None", "Attack_Param": 0,
            "PSNR_Container": res["imperceptibility"]["psnr"], "SSIM_Container": res["imperceptibility"]["ssim"],
            "Retrieval_PSNR": None, "Retrieval_SSIM": None,
            "Tamper_AUC": None, "Tamper_AP": None, "Tamper_F1": None,
        })

        for atk_key in res["retrieval_robustness"].keys():
            parts = atk_key.split("_")
            atk_category = parts[0]
            atk_param = float(parts[1]) if len(parts) > 1 else 0.0

            rows.append({
                "Dataset": dataset_name, "Model": model_name,
                "Attack_Category": atk_category, "Attack_Param": atk_param,
                "PSNR_Container": None, "SSIM_Container": None,
                "Retrieval_PSNR": res["retrieval_robustness"][atk_key]["psnr"],
                "Retrieval_SSIM": res["retrieval_robustness"][atk_key]["ssim"],
                "Tamper_AUC": res["tamper_localization"][atk_key]["pixel_auc"],
                "Tamper_AP": res["tamper_localization"][atk_key]["pixel_ap"],
                "Tamper_F1": res["tamper_localization"][atk_key]["pixel_f1"],
            })

    df = pd.DataFrame(rows)
    excel_path = os.path.join(output_dir, f"metrics_{dataset_name}.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"[*] Metrics saved to: {excel_path}")


def plot_all_results(results, dataset_name, output_dir="results"):
    """Plot imperceptibility vs rf and per-attack robustness and localization grids"""

    os.makedirs(output_dir, exist_ok=True)
    rf_values = [res["rf"] for res in results]

    # Stage 1: imperceptibility, PSNR and SSIM, vs rf.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    psnr_values = [res["imperceptibility"]["psnr"] for res in results]
    ssim_values = [res["imperceptibility"]["ssim"] for res in results]

    ax1.plot(rf_values, psnr_values, marker="o", label="AdaMark (RF)")
    ax1.set_xlabel("Robustness factor (RF)")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title(f"[{dataset_name}] Imperceptibility: PSNR vs RF")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    ax2.plot(rf_values, ssim_values, marker="o", label="AdaMark (RF)")
    ax2.set_xlabel("Robustness factor (RF)")
    ax2.set_ylabel("SSIM")
    ax2.set_title(f"[{dataset_name}] Imperceptibility: SSIM vs RF")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_stage1_imperceptibility.png"), dpi=300)
    plt.close(fig)

    num_attacks = len(ATTACKS)
    cols = 4
    rows = math.ceil(num_attacks / cols)

    def draw_grid_plot(metric_dict_key, metric_key, title_prefix, ylabel, filename):
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten()
        fig.suptitle(f"[{dataset_name}] {title_prefix}", fontsize=16, y=1.02)

        # All rf results share the same set of attack keys; use the first as reference.
        source = next((res[metric_dict_key] for res in results if metric_dict_key in res), None)

        for i, attack in enumerate(ATTACKS):
            ax = axes[i]
            attack_label = ATTACK_LABELS.get(attack, attack)

            if not source:
                ax.set_visible(False)
                continue

            attack_keys = [k for k in source.keys() if k.startswith(attack + "_")]

            descending = attack in DESCENDING_ATTACKS
            sorted_keys = sorted(attack_keys, key=lambda k: float(k.split("_")[1]),
                                 reverse=descending)

            attack_params = [float(k.split("_")[1]) for k in sorted_keys]

            for rf in rf_values:
                rf_result = next((res for res in results if res["rf"] == rf), None)
                if rf_result and metric_dict_key in rf_result:
                    rf_attack_values = [rf_result[metric_dict_key][k][metric_key] for k in sorted_keys]
                    ax.plot(attack_params, rf_attack_values, marker="o", label=f"AdaMark (RF={rf})")

            ax.set_xlabel(f"{attack_label} intensity")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Attack: {attack_label}")
            ax.grid(True, linestyle="--", alpha=0.5)

            if descending:
                ax.invert_xaxis()

            ax.legend(fontsize="small")

        for j in range(num_attacks, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_{filename}"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    draw_grid_plot("retrieval_robustness", "psnr", "Robustness: PSNR of the extracted secret",
                   "PSNR (dB)", "stage2_retrieval_psnr.png")
    draw_grid_plot("retrieval_robustness", "ssim", "Robustness: SSIM of the extracted secret",
                   "SSIM", "stage2_retrieval_ssim.png")
    draw_grid_plot("tamper_localization", "pixel_auc", "Localization accuracy: ROC AUC",
                   "Pixel AUC", "stage3_tamper_auc.png")
    draw_grid_plot("tamper_localization", "pixel_ap", "Localization accuracy: Average Precision (AP)",
                   "Pixel AP", "stage4_tamper_ap.png")
    draw_grid_plot("tamper_localization", "pixel_f1", "Localization accuracy: F1 score",
                   "Pixel F1", "stage5_tamper_f1.png")
