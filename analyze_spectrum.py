"""Entry point for the watermark frequency analysis of Section 4.4

Sweeps the robustness factor and reports how the watermark's spectral energy is
redistributed across frequency bands.

Usage:
    python analyze_spectrum.py --config configs/eval_config.yaml
"""

import argparse

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from adamark.data.dataset import create_tamper_dataset
from adamark.data.transforms import get_adaptive_transforms
from adamark.evaluation.spectral import (band_labels, plot_spectral_distribution,
                                         radial_band_energy, save_spectral_table,
                                         spectral_table)
from adamark.losses import enforce_watermark_budget
from adamark.models import HidingNet
from adamark.utils import generate_qr_payloads, set_seed

RF_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_hiding_net(config, device):
    """Load the embedding network from either the last or the best checkpoint"""

    hiding_net = HidingNet(use_film=bool(config.get("use_film", True))).to(device)

    if config.get("last", False):
        ckpt = torch.load(config["last_path"], map_location=device)
        hiding_net.load_state_dict(ckpt["hiding_net"], strict=True)
    else:
        hiding_net.load_state_dict(
            torch.load(config["hiding_net_path"], map_location=device))

    hiding_net.eval()
    return hiding_net


def main():
    parser = argparse.ArgumentParser(description="Analyse watermark frequency content")
    parser.add_argument("--config", default="./configs/eval_config.yaml",
                        help="Path to the evaluation config YAML")
    parser.add_argument("--bands", type=float, nargs="+", default=None,
                        help="Band edges on the normalised radius; N+1 values give N bands")
    args = parser.parse_args()

    set_seed(42)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hiding_net = load_hiding_net(config, device)
    use_budget = bool(config.get("use_budget", True))

    datasets_to_test = config.get("test_datasets") or {}
    if not datasets_to_test:
        print("[!] No test datasets configured (test_datasets is empty). Nothing to analyse.")
        return

    transform = get_adaptive_transforms(256)

    for dataset_name, base_dir in datasets_to_test.items():
        print(f"\n{'=' * 50}\nSpectral analysis on: {dataset_name.upper()}\n{'=' * 50}")

        dataset = create_tamper_dataset(
            dataset_name=dataset_name,
            base_dir=base_dir,
            transform=transform,
            limit=config.get("limit", None),
            tamper_filter=config.get("tamper_filter"),
        )
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["num_workers"], pin_memory=True)

        shares_per_rf = {}
        num_bands = None

        for rf in RF_VALUES:
            totals, count = None, 0

            with torch.no_grad():
                for batch in tqdm(loader, desc=f"RF={rf:.2f}"):
                    cover_img = batch[0].to(device)
                    B, _, H, W = cover_img.shape

                    secret = generate_qr_payloads(B, H, W, block_size=8, device=device)
                    rf_tensor = torch.full((B, 1), rf, device=device)

                    watermark = hiding_net(secret, rf_tensor)
                    if use_budget:
                        watermark = enforce_watermark_budget(watermark, rf_tensor)

                    # Measure the watermark that actually survives clipping.
                    container = (cover_img + watermark).clamp(0.0, 1.0)
                    shares = radial_band_energy(container - cover_img, args.bands)

                    totals = shares.sum(dim=0) if totals is None else totals + shares.sum(dim=0)
                    count += B

            mean_shares = (totals / max(count, 1)).cpu().tolist()
            shares_per_rf[rf] = mean_shares
            num_bands = len(mean_shares)

        table = spectral_table(shares_per_rf, band_labels(num_bands))
        save_spectral_table(table, dataset_name=dataset_name)
        plot_spectral_distribution(table, dataset_name=dataset_name)


if __name__ == "__main__":
    main()
