"""Radial frequency analysis of the generated watermark

Splits the watermark's 2D power spectrum into concentric frequency bands and
reports the share of total energy per band. The radial coordinate matches the one
used by ``FFTLoss``, so the numbers here are directly comparable to the spectral
penalty that shapes them during training.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

BAND_LABELS = ["very low", "low", "mid-low", "mid", "mid-high", "high"]

# Six equal-width annuli over the normalised radius; the last band is open-ended
# so the spectrum corners beyond radius 1 still contribute their energy.
DEFAULT_BAND_EDGES = [0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, float("inf")]


def radial_band_energy(watermark, band_edges=None):
    """Return the per-band share of spectral energy, shaped B x num_bands"""

    edges = list(DEFAULT_BAND_EDGES if band_edges is None else band_edges)
    w = watermark.float()

    fft = torch.fft.fft2(w, norm="ortho")
    power = fft.real ** 2 + fft.imag ** 2

    fy = torch.fft.fftfreq(w.shape[2], device=w.device)
    fx = torch.fft.fftfreq(w.shape[3], device=w.device)
    Y, X = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(X ** 2 + Y ** 2) * 2.0

    total = power.sum(dim=(1, 2, 3)) + 1e-12

    shares = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = ((radius >= lo) & (radius < hi)).to(power.dtype)
        shares.append(power.mul(band).sum(dim=(1, 2, 3)) / total)

    return torch.stack(shares, dim=1)


def band_labels(num_bands):
    """Human-readable labels for ``num_bands`` bands, ordered low to high"""

    if num_bands == len(BAND_LABELS):
        return list(BAND_LABELS)
    return [f"band {i + 1}" for i in range(num_bands)]


def plot_spectral_distribution(table, dataset_name, output_dir="results"):
    """Grouped bar chart of per-band energy share for every rf value"""

    os.makedirs(output_dir, exist_ok=True)
    labels = list(table.columns)
    rf_values = list(table.index)

    x = np.arange(len(labels))
    width = 0.8 / max(len(rf_values), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, rf in enumerate(rf_values):
        ax.bar(x + i * width - 0.4 + width / 2, table.loc[rf].values * 100.0,
               width=width, label=f"RF={rf}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Frequency band")
    ax.set_ylabel("Share of watermark energy (%)")
    ax.set_title(f"[{dataset_name}] Watermark energy across frequency bands")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_spectral_distribution.png")
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[*] Spectral plot saved to: {path}")


def save_spectral_table(table, dataset_name, output_dir="results"):
    """Write the per-band energy shares to an .xlsx file"""

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"spectral_{dataset_name}.xlsx")
    table.mul(100.0).round(2).to_excel(path, index_label="RF")
    print(f"[*] Spectral table saved to: {path}")


def spectral_table(shares_per_rf, labels):
    """Build a DataFrame of mean per-band shares indexed by rf"""

    return pd.DataFrame(
        {rf: vals for rf, vals in shares_per_rf.items()},
        index=labels,
    ).T
