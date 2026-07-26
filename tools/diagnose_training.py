"""Report AMP dtypes and loss-term magnitudes for one training step

Answers two questions that cannot be read off the source:

1. Does the spectral loss run in float16 under autocast? The watermark has an RMS of
   0.005 to 0.0225, so its power spectrum lands near the float16 subnormal boundary
   of 6.1e-5. If the FFT is not promoted to float32 the spectral penalty is noise.
2. How do the three loss terms compare once their lambdas are applied? The terms have
   very different natural scales, so the configured weights may not balance them.

Run on the training machine; the dtype question needs CUDA.

Usage:
    python tools/diagnose_training.py
    python tools/diagnose_training.py --batch 8 --size 256
"""

import argparse
import sys

import torch
from kornia.metrics import ssim

from adamark.attacks import AttackModule
from adamark.losses import FFTLoss, HidingLoss, RevealingLoss, enforce_watermark_budget
from adamark.models import HidingNet, RevealingNet
from adamark.utils import generate_qr_payloads, set_seed

FP16_MIN_NORMAL = 6.104e-5


def describe(name, t):
    """Print dtype and dynamic range of a tensor"""

    finite = t[torch.isfinite(t)]
    lo = finite.abs().min().item() if finite.numel() else float("nan")
    hi = finite.abs().max().item() if finite.numel() else float("nan")
    print(f"  {name:34s} {str(t.dtype):15s} |x| in [{lo:.3e}, {hi:.3e}]")


def check_spectral_dtype(effective_watermark):
    """Recompute the FFTLoss internals to expose their dtype and magnitudes"""

    print("\n--- 1. spectral loss numerics ---")
    describe("effective watermark", effective_watermark)

    fft = torch.fft.fft2(effective_watermark, norm="ortho")
    power = fft.real ** 2 + fft.imag ** 2
    describe("fft2 output", fft)
    describe("power spectrum", power)

    subnormal = (power > 0) & (power < FP16_MIN_NORMAL)
    frac = subnormal.float().mean().item()
    zeros = (power == 0).float().mean().item()

    print(f"  power below fp16 min normal:       {100 * frac:.1f}%")
    print(f"  power exactly zero:                {100 * zeros:.1f}%")

    if power.dtype == torch.float32:
        print("  VERDICT: promoted to float32, the spectral penalty is safe")
        return True

    print("  VERDICT: running in float16. Values under 6.1e-5 lose precision or")
    print("           collapse to zero, so the spectral penalty is unreliable.")
    print("           Wrap the FFT in torch.autocast(enabled=False) and cast to float().")
    return False


def report_loss_terms(device, hiding_net, revealing_net, attack_module, args):
    """Print each loss term and its lambda-weighted contribution across rf"""

    print("\n--- 2. loss term magnitudes ---")
    hiding_criterion = HidingLoss()
    revealing_criterion = RevealingLoss()
    fft_loss_fn = FFTLoss()

    lambda_r, lambda_fft = args.lambda_r, args.lambda_fft
    print(f"  weights: lambda_h=1.0 (fixed in HidingLoss), "
          f"lambda_r={lambda_r}, lambda_fft={lambda_fft}\n")
    header = (f"  {'rf':>5} | {'L_h':>9} {'L_r':>9} {'L_fft':>9} | "
              f"{'1*L_h':>9} {'lr*L_r':>9} {'lf*L_fft':>9} | {'L_h share':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for rf_value in (0.0, 0.25, 0.5, 1.0):
        set_seed(42)
        cover = torch.rand(args.batch, 3, args.size, args.size, device=device)
        secret = generate_qr_payloads(args.batch, args.size, args.size,
                                      block_size=8, device=device)
        rf = torch.full((args.batch, 1), rf_value, device=device)

        with torch.no_grad(), torch.autocast(device_type=device.type):
            watermark = enforce_watermark_budget(hiding_net(secret, rf), rf)
            container = (cover + watermark).clamp(0.0, 1.0)
            effective = container - cover

            tampered, mask = attack_module(container=container, cover=cover, rf=rf)
            retrieved = revealing_net(tampered)

            l_h = hiding_criterion(container, cover, ssim_func=ssim).item()
            l_r = revealing_criterion(retrieved, secret, mask,
                                      lambda_r_base=1.0).item()
            l_fft = fft_loss_fn(effective, rf).item()

        w_h, w_r, w_fft = l_h, lambda_r * l_r, lambda_fft * l_fft
        total = w_h + w_r + w_fft
        share = 100 * w_h / total if total else float("nan")

        print(f"  {rf_value:5.2f} | {l_h:9.5f} {l_r:9.5f} {l_fft:9.5f} | "
              f"{w_h:9.5f} {w_r:9.5f} {w_fft:9.5f} | {share:9.1f}%")

    print("\n  L_h share is the fraction of the objective spent on imperceptibility.")
    print("  A few percent means visual quality is barely being optimised.")


def report_raw_amplitude(device, hiding_net, args):
    """Print raw tanh statistics, which the budget hides from the gradient"""

    print("\n--- 3. raw watermark amplitude (unconstrained by design) ---")
    print("  enforce_watermark_budget detaches the RMS, so the network gets no")
    print("  gradient about its own raw amplitude. Saturation here kills gradients.\n")

    for rf_value in (0.0, 0.5, 1.0):
        set_seed(42)
        secret = generate_qr_payloads(args.batch, args.size, args.size,
                                      block_size=8, device=device)
        rf = torch.full((args.batch, 1), rf_value, device=device)

        with torch.no_grad(), torch.autocast(device_type=device.type):
            raw = hiding_net(secret, rf).float()

        saturated = (raw.abs() > 0.99).float().mean().item()
        print(f"  rf={rf_value:.2f}  |raw| mean {raw.abs().mean():.4f}  "
              f"rms {raw.pow(2).mean().sqrt():.4f}  saturated {100 * saturated:.2f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lambda-r", type=float, default=3.0, dest="lambda_r")
    ap.add_argument("--lambda-fft", type=float, default=3.0, dest="lambda_fft")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  torch: {torch.__version__}")
    if device.type != "cuda":
        print("[!] No CUDA. autocast on CPU uses bfloat16, so the float16 question")
        print("    below is not answered by this run.")

    set_seed(42)
    hiding_net = HidingNet().to(device).eval()
    revealing_net = RevealingNet().to(device).eval()
    attack_module = AttackModule().to(device)

    cover = torch.rand(args.batch, 3, args.size, args.size, device=device)
    secret = generate_qr_payloads(args.batch, args.size, args.size,
                                  block_size=8, device=device)
    rf = torch.full((args.batch, 1), 0.0, device=device)

    with torch.no_grad(), torch.autocast(device_type=device.type):
        watermark = enforce_watermark_budget(hiding_net(secret, rf), rf)
        container = (cover + watermark).clamp(0.0, 1.0)
        effective = container - cover

        print("\n--- dtypes inside the autocast region ---")
        describe("hiding net output", watermark)
        describe("container", container)

        spectral_ok = check_spectral_dtype(effective)

    report_loss_terms(device, hiding_net, revealing_net, attack_module, args)
    report_raw_amplitude(device, hiding_net, args)

    return 0 if spectral_ok else 1


if __name__ == "__main__":
    sys.exit(main())
