"""Dump and compare forward passes across two AdaMark revisions

The cleanup refactor must be numerically inert. This script dumps every network
output, loss scalar and attack output for a fixed seed, so two revisions can be
compared bit-for-bit. It lives outside the package and selects the code under test
via ``--src``, so the same script runs against both revisions.

Signatures differ between revisions because the refactor drops unused parameters,
so arguments are bound through ``inspect.signature`` rather than positionally.

Usage:
    git worktree add /tmp/adamark-base <baseline-commit>
    python tools/equivalence_dump.py --src /tmp/adamark-base/src --out /tmp/before.pt
    python tools/equivalence_dump.py --src ./src --out /tmp/after.pt
    python tools/equivalence_dump.py --compare /tmp/before.pt /tmp/after.pt

Note: ``ImageDataset`` is deliberately excluded. The refactor removes its
``random.seed(42)`` side effect on the global RNG, which is a known and intended
difference; every stochastic op below is re-seeded explicitly so it stays isolated.
"""

import argparse
import inspect
import os
import random
import sys

SEED = 1234
B, C, H, W = 2, 3, 64, 64


def bind(fn, **candidates):
    """Keep only the keyword arguments this revision's signature actually accepts"""

    params = inspect.signature(fn).parameters
    return {k: v for k, v in candidates.items() if k in params}


def seed_all(torch):
    random.seed(SEED)
    torch.manual_seed(SEED)


def build_dump(src_path):
    src_path = os.path.abspath(src_path)
    sys.path.insert(0, src_path)
    for mod in [m for m in sys.modules if m == "adamark" or m.startswith("adamark.")]:
        del sys.modules[mod]

    import torch
    from kornia.metrics import ssim

    import adamark
    resolved = os.path.abspath(os.path.dirname(os.path.dirname(adamark.__file__)))
    assert resolved == src_path, f"imported adamark from {resolved}, expected {src_path}"

    from adamark.attacks import AttackModule, Distortion, JpegDistortion, Tampering
    from adamark.attacks.distortion import distortion_deterministic_oneof
    from adamark.losses import (FFTLoss, HidingLoss, RevealingLoss,
                                enforce_watermark_budget)
    from adamark.models import HidingNet, RevealingNet
    from adamark.utils import generate_qr_payloads

    device = torch.device("cpu")
    out = {}

    # ---- deterministic inputs ----
    seed_all(torch)
    cover = torch.rand(B, C, H, W)
    rf = torch.tensor([[0.0], [0.7]])
    out["input/cover"] = cover
    out["input/rf"] = rf

    seed_all(torch)
    secret = generate_qr_payloads(B, H, W, block_size=8, device=device)
    out["payload/secret"] = secret

    # ---- HidingNet: construction must consume RNG identically ----
    seed_all(torch)
    hiding_net = HidingNet(**bind(HidingNet.__init__, film_scale=1.0))
    hiding_net.eval()
    for name, p in sorted(hiding_net.state_dict().items()):
        out[f"hiding_net/param/{name}"] = p.clone()

    seed_all(torch)
    revealing_net = RevealingNet()
    revealing_net.eval()
    for name, p in sorted(revealing_net.state_dict().items()):
        out[f"revealing_net/param/{name}"] = p.clone()

    # ---- forward passes ----
    with torch.no_grad():
        seed_all(torch)
        watermark_raw = hiding_net(secret, rf)
        out["hiding_net/raw"] = watermark_raw

        seed_all(torch)
        watermark = enforce_watermark_budget(watermark_raw, rf)
        out["budget/watermark"] = watermark
        out["budget/rms"] = watermark.pow(2).mean(dim=(1, 2, 3)).sqrt()

        container = (cover + watermark).clamp(0.0, 1.0)
        out["container"] = container

        seed_all(torch)
        retrieved = revealing_net(container)
        out["revealing_net/retrieved"] = retrieved

    # ---- losses ----
    hiding_criterion = HidingLoss()
    revealing_criterion = RevealingLoss()
    fft_loss_fn = FFTLoss()
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, 16:40, 16:40] = 1.0

    with torch.no_grad():
        seed_all(torch)
        out["loss/hiding"] = hiding_criterion(
            container, cover,
            **bind(hiding_criterion.forward, rf_vec=rf, lambda_h_base=1.0, ssim_func=ssim))

        seed_all(torch)
        out["loss/revealing"] = revealing_criterion(
            retrieved, secret, mask,
            **bind(revealing_criterion.forward, rf_vec=rf, lambda_r_base=3.0))

        seed_all(torch)
        out["loss/fft"] = fft_loss_fn(container - cover, rf)

    # ---- attacks ----
    with torch.no_grad():
        for force_max in (False, True):
            seed_all(torch)
            out[f"distortion/force_max={force_max}"] = Distortion()(container, rf,
                                                                    force_max=force_max)

        for mode in ("jpeg", "blur"):
            seed_all(torch)
            out[f"deterministic/{mode}"] = distortion_deterministic_oneof(
                container, rf_attack=1.0, mode=mode)

        for q in (100, 50):
            jd = JpegDistortion(jpeg_quality=q)
            seed_all(torch)
            out[f"jpeg_eval/q={q}"] = jd(container, **bind(jd.forward, rf=None))

        seed_all(torch)
        t_img, t_mask = Tampering()(container, cover)
        out["tampering/img"], out["tampering/mask"] = t_img, t_mask

        seed_all(torch)
        a_img, a_mask = AttackModule()(container=container, cover=cover, rf=rf)
        out["attack_module/img"], out["attack_module/mask"] = a_img, a_mask

    sys.path.remove(src_path)
    return out


def compare(path_a, path_b):
    import torch

    a = torch.load(path_a, weights_only=False)
    b = torch.load(path_b, weights_only=False)

    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    for k in only_a:
        print(f"  MISSING in B: {k}")
    for k in only_b:
        print(f"  MISSING in A: {k}")

    mismatched, checked = [], 0
    for k in sorted(set(a) & set(b)):
        ta, tb = a[k], b[k]
        checked += 1
        if ta.shape != tb.shape:
            mismatched.append((k, f"shape {tuple(ta.shape)} vs {tuple(tb.shape)}"))
            continue
        if not torch.equal(ta, tb):
            diff = (ta.float() - tb.float()).abs().max().item()
            mismatched.append((k, f"max abs diff {diff:.3e}"))

    print(f"\nchecked {checked} tensors")
    if mismatched or only_a or only_b:
        print(f"MISMATCH in {len(mismatched)} tensor(s):")
        for k, why in mismatched:
            print(f"  - {k}: {why}")
        return 1

    print("ALL BIT-IDENTICAL")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", help="Path to the src/ directory of the revision under test")
    ap.add_argument("--out", help="Where to write the dump")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Compare two dumps")
    args = ap.parse_args()

    if args.compare:
        sys.exit(compare(*args.compare))

    if not args.src or not args.out:
        ap.error("--src and --out are both required unless --compare is used")

    import torch

    dump = build_dump(args.src)
    torch.save(dump, args.out)
    print(f"wrote {len(dump)} tensors from {args.src} to {args.out}")


if __name__ == "__main__":
    main()
