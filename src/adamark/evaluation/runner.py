"""End-to-end evaluation: load models, build attacks, sweep rf and report metrics"""

import os

import torch
from torch.utils.data import DataLoader

from adamark.data.dataset import create_tamper_dataset
from adamark.data.transforms import get_adaptive_transforms
from adamark.models import HidingNet, RevealingNet
from adamark.evaluation.attack_suite import build_eval_attacks
from adamark.evaluation.evaluator import evaluate_model_single_pass
from adamark.evaluation.reporting import plot_all_results, save_results_to_excel

RF_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]


def _load_models(config, device):
    hiding_net = HidingNet().to(device)
    revealing_net = RevealingNet().to(device)

    if config.get("last", False):
        ckpt = torch.load(config["last_path"], map_location=device)
        hiding_net.load_state_dict(ckpt["hiding_net"], strict=True)
        revealing_net.load_state_dict(ckpt["revealing_net"], strict=True)
    else:
        hiding_net.load_state_dict(torch.load(config["hiding_net_path"], map_location=device))
        revealing_net.load_state_dict(torch.load(config["revealing_net_path"], map_location=device))

    return hiding_net, revealing_net


def run_evaluation(config):
    os.makedirs("results", exist_ok=True)

    verbose = config.get("verbose", False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hiding_net, revealing_net = _load_models(config, device)

    adaptive_transform = get_adaptive_transforms(256)
    datasets_to_test = config.get("test_datasets") or {}

    if not datasets_to_test:
        print("[!] No test datasets configured (test_datasets is empty). Nothing to evaluate.")
        return

    for dataset_name, base_dir in datasets_to_test.items():
        print(f"\n{'=' * 50}\nEvaluating on dataset: {dataset_name.upper()}\n{'=' * 50}")

        dataset = create_tamper_dataset(
            dataset_name=dataset_name,
            base_dir=base_dir,
            transform=adaptive_transform,
            limit=config.get("limit", None),
            tamper_filter=None,  # e.g. "splicing" / "copymove"
        )

        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
            persistent_workers=config["num_workers"] > 0,
            prefetch_factor=4,
        )

        attacks_dict = build_eval_attacks(device)

        results_rf = []
        for rf in RF_VALUES:
            print(f"--- Evaluating adaptive model | RF={rf} ---")
            res = evaluate_model_single_pass(
                hiding_net, revealing_net, loader, device, attacks_dict,
                rf_value=rf, verbose=verbose, tamper_threshold=0.5)
            res["rf"] = rf
            results_rf.append(res)

        save_results_to_excel(results_rf, dataset_name=dataset_name)
        plot_all_results(results=results_rf, dataset_name=dataset_name)
