"""Entry point for evaluation.

Usage:
    python evaluate.py --config configs/eval_config.yaml
"""

import argparse

import yaml

from adamark.evaluation import run_evaluation
from adamark.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate the adaptive watermarking model.")
    parser.add_argument("--config", default="./configs/eval_config.yaml",
                        help="Path to the evaluation config YAML.")
    args = parser.parse_args()

    set_seed(42)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_evaluation(config)


if __name__ == "__main__":
    main()
