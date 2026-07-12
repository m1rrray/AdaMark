"""Entry point for distributed training.

Usage:
    torchrun --nproc_per_node=<N> train.py --config configs/train_config.yaml
"""

import argparse
import logging

import yaml

from adamark.training import train


def main():
    parser = argparse.ArgumentParser(description="Train the adaptive watermarking model.")
    parser.add_argument("--config", default="./configs/train_config.yaml",
                        help="Path to the training config YAML.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
