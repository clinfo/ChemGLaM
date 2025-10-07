#!/usr/bin/env python3
"""
Create prediction config from best training config.

Usage:
    python create_predict_config.py <best_config.json> [--split valid|test]

Example:
    python create_predict_config.py logs/optuna/davis_cv0/best_config.json
    python create_predict_config.py logs/optuna/davis_cv0/best_config.json --split valid
"""

import json
import sys
import os
import argparse
from pathlib import Path


def create_predict_config(train_config_path, split='test'):
    """Create prediction config from training config.

    Args:
        train_config_path: Path to the training config file
        split: Which split to use ('test' or 'valid')
    """

    # Load training config
    with open(train_config_path, 'r') as f:
        config = json.load(f)

    # Extract experiment info (e.g., "davis_cv0" -> cv0)
    exp_name = config["experiment_name"]

    # Modify for prediction
    config["experiment_name"] = f"{exp_name}_optuna_{split}"
    config["cache_dir"] = f"{exp_name}_optuna_{split}"

    # Change dataset path from train_valid.csv to test.csv or valid.csv
    dataset_path = config["dataset_csv_path"]
    config["dataset_csv_path"] = dataset_path.replace("train_valid.csv", f"{split}.csv")

    # Prediction-specific settings
    config["deterministic_eval"] = True
    config["split_json_path"] = None
    config["batch_size"] = 16  # Fixed for memory constraints

    # Set checkpoint path to best_checkpoint.ckpt in the same directory
    input_path = Path(train_config_path)
    checkpoint_path = input_path.parent / "best_checkpoint.ckpt"
    config["checkpoint_path"] = str(checkpoint_path)

    # Save to same directory as input with _predict suffix
    output_path = input_path.parent / f"best_config_{split}_predict.json"

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Created prediction config: {output_path}")
    print(f"\nKey changes:")
    print(f"  experiment_name: {exp_name} -> {config['experiment_name']}")
    print(f"  dataset_csv_path: {dataset_path} -> {config['dataset_csv_path']}")
    print(f"  batch_size: -> 16 (fixed)")
    print(f"  deterministic_eval: False -> True")
    print(f"  split_json_path: ... -> null")
    print(f"  checkpoint_path: null -> {config['checkpoint_path']}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create prediction config from training config")
    parser.add_argument("config_path", help="Path to best_config.json")
    parser.add_argument("--split", choices=['test', 'valid'], default='test',
                        help="Which split to use (default: test)")

    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        print(f"Error: File not found: {args.config_path}")
        sys.exit(1)

    create_predict_config(args.config_path, args.split)
