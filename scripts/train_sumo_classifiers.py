#!/usr/bin/env python3
"""CLI wrapper for SUMO hierarchical classifier training."""

import argparse

from src.training.sumo_classifiers import train_sumo_classifiers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SUMO classifiers for hierarchical concepts")
    parser.add_argument('--layers', nargs='+', type=int, default=[0, 1, 2],
                        help='Which layers to train (default: 0 1 2)')
    parser.add_argument('--model', default="google/gemma-3-4b-pt",
                        help='Model name (default: gemma-3-4b-pt)')
    parser.add_argument('--device', default="cuda",
                        help='Device (default: cuda)')
    parser.add_argument('--n-train-pos', type=int, default=10,
                        help='Positive training samples per concept (default: 10)')
    parser.add_argument('--n-train-neg', type=int, default=10,
                        help='Negative training samples per concept (default: 10)')
    parser.add_argument('--n-test-pos', type=int, default=20,
                        help='Positive test samples per concept (default: 20)')
    parser.add_argument('--n-test-neg', type=int, default=20,
                        help='Negative test samples per concept (default: 20)')
    parser.add_argument('--output-dir', type=str, default="results/sumo_classifiers",
                        help='Output directory (default: results/sumo_classifiers)')
    parser.add_argument('--train-text-probes', action='store_true',
                        help='Train text probes alongside activation probes')
    parser.add_argument('--use-adaptive-training', action='store_true',
                        help='Use DualAdaptiveTrainer with independent graduation for activation and text probes')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_sumo_classifiers(
        layers=args.layers,
        model_name=args.model,
        device=args.device,
        n_train_pos=args.n_train_pos,
        n_train_neg=args.n_train_neg,
        n_test_pos=args.n_test_pos,
        n_test_neg=args.n_test_neg,
        output_dir=args.output_dir,
        train_text_probes=args.train_text_probes,
        use_adaptive_training=args.use_adaptive_training,
    )


if __name__ == '__main__':
    main()
