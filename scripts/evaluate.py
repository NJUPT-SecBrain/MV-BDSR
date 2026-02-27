#!/usr/bin/env python3
"""Script to evaluate repair results on test set."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from loguru import logger
from tqdm import tqdm

from utils.helpers import load_yaml, save_json
from utils.logger import setup_logger
from utils.metrics import evaluate_repair, compute_metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="MV-BDSR Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        required=True,
        help="Path to repair results JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for evaluation metrics",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_yaml(args.config)
    
    # Setup logging
    setup_logger(level=config["logging"]["level"])

    logger.info("Starting evaluation")
    logger.info(f"Results file: {args.results_file}")

    # Load results
    from utils.helpers import load_json
    results = load_json(args.results_file)

    logger.info(f"Loaded {len(results)} repair results")

    # Evaluate each result
    evaluations = []
    for result in tqdm(results, desc="Evaluating"):
        eval_result = evaluate_repair(
            result["repair_result"],
            result.get("ground_truth_patch"),
        )
        evaluations.append(eval_result)

    # Compute aggregate metrics
    total = len(evaluations)
    plausible = sum(1 for e in evaluations if e["plausible"])
    correct = sum(1 for e in evaluations if e["correct"])

    aggregate_metrics = {
        "total_samples": total,
        "plausible_patches": plausible,
        "correct_patches": correct,
        "plausibility_rate": plausible / total if total > 0 else 0.0,
        "correctness_rate": correct / total if total > 0 else 0.0,
        "average_iterations": sum(e["iterations"] for e in evaluations) / total if total > 0 else 0.0,
    }

    # Save results
    output_data = {
        "aggregate_metrics": aggregate_metrics,
        "per_sample_evaluations": evaluations,
    }

    save_json(output_data, args.output)
    
    logger.info("=== Evaluation Results ===")
    logger.info(f"Total samples: {aggregate_metrics['total_samples']}")
    logger.info(f"Plausible patches: {aggregate_metrics['plausible_patches']} ({aggregate_metrics['plausibility_rate']:.2%})")
    logger.info(f"Correct patches: {aggregate_metrics['correct_patches']} ({aggregate_metrics['correctness_rate']:.2%})")
    logger.info(f"Average iterations: {aggregate_metrics['average_iterations']:.2f}")
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
