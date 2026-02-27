"""Metrics for evaluating repair quality."""

from typing import Dict, List, Optional
import difflib
from loguru import logger


def compute_metrics(
    predictions: List[str],
    ground_truths: List[str],
    exact_match: bool = True,
) -> Dict:
    """
    Compute evaluation metrics for repair results.

    Args:
        predictions: List of predicted patches
        ground_truths: List of ground truth patches
        exact_match: Whether to use exact match or fuzzy match

    Returns:
        Dictionary of metrics
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    correct = 0
    partial_matches = 0
    total = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        if exact_match:
            if pred.strip() == gt.strip():
                correct += 1
        else:
            # Fuzzy matching using sequence matcher
            similarity = difflib.SequenceMatcher(None, pred, gt).ratio()
            if similarity >= 0.9:
                correct += 1
            elif similarity >= 0.5:
                partial_matches += 1

    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "partial_matches": partial_matches,
    }

    return metrics


def evaluate_repair(
    repair_result: Dict,
    ground_truth_patch: Optional[str] = None,
) -> Dict:
    """
    Evaluate a single repair result.

    Args:
        repair_result: Repair result from RepairAgent
        ground_truth_patch: Optional ground truth patch for comparison

    Returns:
        Evaluation metrics
    """
    evaluation = {
        "success": repair_result.get("success", False),
        "iterations": repair_result.get("iterations", 0),
        "plausible": False,  # Passes tests
        "correct": False,  # Matches ground truth
    }

    # Check if plausible (passes tests)
    if repair_result.get("success"):
        evaluation["plausible"] = True

    # Check if correct (matches ground truth)
    if ground_truth_patch and repair_result.get("final_patch"):
        similarity = difflib.SequenceMatcher(
            None,
            repair_result["final_patch"].strip(),
            ground_truth_patch.strip(),
        ).ratio()

        evaluation["similarity"] = similarity
        if similarity >= 0.9:
            evaluation["correct"] = True

    return evaluation


def compute_precision_recall(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> Dict:
    """
    Compute precision and recall.

    Args:
        true_positives: Number of true positives
        false_positives: Number of false positives
        false_negatives: Number of false negatives

    Returns:
        Dictionary with precision, recall, and F1 score
    """
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def compute_ranking_metrics(
    relevance_scores: List[float],
    k: int = 3,
) -> Dict:
    """
    Compute ranking metrics (MRR, Precision@K, etc.).

    Args:
        relevance_scores: List of relevance scores (1 for relevant, 0 for not)
        k: Top-k for precision

    Returns:
        Dictionary of ranking metrics
    """
    # Mean Reciprocal Rank
    mrr = 0.0
    for i, score in enumerate(relevance_scores, 1):
        if score > 0:
            mrr = 1.0 / i
            break

    # Precision@K
    top_k = relevance_scores[:k]
    precision_at_k = sum(top_k) / k if k > 0 else 0.0

    # Average Precision
    precisions = []
    relevant_count = 0
    for i, score in enumerate(relevance_scores, 1):
        if score > 0:
            relevant_count += 1
            precisions.append(relevant_count / i)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0

    return {
        "mrr": mrr,
        f"precision@{k}": precision_at_k,
        "average_precision": avg_precision,
    }
