"""
Active learning strategies for efficient labeling.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Set
import random

from app.core.linkage.blocking import create_candidate_pairs
from app.core.linkage.comparators import compare_records


def uncertainty_sampling(
    probabilities: List[float],
    indices: List[Tuple[int, int]]
) -> Optional[Tuple[int, int]]:
    """
    Select the pair with highest uncertainty (closest to 0.5 probability).

    Args:
        probabilities: Match probabilities for each pair
        indices: List of (left_idx, right_idx) tuples

    Returns:
        The most uncertain (left_idx, right_idx) pair or None
    """
    if not probabilities or not indices:
        return None

    # Calculate uncertainty as distance from 0.5
    uncertainties = [abs(p - 0.5) for p in probabilities]

    # Find minimum uncertainty (most uncertain)
    min_idx = np.argmin(uncertainties)

    return indices[min_idx]


def random_sampling(
    indices: List[Tuple[int, int]]
) -> Optional[Tuple[int, int]]:
    """
    Randomly select a pair.

    Args:
        indices: List of (left_idx, right_idx) tuples

    Returns:
        A random (left_idx, right_idx) pair or None
    """
    if not indices:
        return None

    return random.choice(indices)


def diversity_sampling(
    comparison_vectors: List[Dict[str, float]],
    indices: List[Tuple[int, int]],
    n_select: int = 1
) -> List[Tuple[int, int]]:
    """
    Select diverse pairs based on comparison vectors.

    Args:
        comparison_vectors: Comparison vectors for each pair
        indices: List of (left_idx, right_idx) tuples
        n_select: Number of pairs to select

    Returns:
        List of selected pairs
    """
    if not comparison_vectors or not indices:
        return []

    # Convert to numpy array
    X = np.array([list(cv.values()) for cv in comparison_vectors])

    if len(X) <= n_select:
        return list(indices)

    # Use k-means style selection for diversity
    selected_indices = []
    remaining = list(range(len(X)))

    # Start with random point
    first_idx = random.choice(remaining)
    selected_indices.append(first_idx)
    remaining.remove(first_idx)

    # Greedily select points furthest from already selected
    while len(selected_indices) < n_select and remaining:
        max_min_dist = -1
        best_idx = remaining[0]

        for idx in remaining:
            # Distance to nearest selected point
            min_dist = min(
                np.linalg.norm(X[idx] - X[sel_idx])
                for sel_idx in selected_indices
            )
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = idx

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [indices[i] for i in selected_indices]


def select_informative_pair(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    column_mappings: Dict[str, str],
    comparison_config: Dict[str, Any],
    blocking_config: Dict[str, Any],
    labeled_pairs: Set[Tuple[int, int]],
    model: Any,
    strategy: str = "uncertainty",
    is_dedup: bool = False,
    sample_size: int = 1000
) -> Optional[Tuple[int, int, Dict[str, float]]]:
    """
    Select the most informative pair for labeling.

    Args:
        source_df: Source DataFrame
        target_df: Target DataFrame
        column_mappings: Column mappings configuration
        comparison_config: Comparison methods configuration
        blocking_config: Blocking configuration
        labeled_pairs: Set of already labeled (left_idx, right_idx) pairs
        model: Trained model for uncertainty estimation (optional)
        strategy: "uncertainty" or "random"
        is_dedup: Whether this is deduplication
        sample_size: Max pairs to evaluate

    Returns:
        Tuple of (left_idx, right_idx, comparison_vector) or None
    """
    # Generate candidate pairs
    all_pairs = create_candidate_pairs(
        source_df, target_df, blocking_config, is_dedup
    )

    # Filter out already labeled pairs
    unlabeled_pairs = [p for p in all_pairs if p not in labeled_pairs]

    if not unlabeled_pairs:
        return None

    # Sample if too many pairs
    if len(unlabeled_pairs) > sample_size:
        unlabeled_pairs = random.sample(unlabeled_pairs, sample_size)

    # Compute comparison vectors
    comparison_vectors = []
    valid_pairs = []

    for left_idx, right_idx in unlabeled_pairs:
        try:
            left_record = source_df.iloc[left_idx].to_dict()
            right_record = target_df.iloc[right_idx].to_dict()

            vector = compare_records(
                left_record, right_record,
                column_mappings, comparison_config
            )
            comparison_vectors.append(vector)
            valid_pairs.append((left_idx, right_idx))
        except Exception:
            continue

    if not valid_pairs:
        return None

    # Select based on strategy
    if strategy == "random" or model is None:
        idx = random.randrange(len(valid_pairs))
        return (*valid_pairs[idx], comparison_vectors[idx])

    elif strategy == "uncertainty":
        # Get model predictions
        try:
            X = [list(cv.values()) for cv in comparison_vectors]
            probabilities = model.predict_proba(X)

            # Find most uncertain
            uncertainties = [abs(p - 0.5) for p in probabilities]
            min_idx = np.argmin(uncertainties)

            return (*valid_pairs[min_idx], comparison_vectors[min_idx])
        except Exception:
            # Fall back to random if model fails
            idx = random.randrange(len(valid_pairs))
            return (*valid_pairs[idx], comparison_vectors[idx])

    else:
        # Default to random
        idx = random.randrange(len(valid_pairs))
        return (*valid_pairs[idx], comparison_vectors[idx])


def calculate_model_uncertainty(model: Any, X: List[List[float]]) -> List[float]:
    """
    Calculate prediction uncertainty for each sample.

    Returns:
        List of uncertainty scores (0 = certain, 0.5 = most uncertain)
    """
    try:
        probabilities = model.predict_proba(X)
        return [abs(p - 0.5) for p in probabilities]
    except Exception:
        return [0.5] * len(X)  # Maximum uncertainty as fallback


def should_retrain(
    total_labeled: int,
    last_retrain_count: int,
    retrain_interval: int = 20
) -> bool:
    """
    Determine if the model should be retrained.

    Args:
        total_labeled: Total number of labeled pairs
        last_retrain_count: Number of labels when last retrained
        retrain_interval: Number of new labels before retraining

    Returns:
        True if should retrain
    """
    return (total_labeled - last_retrain_count) >= retrain_interval
