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


class PairSelectionResult:
    """Result of pair selection with explanation."""
    def __init__(
        self,
        left_idx: int,
        right_idx: int,
        comparison_vector: Dict[str, float],
        selection_reason: str,
        uncertainty_score: Optional[float] = None,
        model_probability: Optional[float] = None,
        candidates_evaluated: int = 0
    ):
        self.left_idx = left_idx
        self.right_idx = right_idx
        self.comparison_vector = comparison_vector
        self.selection_reason = selection_reason
        self.uncertainty_score = uncertainty_score
        self.model_probability = model_probability
        self.candidates_evaluated = candidates_evaluated


def count_candidate_pairs(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    blocking_config: Dict[str, Any],
    labeled_pairs: Set[Tuple[int, int]],
    is_dedup: bool = False
) -> Tuple[int, int]:
    """
    Count total and remaining candidate pairs.

    Args:
        source_df: Source DataFrame
        target_df: Target DataFrame
        blocking_config: Blocking configuration
        labeled_pairs: Set of already labeled pairs
        is_dedup: Whether this is deduplication

    Returns:
        Tuple of (total_candidates, remaining_candidates)
    """
    all_pairs = create_candidate_pairs(
        source_df, target_df, blocking_config, is_dedup
    )
    total = len(all_pairs)
    remaining = len([p for p in all_pairs if p not in labeled_pairs])
    return total, remaining


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
    result = select_informative_pair_with_explanation(
        source_df, target_df, column_mappings, comparison_config,
        blocking_config, labeled_pairs, model, strategy, is_dedup, sample_size
    )
    if result is None:
        return None
    return (result.left_idx, result.right_idx, result.comparison_vector)


def select_informative_pair_with_explanation(
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
) -> Optional[PairSelectionResult]:
    """
    Select the most informative pair for labeling with explanation.

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
        PairSelectionResult with explanation or None
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
    sampled_pairs = unlabeled_pairs
    if len(unlabeled_pairs) > sample_size:
        sampled_pairs = random.sample(unlabeled_pairs, sample_size)

    # Compute comparison vectors
    comparison_vectors = []
    valid_pairs = []

    for left_idx, right_idx in sampled_pairs:
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

    candidates_evaluated = len(valid_pairs)

    # Select based on strategy
    if strategy == "random" or model is None:
        idx = random.randrange(len(valid_pairs))
        reason = "random" if strategy == "random" else "no_model"
        return PairSelectionResult(
            left_idx=valid_pairs[idx][0],
            right_idx=valid_pairs[idx][1],
            comparison_vector=comparison_vectors[idx],
            selection_reason=reason,
            candidates_evaluated=candidates_evaluated
        )

    elif strategy == "uncertainty":
        # Get model predictions
        try:
            X = [list(cv.values()) for cv in comparison_vectors]
            probabilities = model.predict_proba(X)

            # Find most uncertain
            uncertainties = [abs(p - 0.5) for p in probabilities]
            min_idx = np.argmin(uncertainties)

            return PairSelectionResult(
                left_idx=valid_pairs[min_idx][0],
                right_idx=valid_pairs[min_idx][1],
                comparison_vector=comparison_vectors[min_idx],
                selection_reason="uncertainty_sampling",
                uncertainty_score=float(uncertainties[min_idx]),
                model_probability=float(probabilities[min_idx]),
                candidates_evaluated=candidates_evaluated
            )
        except Exception:
            # Fall back to random if model fails
            idx = random.randrange(len(valid_pairs))
            return PairSelectionResult(
                left_idx=valid_pairs[idx][0],
                right_idx=valid_pairs[idx][1],
                comparison_vector=comparison_vectors[idx],
                selection_reason="random_fallback",
                candidates_evaluated=candidates_evaluated
            )

    else:
        # Default to random
        idx = random.randrange(len(valid_pairs))
        return PairSelectionResult(
            left_idx=valid_pairs[idx][0],
            right_idx=valid_pairs[idx][1],
            comparison_vector=comparison_vectors[idx],
            selection_reason="random",
            candidates_evaluated=candidates_evaluated
        )


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


def select_from_linkage_results(
    db,  # SQLAlchemy session
    project_id: int,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    column_mappings: Dict[str, str],
    comparison_config: Dict[str, Any],
    labeled_pairs: Set[Tuple[int, int]]
) -> Optional[PairSelectionResult]:
    """
    Select the next pair from linkage job results, prioritizing matches for confirmation.

    This function queries the database for pairs classified as "match" from
    the most recent completed linkage job and returns them for user confirmation.

    Args:
        db: SQLAlchemy database session
        project_id: Project ID to query linkage results for
        source_df: Source DataFrame
        target_df: Target DataFrame
        column_mappings: Column mappings configuration
        comparison_config: Comparison methods configuration
        labeled_pairs: Set of already labeled (left_idx, right_idx) pairs

    Returns:
        PairSelectionResult with explanation or None if no linkage matches available
    """
    # Import here to avoid circular imports
    from app.db.models import LinkageJob, RecordPair, JobStatus

    # Get the most recent completed linkage job
    latest_job = db.query(LinkageJob).filter(
        LinkageJob.project_id == project_id,
        LinkageJob.status == JobStatus.COMPLETED
    ).order_by(LinkageJob.completed_at.desc()).first()

    if not latest_job:
        return None

    # Get matched pairs from linkage results that haven't been labeled yet
    match_pairs = db.query(RecordPair).filter(
        RecordPair.job_id == latest_job.id,
        RecordPair.classification == "match"
    ).order_by(RecordPair.match_score.desc()).all()

    # Find the first unlabeled pair
    for pair in match_pairs:
        if (pair.left_record_idx, pair.right_record_idx) not in labeled_pairs:
            # Compute comparison vector if not already stored
            if pair.comparison_vector:
                comparison_vector = pair.comparison_vector
            else:
                try:
                    left_record = source_df.iloc[pair.left_record_idx].to_dict()
                    right_record = target_df.iloc[pair.right_record_idx].to_dict()
                    comparison_vector = compare_records(
                        left_record, right_record,
                        column_mappings, comparison_config
                    )
                except Exception:
                    continue

            return PairSelectionResult(
                left_idx=pair.left_record_idx,
                right_idx=pair.right_record_idx,
                comparison_vector=comparison_vector,
                selection_reason="linkage_result_confirmation",
                model_probability=pair.match_score,
                candidates_evaluated=1
            )

    return None
