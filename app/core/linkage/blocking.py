"""
Blocking/indexing strategies to reduce comparison space.
"""
import pandas as pd
from typing import List, Tuple, Dict, Any, Set, Optional
from collections import defaultdict


def create_blocking_key(record: pd.Series, blocking_keys: List[str]) -> str:
    """Create a blocking key from record values."""
    parts = []
    for key in blocking_keys:
        value = record.get(key, '')
        if pd.isna(value):
            value = ''
        parts.append(str(value).strip().lower())
    return '|'.join(parts)


def standard_blocking(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    blocking_keys: List[str],
    is_dedup: bool = False
) -> List[Tuple[int, int]]:
    """
    Standard blocking: only compare records with identical blocking keys.

    Args:
        source_df: Source DataFrame
        target_df: Target DataFrame
        blocking_keys: List of column names to use for blocking
        is_dedup: If True, this is deduplication (same dataset)

    Returns:
        List of (source_idx, target_idx) pairs to compare
    """
    pairs = []

    # Build blocking index for target
    target_blocks = defaultdict(list)
    for idx, row in target_df.iterrows():
        key = create_blocking_key(row, blocking_keys)
        if key:  # Skip empty keys
            target_blocks[key].append(idx)

    # Find matches in source
    for source_idx, source_row in source_df.iterrows():
        key = create_blocking_key(source_row, blocking_keys)
        if key and key in target_blocks:
            for target_idx in target_blocks[key]:
                # For dedup, avoid self-comparison and duplicate pairs
                if is_dedup:
                    if source_idx < target_idx:
                        pairs.append((source_idx, target_idx))
                else:
                    pairs.append((source_idx, target_idx))

    return pairs


def sorted_neighborhood_blocking(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    blocking_keys: List[str],
    window_size: int = 5,
    is_dedup: bool = False
) -> List[Tuple[int, int]]:
    """
    Sorted neighborhood blocking: sort by blocking key and compare within window.

    Args:
        source_df: Source DataFrame
        target_df: Target DataFrame
        blocking_keys: List of column names for sorting key
        window_size: Size of comparison window
        is_dedup: If True, this is deduplication

    Returns:
        List of (source_idx, target_idx) pairs to compare
    """
    pairs = set()

    if is_dedup:
        # For dedup, work with single dataset
        df = source_df.copy()
        df['_blocking_key'] = df.apply(
            lambda row: create_blocking_key(row, blocking_keys), axis=1
        )
        df['_original_idx'] = df.index
        df_sorted = df.sort_values('_blocking_key').reset_index(drop=True)

        for i in range(len(df_sorted)):
            for j in range(i + 1, min(i + window_size, len(df_sorted))):
                idx1 = df_sorted.iloc[i]['_original_idx']
                idx2 = df_sorted.iloc[j]['_original_idx']
                if idx1 != idx2:
                    pairs.add((min(idx1, idx2), max(idx1, idx2)))
    else:
        # For linkage, merge and sort both datasets
        source_df = source_df.copy()
        target_df = target_df.copy()

        source_df['_blocking_key'] = source_df.apply(
            lambda row: create_blocking_key(row, blocking_keys), axis=1
        )
        target_df['_blocking_key'] = target_df.apply(
            lambda row: create_blocking_key(row, blocking_keys), axis=1
        )

        source_df['_original_idx'] = source_df.index
        target_df['_original_idx'] = target_df.index
        source_df['_source'] = 'source'
        target_df['_source'] = 'target'

        combined = pd.concat([source_df, target_df], ignore_index=True)
        combined_sorted = combined.sort_values('_blocking_key').reset_index(drop=True)

        for i in range(len(combined_sorted)):
            if combined_sorted.iloc[i]['_source'] == 'source':
                source_idx = combined_sorted.iloc[i]['_original_idx']
                for j in range(max(0, i - window_size), min(len(combined_sorted), i + window_size)):
                    if i != j and combined_sorted.iloc[j]['_source'] == 'target':
                        target_idx = combined_sorted.iloc[j]['_original_idx']
                        pairs.add((source_idx, target_idx))

    return list(pairs)


def phonetic_blocking(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    blocking_keys: List[str],
    is_dedup: bool = False
) -> List[Tuple[int, int]]:
    """
    Phonetic blocking using Soundex.

    Args:
        source_df: Source DataFrame
        target_df: Target DataFrame
        blocking_keys: List of column names (typically name columns)
        is_dedup: If True, this is deduplication

    Returns:
        List of (source_idx, target_idx) pairs
    """
    import jellyfish

    def create_phonetic_key(record: pd.Series, keys: List[str]) -> str:
        parts = []
        for key in keys:
            value = str(record.get(key, '')).strip()
            # Extract only alphabetic characters
            alpha_value = ''.join(c for c in value if c.isalpha())
            if alpha_value:
                parts.append(jellyfish.soundex(alpha_value))
        return '|'.join(parts)

    pairs = []

    # Build blocking index for target
    target_blocks = defaultdict(list)
    for idx, row in target_df.iterrows():
        key = create_phonetic_key(row, blocking_keys)
        if key:
            target_blocks[key].append(idx)

    # Find matches in source
    for source_idx, source_row in source_df.iterrows():
        key = create_phonetic_key(source_row, blocking_keys)
        if key and key in target_blocks:
            for target_idx in target_blocks[key]:
                if is_dedup:
                    if source_idx < target_idx:
                        pairs.append((source_idx, target_idx))
                else:
                    pairs.append((source_idx, target_idx))

    return pairs


def create_candidate_pairs(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    blocking_config: Dict[str, Any],
    is_dedup: bool = False
) -> List[Tuple[int, int]]:
    """
    Create candidate pairs using configured blocking strategy.

    Args:
        source_df: Source DataFrame
        target_df: Target DataFrame
        blocking_config: Configuration dict with:
            - method: "standard", "sorted_neighborhood", "phonetic", or "none"
            - blocking_keys: List of column names
            - window_size: For sorted neighborhood (default: 5)
        is_dedup: If True, this is deduplication

    Returns:
        List of (source_idx, target_idx) pairs
    """
    method = blocking_config.get('method', 'none')
    blocking_keys = blocking_config.get('blocking_keys', [])

    if method == 'none' or not blocking_keys:
        # Full cartesian product (expensive!)
        pairs = []
        for source_idx in source_df.index:
            for target_idx in target_df.index:
                if is_dedup:
                    if source_idx < target_idx:
                        pairs.append((source_idx, target_idx))
                else:
                    pairs.append((source_idx, target_idx))
        return pairs

    elif method == 'standard':
        return standard_blocking(source_df, target_df, blocking_keys, is_dedup)

    elif method == 'sorted_neighborhood':
        window_size = blocking_config.get('window_size', 5)
        return sorted_neighborhood_blocking(
            source_df, target_df, blocking_keys, window_size, is_dedup
        )

    elif method == 'phonetic':
        return phonetic_blocking(source_df, target_df, blocking_keys, is_dedup)

    else:
        raise ValueError(f"Unknown blocking method: {method}")
