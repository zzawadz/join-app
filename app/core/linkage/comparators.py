"""
String and field comparison functions for record linkage.
"""
import jellyfish
from typing import Any, Dict, Optional
from datetime import datetime
import re


def exact_match(value1: Any, value2: Any) -> float:
    """Exact string match (case-insensitive)."""
    if value1 is None or value2 is None:
        return 0.0

    str1 = str(value1).strip().lower()
    str2 = str(value2).strip().lower()

    return 1.0 if str1 == str2 else 0.0


def jaro_similarity(value1: Any, value2: Any) -> float:
    """Jaro string similarity."""
    if value1 is None or value2 is None:
        return 0.0

    str1 = str(value1).strip().lower()
    str2 = str(value2).strip().lower()

    if not str1 or not str2:
        return 0.0

    return jellyfish.jaro_similarity(str1, str2)


def jaro_winkler_similarity(value1: Any, value2: Any) -> float:
    """Jaro-Winkler string similarity (good for names)."""
    if value1 is None or value2 is None:
        return 0.0

    str1 = str(value1).strip().lower()
    str2 = str(value2).strip().lower()

    if not str1 or not str2:
        return 0.0

    return jellyfish.jaro_winkler_similarity(str1, str2)


def levenshtein_distance(value1: Any, value2: Any) -> int:
    """Levenshtein (edit) distance between two strings."""
    if value1 is None or value2 is None:
        return -1

    str1 = str(value1).strip().lower()
    str2 = str(value2).strip().lower()

    return jellyfish.levenshtein_distance(str1, str2)


def levenshtein_similarity(value1: Any, value2: Any) -> float:
    """Normalized Levenshtein similarity (0-1 range)."""
    if value1 is None or value2 is None:
        return 0.0

    str1 = str(value1).strip().lower()
    str2 = str(value2).strip().lower()

    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0

    distance = jellyfish.levenshtein_distance(str1, str2)
    max_len = max(len(str1), len(str2))

    return 1.0 - (distance / max_len)


def damerau_levenshtein_similarity(value1: Any, value2: Any) -> float:
    """Damerau-Levenshtein similarity (allows transpositions)."""
    if value1 is None or value2 is None:
        return 0.0

    str1 = str(value1).strip().lower()
    str2 = str(value2).strip().lower()

    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0

    distance = jellyfish.damerau_levenshtein_distance(str1, str2)
    max_len = max(len(str1), len(str2))

    return 1.0 - (distance / max_len)


def soundex_match(value1: Any, value2: Any) -> float:
    """Soundex phonetic match (English names)."""
    if value1 is None or value2 is None:
        return 0.0

    str1 = str(value1).strip()
    str2 = str(value2).strip()

    # Remove non-alphabetic characters
    str1 = re.sub(r'[^a-zA-Z]', '', str1)
    str2 = re.sub(r'[^a-zA-Z]', '', str2)

    if not str1 or not str2:
        return 0.0

    return 1.0 if jellyfish.soundex(str1) == jellyfish.soundex(str2) else 0.0


def metaphone_match(value1: Any, value2: Any) -> float:
    """Metaphone phonetic match."""
    if value1 is None or value2 is None:
        return 0.0

    str1 = str(value1).strip()
    str2 = str(value2).strip()

    str1 = re.sub(r'[^a-zA-Z]', '', str1)
    str2 = re.sub(r'[^a-zA-Z]', '', str2)

    if not str1 or not str2:
        return 0.0

    return 1.0 if jellyfish.metaphone(str1) == jellyfish.metaphone(str2) else 0.0


def numeric_similarity(value1: Any, value2: Any, tolerance: float = 0.0) -> float:
    """
    Numeric similarity with optional tolerance.

    Args:
        tolerance: Absolute tolerance for considering values equal
    """
    if value1 is None or value2 is None:
        return 0.0

    try:
        num1 = float(str(value1).replace(',', ''))
        num2 = float(str(value2).replace(',', ''))
    except (ValueError, TypeError):
        return 0.0

    if num1 == num2:
        return 1.0

    if tolerance > 0:
        if abs(num1 - num2) <= tolerance:
            return 1.0

    # Relative similarity
    max_val = max(abs(num1), abs(num2))
    if max_val == 0:
        return 1.0

    diff = abs(num1 - num2) / max_val
    return max(0.0, 1.0 - diff)


def date_similarity(value1: Any, value2: Any, day_tolerance: int = 0) -> float:
    """
    Date similarity with optional day tolerance.

    Args:
        day_tolerance: Number of days difference allowed for full match
    """
    if value1 is None or value2 is None:
        return 0.0

    # Try to parse dates
    date_formats = [
        '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y',
        '%Y/%m/%d', '%m-%d-%Y', '%d-%m-%Y',
        '%Y%m%d'
    ]

    def parse_date(val):
        if isinstance(val, datetime):
            return val
        for fmt in date_formats:
            try:
                return datetime.strptime(str(val).strip(), fmt)
            except ValueError:
                continue
        return None

    date1 = parse_date(value1)
    date2 = parse_date(value2)

    if date1 is None or date2 is None:
        return 0.0

    diff_days = abs((date1 - date2).days)

    if diff_days <= day_tolerance:
        return 1.0

    # Decay similarity over 365 days
    return max(0.0, 1.0 - (diff_days / 365.0))


# Mapping of comparator names to functions
COMPARATORS = {
    'exact': exact_match,
    'jaro': jaro_similarity,
    'jaro_winkler': jaro_winkler_similarity,
    'levenshtein': levenshtein_similarity,
    'damerau_levenshtein': damerau_levenshtein_similarity,
    'soundex': soundex_match,
    'metaphone': metaphone_match,
    'numeric': numeric_similarity,
    'date': date_similarity,
}


def get_comparator(name: str):
    """Get a comparator function by name."""
    return COMPARATORS.get(name, exact_match)


def compare_records(
    record1: Dict[str, Any],
    record2: Dict[str, Any],
    column_mappings: Dict[str, str],
    comparison_config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Compare two records and return comparison vector.

    Args:
        record1: First record (source)
        record2: Second record (target)
        column_mappings: Mapping of source columns to target columns
        comparison_config: Configuration for each column comparison
            Example: {"name": {"method": "jaro_winkler", "threshold": 0.85}}

    Returns:
        Dictionary of column -> similarity score
    """
    comparison_vector = {}

    for source_col, target_col in column_mappings.items():
        value1 = record1.get(source_col)
        value2 = record2.get(target_col)

        # Get comparison method for this column
        col_config = comparison_config.get(source_col, {})
        method = col_config.get('method', 'jaro_winkler')
        comparator = get_comparator(method)

        # Get additional parameters
        if method == 'numeric':
            tolerance = col_config.get('threshold', 0.0)
            score = numeric_similarity(value1, value2, tolerance)
        elif method == 'date':
            day_tolerance = int(col_config.get('threshold', 0))
            score = date_similarity(value1, value2, day_tolerance)
        else:
            score = comparator(value1, value2)

        comparison_vector[source_col] = score

    return comparison_vector
