from typing import List, Dict, Any, Optional
import jellyfish
import re


def normalize_column_name(name: str) -> str:
    """Normalize a column name for comparison."""
    # Convert to lowercase
    name = name.lower()
    # Remove common prefixes/suffixes
    name = re.sub(r'^(col_|column_|field_|fld_)', '', name)
    # Replace separators with spaces
    name = re.sub(r'[_\-\.]', ' ', name)
    # Remove extra whitespace
    name = ' '.join(name.split())
    return name


def column_name_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between two column names."""
    norm1 = normalize_column_name(name1)
    norm2 = normalize_column_name(name2)

    # Exact match after normalization
    if norm1 == norm2:
        return 1.0

    # Jaro-Winkler similarity
    jw_sim = jellyfish.jaro_winkler_similarity(norm1, norm2)

    # Check for common synonyms
    synonyms = [
        ({'first', 'fname', 'firstname', 'given', 'givenname'}, 'first name'),
        ({'last', 'lname', 'lastname', 'surname', 'family', 'familyname'}, 'last name'),
        ({'middle', 'mname', 'middlename', 'mi'}, 'middle name'),
        ({'address', 'addr', 'street', 'streetaddress'}, 'address'),
        ({'city', 'town'}, 'city'),
        ({'state', 'province', 'st'}, 'state'),
        ({'zip', 'zipcode', 'postalcode', 'postal'}, 'zip'),
        ({'phone', 'telephone', 'tel', 'phonenumber'}, 'phone'),
        ({'email', 'emailaddress', 'mail'}, 'email'),
        ({'dob', 'dateofbirth', 'birthdate', 'birthday'}, 'date of birth'),
        ({'ssn', 'socialsecurity', 'ssnum'}, 'ssn'),
        ({'id', 'identifier', 'recordid', 'record id'}, 'id'),
    ]

    for syn_set, _ in synonyms:
        norm1_words = set(norm1.split())
        norm2_words = set(norm2.split())
        if (norm1_words & syn_set) and (norm2_words & syn_set):
            return 0.95

    return jw_sim


def column_content_similarity(sample1: List[Any], sample2: List[Any]) -> float:
    """
    Calculate similarity between column contents based on sample values.
    """
    if not sample1 or not sample2:
        return 0.0

    # Convert to strings, filter nulls
    values1 = [str(v) for v in sample1 if v is not None and str(v).strip()]
    values2 = [str(v) for v in sample2 if v is not None and str(v).strip()]

    if not values1 or not values2:
        return 0.0

    # Check value length similarity
    avg_len1 = sum(len(v) for v in values1) / len(values1)
    avg_len2 = sum(len(v) for v in values2) / len(values2)
    len_ratio = min(avg_len1, avg_len2) / max(avg_len1, avg_len2) if max(avg_len1, avg_len2) > 0 else 0

    # Check character type distribution
    def char_distribution(values):
        total_alpha = sum(c.isalpha() for v in values for c in v)
        total_digit = sum(c.isdigit() for v in values for c in v)
        total_chars = total_alpha + total_digit + 1
        return (total_alpha / total_chars, total_digit / total_chars)

    dist1 = char_distribution(values1)
    dist2 = char_distribution(values2)
    dist_sim = 1 - (abs(dist1[0] - dist2[0]) + abs(dist1[1] - dist2[1])) / 2

    # Check for value overlap
    set1 = set(v.lower() for v in values1)
    set2 = set(v.lower() for v in values2)
    if set1 and set2:
        overlap = len(set1 & set2) / min(len(set1), len(set2))
    else:
        overlap = 0

    return (len_ratio + dist_sim + overlap) / 3


def suggest_column_mappings(
    source_columns: List[str],
    target_columns: List[str],
    source_sample: Optional[List[Dict]] = None,
    target_sample: Optional[List[Dict]] = None
) -> List[Dict[str, Any]]:
    """
    Suggest column mappings between source and target datasets.

    Returns:
        List of mapping suggestions with confidence scores
    """
    suggestions = []

    for source_col in source_columns:
        best_match = None
        best_score = 0.0

        for target_col in target_columns:
            # Name similarity
            name_sim = column_name_similarity(source_col, target_col)

            # Content similarity (if samples provided)
            content_sim = 0.0
            if source_sample and target_sample:
                source_values = [row.get(source_col) for row in source_sample]
                target_values = [row.get(target_col) for row in target_sample]
                content_sim = column_content_similarity(source_values, target_values)

            # Combined score (weight name similarity more heavily)
            if source_sample and target_sample:
                score = name_sim * 0.6 + content_sim * 0.4
            else:
                score = name_sim

            if score > best_score:
                best_score = score
                best_match = target_col

        if best_match and best_score > 0.5:
            suggestions.append({
                "source_column": source_col,
                "target_column": best_match,
                "confidence": round(best_score, 3)
            })

    # Sort by confidence
    suggestions.sort(key=lambda x: x["confidence"], reverse=True)

    return suggestions
