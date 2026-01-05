import pandas as pd
import chardet
from typing import Dict, Any, List, Optional
from pathlib import Path


def detect_encoding(file_path: str) -> str:
    """Detect the encoding of a file."""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return result['encoding'] or 'utf-8'


def detect_delimiter(file_path: str, encoding: str) -> str:
    """Detect the delimiter used in a CSV file."""
    import csv

    with open(file_path, 'r', encoding=encoding) as f:
        sample = f.read(8192)

    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample, delimiters=',;\t|')
        return dialect.delimiter
    except csv.Error:
        return ','


def infer_column_type(series: pd.Series) -> str:
    """Infer the semantic type of a column."""
    # Drop nulls for analysis
    non_null = series.dropna()

    if len(non_null) == 0:
        return "unknown"

    # Check if numeric
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # Sample for analysis
    sample = non_null.astype(str).head(100)

    # Check for date patterns
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',
        r'\d{2}/\d{2}/\d{4}',
        r'\d{2}-\d{2}-\d{4}'
    ]
    for pattern in date_patterns:
        if sample.str.match(pattern).mean() > 0.8:
            return "date"

    # Check for email
    if sample.str.contains(r'@.*\.', regex=True).mean() > 0.5:
        return "email"

    # Check for phone
    if sample.str.contains(r'\d{3}.*\d{3}.*\d{4}', regex=True).mean() > 0.5:
        return "phone"

    # Default to string
    avg_len = sample.str.len().mean()
    if avg_len > 50:
        return "text"

    return "string"


def process_csv_upload(file_path: str, sample_rows: int = 20) -> Dict[str, Any]:
    """
    Process an uploaded CSV file and extract metadata.

    Returns:
        dict with:
        - row_count: total number of rows
        - column_names: list of column names
        - column_types: dict mapping column names to inferred types
        - sample_data: first N rows as list of dicts
    """
    # Detect encoding
    encoding = detect_encoding(file_path)

    # Detect delimiter
    delimiter = detect_delimiter(file_path, encoding)

    # Read CSV
    df = pd.read_csv(
        file_path,
        encoding=encoding,
        delimiter=delimiter,
        low_memory=False,
        nrows=10000  # Read first 10k rows for type inference
    )

    # Get full row count
    with open(file_path, 'r', encoding=encoding) as f:
        row_count = sum(1 for _ in f) - 1  # Subtract header

    # Infer column types
    column_types = {}
    for col in df.columns:
        column_types[col] = infer_column_type(df[col])

    # Get sample data
    sample_df = df.head(sample_rows)
    # Convert to list of dicts, handling NaN values
    sample_data = sample_df.where(pd.notnull(sample_df), None).to_dict('records')

    return {
        "row_count": row_count,
        "column_names": list(df.columns),
        "column_types": column_types,
        "sample_data": sample_data
    }


def load_dataframe(file_path: str) -> pd.DataFrame:
    """Load a CSV file as a DataFrame with proper encoding detection."""
    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)

    return pd.read_csv(
        file_path,
        encoding=encoding,
        delimiter=delimiter,
        low_memory=False
    )
