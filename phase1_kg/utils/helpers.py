"""
Utility functions for Phase 1 implementation
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from typing import Any, Dict, List, Union
import hashlib


def load_master_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load and preprocess master portfolio data"""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Fund_Name', 'ISIN', 'Date'])
    return df


def save_to_cache(data: Any, cache_name: str, cache_dir: Path) -> None:
    """Save data to cache directory"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_name}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Cached: {cache_name}")


def load_from_cache(cache_name: str, cache_dir: Path) -> Any:
    """Load data from cache if exists"""
    cache_file = cache_dir / f"{cache_name}.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            print(f"Loaded from cache: {cache_name}")
            return pickle.load(f)
    return None


def get_cache_key(*args) -> str:
    """Generate a cache key from arguments"""
    key_str = "_".join(str(arg) for arg in args)
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def calculate_portfolio_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate portfolio changes between consecutive periods

    Returns DataFrame with additional columns:
    - prev_weight: Previous period weight
    - weight_change: Change in weight
    - weight_change_pct: Percentage change
    - action: INCREASED, DECREASED, ENTERED, EXITED, UNCHANGED
    """
    df = df.sort_values(['Fund_Name', 'ISIN', 'Date'])

    # Calculate previous weight
    df['prev_weight'] = df.groupby(['Fund_Name', 'ISIN'])['Portfolio_Weight'].shift(1)

    # Calculate changes
    df['weight_change'] = df['Portfolio_Weight'] - df['prev_weight']
    df['weight_change_pct'] = (df['weight_change'] / df['prev_weight']) * 100

    # Determine action
    def classify_action(row):
        if pd.isna(row['prev_weight']):
            return 'ENTERED'
        elif pd.isna(row['Portfolio_Weight']):
            return 'EXITED'
        elif abs(row['weight_change']) < 0.001:  # Less than 0.1%
            return 'UNCHANGED'
        elif row['weight_change'] > 0:
            return 'INCREASED'
        else:
            return 'DECREASED'

    df['action'] = df.apply(classify_action, axis=1)

    return df


def get_sector_mapping(isin_list: List[str]) -> Dict[str, str]:
    """
    Map ISINs to standard sectors
    Note: This is a placeholder. In production, use actual sector data.
    """
    # TODO: Implement actual sector mapping using NSE/BSE data
    # For now, return a dummy mapping
    sector_map = {}
    sectors = [
        "Banking", "IT", "Pharma", "Energy", "Consumer",
        "Automobile", "Infrastructure", "Metals", "FMCG", "Healthcare"
    ]

    for isin in isin_list:
        # Simple hash-based assignment for demo
        sector_idx = hash(isin) % len(sectors)
        sector_map[isin] = sectors[sector_idx]

    return sector_map


def forward_fill_missing(df: pd.DataFrame, column: str, group_by: List[str]) -> pd.DataFrame:
    """Forward fill missing values within groups"""
    df[column] = df.groupby(group_by)[column].ffill()
    return df


def save_json(data: Dict, file_path: Union[str, Path]) -> None:
    """Save data as JSON"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved JSON: {file_path}")


def load_json(file_path: Union[str, Path]) -> Dict:
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def create_date_range_chunks(start_date: str, end_date: str, freq: str = 'M') -> List:
    """Create chunks of date ranges for parallel processing"""
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    return list(date_range)


def log_step(step_name: str, status: str = "START") -> None:
    """Log execution steps"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*80}")
    print(f"[{timestamp}] {status}: {step_name}")
    print(f"{'='*80}\n")


def calculate_statistics(values: List[float]) -> Dict:
    """Calculate basic statistics"""
    if not values:
        return {}

    return {
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "count": len(values)
    }


def filter_outliers(df: pd.DataFrame, column: str, n_std: float = 3) -> pd.DataFrame:
    """Remove outliers beyond n standard deviations"""
    mean = df[column].mean()
    std = df[column].std()
    df_filtered = df[
        (df[column] >= mean - n_std * std) &
        (df[column] <= mean + n_std * std)
    ]
    removed = len(df) - len(df_filtered)
    if removed > 0:
        print(f"Removed {removed} outliers from {column}")
    return df_filtered


if __name__ == "__main__":
    print("Utility functions loaded successfully")
