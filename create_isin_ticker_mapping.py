"""
Create ISIN to Yahoo Finance Ticker Mapping
Uses BSE and NSE BhavCopy files to create a mapping dictionary
"""
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_bhav_copies() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load BSE and NSE BhavCopy files

    Returns:
        Tuple of (BSE dataframe, NSE dataframe)
    """
    data_dir = Path(__file__).parent / "data"

    # Find BhavCopy files
    bse_files = list(data_dir.glob("BhavCopy_BSE_*.CSV")) + list(data_dir.glob("BhavCopy_BSE_*.csv"))
    nse_files = list(data_dir.glob("BhavCopy_NSE_*.CSV")) + list(data_dir.glob("BhavCopy_NSE_*.csv"))

    if not bse_files or not nse_files:
        raise FileNotFoundError("BhavCopy files not found in data directory")

    # Use the most recent files
    bse_file = sorted(bse_files)[-1]
    nse_file = sorted(nse_files)[-1]

    print(f"Loading BSE BhavCopy: {bse_file.name}")
    print(f"Loading NSE BhavCopy: {nse_file.name}")

    # Read files
    df_bse = pd.read_csv(bse_file)
    df_nse = pd.read_csv(nse_file)

    return df_bse, df_nse


def create_ticker_mapping(df_bse: pd.DataFrame, df_nse: pd.DataFrame) -> Dict[str, str]:
    """
    Create ISIN to Yahoo Finance ticker mapping

    Prioritizes NSE tickers (.NS) over BSE tickers (.BO)

    Args:
        df_bse: BSE BhavCopy dataframe
        df_nse: NSE BhavCopy dataframe

    Returns:
        Dictionary mapping ISIN to Yahoo ticker
    """
    mapping = {}

    # Process NSE data (preferred)
    print("\nProcessing NSE data...")
    nse_valid = df_nse[df_nse['ISIN'].notna() & df_nse['TckrSymb'].notna()].copy()

    for _, row in nse_valid.iterrows():
        isin = row['ISIN']
        ticker = row['TckrSymb']

        # Skip if ticker is empty or invalid
        if pd.isna(ticker) or str(ticker).strip() == '':
            continue

        # Add .NS suffix for NSE
        yahoo_ticker = f"{ticker}.NS"
        mapping[isin] = yahoo_ticker

    print(f"✓ Added {len(mapping)} NSE mappings")

    # Process BSE data (fallback for stocks not on NSE)
    print("\nProcessing BSE data...")
    bse_valid = df_bse[df_bse['ISIN'].notna() & df_bse['TckrSymb'].notna()].copy()

    bse_added = 0
    for _, row in bse_valid.iterrows():
        isin = row['ISIN']
        ticker = row['TckrSymb']

        # Skip if already mapped (NSE takes priority)
        if isin in mapping:
            continue

        # Skip if ticker is empty or invalid
        if pd.isna(ticker) or str(ticker).strip() == '':
            continue

        # Add .BO suffix for BSE
        yahoo_ticker = f"{ticker}.BO"
        mapping[isin] = yahoo_ticker
        bse_added += 1

    print(f"✓ Added {bse_added} BSE mappings (not on NSE)")
    print(f"\n✓ Total mappings created: {len(mapping)}")

    return mapping


def save_mapping(mapping: Dict[str, str], output_path: Path):
    """
    Save mapping to JSON file

    Args:
        mapping: ISIN to ticker dictionary
        output_path: Path to save JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"\n✓ Mapping saved to: {output_path}")


def load_mapping(mapping_path: Path) -> Dict[str, str]:
    """
    Load existing mapping from JSON file

    Args:
        mapping_path: Path to JSON file

    Returns:
        ISIN to ticker dictionary
    """
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    return mapping


def verify_mapping_coverage(master_csv_path: Path, mapping: Dict[str, str]):
    """
    Verify what percentage of stocks in master data have ticker mappings

    Args:
        master_csv_path: Path to master portfolio CSV
        mapping: ISIN to ticker dictionary
    """
    print("\n" + "="*80)
    print("MAPPING COVERAGE VERIFICATION")
    print("="*80)

    df_master = pd.read_csv(master_csv_path)
    unique_isins = df_master['ISIN'].unique()

    mapped_count = sum(1 for isin in unique_isins if isin in mapping)
    total_count = len(unique_isins)
    coverage = (mapped_count / total_count) * 100

    print(f"Total unique ISINs in portfolio: {total_count}")
    print(f"ISINs with ticker mapping: {mapped_count}")
    print(f"Coverage: {coverage:.2f}%")

    # Show some unmapped ISINs
    unmapped = [isin for isin in unique_isins if isin not in mapping]
    if unmapped:
        print(f"\nSample unmapped ISINs (first 10):")
        for isin in unmapped[:10]:
            stock_name = df_master[df_master['ISIN'] == isin]['Stock_Name'].iloc[0]
            print(f"  {isin}: {stock_name}")

    return coverage


def main():
    """Main execution"""
    print("="*80)
    print("CREATING ISIN TO YAHOO FINANCE TICKER MAPPING")
    print("="*80)

    # Load BhavCopy files
    df_bse, df_nse = load_bhav_copies()

    print(f"\nBSE records: {len(df_bse)}")
    print(f"NSE records: {len(df_nse)}")

    # Create mapping
    mapping = create_ticker_mapping(df_bse, df_nse)

    # Save to data directory
    output_path = Path(__file__).parent / "data" / "isin_ticker_mapping.json"
    save_mapping(mapping, output_path)

    # Verify coverage
    master_csv = Path(__file__).parent / "fund_portfolio_master_smart.csv"
    if master_csv.exists():
        coverage = verify_mapping_coverage(master_csv, mapping)

        if coverage < 80:
            print(f"\n⚠️  Warning: Only {coverage:.2f}% coverage. Some stocks may not have data.")
        else:
            print(f"\n✓ Good coverage: {coverage:.2f}%")

    print("\n" + "="*80)
    print("MAPPING CREATION COMPLETE")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Review the mapping file: {output_path}")
    print(f"2. Run the data collection pipeline: python run_phase1.py")


if __name__ == "__main__":
    main()
