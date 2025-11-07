"""
Fundamental Data Collection using yfinance
Fetches company financials, valuation metrics, and sector classification
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from tqdm import tqdm
import time
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import FUNDAMENTAL_METRICS, CACHE_DIR, PROCESSED_DATA_DIR, START_DATE, END_DATE
from utils.helpers import save_to_cache, load_from_cache, log_step


class FundamentalDataCollector:
    """
    Collects fundamental data for stocks using yfinance
    - Fetches key financial metrics
    - Extracts sector and industry classification
    - Handles missing data
    - Caches results for efficiency
    """

    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self.metrics = FUNDAMENTAL_METRICS
        log_step("Initializing Fundamental Data Collector")
        print(f"✓ Metrics to collect: {self.metrics}")

    def load_ticker_mapping(self) -> Dict[str, str]:
        """
        Load ISIN to Yahoo Finance ticker mapping from JSON file

        Returns:
            Dictionary mapping ISIN to Yahoo ticker
        """
        mapping_path = Path(__file__).parent.parent.parent / "data" / "isin_ticker_mapping.json"

        if not mapping_path.exists():
            print(f"⚠️  Warning: Ticker mapping not found at {mapping_path}")
            print("Please run: python create_isin_ticker_mapping.py")
            return {}

        with open(mapping_path, 'r') as f:
            mapping = json.load(f)

        return mapping

    def get_stock_fundamentals(self, ticker: str, isin: str, dates: List[str]) -> List[Dict]:
        """
        Get historical fundamental data for a stock

        Args:
            ticker: Yahoo Finance ticker symbol
            isin: ISIN code
            dates: List of dates to fetch data for

        Returns:
            List of dictionaries with fundamental data for each date
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or 'symbol' not in info:
                return []

            # Get historical data for price-based calculations
            hist = stock.history(period="max")

            fundamentals = []

            for date_str in dates:
                date = pd.to_datetime(date_str)

                # Extract metrics
                metrics_data = {
                    'ISIN': isin,
                    'ticker': ticker,
                    'Date': date_str
                }

                for metric in self.metrics:
                    metrics_data[metric] = info.get(metric, np.nan)

                fundamentals.append(metrics_data)

            return fundamentals

        except Exception as e:
            # Silently skip - expected for many stocks
            return []

    def is_valid_stock_name(self, name: str) -> bool:
        """
        Check if a stock name is valid (not an ISIN code or empty)
        Valid names should contain company name indicators like Limited, Ltd, Inc, Bank, Fund, etc.

        Args:
            name: Stock name to validate

        Returns:
            True if valid, False otherwise
        """
        if name is None or pd.isna(name):
            return False

        name_str = str(name).strip()

        # Check if it's empty or just whitespace
        if not name_str:
            return False

        # Check if it's an ISIN code (starts with INE or similar patterns)
        if name_str.startswith('INE') or name_str.startswith('IEX'):
            return False

        # Check if it's just a pipe or special character
        if name_str in ['|', '-', ' ', 'nan']:
            return False

        # Valid if contains common company name indicators
        valid_keywords = [
            'limited', 'ltd', 'inc', 'corp', 'corporation',
            'bank', 'financial', 'group', 'company',
            'industries', 'services', 'systems',
            'pharma', 'laboratory', 'labs',
            'energy', 'power', 'oil',
            'tech', 'technology', 'infosys', 'tcs', 'wipro',
            'reliance', 'hdfc', 'icici', 'axis',
            'motors', 'auto', 'steel', 'cement',
            'textiles', 'chemicals', 'agro',
            'infrastructure', 'realty', 'estate',
            'media', 'broadcasting'
        ]

        name_lower = name_str.lower()
        if any(kw in name_lower for kw in valid_keywords):
            return True

        # If it has reasonable length and no numeric-only pattern, might be valid
        if len(name_str) > 5 and not name_str.replace('.', '').replace(',', '').isnumeric():
            return True

        return False

    def get_stock_name_from_yahoo(self, ticker: str) -> str:
        """
        Get proper stock name from Yahoo Finance

        Args:
            ticker: Yahoo Finance ticker symbol

        Returns:
            Stock name or None if not found
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info:
                return None

            # Try to get the long name (most descriptive)
            name = info.get('longName')
            if name and self.is_valid_stock_name(name):
                return name.strip()

            # Fallback to short name
            name = info.get('shortName')
            if name and self.is_valid_stock_name(name):
                return name.strip()

            # Fallback to company name
            name = info.get('companyName')
            if name and self.is_valid_stock_name(name):
                return name.strip()

            return None

        except Exception as e:
            return None

    def get_stock_sector_info(self, ticker: str) -> Dict:
        """
        Extract sector and industry information from Yahoo Finance

        Args:
            ticker: Yahoo Finance ticker symbol

        Returns:
            Dictionary with sector and industry info
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info:
                return {
                    'sector': 'Other',
                    'industry': 'Other'
                }

            # Extract sector and industry from Yahoo Finance
            sector = info.get('sector', 'Other')
            industry = info.get('industry', 'Other')

            # Handle None/missing values
            sector = sector if sector else 'Other'
            industry = industry if industry else 'Other'

            return {
                'sector': sector,
                'industry': industry
            }

        except Exception as e:
            # Return default on error
            return {
                'sector': 'Other',
                'industry': 'Other'
            }

    def get_stock_info(self, ticker: str) -> Dict:
        """
        Get current stock information from yfinance (including sector)

        Args:
            ticker: Yahoo Finance ticker symbol

        Returns:
            Dictionary with stock info and sector classification
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or 'symbol' not in info:
                return None

            # Extract relevant metrics
            metrics_data = {}
            for metric in self.metrics:
                metrics_data[metric] = info.get(metric, np.nan)

            metrics_data['ticker'] = ticker
            metrics_data['fetch_date'] = datetime.now().strftime('%Y-%m-%d')

            # Add sector information (NEW)
            sector_info = self.get_stock_sector_info(ticker)
            metrics_data['sector'] = sector_info['sector']
            metrics_data['industry'] = sector_info['industry']

            return metrics_data

        except Exception as e:
            # Silently skip - expected for many stocks
            return None

    def collect_fundamentals_for_stocks(
        self,
        master_df: pd.DataFrame,
        ticker_mapping: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        Collect fundamental data for stocks in the portfolio over time

        Args:
            master_df: Master portfolio dataframe with ISIN and Date columns
            ticker_mapping: Dictionary mapping ISIN to Yahoo ticker (optional)

        Returns:
            DataFrame with fundamental data by ISIN and Date
        """
        if ticker_mapping is None:
            ticker_mapping = self.load_ticker_mapping()

        if not ticker_mapping:
            print("❌ No ticker mapping available. Cannot fetch fundamental data.")
            print("Run: python create_isin_ticker_mapping.py")
            return pd.DataFrame()

        # Get unique ISINs and dates from master data
        unique_isins = master_df['ISIN'].unique()
        dates = pd.to_datetime(master_df['Date']).dt.strftime('%Y-%m-%d').unique()
        dates = sorted(dates)

        # Filter ISINs that have ticker mappings
        mapped_isins = [isin for isin in unique_isins if isin in ticker_mapping]

        print(f"Total unique ISINs: {len(unique_isins)}")
        print(f"ISINs with ticker mapping: {len(mapped_isins)}")
        print(f"Coverage: {len(mapped_isins)/len(unique_isins)*100:.1f}%")

        cache_key = f"fundamentals_{len(mapped_isins)}_{dates[0]}_{dates[-1]}"

        # Try loading from cache
        if self.cache_enabled:
            cached_data = load_from_cache(cache_key, CACHE_DIR)
            if cached_data is not None:
                print(f"✓ Loaded fundamental data from cache")
                return cached_data

        log_step(f"Collecting fundamental data for {len(mapped_isins)} stocks")

        all_fundamentals = []
        success_count = 0
        fail_count = 0

        for isin in tqdm(mapped_isins, desc="Fetching fundamentals"):
            ticker = ticker_mapping[isin]

            # Fetch data for this stock
            data = self.get_stock_info(ticker)

            if data:
                # Replicate data for each date in the portfolio
                for date in dates:
                    row = data.copy()
                    row['ISIN'] = isin
                    row['Date'] = date
                    all_fundamentals.append(row)
                success_count += 1
            else:
                fail_count += 1

            # Rate limiting to avoid overwhelming Yahoo Finance
            time.sleep(0.2)

        df = pd.DataFrame(all_fundamentals)

        print(f"\n✓ Successfully fetched: {success_count} stocks")
        print(f"✗ Failed to fetch: {fail_count} stocks")

        # Cache results
        if self.cache_enabled and not df.empty:
            save_to_cache(df, cache_key, CACHE_DIR)

        return df

    def collect_sector_mapping(
        self,
        master_df: pd.DataFrame,
        ticker_mapping: Dict[str, str] = None
    ) -> Dict[str, str]:
        """
        Collect sector mapping for all stocks in the portfolio

        Args:
            master_df: Master portfolio dataframe with ISIN column
            ticker_mapping: Dictionary mapping ISIN to Yahoo ticker (optional)

        Returns:
            Dictionary mapping ISIN to Sector
        """
        if ticker_mapping is None:
            ticker_mapping = self.load_ticker_mapping()

        if not ticker_mapping:
            print("❌ No ticker mapping available. Cannot fetch sector data.")
            return {}

        # Get unique ISINs from master data
        unique_isins = master_df['ISIN'].unique()

        # Filter ISINs that have ticker mappings
        mapped_isins = [isin for isin in unique_isins if isin in ticker_mapping]

        print(f"Total unique ISINs: {len(unique_isins)}")
        print(f"ISINs with ticker mapping: {len(mapped_isins)}")

        cache_key = f"sector_mapping_{len(mapped_isins)}"

        # Try loading from cache
        if self.cache_enabled:
            cached_mapping = load_from_cache(cache_key, CACHE_DIR)
            if cached_mapping is not None:
                print(f"✓ Loaded sector mapping from cache")
                return cached_mapping

        log_step(f"Collecting sector data for {len(mapped_isins)} stocks")

        sector_mapping = {}
        success_count = 0
        fail_count = 0

        for isin in tqdm(mapped_isins, desc="Fetching sector data"):
            ticker = ticker_mapping[isin]

            # Fetch sector data
            sector_info = self.get_stock_sector_info(ticker)
            sector = sector_info.get('sector', 'Other')

            if sector and sector != 'Other':
                sector_mapping[isin] = sector
                success_count += 1
            else:
                # Default to 'Other' if sector not found
                sector_mapping[isin] = 'Other'
                fail_count += 1

            # Rate limiting to avoid overwhelming Yahoo Finance
            time.sleep(0.1)

        print(f"\n✓ Successfully mapped: {success_count} stocks")
        print(f"⚠ Defaulted to 'Other': {fail_count} stocks")

        # Cache results
        if self.cache_enabled:
            save_to_cache(sector_mapping, cache_key, CACHE_DIR)

        return sector_mapping



def main():
    """Test fundamental data and sector collection"""
    log_step("Testing Fundamental Data Collector with Sector Information")

    from config import MASTER_CSV_PATH
    from utils.helpers import load_master_data

    # Load master data
    master_df = load_master_data(MASTER_CSV_PATH)

    # Initialize collector
    collector = FundamentalDataCollector()

    # Step 1: Collect sector mapping (NEW)
    print("\n" + "="*80)
    print("STEP 1: COLLECTING SECTOR MAPPING")
    print("="*80)
    sector_mapping = collector.collect_sector_mapping(master_df)

    if sector_mapping:
        print(f"\n✓ Sector mapping collected for {len(sector_mapping)} ISINs")
        # Show sample sectors
        sample_sectors = dict(list(sector_mapping.items())[:10])
        print("\nSample sector mappings:")
        for isin, sector in sample_sectors.items():
            print(f"  {isin}: {sector}")

        # Count sectors
        sector_counts = pd.Series(sector_mapping).value_counts()
        print(f"\nSector distribution:")
        for sector, count in sector_counts.items():
            print(f"  {sector}: {count} stocks")

        # Save sector mapping to JSON
        sector_mapping_path = Path(__file__).parent.parent.parent / "data" / "isin_sector_mapping.json"
        with open(sector_mapping_path, 'w') as f:
            json.dump(sector_mapping, f, indent=2)
        print(f"\n✓ Sector mapping saved to: {sector_mapping_path}")
    else:
        print("\n⚠ No sector mapping collected. Check ticker mapping.")

    # Step 2: Collect fundamental data (with sectors now included)
    print("\n" + "="*80)
    print("STEP 2: COLLECTING FUNDAMENTAL DATA")
    print("="*80)
    df_fundamentals = collector.collect_fundamentals_for_stocks(master_df)

    if df_fundamentals.empty:
        print("\n❌ No fundamental data collected. Check ticker mapping.")
        return

    print("\n" + "="*80)
    print("FUNDAMENTAL DATA SAMPLE (with sectors)")
    print("="*80)
    # Show sample with sector info
    cols_to_show = ['ISIN', 'ticker', 'sector', 'industry', 'trailingPE', 'returnOnEquity']
    available_cols = [col for col in cols_to_show if col in df_fundamentals.columns]
    print(df_fundamentals[available_cols].head(20))
    print(f"\nShape: {df_fundamentals.shape}")
    print(f"Date range: {df_fundamentals['Date'].min()} to {df_fundamentals['Date'].max()}")
    print(f"Unique stocks: {df_fundamentals['ISIN'].nunique()}")
    print(f"Unique sectors: {df_fundamentals['sector'].nunique()}")

    # Save
    output_path = PROCESSED_DATA_DIR / "fundamental_data.csv"
    df_fundamentals.to_csv(output_path, index=False)
    print(f"\n✓ Fundamental data saved to: {output_path}")

    # Show statistics
    print("\n" + "="*80)
    print("SECTOR DISTRIBUTION IN FUNDAMENTAL DATA")
    print("="*80)
    if 'sector' in df_fundamentals.columns:
        sector_dist = df_fundamentals.groupby('sector')['ISIN'].nunique()
        for sector, count in sector_dist.items():
            pct = count / df_fundamentals['ISIN'].nunique() * 100
            print(f"  {sector}: {count} stocks ({pct:.1f}%)")

    print("\n" + "="*80)
    print("FUNDAMENTAL METRICS STATISTICS")
    print("="*80)
    numeric_cols = df_fundamentals.select_dtypes(include=[np.number]).columns
    print(df_fundamentals[numeric_cols].describe())


if __name__ == "__main__":
    main()
