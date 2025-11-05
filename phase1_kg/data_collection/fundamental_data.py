"""
Fundamental Data Collection using yfinance
Fetches company financials and valuation metrics
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

    def get_stock_info(self, ticker: str) -> Dict:
        """
        Get current stock information from yfinance

        Args:
            ticker: Yahoo Finance ticker symbol

        Returns:
            Dictionary with stock info
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



def main():
    """Test fundamental data collection"""
    log_step("Testing Fundamental Data Collector")

    from config import MASTER_CSV_PATH
    from utils.helpers import load_master_data

    # Load master data
    master_df = load_master_data(MASTER_CSV_PATH)

    # Initialize collector
    collector = FundamentalDataCollector()

    # Collect real fundamental data
    df_fundamentals = collector.collect_fundamentals_for_stocks(master_df)

    if df_fundamentals.empty:
        print("\n❌ No fundamental data collected. Check ticker mapping.")
        return

    print("\n" + "="*80)
    print("FUNDAMENTAL DATA SAMPLE")
    print("="*80)
    print(df_fundamentals.head(20))
    print(f"\nShape: {df_fundamentals.shape}")
    print(f"Date range: {df_fundamentals['Date'].min()} to {df_fundamentals['Date'].max()}")
    print(f"Unique stocks: {df_fundamentals['ISIN'].nunique()}")

    # Save
    output_path = PROCESSED_DATA_DIR / "fundamental_data.csv"
    df_fundamentals.to_csv(output_path, index=False)
    print(f"\n✓ Fundamental data saved to: {output_path}")

    # Show statistics
    print("\n" + "="*80)
    print("FUNDAMENTAL METRICS STATISTICS")
    print("="*80)
    numeric_cols = df_fundamentals.select_dtypes(include=[np.number]).columns
    print(df_fundamentals[numeric_cols].describe())


if __name__ == "__main__":
    main()
