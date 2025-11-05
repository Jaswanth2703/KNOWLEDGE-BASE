"""
Macroeconomic Indicators Collection
Fetches market-wide indicators using yfinance and other sources
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import MACRO_INDICATORS, CACHE_DIR, PROCESSED_DATA_DIR, START_DATE, END_DATE
from utils.helpers import save_to_cache, load_from_cache, log_step


class MacroDataCollector:
    """
    Collects macroeconomic indicators
    - Market indices (NIFTY50, VIX)
    - Currency rates (USD/INR)
    - Interest rate proxies
    """

    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self.indicators = MACRO_INDICATORS
        log_step("Initializing Macroeconomic Data Collector")
        print(f"✓ Indicators to collect: {list(self.indicators.keys())}")

    def fetch_indicator_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical data for a single indicator

        Args:
            ticker: Yahoo Finance ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with historical data
        """
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )

            if data.empty:
                print(f"No data found for {ticker}")
                return pd.DataFrame()

            # Keep only close price
            df = data[['Close']].copy()
            df.reset_index(inplace=True)
            df.columns = ['Date', 'value']
            df['indicator'] = ticker

            return df

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()

    def collect_all_indicators(
        self,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Collect all macroeconomic indicators

        Args:
            start_date: Start date (default from config)
            end_date: End date (default from config)

        Returns:
            DataFrame with all indicators
        """
        start_date = start_date or START_DATE
        end_date = end_date or END_DATE

        cache_key = f"macro_data_{start_date}_{end_date}"

        # Try loading from cache
        if self.cache_enabled:
            cached_data = load_from_cache(cache_key, CACHE_DIR)
            if cached_data is not None:
                print(f"✓ Loaded macro data from cache")
                return cached_data

        log_step(f"Collecting macroeconomic data from {start_date} to {end_date}")

        all_data = []
        success_count = 0
        fail_count = 0

        for name, ticker in tqdm(self.indicators.items(), desc="Fetching indicators"):
            print(f"\nFetching: {name} ({ticker})")
            df = self.fetch_indicator_data(ticker, start_date, end_date)

            if not df.empty:
                df['indicator_name'] = name
                all_data.append(df)
                success_count += 1
                print(f"  ✓ Success: {len(df)} records")
            else:
                fail_count += 1
                print(f"  ✗ Failed or no data")

        if not all_data:
            print("\n❌ No macro data collected! Check tickers and internet connection.")
            return pd.DataFrame()

        print(f"\n✓ Successfully fetched: {success_count}/{len(self.indicators)} indicators")
        print(f"✗ Failed to fetch: {fail_count}/{len(self.indicators)} indicators")

        # Combine all indicators
        df_combined = pd.concat(all_data, ignore_index=True)
        df_combined['Date'] = pd.to_datetime(df_combined['Date'])

        # Pivot to wide format
        df_wide = df_combined.pivot(
            index='Date',
            columns='indicator_name',
            values='value'
        ).reset_index()

        # Resample to monthly (align with portfolio data)
        df_wide.set_index('Date', inplace=True)
        df_monthly = df_wide.resample('M').last()  # Use last value of month
        df_monthly.reset_index(inplace=True)

        # Calculate returns and changes
        for col in df_monthly.columns:
            if col != 'Date':
                df_monthly[f'{col}_return'] = df_monthly[col].pct_change()
                df_monthly[f'{col}_change'] = df_monthly[col].diff()

        # Cache results
        if self.cache_enabled and not df_monthly.empty:
            save_to_cache(df_monthly, cache_key, CACHE_DIR)

        return df_monthly



def main():
    """Test macro data collection"""
    log_step("Testing Macroeconomic Data Collector")

    from config import START_DATE, END_DATE

    # Initialize collector
    collector = MacroDataCollector()

    # Collect real data
    print("\nFetching real macroeconomic data...")
    df_macro = collector.collect_all_indicators(START_DATE, END_DATE)

    if df_macro.empty:
        print("\n❌ Failed to fetch macroeconomic data. Check internet connection and ticker symbols.")
        return

    print("\n" + "="*80)
    print("MACROECONOMIC DATA SAMPLE")
    print("="*80)
    print(df_macro.head(20))
    print(f"\nShape: {df_macro.shape}")
    print(f"Columns: {df_macro.columns.tolist()}")
    print(f"Date range: {df_macro['Date'].min()} to {df_macro['Date'].max()}")

    # Save
    output_path = PROCESSED_DATA_DIR / "macro_data.csv"
    df_macro.to_csv(output_path, index=False)
    print(f"\n✓ Macro data saved to: {output_path}")

    # Show statistics
    print("\n" + "="*80)
    print("MACRO INDICATORS STATISTICS")
    print("="*80)
    numeric_cols = df_macro.select_dtypes(include=[np.number]).columns
    print(df_macro[numeric_cols].describe())


if __name__ == "__main__":
    main()
