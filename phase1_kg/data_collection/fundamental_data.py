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
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import FUNDAMENTAL_METRICS, CACHE_DIR, PROCESSED_DATA_DIR
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

    def isin_to_yahoo_ticker(self, isin: str) -> str:
        """
        Convert ISIN to Yahoo Finance ticker
        Indian stocks: ISIN INE123456789 -> *.NS or *.BO

        Note: This is a simplification. In production, use a proper mapping database
        """
        if isin.startswith('INE'):
            # Extract numeric portion and add .NS suffix for NSE
            # This is a heuristic - actual mapping requires a lookup table
            # For now, we'll try both .NS (NSE) and .BO (BSE)
            return None  # Return None to indicate need for manual mapping

        return isin

    def get_stock_info(self, ticker: str) -> Dict:
        """
        Get stock information from yfinance

        Args:
            ticker: Yahoo Finance ticker symbol

        Returns:
            Dictionary with stock info
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Extract relevant metrics
            metrics_data = {}
            for metric in self.metrics:
                metrics_data[metric] = info.get(metric, np.nan)

            metrics_data['ticker'] = ticker
            metrics_data['fetch_date'] = datetime.now().strftime('%Y-%m-%d')

            return metrics_data

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

    def collect_fundamentals_for_stocks(
        self,
        isin_list: List[str],
        ticker_mapping: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        Collect fundamental data for a list of stocks

        Args:
            isin_list: List of ISIN codes
            ticker_mapping: Dictionary mapping ISIN to Yahoo ticker (optional)

        Returns:
            DataFrame with fundamental data
        """
        cache_key = f"fundamentals_{len(isin_list)}_stocks"

        # Try loading from cache
        if self.cache_enabled:
            cached_data = load_from_cache(cache_key, CACHE_DIR)
            if cached_data is not None:
                return cached_data

        log_step(f"Collecting fundamental data for {len(isin_list)} stocks")

        all_fundamentals = []

        for isin in tqdm(isin_list, desc="Fetching fundamentals"):
            # Get ticker from mapping or convert
            if ticker_mapping and isin in ticker_mapping:
                ticker = ticker_mapping[isin]
            else:
                # Skip if no mapping available
                # In production, use proper ISIN-to-ticker database
                continue

            # Fetch data
            data = self.get_stock_info(ticker)

            if data:
                data['ISIN'] = isin
                all_fundamentals.append(data)

            # Rate limiting
            time.sleep(0.1)

        df = pd.DataFrame(all_fundamentals)

        # Cache results
        if self.cache_enabled and not df.empty:
            save_to_cache(df, cache_key, CACHE_DIR)

        return df

    def create_synthetic_fundamentals(
        self,
        master_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create synthetic fundamental data for demonstration purposes
        In production, replace with actual data fetching

        Args:
            master_df: Master portfolio dataframe

        Returns:
            DataFrame with synthetic fundamental metrics
        """
        log_step("Creating synthetic fundamental data (for demonstration)")

        unique_isins = master_df['ISIN'].unique()
        dates = pd.date_range(start=master_df['Date'].min(),
                               end=master_df['Date'].max(),
                               freq='M')

        fundamentals = []

        for isin in tqdm(unique_isins, desc="Generating synthetic data"):
            # Create time series for each stock
            np.random.seed(hash(isin) % (2**32))

            for date in dates:
                fundamentals.append({
                    'ISIN': isin,
                    'Date': date,
                    'trailingPE': np.random.uniform(10, 40),
                    'priceToBook': np.random.uniform(1, 10),
                    'returnOnEquity': np.random.uniform(0.05, 0.30),
                    'revenueGrowth': np.random.uniform(-0.10, 0.50),
                    'debtToEquity': np.random.uniform(0, 2),
                    'profitMargins': np.random.uniform(0.02, 0.25),
                    'marketCap': np.random.uniform(1e9, 1e12),
                    'beta': np.random.uniform(0.5, 2.0)
                })

        df = pd.DataFrame(fundamentals)
        df['Date'] = pd.to_datetime(df['Date'])

        # Add some temporal correlation (values change slowly)
        df = df.sort_values(['ISIN', 'Date'])
        for metric in FUNDAMENTAL_METRICS:
            df[metric] = df.groupby('ISIN')[metric].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )

        return df


def load_isin_ticker_mapping() -> Dict[str, str]:
    """
    Load ISIN to Yahoo ticker mapping
    In production, this should load from a database or CSV file

    Returns:
        Dictionary mapping ISIN to ticker
    """
    # TODO: Implement actual mapping
    # For now, return empty dict to use synthetic data
    return {}


def main():
    """Test fundamental data collection"""
    log_step("Testing Fundamental Data Collector")

    from config import MASTER_CSV_PATH
    from utils.helpers import load_master_data

    # Load master data
    master_df = load_master_data(MASTER_CSV_PATH)

    # Initialize collector
    collector = FundamentalDataCollector()

    # Create synthetic data (for demonstration)
    df_fundamentals = collector.create_synthetic_fundamentals(master_df)

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
    print(df_fundamentals[FUNDAMENTAL_METRICS].describe())


if __name__ == "__main__":
    main()
