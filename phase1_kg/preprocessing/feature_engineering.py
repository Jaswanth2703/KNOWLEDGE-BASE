"""
Feature Engineering for Knowledge Graph Construction
- Portfolio change features
- Relative performance metrics
- Temporal features
- Integration of all data sources
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import MASTER_CSV_PATH, PROCESSED_DATA_DIR, CACHE_DIR
from utils.helpers import (
    load_master_data, calculate_portfolio_changes,
    save_to_cache, load_from_cache, log_step
)


class FeatureEngineer:
    """
    Engineer features from raw data for knowledge graph construction
    """

    def __init__(self):
        log_step("Initializing Feature Engineer")

    def calculate_portfolio_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate portfolio-level metrics

        Args:
            df: Portfolio holdings dataframe

        Returns:
            DataFrame with additional portfolio metrics
        """
        log_step("Calculating portfolio metrics")

        # Sort by fund, date, and weight
        df = df.sort_values(['Fund_Name', 'Date', 'Portfolio_Weight'], ascending=[True, True, False])

        # Calculate portfolio changes
        df = calculate_portfolio_changes(df)

        # Calculate relative metrics per fund per month
        df['rank_in_portfolio'] = df.groupby(['Fund_Name', 'Date'])['Portfolio_Weight'].rank(
            ascending=False, method='dense'
        )

        df['weight_percentile'] = df.groupby(['Fund_Name', 'Date'])['Portfolio_Weight'].rank(
            pct=True
        )

        # Calculate concentration metrics
        df['is_top_10'] = df['rank_in_portfolio'] <= 10
        df['is_top_20'] = df['rank_in_portfolio'] <= 20

        # Number of holdings per fund per month
        holdings_count = df.groupby(['Fund_Name', 'Date']).size().reset_index(name='total_holdings')
        df = df.merge(holdings_count, on=['Fund_Name', 'Date'], how='left')

        print(f"✓ Portfolio metrics calculated")
        return df

    def add_sector_information(self, df: pd.DataFrame, sector_map: Dict[str, str] = None) -> pd.DataFrame:
        """
        Add sector information to stocks

        Args:
            df: Portfolio dataframe
            sector_map: Dictionary mapping ISIN to sector

        Returns:
            DataFrame with sector information
        """
        log_step("Adding sector information")

        if sector_map is None:
            # Create a simple sector mapping based on Sheet_Name patterns
            # In production, use actual sector data
            def infer_sector(row):
                sheet = str(row['Sheet_Name']).upper()
                # This is a placeholder - in production, map ISINs to actual sectors
                if 'BANK' in sheet or 'FINANC' in sheet:
                    return 'Financial Services'
                elif 'IT' in sheet or 'TECH' in sheet:
                    return 'IT'
                elif 'PHARMA' in sheet or 'HEALTH' in sheet:
                    return 'Pharma'
                elif 'AUTO' in sheet:
                    return 'Automobile'
                elif 'ENERGY' in sheet:
                    return 'Energy'
                else:
                    return 'Other'

            df['sector'] = df.apply(infer_sector, axis=1)
        else:
            df['sector'] = df['ISIN'].map(sector_map)

        print(f"✓ Sector information added. Unique sectors: {df['sector'].nunique()}")
        return df

    def calculate_sector_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sector-level aggregations

        Args:
            df: Portfolio dataframe with sector information

        Returns:
            DataFrame with sector metrics
        """
        log_step("Calculating sector metrics")

        # Sector weight per fund per month
        sector_weights = df.groupby(['Fund_Name', 'Date', 'sector'])['Portfolio_Weight'].sum().reset_index()
        sector_weights.rename(columns={'Portfolio_Weight': 'sector_weight'}, inplace=True)

        df = df.merge(sector_weights, on=['Fund_Name', 'Date', 'sector'], how='left')

        # Sector concentration
        df['sector_concentration'] = df['Portfolio_Weight'] / df['sector_weight']

        print(f"✓ Sector metrics calculated")
        return df

    def calculate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate temporal features

        Args:
            df: Portfolio dataframe

        Returns:
            DataFrame with temporal features
        """
        log_step("Calculating temporal features")

        df['Date'] = pd.to_datetime(df['Date'])

        # Time-based features
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter

        # Time index (months since start)
        min_date = df['Date'].min()
        df['months_since_start'] = ((df['Date'].dt.year - min_date.year) * 12 +
                                     (df['Date'].dt.month - min_date.month))

        # Holding duration
        df = df.sort_values(['Fund_Name', 'ISIN', 'Date'])
        df['months_held'] = df.groupby(['Fund_Name', 'ISIN']).cumcount() + 1

        print(f"✓ Temporal features calculated")
        return df

    def integrate_all_data(
        self,
        portfolio_df: pd.DataFrame,
        fundamental_df: pd.DataFrame = None,
        macro_df: pd.DataFrame = None,
        sentiment_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Integrate all data sources into a unified dataset

        Args:
            portfolio_df: Portfolio holdings data
            fundamental_df: Fundamental metrics data
            macro_df: Macroeconomic indicators
            sentiment_df: Sentiment analysis data

        Returns:
            Integrated DataFrame
        """
        log_step("Integrating all data sources")

        df = portfolio_df.copy()
        df['Date'] = pd.to_datetime(df['Date'])

        # Merge fundamental data
        if fundamental_df is not None and not fundamental_df.empty:
            fundamental_df['Date'] = pd.to_datetime(fundamental_df['Date'])
            # Merge on ISIN and Date
            df = df.merge(
                fundamental_df,
                on=['ISIN', 'Date'],
                how='left',
                suffixes=('', '_fund')
            )
            print(f"✓ Fundamental data merged ({len(fundamental_df)} records)")
        else:
            print(f"⚠️  No fundamental data to merge")

        # Merge macroeconomic data
        if macro_df is not None and not macro_df.empty:
            macro_df['Date'] = pd.to_datetime(macro_df['Date'])
            # Merge on Date only (macro data is not stock-specific)
            df = df.merge(
                macro_df,
                on='Date',
                how='left',
                suffixes=('', '_macro')
            )
            print(f"✓ Macroeconomic data merged ({len(macro_df)} records)")
        else:
            print(f"⚠️  No macroeconomic data to merge")

        # Merge sentiment data
        if sentiment_df is not None and not sentiment_df.empty:
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            # Merge on sector and date
            df = df.merge(
                sentiment_df,
                left_on=['sector', 'Date'],
                right_on=['sector', 'date'],
                how='left',
                suffixes=('', '_sentiment')
            )
            print(f"✓ Sentiment data merged ({len(sentiment_df)} records)")
        else:
            print(f"⚠️  No sentiment data to merge")

        print(f"✓ Data integration complete. Final shape: {df.shape}")
        return df

    def create_integrated_dataset(self) -> pd.DataFrame:
        """
        Main function to create the complete integrated dataset
        Loads all data sources and performs feature engineering

        Returns:
            Integrated DataFrame ready for KG construction
        """
        log_step("Creating Integrated Dataset for Knowledge Graph Construction")

        # Load master portfolio data
        df_portfolio = load_master_data(MASTER_CSV_PATH)
        print(f"✓ Loaded portfolio data: {df_portfolio.shape}")

        # Add sector information
        df_portfolio = self.add_sector_information(df_portfolio)

        # Calculate portfolio metrics and changes
        df_portfolio = self.calculate_portfolio_metrics(df_portfolio)

        # Calculate sector metrics
        df_portfolio = self.calculate_sector_metrics(df_portfolio)

        # Calculate temporal features
        df_portfolio = self.calculate_temporal_features(df_portfolio)

        # Load other data sources if available
        fundamental_path = PROCESSED_DATA_DIR / "fundamental_data.csv"
        macro_path = PROCESSED_DATA_DIR / "macro_data.csv"
        sentiment_path = PROCESSED_DATA_DIR / "sentiment_data.csv"

        df_fundamental = pd.read_csv(fundamental_path) if fundamental_path.exists() else None
        df_macro = pd.read_csv(macro_path) if macro_path.exists() else None
        df_sentiment = pd.read_csv(sentiment_path) if sentiment_path.exists() else None

        # Integrate all data
        df_integrated = self.integrate_all_data(
            portfolio_df=df_portfolio,
            fundamental_df=df_fundamental,
            macro_df=df_macro,
            sentiment_df=df_sentiment
        )

        # Save integrated dataset
        output_path = PROCESSED_DATA_DIR / "integrated_dataset.csv"
        df_integrated.to_csv(output_path, index=False)
        print(f"\n✓ Integrated dataset saved to: {output_path}")

        return df_integrated


def main():
    """Test feature engineering"""
    log_step("Testing Feature Engineering")

    engineer = FeatureEngineer()

    # Create integrated dataset
    df = engineer.create_integrated_dataset()

    print("\n" + "="*80)
    print("INTEGRATED DATASET SAMPLE")
    print("="*80)
    print(df.head(20))
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")

    # Show some statistics
    print("\n" + "="*80)
    print("PORTFOLIO CHANGE STATISTICS")
    print("="*80)
    print(df['action'].value_counts())

    print("\n" + "="*80)
    print("SECTOR DISTRIBUTION")
    print("="*80)
    print(df.groupby('sector')['Portfolio_Weight'].sum().sort_values(ascending=False))

    print("\n" + "="*80)
    print("TOP 10 HOLDINGS BY WEIGHT")
    print("="*80)
    latest_date = df['Date'].max()
    top_holdings = df[df['Date'] == latest_date].nlargest(10, 'Portfolio_Weight')
    print(top_holdings[['Fund_Name', 'Stock_Name', 'sector', 'Portfolio_Weight']])


if __name__ == "__main__":
    main()
