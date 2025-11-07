"""
Temporal Knowledge Graph Construction
Models portfolio evolution through time-stamped relationships
"""
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import pickle
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import TEMPORAL_KG_CONFIG, GRAPHS_DIR, PROCESSED_DATA_DIR
from utils.helpers import log_step
import pickle

class TemporalKG:
    """
    Temporal Knowledge Graph for Fund Manager Decisions

    Nodes:
    - Fund: Mutual fund entities
    - Stock: Individual stocks (ISIN)
    - Sector: Industry sectors
    - TimePeriod: Monthly time periods

    Edges:
    - HOLDS: Fund holds stock at time T with weight
    - INCREASED/DECREASED: Change in position
    - ENTERED/EXITED: New position or position closure
    - BELONGS_TO_SECTOR: Stock sector membership
    """

    def __init__(self, config: Dict = None):
        self.config = config or TEMPORAL_KG_CONFIG
        self.graph = nx.MultiDiGraph()  # Allow multiple edges between nodes
        self.change_threshold = self.config['change_threshold']
        log_step("Initializing Temporal Knowledge Graph")
        print(f"✓ Change threshold: {self.change_threshold}")

    def add_fund_nodes(self, df: pd.DataFrame) -> None:
        """Add fund nodes to the graph"""
        funds = df['Fund_Name'].unique()

        for fund in funds:
            fund_data = df[df['Fund_Name'] == fund]

            self.graph.add_node(
                fund,
                node_type='Fund',
                fund_type=fund_data['Fund_Type'].iloc[0],
                first_date=fund_data['Date'].min(),
                last_date=fund_data['Date'].max(),
                total_holdings=fund_data['ISIN'].nunique()
            )

        print(f"✓ Added {len(funds)} fund nodes")

    def add_stock_nodes(self, df: pd.DataFrame) -> None:
        """Add stock nodes to the graph"""
        stocks = df[['ISIN', 'Stock_Name', 'sector']].drop_duplicates()

        for _, row in stocks.iterrows():
            isin = row['ISIN']
            stock_name = row['Stock_Name'] if pd.notna(row['Stock_Name']) else isin

            self.graph.add_node(
                isin,
                node_type='Stock',
                stock_name=stock_name,
                sector=row['sector'] if pd.notna(row['sector']) else 'Unknown'
            )

        print(f"✓ Added {len(stocks)} stock nodes")

    def add_sector_nodes(self, df: pd.DataFrame) -> None:
        """Add sector nodes to the graph"""
        sectors = df['sector'].dropna().unique()

        for sector in sectors:
            sector_stocks = df[df['sector'] == sector]['ISIN'].nunique()

            self.graph.add_node(
                f"SECTOR_{sector}",
                node_type='Sector',
                sector_name=sector,
                num_stocks=sector_stocks
            )

        print(f"✓ Added {len(sectors)} sector nodes")

    def add_time_period_nodes(self, df: pd.DataFrame) -> None:
        """Add time period nodes to the graph"""
        dates = df['Date'].unique()

        for date in dates:
            date_str = pd.to_datetime(date).strftime('%Y-%m')

            self.graph.add_node(
                f"TIME_{date_str}",
                node_type='TimePeriod',
                date=date,
                year=pd.to_datetime(date).year,
                month=pd.to_datetime(date).month,
                quarter=pd.to_datetime(date).quarter
            )

        print(f"✓ Added {len(dates)} time period nodes")

    def add_sector_membership_edges(self, df: pd.DataFrame) -> None:
        """Add edges connecting stocks to sectors"""
        stock_sectors = df[['ISIN', 'sector']].dropna().drop_duplicates()

        for _, row in stock_sectors.iterrows():
            self.graph.add_edge(
                row['ISIN'],
                f"SECTOR_{row['sector']}",
                edge_type='BELONGS_TO_SECTOR',
                sector=row['sector']
            )

        print(f"✓ Added {len(stock_sectors)} sector membership edges")

    def add_holding_edges(self, df: pd.DataFrame) -> None:
        """Add HOLDS edges for fund-stock holdings"""
        holdings = df[['Fund_Name', 'ISIN', 'Date', 'Portfolio_Weight', 'Market_Value']].dropna(subset=['Portfolio_Weight'])

        edge_count = 0
        for _, row in holdings.iterrows():
            date_str = pd.to_datetime(row['Date']).strftime('%Y-%m')

            self.graph.add_edge(
                row['Fund_Name'],
                row['ISIN'],
                key=f"HOLDS_{date_str}",  # Unique key for MultiDiGraph
                edge_type='HOLDS',
                date=row['Date'],
                date_str=date_str,
                weight=row['Portfolio_Weight'],
                market_value=row['Market_Value'] if pd.notna(row['Market_Value']) else 0
            )
            edge_count += 1

        print(f"✓ Added {edge_count} holding edges")

    def add_change_edges(self, df: pd.DataFrame) -> None:
        """Add INCREASED/DECREASED/ENTERED/EXITED edges based on portfolio changes"""
        # Filter for meaningful changes
        changes = df[df['action'].isin(['INCREASED', 'DECREASED', 'ENTERED', 'EXITED'])].copy()

        edge_count = 0
        for _, row in changes.iterrows():
            if pd.isna(row['Fund_Name']) or pd.isna(row['ISIN']):
                continue

            date_str = pd.to_datetime(row['Date']).strftime('%Y-%m')
            action = row['action']

            # Only add edge if change is above threshold (except for ENTERED/EXITED)
            if action in ['INCREASED', 'DECREASED']:
                if abs(row['weight_change']) < self.change_threshold:
                    continue

            self.graph.add_edge(
                row['Fund_Name'],
                row['ISIN'],
                key=f"{action}_{date_str}",
                edge_type=action,
                date=row['Date'],
                date_str=date_str,
                weight_change=row['weight_change'] if pd.notna(row['weight_change']) else 0,
                weight_change_pct=row['weight_change_pct'] if pd.notna(row['weight_change_pct']) else 0,
                prev_weight=row['prev_weight'] if pd.notna(row['prev_weight']) else 0,
                new_weight=row['Portfolio_Weight'] if pd.notna(row['Portfolio_Weight']) else 0
            )
            edge_count += 1

        print(f"✓ Added {edge_count} change edges")

    def build_from_dataframe(self, df: pd.DataFrame) -> nx.MultiDiGraph:
        """
        Build complete temporal knowledge graph from integrated dataset

        Args:
            df: Integrated dataset with portfolio and additional features

        Returns:
            NetworkX MultiDiGraph
        """
        log_step("Building Temporal Knowledge Graph")

        # Add nodes
        self.add_fund_nodes(df)
        self.add_stock_nodes(df)
        self.add_sector_nodes(df)
        self.add_time_period_nodes(df)

        # Add edges
        self.add_sector_membership_edges(df)
        self.add_holding_edges(df)
        self.add_change_edges(df)

        # Print summary
        print(f"\n{'='*80}")
        print("TEMPORAL KG CONSTRUCTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total nodes: {self.graph.number_of_nodes()}")
        print(f"Total edges: {self.graph.number_of_edges()}")
        print(f"\nNode types:")
        for node_type in ['Fund', 'Stock', 'Sector', 'TimePeriod']:
            count = sum(1 for n, d in self.graph.nodes(data=True) if d.get('node_type') == node_type)
            print(f"  {node_type}: {count}")

        print(f"\nEdge types:")
        for edge_type in self.config['relationship_types']:
            count = sum(1 for u, v, d in self.graph.edges(data=True) if d.get('edge_type') == edge_type)
            print(f"  {edge_type}: {count}")

        return self.graph

    def save(self, filename: str = "temporal_kg.gpickle") -> None:
        """Save graph to file"""
        filepath = GRAPHS_DIR / filename
        with open(filepath, 'wb') as f:
            pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\n✓ Temporal KG saved to: {filepath}")

    def load(self, filename: str = "temporal_kg.gpickle") -> nx.MultiDiGraph:
        """Load graph from file"""
        filepath = GRAPHS_DIR / filename
        with open(filepath, 'rb') as f:
            self.graph = pickle.load(f)
        print(f"✓ Temporal KG loaded from: {filepath}")
        return self.graph

    def query_fund_holdings(self, fund_name: str, date: str = None) -> List[Tuple]:
        """
        Query holdings for a specific fund at a specific time

        Args:
            fund_name: Name of the fund
            date: Date string (YYYY-MM), if None returns all

        Returns:
            List of (stock, weight) tuples
        """
        holdings = []

        if fund_name not in self.graph:
            return holdings

        for neighbor in self.graph.neighbors(fund_name):
            # Get all edges between fund and stock
            edges = self.graph.get_edge_data(fund_name, neighbor)

            if edges is None:
                continue

            for key, edge_data in edges.items():
                if edge_data.get('edge_type') == 'HOLDS':
                    if date is None or edge_data.get('date_str') == date:
                        holdings.append((
                            neighbor,
                            edge_data.get('weight', 0),
                            edge_data.get('date_str')
                        ))

        return sorted(holdings, key=lambda x: x[1], reverse=True)

    def query_stock_holders(self, isin: str, date: str = None) -> List[Tuple]:
        """
        Query which funds hold a specific stock

        Args:
            isin: Stock ISIN
            date: Date string (YYYY-MM)

        Returns:
            List of (fund, weight) tuples
        """
        holders = []

        if isin not in self.graph:
            return holders

        # Get predecessors (funds that hold this stock)
        for fund in self.graph.predecessors(isin):
            edges = self.graph.get_edge_data(fund, isin)

            if edges is None:
                continue

            for key, edge_data in edges.items():
                if edge_data.get('edge_type') == 'HOLDS':
                    if date is None or edge_data.get('date_str') == date:
                        holders.append((
                            fund,
                            edge_data.get('weight', 0),
                            edge_data.get('date_str')
                        ))

        return sorted(holders, key=lambda x: x[1], reverse=True)

    def query_portfolio_changes(self, fund_name: str, start_date: str, end_date: str) -> Dict:
        """
        Query portfolio changes for a fund between two dates

        Args:
            fund_name: Name of the fund
            start_date: Start date (YYYY-MM)
            end_date: End date (YYYY-MM)

        Returns:
            Dictionary with change statistics
        """
        changes = {'INCREASED': [], 'DECREASED': [], 'ENTERED': [], 'EXITED': []}

        if fund_name not in self.graph:
            return changes

        for neighbor in self.graph.neighbors(fund_name):
            edges = self.graph.get_edge_data(fund_name, neighbor)

            if edges is None:
                continue

            for key, edge_data in edges.items():
                edge_type = edge_data.get('edge_type')
                date_str = edge_data.get('date_str')

                if edge_type in changes and date_str >= start_date and date_str <= end_date:
                    changes[edge_type].append({
                        'stock': neighbor,
                        'date': date_str,
                        'weight_change': edge_data.get('weight_change', 0),
                        'weight_change_pct': edge_data.get('weight_change_pct', 0)
                    })

        return changes


def main():
    """Test temporal KG construction"""
    log_step("Testing Temporal Knowledge Graph Construction")

    # Load integrated dataset
    data_path = PROCESSED_DATA_DIR / "integrated_dataset.csv"

    if not data_path.exists():
        print(f"Error: Integrated dataset not found at {data_path}")
        print("Please run feature_engineering.py first to create the integrated dataset.")
        return

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"✓ Loaded integrated dataset: {df.shape}")

    # Build temporal KG
    tkg = TemporalKG()
    graph = tkg.build_from_dataframe(df)

    # Save
    tkg.save()

    # Test queries
    print(f"\n{'='*80}")
    print("TESTING QUERIES")
    print(f"{'='*80}")

    # Query 1: Fund holdings
    funds = df['Fund_Name'].unique()[:2]
    for fund in funds:
        print(f"\nHoldings for {fund} in 2024-12:")
        holdings = tkg.query_fund_holdings(fund, "2024-12")
        for stock, weight, date in holdings[:5]:
            stock_name = graph.nodes[stock].get('stock_name', stock)
            print(f"  {stock_name}: {weight:.4f} ({date})")

    # Query 2: Stock holders
    stocks = df['ISIN'].unique()[:2]
    for stock in stocks:
        print(f"\nFunds holding {stock} in 2024-12:")
        holders = tkg.query_stock_holders(stock, "2024-12")
        for fund, weight, date in holders[:5]:
            print(f"  {fund}: {weight:.4f} ({date})")

    print(f"\n{'='*80}")
    print("✓ Temporal KG construction and testing complete!")


if __name__ == "__main__":
    main()
