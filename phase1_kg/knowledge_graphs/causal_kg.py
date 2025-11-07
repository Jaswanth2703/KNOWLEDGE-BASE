"""
Causal Knowledge Graph Construction
Models cause-effect relationships between factors and portfolio decisions
"""
import networkx as nx
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import pearsonr
from typing import Dict, List, Tuple
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import pickle
# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import CAUSAL_KG_CONFIG, GRAPHS_DIR, PROCESSED_DATA_DIR
from utils.helpers import log_step


class CausalKG:
    """
    Causal Knowledge Graph for Fund Manager Decision Rationale

    Nodes:
    - Observable Factors: Macro indicators, sector metrics, stock fundamentals
    - Intermediate Signals: Derived concepts like value scores, risk appetite
    - Portfolio Actions: Allocation decisions (sector weights, stock selections)

    Edges:
    - CAUSES: Direct causal relationship (Granger causality)
    - INFLUENCES: Strong correlation
    - PRECEDES: Temporal precedence
    - CORRELATED_WITH: Statistical correlation
    """

    def __init__(self, config: Dict = None):
        self.config = config or CAUSAL_KG_CONFIG
        self.graph = nx.DiGraph()
        self.max_lag = self.config['granger_max_lag']
        self.significance = self.config['granger_significance']
        self.correlation_threshold = self.config['correlation_threshold']
        self.min_observations = self.config['min_observations']
        log_step("Initializing Causal Knowledge Graph")

    def test_granger_causality(
        self,
        cause_series: pd.Series,
        effect_series: pd.Series,
        max_lag: int = None
    ) -> Tuple[bool, float, int]:
        """
        Test Granger causality between two time series

        Args:
            cause_series: Potential cause variable
            effect_series: Potential effect variable
            max_lag: Maximum lag to test

        Returns:
            (is_causal, min_p_value, best_lag)
        """
        max_lag = max_lag or self.max_lag

        # Remove NaN values
        combined = pd.DataFrame({
            'cause': cause_series,
            'effect': effect_series
        }).dropna()

        if len(combined) < self.min_observations:
            return False, 1.0, 0

        try:
            # Prepare data for Granger test
            data = combined[['effect', 'cause']].values

            # Run Granger causality test
            test_result = grangercausalitytests(data, max_lag, verbose=False)

            # Extract p-values for each lag
            p_values = {}
            for lag in range(1, max_lag + 1):
                # Use F-test p-value
                p_value = test_result[lag][0]['ssr_ftest'][1]
                p_values[lag] = p_value

            # Find minimum p-value
            best_lag = min(p_values, key=p_values.get)
            min_p_value = p_values[best_lag]

            # Is causal if p-value < significance level
            is_causal = min_p_value < self.significance

            return is_causal, min_p_value, best_lag

        except Exception as e:
            # Granger test can fail for various reasons
            return False, 1.0, 0

    def calculate_correlation(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> Tuple[float, float]:
        """
        Calculate Pearson correlation between two series

        Args:
            series1: First series
            series2: Second series

        Returns:
            (correlation, p_value)
        """
        # Remove NaN values
        combined = pd.DataFrame({'s1': series1, 's2': series2}).dropna()

        if len(combined) < self.min_observations:
            return 0.0, 1.0

        try:
            corr, p_value = pearsonr(combined['s1'], combined['s2'])
            return corr, p_value
        except:
            return 0.0, 1.0

    def extract_macro_to_sector_causality(self, df: pd.DataFrame) -> List[Tuple]:
        """
        Extract causal relationships from macro indicators to sector allocations

        Args:
            df: Integrated dataset

        Returns:
            List of (cause, effect, relationship_type, strength, lag) tuples
        """
        log_step("Extracting Macro → Sector causality")

        relationships = []

        # Macro indicators to test
        macro_vars = [col for col in df.columns if any(
            x in col for x in ['NIFTY', 'VIX', 'USD_INR', 'BOND', '_return', '_change']
        )]

        # Get sector allocation time series
        sector_allocations = df.groupby(['Date', 'sector'])['Portfolio_Weight'].sum().reset_index()
        sectors = sector_allocations['sector'].unique()

        for macro_var in macro_vars[:10]:  # Limit for performance
            if macro_var not in df.columns:
                continue

            # Aggregate macro variable by date
            macro_series = df.groupby('Date')[macro_var].mean()

            for sector in sectors:
                # Get sector allocation series
                sector_data = sector_allocations[sector_allocations['sector'] == sector]
                sector_series = sector_data.set_index('Date')['Portfolio_Weight']

                # Align time series
                aligned = pd.DataFrame({
                    'macro': macro_series,
                    'sector': sector_series
                }).dropna()

                if len(aligned) < self.min_observations:
                    continue

                # Test Granger causality
                is_causal, p_value, lag = self.test_granger_causality(
                    aligned['macro'],
                    aligned['sector']
                )

                if is_causal:
                    relationships.append((
                        f"MACRO_{macro_var}",
                        f"SECTOR_{sector}",
                        "CAUSES",
                        1 - p_value,  # Strength: higher is stronger
                        lag
                    ))

                # Also test correlation
                corr, corr_p = self.calculate_correlation(
                    aligned['macro'],
                    aligned['sector']
                )

                if abs(corr) > self.correlation_threshold and corr_p < 0.05:
                    relationships.append((
                        f"MACRO_{macro_var}",
                        f"SECTOR_{sector}",
                        "INFLUENCES",
                        abs(corr),
                        0
                    ))

        print(f"✓ Found {len(relationships)} macro → sector relationships")
        return relationships

    def extract_sector_to_allocation_causality(self, df: pd.DataFrame) -> List[Tuple]:
        """
        Extract causal relationships from sector performance to allocations

        Args:
            df: Integrated dataset

        Returns:
            List of relationships
        """
        log_step("Extracting Sector Performance → Allocation causality")

        relationships = []

        # For each sector, test if sector weight changes are caused by sentiment
        if 'sentiment_score' in df.columns and 'sector_weight' in df.columns:
            sectors = df['sector'].unique()

            for sector in sectors:
                sector_data = df[df['sector'] == sector].copy()

                if len(sector_data) < self.min_observations:
                    continue

                # Aggregate by date
                sector_ts = sector_data.groupby('Date').agg({
                    'sentiment_score': 'mean',
                    'sector_weight': 'mean'
                }).dropna()

                if len(sector_ts) < self.min_observations:
                    continue

                # Test if sentiment causes sector weight changes
                is_causal, p_value, lag = self.test_granger_causality(
                    sector_ts['sentiment_score'],
                    sector_ts['sector_weight']
                )

                if is_causal:
                    relationships.append((
                        f"SENTIMENT_{sector}",
                        f"SECTOR_ALLOCATION_{sector}",
                        "CAUSES",
                        1 - p_value,
                        lag
                    ))

        print(f"✓ Found {len(relationships)} sector → allocation relationships")
        return relationships

    def extract_fundamental_to_selection_causality(self, df: pd.DataFrame) -> List[Tuple]:
        """
        Extract relationships between fundamentals and stock selection

        Args:
            df: Integrated dataset

        Returns:
            List of relationships
        """
        log_step("Extracting Fundamental Metrics → Stock Selection causality")

        relationships = []

        # Fundamental metrics
        fundamental_cols = ['trailingPE', 'priceToBook', 'returnOnEquity',
                           'revenueGrowth', 'debtToEquity', 'profitMargins']

        available_fundamentals = [col for col in fundamental_cols if col in df.columns]

        if not available_fundamentals:
            print("Warning: No fundamental metrics found in dataset")
            return relationships

        # For each stock, test if fundamentals correlate with allocation changes
        stocks = df['ISIN'].unique()[:50]  # Limit for performance

        for stock in stocks:
            stock_data = df[df['ISIN'] == stock].copy()

            if len(stock_data) < self.min_observations:
                continue

            # Sort by date
            stock_data = stock_data.sort_values('Date')

            for fund_metric in available_fundamentals:
                if stock_data[fund_metric].isna().all():
                    continue

                # Test correlation with portfolio weight
                corr, p_value = self.calculate_correlation(
                    stock_data[fund_metric],
                    stock_data['Portfolio_Weight']
                )

                if abs(corr) > self.correlation_threshold and p_value < 0.05:
                    relationships.append((
                        f"FUNDAMENTAL_{fund_metric}",
                        f"STOCK_SELECTION_{stock[:15]}",  # Truncate ISIN for readability
                        "INFLUENCES",
                        abs(corr),
                        0
                    ))

        print(f"✓ Found {len(relationships)} fundamental → selection relationships")
        return relationships

    def add_domain_knowledge_rules(self) -> List[Tuple]:
        """
        Add known causal relationships from financial theory

        Returns:
            List of relationships
        """
        log_step("Adding domain knowledge rules")

        # Known relationships from financial theory
        rules = [
            ("MACRO_interest_rate", "SECTOR_Banking", "INFLUENCES", 0.8, 0),
            ("MACRO_interest_rate", "SECTOR_Real Estate", "INFLUENCES", 0.7, 0),
            ("MACRO_VIX", "SECTOR_Defensive", "INFLUENCES", 0.6, 0),
            ("MACRO_GDP_growth", "SECTOR_Cyclical", "INFLUENCES", 0.7, 0),
            ("SENTIMENT_positive", "ALLOCATION_increase", "INFLUENCES", 0.5, 1),
            ("FUNDAMENTAL_valuation", "STOCK_selection", "INFLUENCES", 0.6, 0),
        ]

        print(f"✓ Added {len(rules)} domain knowledge rules")
        return rules

    def build_from_dataframe(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Build complete causal knowledge graph from integrated dataset

        Args:
            df: Integrated dataset

        Returns:
            NetworkX DiGraph
        """
        log_step("Building Causal Knowledge Graph")

        # Extract different types of causality
        macro_sector_rels = self.extract_macro_to_sector_causality(df)
        sector_allocation_rels = self.extract_sector_to_allocation_causality(df)
        fundamental_selection_rels = self.extract_fundamental_to_selection_causality(df)
        domain_rules = self.add_domain_knowledge_rules()

        # Combine all relationships
        all_relationships = (
            macro_sector_rels +
            sector_allocation_rels +
            fundamental_selection_rels +
            domain_rules
        )

        # Add nodes and edges to graph
        for cause, effect, rel_type, strength, lag in all_relationships:
            # Add nodes if not exist
            if cause not in self.graph:
                self.graph.add_node(cause, node_type='Factor')

            if effect not in self.graph:
                self.graph.add_node(effect, node_type='Action')

            # Add edge
            self.graph.add_edge(
                cause,
                effect,
                edge_type=rel_type,
                strength=strength,
                lag=lag,
                confidence='high' if strength > 0.7 else 'medium' if strength > 0.4 else 'low'
            )

        # Check for cycles and remove weak edges creating cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            print("Warning: Graph contains cycles. Removing weakest edges to create DAG...")
            self._remove_cycles()

        # Print summary
        print(f"\n{'='*80}")
        print("CAUSAL KG CONSTRUCTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total nodes: {self.graph.number_of_nodes()}")
        print(f"Total edges: {self.graph.number_of_edges()}")
        print(f"Is DAG: {nx.is_directed_acyclic_graph(self.graph)}")

        print(f"\nEdge types:")
        for edge_type in self.config['relationship_types']:
            count = sum(1 for u, v, d in self.graph.edges(data=True)
                       if d.get('edge_type') == edge_type)
            print(f"  {edge_type}: {count}")

        return self.graph

    def _remove_cycles(self) -> None:
        """Remove cycles from graph by removing weakest edges"""
        while not nx.is_directed_acyclic_graph(self.graph):
            try:
                cycle = nx.find_cycle(self.graph)
                # Find weakest edge in cycle
                weakest_edge = min(
                    cycle,
                    key=lambda e: self.graph[e[0]][e[1]].get('strength', 0)
                )
                self.graph.remove_edge(weakest_edge[0], weakest_edge[1])
            except:
                break

    def save(self, filename: str = "causal_kg.gpickle") -> None:
        """Save graph to file"""
        filepath = GRAPHS_DIR / filename
        with open(filepath, 'wb') as f:
            pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\n✓ Causal KG saved to: {filepath}")

    def load(self, filename: str = "causal_kg.gpickle") -> nx.DiGraph:
        """Load graph from file"""
        filepath = GRAPHS_DIR / filename
        with open(filepath, 'rb') as f:
            self.graph = pickle.load(f)
        print(f"✓ Causal KG loaded from: {filepath}")
        return self.graph

    def find_causal_paths(self, source: str, target: str, max_length: int = 4) -> List[List]:
        """
        Find all causal paths between source and target

        Args:
            source: Source node
            target: Target node
            max_length: Maximum path length

        Returns:
            List of paths
        """
        if source not in self.graph or target not in self.graph:
            return []

        try:
            paths = nx.all_simple_paths(
                self.graph,
                source=source,
                target=target,
                cutoff=max_length
            )
            return list(paths)
        except:
            return []

    def get_influencing_factors(self, action_node: str) -> List[Tuple]:
        """
        Get all factors that influence a specific action

        Args:
            action_node: Action node name

        Returns:
            List of (factor, edge_data) tuples sorted by strength
        """
        if action_node not in self.graph:
            return []

        factors = []
        for predecessor in self.graph.predecessors(action_node):
            edge_data = self.graph[predecessor][action_node]
            factors.append((predecessor, edge_data))

        # Sort by strength
        factors.sort(key=lambda x: x[1].get('strength', 0), reverse=True)
        return factors


def main():
    """Test causal KG construction"""
    log_step("Testing Causal Knowledge Graph Construction")

    # Load integrated dataset
    data_path = PROCESSED_DATA_DIR / "integrated_dataset.csv"

    if not data_path.exists():
        print(f"Error: Integrated dataset not found at {data_path}")
        print("Please run feature_engineering.py first.")
        return

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"✓ Loaded integrated dataset: {df.shape}")

    # Build causal KG
    ckg = CausalKG()
    graph = ckg.build_from_dataframe(df)

    # Save
    ckg.save()

    # Test queries
    print(f"\n{'='*80}")
    print("TESTING QUERIES")
    print(f"{'='*80}")

    # Query: Show some causal relationships
    print("\nSample causal relationships:")
    for u, v, data in list(graph.edges(data=True))[:10]:
        print(f"  {u} --[{data['edge_type']}, strength={data['strength']:.3f}]--> {v}")

    print(f"\n{'='*80}")
    print("✓ Causal KG construction and testing complete!")


if __name__ == "__main__":
    main()
