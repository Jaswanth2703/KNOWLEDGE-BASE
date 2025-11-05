"""
Knowledge Graph Integration
Connects Temporal and Causal KGs through shared entities and cross-references
"""
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import GRAPHS_DIR
from utils.helpers import log_step
from .temporal_kg import TemporalKG
from .causal_kg import CausalKG


class IntegratedKG:
    """
    Integrated Knowledge Graph combining Temporal and Causal perspectives
    """

    def __init__(self, temporal_kg: nx.MultiDiGraph, causal_kg: nx.DiGraph):
        self.temporal_kg = temporal_kg
        self.causal_kg = causal_kg
        self.graph = nx.DiGraph()  # Integrated graph
        log_step("Initializing Integrated Knowledge Graph")

    def identify_shared_entities(self) -> Dict[str, List]:
        """
        Identify entities that appear in both graphs

        Returns:
            Dictionary of shared entity types and their instances
        """
        temporal_nodes = set(self.temporal_kg.nodes())
        causal_nodes = set(self.causal_kg.nodes())

        # Find overlaps by entity type
        shared = {
            'sectors': [],
            'stocks': [],
            'funds': [],
            'factors': []
        }

        for node in temporal_nodes:
            node_str = str(node)
            # Check if exists in causal graph (may have different prefix)
            for causal_node in causal_nodes:
                causal_str = str(causal_node)
                # Match sector nodes
                if 'SECTOR_' in node_str and node_str in causal_str:
                    shared['sectors'].append((node, causal_node))
                # Match stock nodes (by ISIN)
                elif node.startswith('INE') and node[:10] in causal_str:
                    shared['stocks'].append((node, causal_node))

        print(f"✓ Identified shared entities:")
        for entity_type, entities in shared.items():
            print(f"  {entity_type}: {len(entities)}")

        return shared

    def create_cross_references(self) -> List[Tuple]:
        """
        Create cross-reference edges between temporal events and causal explanations

        Returns:
            List of (temporal_node, causal_node, relationship) tuples
        """
        log_step("Creating cross-references between Temporal and Causal KGs")

        cross_refs = []

        # For each portfolio change in temporal KG, link to potential causal factors
        for u, v, key, data in self.temporal_kg.edges(keys=True, data=True):
            edge_type = data.get('edge_type')

            # Focus on actionable edges
            if edge_type in ['INCREASED', 'DECREASED', 'ENTERED', 'EXITED']:
                date = data.get('date_str')
                stock = v  # The stock node
                fund = u   # The fund node

                # Get stock's sector from temporal KG
                if stock in self.temporal_kg:
                    stock_data = self.temporal_kg.nodes[stock]
                    sector = stock_data.get('sector')

                    if sector:
                        # Look for causal factors related to this sector
                        causal_sector_node = f"SECTOR_{sector}"

                        if causal_sector_node in self.causal_kg:
                            # Find factors that influence this sector
                            for predecessor in self.causal_kg.predecessors(causal_sector_node):
                                causal_edge = self.causal_kg[predecessor][causal_sector_node]

                                cross_refs.append((
                                    f"{fund}-{stock}-{edge_type}-{date}",  # Temporal event ID
                                    predecessor,  # Causal factor
                                    {
                                        'type': 'EXPLAINED_BY',
                                        'strength': causal_edge.get('strength', 0.5),
                                        'lag': causal_edge.get('lag', 0)
                                    }
                                ))

        print(f"✓ Created {len(cross_refs)} cross-references")
        return cross_refs

    def build_integrated_graph(self) -> nx.DiGraph:
        """
        Build unified graph combining both perspectives

        Returns:
            NetworkX DiGraph
        """
        log_step("Building Integrated Knowledge Graph")

        # Add all temporal nodes
        for node, data in self.temporal_kg.nodes(data=True):
            self.graph.add_node(node, **data, source='temporal')

        # Add all causal nodes
        for node, data in self.causal_kg.nodes(data=True):
            if node not in self.graph:
                self.graph.add_node(node, **data, source='causal')
            else:
                # Mark as shared
                self.graph.nodes[node]['source'] = 'both'

        # Add temporal edges
        for u, v, key, data in self.temporal_kg.edges(keys=True, data=True):
            self.graph.add_edge(u, v, **data, source='temporal')

        # Add causal edges
        for u, v, data in self.causal_kg.edges(data=True):
            self.graph.add_edge(u, v, **data, source='causal')

        # Add cross-reference edges
        cross_refs = self.create_cross_references()
        for temporal_event, causal_factor, ref_data in cross_refs:
            # Add cross-reference edge
            self.graph.add_edge(
                causal_factor,
                temporal_event,
                **ref_data,
                source='cross_reference'
            )

        # Print summary
        print(f"\n{'='*80}")
        print("INTEGRATED KG CONSTRUCTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total nodes: {self.graph.number_of_nodes()}")
        print(f"Total edges: {self.graph.number_of_edges()}")

        # Count nodes by source
        for source_type in ['temporal', 'causal', 'both']:
            count = sum(1 for n, d in self.graph.nodes(data=True)
                       if d.get('source') == source_type)
            print(f"Nodes from {source_type}: {count}")

        return self.graph

    def query_decision_explanation(
        self,
        fund_name: str,
        stock_isin: str,
        date: str
    ) -> Dict:
        """
        Query combined temporal-causal explanation for a decision

        Args:
            fund_name: Fund name
            stock_isin: Stock ISIN
            date: Date (YYYY-MM)

        Returns:
            Dictionary with temporal context and causal factors
        """
        explanation = {
            'temporal_context': {},
            'causal_factors': [],
            'decision': None
        }

        # Get temporal information
        if fund_name in self.temporal_kg and stock_isin in self.temporal_kg:
            edges = self.temporal_kg.get_edge_data(fund_name, stock_isin)

            if edges:
                for key, edge_data in edges.items():
                    if edge_data.get('date_str') == date:
                        explanation['decision'] = edge_data.get('edge_type')
                        explanation['temporal_context'] = edge_data

        # Get causal factors
        stock_data = self.temporal_kg.nodes.get(stock_isin, {})
        sector = stock_data.get('sector')

        if sector:
            causal_sector = f"SECTOR_{sector}"
            if causal_sector in self.causal_kg:
                for predecessor in self.causal_kg.predecessors(causal_sector):
                    edge_data = self.causal_kg[predecessor][causal_sector]
                    explanation['causal_factors'].append({
                        'factor': predecessor,
                        'relationship': edge_data.get('edge_type'),
                        'strength': edge_data.get('strength'),
                        'lag': edge_data.get('lag')
                    })

        return explanation

    def save(self, filename: str = "integrated_kg.gpickle") -> None:
        """Save integrated graph"""
        filepath = GRAPHS_DIR / filename
        nx.write_gpickle(self.graph, filepath)
        print(f"\n✓ Integrated KG saved to: {filepath}")

    @staticmethod
    def load_and_integrate() -> 'IntegratedKG':
        """Load both KGs and create integrated version"""
        log_step("Loading Temporal and Causal KGs for integration")

        # Load temporal KG
        tkg = TemporalKG()
        temporal_graph = tkg.load()

        # Load causal KG
        ckg = CausalKG()
        causal_graph = ckg.load()

        # Create integrated KG
        ikg = IntegratedKG(temporal_graph, causal_graph)
        ikg.build_integrated_graph()

        return ikg


def main():
    """Test KG integration"""
    log_step("Testing Knowledge Graph Integration")

    try:
        # Load and integrate
        ikg = IntegratedKG.load_and_integrate()

        # Save
        ikg.save()

        # Test query
        print(f"\n{'='*80}")
        print("TESTING INTEGRATED QUERY")
        print(f"{'='*80}")

        # Get a sample fund and stock
        funds = [n for n, d in ikg.temporal_kg.nodes(data=True)
                if d.get('node_type') == 'Fund']
        stocks = [n for n, d in ikg.temporal_kg.nodes(data=True)
                 if d.get('node_type') == 'Stock']

        if funds and stocks:
            explanation = ikg.query_decision_explanation(
                funds[0],
                stocks[0],
                "2024-12"
            )

            print(f"\nDecision explanation for {funds[0]} - {stocks[0]}:")
            print(f"Decision: {explanation['decision']}")
            print(f"Temporal context: {explanation['temporal_context']}")
            print(f"Causal factors: {len(explanation['causal_factors'])}")
            for factor in explanation['causal_factors'][:5]:
                print(f"  - {factor['factor']}: {factor['relationship']} "
                     f"(strength={factor['strength']:.3f})")

        print(f"\n{'='*80}")
        print("✓ KG integration and testing complete!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run temporal_kg.py and causal_kg.py first to create the individual graphs.")


if __name__ == "__main__":
    main()
