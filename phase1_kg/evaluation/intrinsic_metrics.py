"""
Intrinsic Evaluation Metrics for Knowledge Graphs
Evaluates structural completeness, semantic coherence, consistency, and inferential utility
"""
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import community  # python-louvain
from collections import Counter, defaultdict
import sys
from pathlib import Path
import pickle
# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import EVALUATION_CONFIG, METRICS_DIR
from utils.helpers import log_step, save_json


class KGEvaluator:
    """
    Comprehensive evaluator for knowledge graph quality
    """

    def __init__(self, graph: nx.Graph, graph_name: str = "KG"):
        self.graph = graph
        self.graph_name = graph_name
        self.results = {}
        log_step(f"Initializing evaluator for {graph_name}")

    # ============================================================================
    # 1. STRUCTURAL COMPLETENESS METRICS
    # ============================================================================

    def evaluate_structural_completeness(self) -> Dict:
        """
        Evaluate structural properties and completeness
        """
        log_step("Evaluating Structural Completeness")

        metrics = {}

        # Basic statistics
        metrics['node_count'] = self.graph.number_of_nodes()
        metrics['edge_count'] = self.graph.number_of_edges()

        # Density
        if isinstance(self.graph, nx.DiGraph) or isinstance(self.graph, nx.MultiDiGraph):
            n = metrics['node_count']
            max_edges = n * (n - 1)  # For directed graph
            metrics['density'] = metrics['edge_count'] / max_edges if max_edges > 0 else 0
        else:
            metrics['density'] = nx.density(self.graph)

        # Degree statistics
        if isinstance(self.graph, nx.MultiDiGraph):
            # For MultiDiGraph, count unique neighbors
            degrees = [len(set(self.graph.neighbors(n))) for n in self.graph.nodes()]
        else:
            degrees = [d for n, d in self.graph.degree()]

        metrics['average_degree'] = np.mean(degrees) if degrees else 0
        metrics['median_degree'] = np.median(degrees) if degrees else 0
        metrics['max_degree'] = max(degrees) if degrees else 0
        metrics['min_degree'] = min(degrees) if degrees else 0
        metrics['degree_std'] = np.std(degrees) if degrees else 0

        # Degree distribution
        degree_counts = Counter(degrees)
        metrics['degree_distribution'] = dict(sorted(degree_counts.items())[:20])  # Top 20

        # Connected components
        if isinstance(self.graph, nx.DiGraph) or isinstance(self.graph, nx.MultiDiGraph):
            undirected = self.graph.to_undirected()
            metrics['connected_components'] = nx.number_connected_components(undirected)
            metrics['largest_component_size'] = len(max(nx.connected_components(undirected), key=len))
        else:
            metrics['connected_components'] = nx.number_connected_components(self.graph)
            metrics['largest_component_size'] = len(max(nx.connected_components(self.graph), key=len))

        # Clustering coefficient
        if isinstance(self.graph, nx.MultiDiGraph):
            # Convert to simple graph for clustering
            simple_graph = nx.Graph(self.graph)
            metrics['average_clustering'] = nx.average_clustering(simple_graph)
        elif not isinstance(self.graph, nx.DiGraph):
            metrics['average_clustering'] = nx.average_clustering(self.graph)
        else:
            metrics['average_clustering'] = 0  # Not applicable for directed

        # Coverage by node type
        node_types = defaultdict(int)
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', 'Unknown')
            node_types[node_type] += 1

        metrics['node_type_distribution'] = dict(node_types)

        # Edge type distribution
        edge_types = defaultdict(int)
        if isinstance(self.graph, nx.MultiDiGraph):
            for u, v, key, data in self.graph.edges(keys=True, data=True):
                edge_type = data.get('edge_type', 'Unknown')
                edge_types[edge_type] += 1
        else:
            for u, v, data in self.graph.edges(data=True):
                edge_type = data.get('edge_type', 'Unknown')
                edge_types[edge_type] += 1

        metrics['edge_type_distribution'] = dict(edge_types)

        # Completeness score (0-1)
        # Based on ratio of actual edges to expected edges for a complete representation
        expected_node_types = 4  # Fund, Stock, Sector, TimePeriod (for Temporal KG)
        expected_edge_types = 6  # Various relationship types

        actual_node_types = len(node_types)
        actual_edge_types = len(edge_types)

        type_completeness = (actual_node_types / expected_node_types +
                            actual_edge_types / expected_edge_types) / 2

        connectivity_completeness = (metrics['largest_component_size'] / metrics['node_count']
                                    if metrics['node_count'] > 0 else 0)

        metrics['completeness_score'] = (type_completeness + connectivity_completeness) / 2

        print(f"✓ Structural completeness: {metrics['completeness_score']:.3f}")
        return metrics

    # ============================================================================
    # 2. CONSISTENCY (INTEGRITY) METRICS
    # ============================================================================

    def evaluate_consistency(self) -> Dict:
        """
        Evaluate logical consistency and integrity constraints
        """
        log_step("Evaluating Consistency and Integrity")

        metrics = {}
        violations = []

        # Check for orphan nodes (no edges)
        orphan_nodes = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]
        metrics['orphan_nodes_count'] = len(orphan_nodes)
        metrics['orphan_nodes_pct'] = (len(orphan_nodes) / self.graph.number_of_nodes()
                                       if self.graph.number_of_nodes() > 0 else 0)

        if orphan_nodes:
            violations.append(f"{len(orphan_nodes)} orphan nodes found")

        # Check for self-loops
        self_loops = list(nx.nodes_with_selfloops(self.graph))
        metrics['self_loops_count'] = len(self_loops)

        if self_loops:
            violations.append(f"{len(self_loops)} self-loops found")

        # Check temporal consistency (dates should be chronological)
        temporal_violations = 0
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            if 'first_date' in node_data and 'last_date' in node_data:
                if node_data['first_date'] > node_data['last_date']:
                    temporal_violations += 1

        metrics['temporal_violations'] = temporal_violations

        if temporal_violations > 0:
            violations.append(f"{temporal_violations} temporal ordering violations")

        # Check for causal cycles (only for causal KG/DAG)
        if isinstance(self.graph, nx.DiGraph) and not isinstance(self.graph, nx.MultiDiGraph):
            is_dag = nx.is_directed_acyclic_graph(self.graph)
            metrics['is_dag'] = is_dag

            if not is_dag:
                try:
                    cycles = list(nx.simple_cycles(self.graph))
                    metrics['cycle_count'] = len(cycles)
                    violations.append(f"{len(cycles)} causal cycles found")
                except:
                    metrics['cycle_count'] = 0
            else:
                metrics['cycle_count'] = 0
        else:
            metrics['is_dag'] = None
            metrics['cycle_count'] = 0

        # Check cardinality constraints
        # Example: A fund should have multiple holdings, not just one
        cardinality_violations = 0
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'Fund':
                # Count holdings
                if isinstance(self.graph, nx.MultiDiGraph):
                    holdings = len([n for n in self.graph.neighbors(node)])
                else:
                    holdings = self.graph.out_degree(node)

                if holdings < 5:  # Arbitrary threshold: fund should have at least 5 holdings
                    cardinality_violations += 1

        metrics['cardinality_violations'] = cardinality_violations

        if cardinality_violations > 0:
            violations.append(f"{cardinality_violations} cardinality constraint violations")

        # Check attribute completeness
        missing_attributes = 0
        required_attributes_by_type = {
            'Fund': ['fund_type', 'first_date'],
            'Stock': ['stock_name', 'sector'],
            'Sector': ['sector_name'],
        }

        for node, data in self.graph.nodes(data=True):
            node_type = data.get('node_type')
            if node_type in required_attributes_by_type:
                for attr in required_attributes_by_type[node_type]:
                    if attr not in data or data[attr] is None:
                        missing_attributes += 1

        metrics['missing_attributes_count'] = missing_attributes

        if missing_attributes > 0:
            violations.append(f"{missing_attributes} missing required attributes")

        # Overall consistency score
        total_checks = 6
        failed_checks = len([v for v in [
            metrics['orphan_nodes_count'],
            metrics['self_loops_count'],
            metrics['temporal_violations'],
            metrics['cycle_count'],
            metrics['cardinality_violations'],
            metrics['missing_attributes_count']
        ] if v > 0])

        metrics['consistency_score'] = 1 - (failed_checks / total_checks)
        metrics['violations'] = violations

        print(f"✓ Consistency score: {metrics['consistency_score']:.3f}")
        print(f"  Violations: {len(violations)}")
        return metrics

    # ============================================================================
    # 3. SEMANTIC COHERENCE METRICS
    # ============================================================================

    def evaluate_semantic_coherence(self) -> Dict:
        """
        Evaluate semantic coherence and meaningful clustering
        """
        log_step("Evaluating Semantic Coherence")

        metrics = {}

        # Convert to undirected for community detection
        if isinstance(self.graph, nx.DiGraph) or isinstance(self.graph, nx.MultiDiGraph):
            undirected = nx.Graph(self.graph)
        else:
            undirected = self.graph

        # Community detection (Louvain method)
        try:
            partition = community.best_partition(undirected)
            metrics['num_communities'] = len(set(partition.values()))

            # Modularity score
            metrics['modularity'] = community.modularity(partition, undirected)

            # Sector purity: Check if stocks from same sector cluster together
            sector_purity_scores = []

            for comm_id in set(partition.values()):
                nodes_in_comm = [n for n, c in partition.items() if c == comm_id]

                # Get sectors of stocks in this community
                sectors = []
                for node in nodes_in_comm:
                    node_data = undirected.nodes.get(node, {})
                    if node_data.get('node_type') == 'Stock':
                        sector = node_data.get('sector')
                        if sector:
                            sectors.append(sector)

                if sectors:
                    # Purity = fraction of most common sector
                    most_common = Counter(sectors).most_common(1)[0][1]
                    purity = most_common / len(sectors)
                    sector_purity_scores.append(purity)

            metrics['average_sector_purity'] = (np.mean(sector_purity_scores)
                                               if sector_purity_scores else 0)

        except Exception as e:
            print(f"Warning: Community detection failed: {e}")
            metrics['num_communities'] = 0
            metrics['modularity'] = 0
            metrics['average_sector_purity'] = 0

        # Temporal coherence: Check if temporal relationships are logically ordered
        temporal_coherence_count = 0
        temporal_total = 0

        for u, v, data in self.graph.edges(data=True):
            if 'date' in data or 'date_str' in data:
                temporal_total += 1
                # Basic check: edge should have valid date
                if data.get('date') or data.get('date_str'):
                    temporal_coherence_count += 1

        metrics['temporal_coherence'] = (temporal_coherence_count / temporal_total
                                         if temporal_total > 0 else 1.0)

        # Semantic coherence score (0-1)
        # Weighted combination of modularity, sector purity, and temporal coherence
        metrics['semantic_coherence_score'] = (
            0.4 * max(0, min(1, metrics['modularity'])) +  # Modularity (normalize to 0-1)
            0.3 * metrics['average_sector_purity'] +
            0.3 * metrics['temporal_coherence']
        )

        print(f"✓ Semantic coherence score: {metrics['semantic_coherence_score']:.3f}")
        return metrics

    # ============================================================================
    # 4. INFORMATIVENESS METRICS
    # ============================================================================

    def evaluate_informativeness(self) -> Dict:
        """
        Evaluate richness and non-redundancy of information
        """
        log_step("Evaluating Informativeness")

        metrics = {}

        # Information richness: Count of unique attributes
        all_attributes = set()
        for node, data in self.graph.nodes(data=True):
            all_attributes.update(data.keys())

        for u, v, data in self.graph.edges(data=True):
            all_attributes.update(data.keys())

        metrics['unique_attributes'] = len(all_attributes)

        # Attribute diversity per node type
        attributes_by_type = defaultdict(set)
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', 'Unknown')
            attributes_by_type[node_type].update(data.keys())

        metrics['attributes_per_node_type'] = {
            k: len(v) for k, v in attributes_by_type.items()
        }

        # Average attributes per node
        node_attribute_counts = [len(data) for node, data in self.graph.nodes(data=True)]
        metrics['avg_attributes_per_node'] = (np.mean(node_attribute_counts)
                                             if node_attribute_counts else 0)

        # Edge attribute counts
        edge_attribute_counts = [len(data) for u, v, data in self.graph.edges(data=True)]
        metrics['avg_attributes_per_edge'] = (np.mean(edge_attribute_counts)
                                             if edge_attribute_counts else 0)

        # Redundancy check: Identify duplicate edges
        if isinstance(self.graph, nx.MultiDiGraph):
            # Count multi-edges
            multi_edge_count = 0
            for u, v in self.graph.edges():
                if self.graph.number_of_edges(u, v) > 1:
                    multi_edge_count += 1

            metrics['redundant_edges'] = multi_edge_count
        else:
            metrics['redundant_edges'] = 0

        # Information density: Attributes per entity
        total_entities = metrics['unique_attributes']
        total_data_points = (
            sum(node_attribute_counts) +
            sum(edge_attribute_counts)
        )

        metrics['information_density'] = (total_data_points /
                                         (self.graph.number_of_nodes() + self.graph.number_of_edges())
                                         if (self.graph.number_of_nodes() + self.graph.number_of_edges()) > 0
                                         else 0)

        # Informativeness score (0-1)
        # Normalized combination of metrics
        normalized_unique_attrs = min(1.0, metrics['unique_attributes'] / 20)  # Expect ~20 unique attributes
        normalized_avg_attrs = min(1.0, metrics['avg_attributes_per_node'] / 10)  # Expect ~10 attrs per node
        redundancy_penalty = 1 - min(1.0, metrics['redundant_edges'] / max(1, self.graph.number_of_edges()))

        metrics['informativeness_score'] = (
            0.4 * normalized_unique_attrs +
            0.4 * normalized_avg_attrs +
            0.2 * redundancy_penalty
        )

        print(f"✓ Informativeness score: {metrics['informativeness_score']:.3f}")
        return metrics

    # ============================================================================
    # 5. INFERENTIAL UTILITY METRICS
    # ============================================================================

    def evaluate_inferential_utility(self) -> Dict:
        """
        Evaluate ability to support reasoning and inference
        """
        log_step("Evaluating Inferential Utility")

        metrics = {}

        # Path length distribution (for reachability)
        if isinstance(self.graph, nx.MultiDiGraph):
            # Sample nodes for efficiency
            sample_size = min(50, self.graph.number_of_nodes())
            sample_nodes = list(self.graph.nodes())[:sample_size]
        else:
            sample_nodes = list(self.graph.nodes())[:50]

        path_lengths = []
        reachable_pairs = 0
        total_pairs = 0

        for i, source in enumerate(sample_nodes):
            for target in sample_nodes[i+1:]:
                total_pairs += 1
                try:
                    if isinstance(self.graph, nx.DiGraph) or isinstance(self.graph, nx.MultiDiGraph):
                        path = nx.shortest_path(self.graph, source, target)
                    else:
                        path = nx.shortest_path(self.graph, source, target)

                    path_lengths.append(len(path) - 1)  # Number of edges
                    reachable_pairs += 1
                except nx.NetworkXNoPath:
                    pass

        metrics['average_path_length'] = np.mean(path_lengths) if path_lengths else 0
        metrics['max_path_length'] = max(path_lengths) if path_lengths else 0
        metrics['reachability'] = reachable_pairs / total_pairs if total_pairs > 0 else 0

        # Centrality measures (identify important nodes)
        try:
            if isinstance(self.graph, nx.MultiDiGraph):
                simple_graph = nx.DiGraph(self.graph)
            else:
                simple_graph = self.graph

            # Degree centrality
            degree_cent = nx.degree_centrality(simple_graph)
            metrics['max_degree_centrality'] = max(degree_cent.values()) if degree_cent else 0
            metrics['avg_degree_centrality'] = np.mean(list(degree_cent.values())) if degree_cent else 0

            # Betweenness centrality (computational expensive, sample)
            if simple_graph.number_of_nodes() < 500:
                betweenness = nx.betweenness_centrality(simple_graph)
                metrics['max_betweenness_centrality'] = max(betweenness.values()) if betweenness else 0
            else:
                metrics['max_betweenness_centrality'] = 0

        except Exception as e:
            print(f"Warning: Centrality calculation failed: {e}")
            metrics['max_degree_centrality'] = 0
            metrics['avg_degree_centrality'] = 0
            metrics['max_betweenness_centrality'] = 0

        # Query complexity: Can the KG answer complex multi-hop queries?
        # Proxy: Presence of paths of length 2-4
        multi_hop_paths = [p for p in path_lengths if 2 <= p <= 4]
        metrics['multi_hop_query_support'] = (len(multi_hop_paths) / len(path_lengths)
                                             if path_lengths else 0)

        # Inferential utility score (0-1)
        # Based on reachability, path diversity, and centrality
        normalized_path_length = 1 - min(1.0, metrics['average_path_length'] / 10)  # Shorter is better
        reachability_score = metrics['reachability']
        multi_hop_score = metrics['multi_hop_query_support']

        metrics['inferential_utility_score'] = (
            0.4 * reachability_score +
            0.3 * normalized_path_length +
            0.3 * multi_hop_score
        )

        print(f"✓ Inferential utility score: {metrics['inferential_utility_score']:.3f}")
        return metrics

    # ============================================================================
    # AGGREGATE EVALUATION
    # ============================================================================

    def evaluate_all(self) -> Dict:
        """
        Run all evaluation metrics

        Returns:
            Dictionary with all evaluation results
        """
        log_step(f"Running Complete Evaluation for {self.graph_name}")

        self.results = {
            'graph_name': self.graph_name,
            'structural_completeness': self.evaluate_structural_completeness(),
            'consistency': self.evaluate_consistency(),
            'semantic_coherence': self.evaluate_semantic_coherence(),
            'informativeness': self.evaluate_informativeness(),
            'inferential_utility': self.evaluate_inferential_utility()
        }

        # Calculate overall quality score
        scores = [
            self.results['structural_completeness']['completeness_score'],
            self.results['consistency']['consistency_score'],
            self.results['semantic_coherence']['semantic_coherence_score'],
            self.results['informativeness']['informativeness_score'],
            self.results['inferential_utility']['inferential_utility_score']
        ]

        self.results['overall_quality_score'] = np.mean(scores)

        print(f"\n{'='*80}")
        print(f"EVALUATION SUMMARY FOR {self.graph_name}")
        print(f"{'='*80}")
        print(f"Overall Quality Score: {self.results['overall_quality_score']:.3f}")
        print(f"  Structural Completeness: {scores[0]:.3f}")
        print(f"  Consistency:             {scores[1]:.3f}")
        print(f"  Semantic Coherence:      {scores[2]:.3f}")
        print(f"  Informativeness:         {scores[3]:.3f}")
        print(f"  Inferential Utility:     {scores[4]:.3f}")

        return self.results

    def save_results(self, filename: str = None) -> None:
        """Save evaluation results to JSON"""
        if filename is None:
            filename = f"{self.graph_name}_evaluation_results.json"

        filepath = METRICS_DIR / filename
        save_json(self.results, filepath)


def main():
    """Test evaluation metrics"""
    log_step("Testing Intrinsic Evaluation Metrics")

    from config import GRAPHS_DIR

    # Try to load and evaluate temporal KG
    temporal_kg_path = GRAPHS_DIR / "temporal_kg.gpickle"

    if temporal_kg_path.exists():
        import pickle
        with open(temporal_kg_path, 'rb') as f:
            temporal_kg = pickle.load(f)
        print(f"✓ Loaded Temporal KG: {temporal_kg.number_of_nodes()} nodes, "
              f"{temporal_kg.number_of_edges()} edges")

        evaluator = KGEvaluator(temporal_kg, "Temporal_KG")
        results = evaluator.evaluate_all()
        evaluator.save_results()
    else:
        print(f"Temporal KG not found at {temporal_kg_path}")
        print("Please run temporal_kg.py first.")

    # Try to load and evaluate causal KG
    causal_kg_path = GRAPHS_DIR / "causal_kg.gpickle"

    if causal_kg_path.exists():
        import pickle
        with open(causal_kg_path, 'rb') as f:
            causal_kg = pickle.load(f)
        print(f"\n✓ Loaded Causal KG: {causal_kg.number_of_nodes()} nodes, "
              f"{causal_kg.number_of_edges()} edges")

        evaluator = KGEvaluator(causal_kg, "Causal_KG")
        results = evaluator.evaluate_all()
        evaluator.save_results()
    else:
        print(f"\nCausal KG not found at {causal_kg_path}")

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
