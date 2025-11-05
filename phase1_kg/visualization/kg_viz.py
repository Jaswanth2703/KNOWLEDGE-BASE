"""
Knowledge Graph Visualization
Creates interactive and static visualizations of Temporal and Causal KGs
"""
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import VIZ_CONFIG, FIGURES_DIR, GRAPHS_DIR
from utils.helpers import log_step


class KGVisualizer:
    """
    Visualizer for knowledge graphs
    Supports both static (matplotlib) and interactive (plotly) visualizations
    """

    def __init__(self, graph: nx.Graph, graph_name: str = "KG"):
        self.graph = graph
        self.graph_name = graph_name
        self.config = VIZ_CONFIG
        log_step(f"Initializing visualizer for {graph_name}")

    def plot_degree_distribution(self, save_path: str = None) -> None:
        """Plot degree distribution"""
        degrees = [d for n, d in self.graph.degree()]

        plt.figure(figsize=(10, 6))
        plt.hist(degrees, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Degree', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Degree Distribution - {self.graph_name}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        if save_path is None:
            save_path = FIGURES_DIR / f"{self.graph_name}_degree_distribution.png"

        plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        print(f"✓ Saved degree distribution: {save_path}")

    def plot_node_type_distribution(self, save_path: str = None) -> None:
        """Plot distribution of node types"""
        node_types = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', 'Unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1

        plt.figure(figsize=(10, 6))
        plt.bar(node_types.keys(), node_types.values(), edgecolor='black', alpha=0.7)
        plt.xlabel('Node Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(f'Node Type Distribution - {self.graph_name}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)

        if save_path is None:
            save_path = FIGURES_DIR / f"{self.graph_name}_node_types.png"

        plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        print(f"✓ Saved node type distribution: {save_path}")

    def plot_edge_type_distribution(self, save_path: str = None) -> None:
        """Plot distribution of edge types"""
        edge_types = {}

        if isinstance(self.graph, nx.MultiDiGraph):
            for u, v, key, data in self.graph.edges(keys=True, data=True):
                edge_type = data.get('edge_type', 'Unknown')
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        else:
            for u, v, data in self.graph.edges(data=True):
                edge_type = data.get('edge_type', 'Unknown')
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        plt.figure(figsize=(12, 6))
        plt.bar(edge_types.keys(), edge_types.values(), edgecolor='black', alpha=0.7)
        plt.xlabel('Edge Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(f'Edge Type Distribution - {self.graph_name}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)

        if save_path is None:
            save_path = FIGURES_DIR / f"{self.graph_name}_edge_types.png"

        plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        print(f"✓ Saved edge type distribution: {save_path}")

    def plot_subgraph(
        self,
        center_node: str,
        depth: int = 2,
        save_path: str = None
    ) -> None:
        """
        Plot subgraph around a center node

        Args:
            center_node: Center node
            depth: Neighborhood depth
            save_path: Save path
        """
        # Extract subgraph
        if center_node not in self.graph:
            print(f"Warning: {center_node} not found in graph")
            return

        # Get neighbors up to depth
        neighbors = {center_node}
        for _ in range(depth):
            new_neighbors = set()
            for node in neighbors:
                if isinstance(self.graph, nx.DiGraph) or isinstance(self.graph, nx.MultiDiGraph):
                    new_neighbors.update(self.graph.successors(node))
                    new_neighbors.update(self.graph.predecessors(node))
                else:
                    new_neighbors.update(self.graph.neighbors(node))
            neighbors.update(new_neighbors)

        subgraph = self.graph.subgraph(neighbors)

        # Create layout
        if isinstance(subgraph, nx.MultiDiGraph):
            simple_graph = nx.DiGraph(subgraph)
            pos = nx.spring_layout(simple_graph, k=0.5, iterations=50)
        else:
            pos = nx.spring_layout(subgraph, k=0.5, iterations=50)

        # Plot
        plt.figure(figsize=self.config['figure_size'])

        # Draw nodes by type
        node_colors = []
        node_sizes = []
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            node_type = node_data.get('node_type', 'Unknown')

            # Color by type
            color_map = {
                'Fund': '#FF6B6B',
                'Stock': '#4ECDC4',
                'Sector': '#FFD93D',
                'TimePeriod': '#95E1D3',
                'Factor': '#A8E6CF',
                'Action': '#FDCB9E'
            }
            node_colors.append(color_map.get(node_type, '#CCCCCC'))

            # Size by degree
            if node == center_node:
                node_sizes.append(3000)
            else:
                degree = subgraph.degree(node)
                node_sizes.append(100 + degree * 50)

        nx.draw_networkx_nodes(
            subgraph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.7
        )

        # Draw edges
        nx.draw_networkx_edges(
            subgraph, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            alpha=0.5,
            width=1.5
        )

        # Draw labels for important nodes
        labels = {}
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            if node == center_node:
                labels[node] = node[:30]  # Truncate long names
            elif node_data.get('node_type') in ['Fund', 'Sector']:
                labels[node] = node[:20]

        nx.draw_networkx_labels(
            subgraph, pos,
            labels,
            font_size=8,
            font_weight='bold'
        )

        plt.title(f'Subgraph around {center_node} (depth={depth})\n{self.graph_name}',
                 fontsize=14, fontweight='bold')
        plt.axis('off')

        if save_path is None:
            safe_name = center_node.replace('/', '_').replace(' ', '_')[:30]
            save_path = FIGURES_DIR / f"{self.graph_name}_subgraph_{safe_name}.png"

        plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        print(f"✓ Saved subgraph: {save_path}")

    def create_interactive_visualization(
        self,
        sample_size: int = None,
        save_path: str = None
    ) -> None:
        """
        Create interactive Plotly visualization

        Args:
            sample_size: Sample nodes if graph is too large
            save_path: HTML save path
        """
        log_step(f"Creating interactive visualization for {self.graph_name}")

        # Sample if too large
        if sample_size is None:
            sample_size = self.config['max_nodes_to_display']

        if self.graph.number_of_nodes() > sample_size:
            print(f"Graph too large ({self.graph.number_of_nodes()} nodes). Sampling {sample_size} nodes...")
            nodes = list(self.graph.nodes())[:sample_size]
            subgraph = self.graph.subgraph(nodes)
        else:
            subgraph = self.graph

        # Create layout
        if isinstance(subgraph, nx.MultiDiGraph):
            simple_graph = nx.DiGraph(subgraph)
            pos = nx.spring_layout(simple_graph, k=0.5, iterations=50)
        else:
            pos = nx.spring_layout(subgraph, k=0.5, iterations=50)

        # Create edge traces
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_color = []

        color_map = {
            'Fund': '#FF6B6B',
            'Stock': '#4ECDC4',
            'Sector': '#FFD93D',
            'TimePeriod': '#95E1D3',
            'Factor': '#A8E6CF',
            'Action': '#FDCB9E'
        }

        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            node_data = subgraph.nodes[node]
            node_type = node_data.get('node_type', 'Unknown')
            node_text.append(f"{node}<br>Type: {node_type}<br>Degree: {subgraph.degree(node)}")
            node_color.append(color_map.get(node_type, '#CCCCCC'))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=False,
                color=node_color,
                size=10,
                line_width=2
            )
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f'{self.graph_name} - Interactive Visualization',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800
            )
        )

        if save_path is None:
            save_path = FIGURES_DIR / f"{self.graph_name}_interactive.html"

        fig.write_html(save_path)
        print(f"✓ Saved interactive visualization: {save_path}")

    def plot_temporal_evolution(self, save_path: str = None) -> None:
        """
        Plot temporal evolution of portfolio holdings
        Only applicable for Temporal KG
        """
        if 'Temporal' not in self.graph_name:
            print("Warning: Temporal evolution plot only for Temporal KG")
            return

        log_step("Creating temporal evolution plot")

        # Extract temporal data
        temporal_data = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if data.get('edge_type') == 'HOLDS' and 'date_str' in data:
                temporal_data.append({
                    'fund': u,
                    'stock': v,
                    'date': data['date_str'],
                    'weight': data.get('weight', 0)
                })

        if not temporal_data:
            print("No temporal data found")
            return

        df = pd.DataFrame(temporal_data)

        # Aggregate by date
        df_agg = df.groupby('date')['weight'].sum().reset_index()
        df_agg['date'] = pd.to_datetime(df_agg['date'])
        df_agg = df_agg.sort_values('date')

        # Plot
        plt.figure(figsize=(14, 6))
        plt.plot(df_agg['date'], df_agg['weight'], linewidth=2, marker='o')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Total Portfolio Weight', fontsize=12)
        plt.title(f'Temporal Evolution of Holdings - {self.graph_name}',
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        if save_path is None:
            save_path = FIGURES_DIR / f"{self.graph_name}_temporal_evolution.png"

        plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        print(f"✓ Saved temporal evolution: {save_path}")

    def visualize_all(self) -> None:
        """Generate all visualizations"""
        log_step(f"Generating all visualizations for {self.graph_name}")

        self.plot_degree_distribution()
        self.plot_node_type_distribution()
        self.plot_edge_type_distribution()

        # Plot sample subgraphs
        sample_nodes = list(self.graph.nodes())[:3]
        for node in sample_nodes:
            self.plot_subgraph(node, depth=2)

        # Interactive visualization
        self.create_interactive_visualization()

        # Temporal evolution (if applicable)
        if 'Temporal' in self.graph_name:
            self.plot_temporal_evolution()

        print(f"✓ All visualizations complete for {self.graph_name}")


def main():
    """Test visualization"""
    log_step("Testing Knowledge Graph Visualization")

    # Visualize Temporal KG
    temporal_kg_path = GRAPHS_DIR / "temporal_kg.gpickle"
    if temporal_kg_path.exists():
        temporal_kg = nx.read_gpickle(temporal_kg_path)
        viz = KGVisualizer(temporal_kg, "Temporal_KG")
        viz.visualize_all()
    else:
        print(f"Temporal KG not found at {temporal_kg_path}")

    # Visualize Causal KG
    causal_kg_path = GRAPHS_DIR / "causal_kg.gpickle"
    if causal_kg_path.exists():
        causal_kg = nx.read_gpickle(causal_kg_path)
        viz = KGVisualizer(causal_kg, "Causal_KG")
        viz.visualize_all()
    else:
        print(f"Causal KG not found at {causal_kg_path}")

    print("\n✓ Visualization testing complete!")


if __name__ == "__main__":
    main()
