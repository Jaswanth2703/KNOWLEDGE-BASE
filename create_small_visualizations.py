"""
Create SMALL, UNDERSTANDABLE Visualizations
Extract meaningful subgraphs instead of showing entire massive graphs
"""
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).parent / "phase1_kg"))
from phase1_kg.config import GRAPHS_DIR

# Fix pandas pickle compatibility for loading old pickles
# Monkey patch to handle pandas Timestamp unpickling from older versions
import pandas._libs.tslibs.timestamps as timestamps_mod

_original_unpickle_timestamp = timestamps_mod._unpickle_timestamp

def _compatible_unpickle_timestamp(*args):
    """Handle both old (3 args) and new (4 args) pandas Timestamp pickle formats"""
    if len(args) == 3:
        # Old format: (value, freq, tz)
        return _original_unpickle_timestamp(args[0], args[1], args[2], None)
    else:
        # New format: (value, freq, tz, unit)
        return _original_unpickle_timestamp(*args)

timestamps_mod._unpickle_timestamp = _compatible_unpickle_timestamp

# Create new output directory for small visualizations
SMALL_VIZ_DIR = Path(__file__).parent / "outputs" / "small_visualizations"
SMALL_VIZ_DIR.mkdir(parents=True, exist_ok=True)

def visualize_temporal_kg_small():
    """
    Create small, understandable Temporal KG visualization
    Show: 1 fund + top 10 stocks + their actions over 3 recent months
    """
    print("\n" + "="*80)
    print("CREATING SMALL TEMPORAL KG VISUALIZATION")
    print("="*80)

    # Load graph with pandas compatibility
    tkg_path = GRAPHS_DIR / "temporal_kg.gpickle"
    try:
        with open(tkg_path, 'rb') as f:
            full_graph = pickle.load(f)
    except (TypeError, ValueError) as e:
        print(f"  Note: Loading with compatibility mode due to pandas version mismatch")
        # Try loading with bytes encoding for compatibility
        with open(tkg_path, 'rb') as f:
            full_graph = pickle.load(f, encoding='latin1')

    # Select 1 fund
    funds = [n for n, d in full_graph.nodes(data=True) if d.get('node_type') == 'Fund']
    sample_fund = funds[0]

    print(f"\nSelected Fund: {sample_fund}")

    # Get top 10 stocks held by this fund (by total weight over time)
    stock_weights = {}
    for u, v, key, data in full_graph.edges(sample_fund, keys=True, data=True):
        if data.get('edge_type') == 'HOLDS':
            stock = v
            weight = data.get('weight', 0)
            stock_weights[stock] = stock_weights.get(stock, 0) + weight

    top_stocks = sorted(stock_weights.items(), key=lambda x: x[1], reverse=True)[:10]
    top_stock_ids = [s[0] for s in top_stocks]

    print(f"Selected Top 10 Stocks by holdings weight")

    # Get recent 3 months
    time_periods = sorted([n for n, d in full_graph.nodes(data=True)
                          if d.get('node_type') == 'TimePeriod'],
                         key=lambda n: full_graph.nodes[n].get('date'))[-3:]

    recent_dates = [full_graph.nodes[t].get('date') for t in time_periods]
    print(f"Recent 3 months: {[str(d)[:7] for d in recent_dates]}")

    # Create subgraph
    subgraph = nx.MultiDiGraph()

    # Add fund node
    subgraph.add_node(sample_fund, **full_graph.nodes[sample_fund])

    # Add stock nodes
    for stock in top_stock_ids:
        subgraph.add_node(stock, **full_graph.nodes[stock])

    # Add relevant edges (only for recent months)
    edge_count = 0
    for u, v, key, data in full_graph.edges(sample_fund, keys=True, data=True):
        if v in top_stock_ids:
            edge_date = data.get('date')
            if edge_date in recent_dates:
                edge_type = data.get('edge_type')
                subgraph.add_edge(u, v, key=key, **data)
                edge_count += 1

    print(f"\nSubgraph created:")
    print(f"  Nodes: {subgraph.number_of_nodes()}")
    print(f"  Edges: {subgraph.number_of_edges()}")

    # Visualize
    plt.figure(figsize=(20, 12))

    # Layout
    pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)

    # Node colors by type
    node_colors = []
    node_sizes = []
    for node in subgraph.nodes():
        node_data = subgraph.nodes[node]
        node_type = node_data.get('node_type')

        if node_type == 'Fund':
            node_colors.append('#FF4444')  # Red for fund
            node_sizes.append(3000)
        else:  # Stock
            node_colors.append('#4444FF')  # Blue for stocks
            node_sizes.append(1500)

    # Draw nodes
    nx.draw_networkx_nodes(subgraph, pos,
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.9)

    # Draw edges by type with different colors
    edge_colors = {
        'HOLDS': '#888888',
        'INCREASED': '#00AA00',
        'DECREASED': '#AA0000',
        'ENTERED': '#0000AA',
        'EXITED': '#AA00AA'
    }

    for edge_type, color in edge_colors.items():
        edges_of_type = [(u, v) for u, v, k, d in subgraph.edges(keys=True, data=True)
                        if d.get('edge_type') == edge_type]
        if edges_of_type:
            nx.draw_networkx_edges(subgraph, pos,
                                  edgelist=edges_of_type,
                                  edge_color=color,
                                  arrows=True,
                                  arrowsize=20,
                                  width=2,
                                  alpha=0.6,
                                  label=edge_type)

    # Labels
    labels = {}
    labels[sample_fund] = sample_fund[:30]  # Fund name
    for stock in top_stock_ids:
        stock_name = subgraph.nodes[stock].get('stock_name', stock)
        labels[stock] = stock_name[:20] if stock_name else stock[:15]

    nx.draw_networkx_labels(subgraph, pos, labels, font_size=9, font_weight='bold')

    plt.title(f'Temporal KG - Small Sample\n{sample_fund}\nTop 10 Holdings, Recent 3 Months',
             fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.axis('off')
    plt.tight_layout()

    # Save
    save_path = SMALL_VIZ_DIR / "temporal_kg_small.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved to: {save_path}")

    # Create legend explanation
    plt.figure(figsize=(10, 6))
    plt.text(0.1, 0.9, 'TEMPORAL KG LEGEND', fontsize=16, fontweight='bold')

    legend_text = """
Node Colors:
  🔴 Red = Fund (Mutual Fund)
  🔵 Blue = Stock (Individual Security)

Edge Colors (Actions):
  ⚫ Gray = HOLDS (holds position at time T)
  🟢 Green = INCREASED (increased position size)
  🔴 Red = DECREASED (decreased position size)
  🔵 Blue = ENTERED (new position initiated)
  🟣 Purple = EXITED (position closed)

Example Reading:
  Fund --[HOLDS, weight=2.5%]--> Stock X
    → Fund holds 2.5% of portfolio in Stock X at that time

  Fund --[INCREASED, +0.5%]--> Stock Y
    → Fund increased allocation to Stock Y by 0.5%

This graph shows:
  • 1 fund's portfolio actions
  • Top 10 stocks by holding weight
  • Recent 3 months of actions
  • Temporal evolution of holdings
"""

    plt.text(0.1, 0.1, legend_text, fontsize=11, verticalalignment='bottom',
             family='monospace')
    plt.axis('off')

    legend_path = SMALL_VIZ_DIR / "temporal_kg_legend.png"
    plt.savefig(legend_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Legend saved to: {legend_path}")

    return subgraph


def visualize_causal_kg_small():
    """
    Create small, understandable Causal KG visualization
    Show: One sector + its direct causes + effects
    """
    print("\n" + "="*80)
    print("CREATING SMALL CAUSAL KG VISUALIZATION")
    print("="*80)

    # Load graph
    ckg_path = GRAPHS_DIR / "causal_kg.gpickle"
    with open(ckg_path, 'rb') as f:
        full_graph = pickle.load(f)

    # Find a sector node (common connection point)
    sector_nodes = [n for n in full_graph.nodes() if 'SECTOR_' in str(n) and 'ALLOCATION' not in str(n)]

    if not sector_nodes:
        print("No sector nodes found. Using sample of all nodes.")
        # Just take first 15 nodes and their connections
        sample_nodes = list(full_graph.nodes())[:15]
        subgraph = full_graph.subgraph(sample_nodes).copy()
    else:
        sample_sector = sector_nodes[0]
        print(f"\nSelected Sector: {sample_sector}")

        # Get predecessors (what causes this sector)
        predecessors = list(full_graph.predecessors(sample_sector))[:5]  # Max 5

        # Get successors (what this sector causes)
        successors = list(full_graph.successors(sample_sector))[:5]  # Max 5

        print(f"Causes (predecessors): {len(predecessors)}")
        print(f"Effects (successors): {len(successors)}")

        # Create subgraph
        nodes_to_include = [sample_sector] + predecessors + successors
        subgraph = full_graph.subgraph(nodes_to_include).copy()

    print(f"\nSubgraph created:")
    print(f"  Nodes: {subgraph.number_of_nodes()}")
    print(f"  Edges: {subgraph.number_of_edges()}")

    # Visualize
    plt.figure(figsize=(20, 14))

    # Hierarchical layout
    try:
        pos = nx.spring_layout(subgraph, k=3, iterations=100, seed=42)
    except:
        pos = nx.circular_layout(subgraph)

    # Node colors by position
    node_colors = []
    node_sizes = []
    for node in subgraph.nodes():
        if 'SECTOR_' in str(node) and 'ALLOCATION' not in str(node):
            node_colors.append('#FFD700')  # Gold for sector
            node_sizes.append(3000)
        elif 'MACRO_' in str(node) or 'SENTIMENT_' in str(node):
            node_colors.append('#FF6B6B')  # Red for causes
            node_sizes.append(2000)
        else:
            node_colors.append('#4ECDC4')  # Cyan for effects
            node_sizes.append(2000)

    # Draw nodes
    nx.draw_networkx_nodes(subgraph, pos,
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.9)

    # Draw edges by type
    edge_colors_map = {
        'CAUSES': '#00AA00',
        'INFLUENCES': '#0000AA',
        'PRECEDES': '#AA00AA',
        'CORRELATED_WITH': '#888888'
    }

    for edge_type, color in edge_colors_map.items():
        edges_of_type = [(u, v) for u, v, d in subgraph.edges(data=True)
                        if d.get('edge_type') == edge_type]
        if edges_of_type:
            nx.draw_networkx_edges(subgraph, pos,
                                  edgelist=edges_of_type,
                                  edge_color=color,
                                  arrows=True,
                                  arrowsize=25,
                                  width=3,
                                  alpha=0.7,
                                  label=edge_type)

    # Edge labels (show strength)
    edge_labels = {}
    for u, v, data in subgraph.edges(data=True):
        strength = data.get('strength', 0)
        lag = data.get('lag', 0)
        edge_labels[(u, v)] = f"{strength:.2f}"
        if lag > 0:
            edge_labels[(u, v)] += f"\nlag:{lag}m"

    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels, font_size=8)

    # Node labels
    labels = {}
    for node in subgraph.nodes():
        # Truncate long names
        label = str(node)
        if len(label) > 30:
            label = label[:27] + "..."
        labels[node] = label

    nx.draw_networkx_labels(subgraph, pos, labels, font_size=9, font_weight='bold')

    plt.title('Causal KG - Small Sample\nOne Sector with Causes & Effects',
             fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.axis('off')
    plt.tight_layout()

    # Save
    save_path = SMALL_VIZ_DIR / "causal_kg_small.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved to: {save_path}")

    # Create legend
    plt.figure(figsize=(10, 6))
    plt.text(0.1, 0.9, 'CAUSAL KG LEGEND', fontsize=16, fontweight='bold')

    legend_text = """
Node Colors:
  🟡 Gold = Sector (Industry sector being analyzed)
  🔴 Red = Observable Factors (Macro indicators, Sentiment)
  🔵 Cyan = Portfolio Actions (Allocations, Selections)

Edge Colors (Relationships):
  🟢 Green = CAUSES (Granger causality proven, p<0.05)
  🔵 Blue = INFLUENCES (Strong correlation, |r|>0.3)
  🟣 Purple = PRECEDES (Temporal precedence)
  ⚫ Gray = CORRELATED_WITH (Statistical correlation)

Edge Labels:
  Number = Strength (0-1, higher is stronger)
  lag:Xm = Time lag in months

Example Reading:
  MACRO_NIFTY --[CAUSES, 0.85, lag:1m]--> SECTOR_Banking
    → NIFTY index movements CAUSE Banking sector allocation changes
    → Strength: 0.85 (very strong)
    → Lag: 1 month (NIFTY changes predict Banking changes 1 month later)

This graph shows:
  • One central sector
  • What factors CAUSE changes in that sector
  • What this sector influences (downstream effects)
  • Strength and time lag of relationships
"""

    plt.text(0.1, 0.1, legend_text, fontsize=11, verticalalignment='bottom',
             family='monospace')
    plt.axis('off')

    legend_path = SMALL_VIZ_DIR / "causal_kg_legend.png"
    plt.savefig(legend_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Legend saved to: {legend_path}")

    return subgraph


def create_interpretation_guide():
    """
    Create visual guide for interpreting the graphs
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Temporal KG interpretation
    ax1 = axes[0]
    ax1.text(0.5, 0.95, 'HOW TO READ TEMPORAL KG',
            ha='center', fontsize=14, fontweight='bold',
            transform=ax1.transAxes)

    interpretation_temporal = """
QUESTION: What did this fund do with this stock?

LOOK FOR:
1. HOLDS edges → Shows portfolio weight at specific times
   Example: "Fund holds 2.5% in Stock X on 2024-12-31"

2. INCREASED edges → Position size went up
   Example: "Fund increased Stock Y by 0.8% in 2024-11"
   Interpretation: Fund is bullish, adding to position

3. DECREASED edges → Position size went down
   Example: "Fund decreased Stock Z by 1.2% in 2024-10"
   Interpretation: Fund is reducing exposure, possibly taking profits

4. ENTERED edges → New position started
   Example: "Fund entered Stock A in 2024-09"
   Interpretation: Fund initiated new investment

5. EXITED edges → Position completely closed
   Example: "Fund exited Stock B in 2024-08"
   Interpretation: Fund sold entire holding

KEY INSIGHT: Follow the timeline of edges to see portfolio evolution
"""

    ax1.text(0.05, 0.05, interpretation_temporal, fontsize=9,
            verticalalignment='bottom', family='monospace',
            transform=ax1.transAxes)
    ax1.axis('off')

    # Causal KG interpretation
    ax2 = axes[1]
    ax2.text(0.5, 0.95, 'HOW TO READ CAUSAL KG',
            ha='center', fontsize=14, fontweight='bold',
            transform=ax2.transAxes)

    interpretation_causal = """
QUESTION: WHY did the fund make this decision?

LOOK FOR:
1. CAUSES edges (Green) → Proven causal relationship
   Example: "Interest_Rate --CAUSES--> Banking_Sector"
   Interpretation: When interest rates change, Banking allocations change
   Strength: How strong the causality (0.9 = very strong)
   Lag: How many months later the effect occurs

2. INFLUENCES edges (Blue) → Strong correlation
   Example: "Sentiment --INFLUENCES--> Allocation"
   Interpretation: Positive sentiment correlates with increased allocation

3. Follow the causal chain:
   MACRO_VIX → SENTIMENT_Defensive → SECTOR_Pharma → ALLOCATION_Increase

   Reading: "High volatility (VIX) → Defensive sentiment →
             More Pharma allocation → Position increases"

KEY INSIGHT: Trace backwards from action to find root causes
            Trace forwards from factor to see its effects
"""

    ax2.text(0.05, 0.05, interpretation_causal, fontsize=9,
            verticalalignment='bottom', family='monospace',
            transform=ax2.transAxes)
    ax2.axis('off')

    plt.tight_layout()
    save_path = SMALL_VIZ_DIR / "interpretation_guide.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Interpretation guide saved to: {save_path}")


def main():
    print("\n" + "="*80)
    print("  CREATING SMALL, UNDERSTANDABLE VISUALIZATIONS")
    print("="*80 + "\n")

    # Create visualizations
    visualize_temporal_kg_small()
    visualize_causal_kg_small()
    create_interpretation_guide()

    print("\n" + "="*80)
    print("ALL VISUALIZATIONS CREATED SUCCESSFULLY")
    print("="*80)
    print(f"\nOutput directory: {SMALL_VIZ_DIR}")
    print("\nGenerated files:")
    print("  1. temporal_kg_small.png - Small temporal graph sample")
    print("  2. temporal_kg_legend.png - How to read temporal graph")
    print("  3. causal_kg_small.png - Small causal graph sample")
    print("  4. causal_kg_legend.png - How to read causal graph")
    print("  5. interpretation_guide.png - Detailed interpretation guide")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
