"""
Simple Knowledge Graph Query Script
Tests that your KGs work and can answer useful questions
"""
import pickle
import sys
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).parent / "phase1_kg"))
from phase1_kg.config import GRAPHS_DIR

print("="*80)
print("KNOWLEDGE GRAPH VERIFICATION")
print("="*80)

# Try to load graphs with multiple methods
print("\nAttempting to load graphs...")

temporal_kg = None
causal_kg = None

# Method 1: Standard pickle
try:
    with open(GRAPHS_DIR / "temporal_kg.gpickle", 'rb') as f:
        temporal_kg = pickle.load(f)
    print("✓ Temporal KG loaded (standard method)")
except Exception as e:
    print(f"× Temporal KG load failed: {type(e).__name__}")
    # Method 2: With encoding
    try:
        with open(GRAPHS_DIR / "temporal_kg.gpickle", 'rb') as f:
            temporal_kg = pickle.load(f, encoding='latin1')
        print("✓ Temporal KG loaded (latin1 encoding)")
    except Exception as e2:
        print(f"× Temporal KG load failed again: {type(e2).__name__}")

# Load causal KG
try:
    with open(GRAPHS_DIR / "causal_kg.gpickle", 'rb') as f:
        causal_kg = pickle.load(f)
    print("✓ Causal KG loaded (standard method)")
except Exception as e:
    print(f"× Causal KG load failed: {type(e).__name__}")
    try:
        with open(GRAPHS_DIR / "causal_kg.gpickle", 'rb') as f:
            causal_kg = pickle.load(f, encoding='latin1')
        print("✓ Causal KG loaded (latin1 encoding)")
    except Exception as e2:
        print(f"× Causal KG load failed again: {type(e2).__name__}")

if temporal_kg is None and causal_kg is None:
    print("\n" + "="*80)
    print("ERROR: Could not load knowledge graphs")
    print("="*80)
    print("\nRECOMMENDATION:")
    print("Your pickle files were created with a different pandas version.")
    print("Please run: python run_phase1.py --skip-data-collection")
    print("This will rebuild the KGs with your current pandas version.")
    sys.exit(1)

# Show what we have
print("\n" + "="*80)
print("KNOWLEDGE GRAPH CONTENTS")
print("="*80)

if temporal_kg:
    print(f"\nTEMPORAL KG:")
    print(f"  Nodes: {temporal_kg.number_of_nodes():,}")
    print(f"  Edges: {temporal_kg.number_of_edges():,}")

    # Node types
    node_types = defaultdict(int)
    for node, data in temporal_kg.nodes(data=True):
        node_types[data.get('node_type', 'Unknown')] += 1

    print(f"\n  Node Types:")
    for ntype, count in sorted(node_types.items()):
        print(f"    {ntype:15s}: {count:5,d}")

    # Edge types
    edge_types = defaultdict(int)
    for u, v, data in temporal_kg.edges(data=True):
        edge_types[data.get('edge_type', 'Unknown')] += 1

    print(f"\n  Edge Types:")
    for etype, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
        print(f"    {etype:20s}: {count:7,d}")

if causal_kg:
    print(f"\nCAUSAL KG:")
    print(f"  Nodes: {causal_kg.number_of_nodes()}")
    print(f"  Edges: {causal_kg.number_of_edges()}")

    # Node types
    node_types = defaultdict(int)
    for node, data in causal_kg.nodes(data=True):
        node_types[data.get('node_type', 'Unknown')] += 1

    print(f"\n  Node Types:")
    for ntype, count in sorted(node_types.items()):
        print(f"    {ntype:15s}: {count:5d}")

    # Edge types
    edge_types = defaultdict(int)
    for u, v, data in causal_kg.edges(data=True):
        edge_types[data.get('edge_type', 'Unknown')] += 1

    print(f"\n  Edge Types:")
    for etype, count in sorted(edge_types.items()):
        print(f"    {etype:20s}: {count:3d}")

# Sample queries if graphs loaded successfully
if temporal_kg:
    print("\n" + "="*80)
    print("SAMPLE QUERIES")
    print("="*80)

    # Get fund nodes
    fund_nodes = [n for n, d in temporal_kg.nodes(data=True)
                  if d.get('node_type') == 'Fund']

    if fund_nodes:
        print(f"\n1. FUNDS IN GRAPH: {len(fund_nodes)} funds")
        for i, fund in enumerate(fund_nodes[:5], 1):
            fund_data = temporal_kg.nodes[fund]
            print(f"   {i}. {fund}")
            print(f"      Holdings: {fund_data.get('total_holdings', 'N/A')} stocks")

        # Query holdings for first fund
        sample_fund = fund_nodes[0]
        print(f"\n2. SAMPLE HOLDINGS for {sample_fund}:")

        holdings = []
        try:
            for neighbor in list(temporal_kg.neighbors(sample_fund))[:10]:
                edges = temporal_kg.get_edge_data(sample_fund, neighbor)
                if edges:
                    for key, edge_data in list(edges.items())[:1]:
                        if edge_data.get('edge_type') == 'HOLDS':
                            stock_name = temporal_kg.nodes.get(neighbor, {}).get('stock_name', neighbor)
                            holdings.append({
                                'stock': stock_name,
                                'weight': edge_data.get('weight', 0),
                                'date': edge_data.get('date_str', 'unknown')
                            })

            if holdings:
                holdings.sort(key=lambda x: x['weight'], reverse=True)
                for h in holdings[:5]:
                    print(f"   {h['stock'][:40]:40s} {h['weight']*100:6.2f}%  ({h['date']})")
            else:
                print("   No HOLDS edges found (try different fund)")
        except Exception as e:
            print(f"   Query error: {e}")

        # Query changes
        print(f"\n3. RECENT PORTFOLIO CHANGES:")
        changes = defaultdict(int)
        try:
            for u, v, data in list(temporal_kg.edges(data=True))[:1000]:
                etype = data.get('edge_type')
                if etype in ['INCREASED', 'DECREASED', 'ENTERED', 'EXITED']:
                    changes[etype] += 1

            for change_type, count in sorted(changes.items(), key=lambda x: x[1], reverse=True):
                print(f"   {change_type:15s}: {count:5,d} actions")
        except Exception as e:
            print(f"   Query error: {e}")

if causal_kg:
    print(f"\n4. CAUSAL RELATIONSHIPS:")

    try:
        for u, v, data in list(causal_kg.edges(data=True))[:5]:
            print(f"\n   {u}")
            print(f"   → {data.get('edge_type', 'UNKNOWN')} → {v}")
            print(f"     Strength: {data.get('strength', 0):.3f}")
            print(f"     Lag: {data.get('lag', 0)} months")
            print(f"     P-value: {data.get('p_value', 1.0):.4f}")
    except Exception as e:
        print(f"   Query error: {e}")

print("\n" + "="*80)
if temporal_kg or causal_kg:
    print("✓ KNOWLEDGE GRAPHS ARE QUERYABLE!")
else:
    print("× KNOWLEDGE GRAPHS COULD NOT BE LOADED")
print("="*80 + "\n")

if temporal_kg or causal_kg:
    print("Next steps:")
    print("1. Read PHASE1_COMPLETE_EXPLANATION.txt for full documentation")
    print("2. Use the query patterns shown above for your own analysis")
    print("3. Proceed to Phase 2 for downstream tasks")
