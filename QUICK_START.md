# Phase 1: Quick Start Guide

## Installation (2 minutes)

```bash
cd "C:\Users\koden\Desktop\Knowledge_base"
pip install -r requirements.txt
```

## Run Complete Pipeline (15-30 minutes)

```bash
python run_phase1.py
```

That's it! The script will:
1. ✅ Collect data (fundamentals, macroeconomics, sentiment)
2. ✅ Engineer features and integrate datasets
3. ✅ Build Temporal Knowledge Graph
4. ✅ Build Causal Knowledge Graph
5. ✅ Integrate both KGs
6. ✅ Evaluate using 5 intrinsic metrics
7. ✅ Generate visualizations
8. ✅ Create comprehensive report

## View Results

### 1. Read the Evaluation Report
```
outputs/reports/phase1_evaluation_report.md
```

### 2. View Visualizations
```
outputs/figures/
├── Temporal_KG_degree_distribution.png
├── Temporal_KG_node_types.png
├── Temporal_KG_interactive.html  ← Open in browser!
├── Causal_KG_degree_distribution.png
└── ... more visualizations
```

### 3. Check Evaluation Metrics
```
outputs/metrics/
├── Temporal_KG_evaluation_results.json
└── Causal_KG_evaluation_results.json
```

### 4. Summary Table
```
outputs/reports/evaluation_summary.csv
```

## If You Want Faster Execution

Skip sentiment analysis (time-consuming):
```bash
python run_phase1.py --skip-sentiment
```

## Run Specific Steps Only

```bash
# Just build the knowledge graphs
python run_phase1.py --only-step kg

# Just generate visualizations
python run_phase1.py --only-step viz

# Just create the report
python run_phase1.py --only-step report
```

## Test Individual Modules

```bash
# Test feature engineering
python phase1_kg/preprocessing/feature_engineering.py

# Test temporal KG construction
python phase1_kg/knowledge_graphs/temporal_kg.py

# Test evaluation metrics
python phase1_kg/evaluation/intrinsic_metrics.py
```

## Expected Outputs

### Knowledge Graphs
- **Temporal KG**: ~2,000+ nodes, ~50,000+ edges
- **Causal KG**: ~50-200 nodes, ~100-500 edges

### Evaluation Scores
All scores are 0-1 (higher is better):
- Structural Completeness
- Consistency
- Semantic Coherence
- Informativeness
- Inferential Utility
- **Overall Quality Score**

### Visualizations
- Static plots (PNG): Degree distributions, node types, etc.
- Interactive plots (HTML): Zoomable network graphs
- Temporal evolution: Portfolio changes over time

## Common Questions

**Q: How long does it take?**
A: 15-30 minutes without real sentiment analysis, 2-4 hours with real sentiment.

**Q: Can I use real stock data instead of synthetic?**
A: Yes! Edit `phase1_kg/data_collection/fundamental_data.py` to use actual yfinance API calls. You'll need an ISIN-to-ticker mapping file.

**Q: The graphs are too large to visualize?**
A: Edit `phase1_kg/config.py` and reduce `VIZ_CONFIG['max_nodes_to_display']` to 200 or 300.

**Q: Where is the master dataset?**
A: `fund_portfolio_master_smart.csv` in the root directory (94,589 records).

**Q: Can I export to Neo4j?**
A: Yes! The code includes Neo4j export functionality (currently commented out). Uncomment in `run_phase1.py` and configure connection in `config.py`.

**Q: How do I query the knowledge graphs?**
A: Load with NetworkX:
```python
import networkx as nx
temporal_kg = nx.read_gpickle("outputs/graphs/temporal_kg.gpickle")

# Example: Get all fund nodes
funds = [n for n, d in temporal_kg.nodes(data=True) if d.get('node_type') == 'Fund']
```

## Need Help?

1. Check `PHASE1_README.md` for detailed documentation
2. Review error messages in console output
3. Check the Troubleshooting section in README

## Next Steps

After completing Phase 1:
1. ✅ Review evaluation report
2. ✅ Examine visualizations
3. ✅ Validate knowledge graph quality
4. 🔜 Proceed to Phase 2 (portfolio construction and extrinsic evaluation)

---

**Ready? Let's go!**

```bash
python run_phase1.py
```
