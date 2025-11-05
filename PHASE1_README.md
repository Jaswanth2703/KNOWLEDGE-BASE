# Phase 1: Knowledge Graph Construction and Evaluation

## Project Overview

This project implements Phase 1 of a research system that captures fund manager decision-making patterns through structured knowledge representations. The system constructs two interconnected knowledge graphs from mutual fund portfolio data:

1. **Temporal Knowledge Graph (KG)**: Models *what* stocks are selected and *when* allocation changes occur
2. **Causal Knowledge Graph (KG)**: Models *why* certain decisions are made based on market conditions, fundamentals, and macroeconomic factors

## Research Context

**Objective**: Imitate fund manager decisions by designing knowledge representation systems that capture both temporal patterns and causal relationships in portfolio construction.

**Data**: 22 mutual funds' monthly portfolio disclosures from January 2022 to November 2025 (94,589 records, 1,762 unique stocks).

**Evaluation**: Phase 1 focuses on intrinsic evaluation using five key metrics:
- Structural Completeness
- Consistency (Integrity)
- Semantic Coherence
- Informativeness
- Inferential Utility

## Project Structure

```
Knowledge_base/
├── phase1_kg/                          # Main package
│   ├── config.py                       # Configuration settings
│   ├── data_collection/                # Data collection modules
│   │   ├── fundamental_data.py         # Stock fundamentals (P/E, ROE, etc.)
│   │   ├── macro_data.py               # Macroeconomic indicators
│   │   └── sentiment_analysis.py       # FinBERT sentiment analysis
│   ├── preprocessing/                  # Data preprocessing
│   │   └── feature_engineering.py      # Portfolio change features
│   ├── knowledge_graphs/               # KG construction
│   │   ├── temporal_kg.py              # Temporal KG builder
│   │   ├── causal_kg.py                # Causal KG with Granger causality
│   │   └── kg_integration.py           # Integrate both KGs
│   ├── evaluation/                     # Evaluation metrics
│   │   ├── intrinsic_metrics.py        # 5 evaluation dimensions
│   │   └── report_generator.py         # Report generation
│   ├── visualization/                  # Visualization tools
│   │   └── kg_viz.py                   # Static & interactive plots
│   └── utils/                          # Utilities
│       └── helpers.py                  # Helper functions
├── data/                               # Data storage
│   ├── raw/                            # Raw data
│   ├── processed/                      # Processed datasets
│   └── cache/                          # Cached results
├── outputs/                            # Generated outputs
│   ├── graphs/                         # Serialized KGs (.gpickle)
│   ├── metrics/                        # Evaluation metrics (.json)
│   ├── figures/                        # Visualizations (.png, .html)
│   └── reports/                        # Evaluation reports (.md, .csv)
├── fund_portfolio_master_smart.csv     # Master dataset
├── run_phase1.py                       # Main execution script
└── requirements.txt                    # Dependencies

```

## Installation

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager

### 2. Install Dependencies

```bash
cd Knowledge_base
pip install -r requirements.txt
```

**Note**: The requirements include:
- NetworkX (graph construction)
- PyTorch & Transformers (FinBERT sentiment)
- statsmodels (Granger causality)
- matplotlib, plotly, seaborn (visualization)
- yfinance (financial data)
- python-louvain (community detection)

## Usage

### Quick Start: Run Complete Pipeline

```bash
python run_phase1.py
```

This executes all steps:
1. Data collection (fundamentals, macro, sentiment)
2. Feature engineering
3. Knowledge graph construction (Temporal + Causal)
4. Intrinsic evaluation
5. Visualization generation
6. Report generation

### Selective Execution

Run specific steps:

```bash
# Run only feature engineering
python run_phase1.py --only-step features

# Run only KG construction
python run_phase1.py --only-step kg

# Run only evaluation
python run_phase1.py --only-step eval

# Run only visualization
python run_phase1.py --only-step viz

# Run only report generation
python run_phase1.py --only-step report
```

Skip specific components:

```bash
# Skip sentiment analysis (time-consuming)
python run_phase1.py --skip-sentiment

# Skip visualizations
python run_phase1.py --skip-viz

# Skip causal KG
python run_phase1.py --skip-causal
```

### Running Individual Modules

Each module can be tested independently:

```bash
# Test fundamental data collection
python phase1_kg/data_collection/fundamental_data.py

# Test sentiment analysis
python phase1_kg/data_collection/sentiment_analysis.py

# Test feature engineering
python phase1_kg/preprocessing/feature_engineering.py

# Test temporal KG
python phase1_kg/knowledge_graphs/temporal_kg.py

# Test causal KG
python phase1_kg/knowledge_graphs/causal_kg.py

# Test evaluation metrics
python phase1_kg/evaluation/intrinsic_metrics.py

# Test visualization
python phase1_kg/visualization/kg_viz.py
```

## Knowledge Graph Structures

### Temporal KG

**Nodes**:
- Fund: Mutual fund entities
- Stock: Individual securities (ISIN)
- Sector: Industry sectors
- TimePeriod: Monthly time periods

**Edges**:
- HOLDS: Fund holds stock with weight at time T
- INCREASED/DECREASED: Portfolio weight changes
- ENTERED/EXITED: New positions or closures
- BELONGS_TO_SECTOR: Stock-sector membership

**Example Query**:
```python
from phase1_kg.knowledge_graphs.temporal_kg import TemporalKG
import networkx as nx

tkg = TemporalKG()
graph = nx.read_gpickle("outputs/graphs/temporal_kg.gpickle")

# Query fund holdings
holdings = tkg.query_fund_holdings("Jaswanth/DSP Small Cap Fund", "2024-12")

# Query portfolio changes
changes = tkg.query_portfolio_changes("Jaswanth/DSP Small Cap Fund", "2024-01", "2024-12")
```

### Causal KG

**Nodes**:
- Observable Factors: Macro indicators, sector metrics
- Intermediate Signals: Derived concepts (valuation scores, risk appetite)
- Portfolio Actions: Allocation decisions

**Edges**:
- CAUSES: Direct causal relationship (Granger causality)
- INFLUENCES: Strong correlation
- PRECEDES: Temporal precedence
- CORRELATED_WITH: Statistical correlation

**Example Query**:
```python
from phase1_kg.knowledge_graphs.causal_kg import CausalKG
import networkx as nx

ckg = CausalKG()
graph = nx.read_gpickle("outputs/graphs/causal_kg.gpickle")

# Find causal paths
paths = ckg.find_causal_paths("MACRO_NIFTY50", "SECTOR_Banking")

# Get influencing factors
factors = ckg.get_influencing_factors("SECTOR_ALLOCATION_IT")
```

## Evaluation Metrics

### 1. Structural Completeness (0-1)
- Node/edge counts and distributions
- Graph density and connectivity
- Coverage of expected entity types

### 2. Consistency (0-1)
- Orphan nodes detection
- Temporal ordering validation
- Causal cycle detection (DAG property)
- Cardinality constraint checking

### 3. Semantic Coherence (0-1)
- Community structure (modularity)
- Sector purity (clustering quality)
- Temporal coherence

### 4. Informativeness (0-1)
- Attribute richness
- Information density
- Redundancy assessment

### 5. Inferential Utility (0-1)
- Path reachability
- Multi-hop query support
- Centrality measures

**Overall Quality Score**: Weighted average of all dimensions

## Output Files

After running the pipeline, you'll find:

### Graphs (outputs/graphs/)
- `temporal_kg.gpickle` - Temporal knowledge graph
- `causal_kg.gpickle` - Causal knowledge graph
- `integrated_kg.gpickle` - Integrated graph

### Metrics (outputs/metrics/)
- `Temporal_KG_evaluation_results.json` - Temporal KG metrics
- `Causal_KG_evaluation_results.json` - Causal KG metrics

### Visualizations (outputs/figures/)
- Degree distributions
- Node/edge type distributions
- Interactive network visualizations (HTML)
- Temporal evolution plots
- Sample subgraphs

### Reports (outputs/reports/)
- `phase1_evaluation_report.md` - Comprehensive markdown report
- `evaluation_summary.csv` - Metrics summary table

## Configuration

Edit `phase1_kg/config.py` to customize:

- Date ranges
- Fundamental metrics to collect
- Macroeconomic indicators
- Sentiment analysis settings
- Granger causality parameters
- Visualization styles
- Neo4j connection (if using)

## Key Design Decisions

### 1. Optimization Strategies
- **Sector-level sentiment**: Instead of per-stock (1,762 × 45 = 79,290 analyses), we analyze per-sector (~10 × 45 = 450 analyses)
- **Synthetic fundamentals**: Due to ISIN-to-ticker mapping challenges, we generate synthetic data. Users can replace with actual yfinance data.
- **Caching**: All data collection results are cached to avoid redundant API calls
- **Sampling**: Large graphs are sampled for visualization to maintain performance

### 2. Storage Choice
- **NetworkX**: Primary storage for flexibility and ease of use
- **pickle/graphml**: Persistence formats
- **Neo4j**: Optional (export functionality included but not required)
- NetworkX provides sufficient query capabilities for Phase 1 evaluation

### 3. Granger Causality
- Tests temporal precedence (does X at t-1 predict Y at t?)
- Lag testing: 1-3 months
- Significance level: 0.05
- DAG enforcement for causal graph integrity

## Troubleshooting

### Common Issues

**1. Missing Data Error**
```
Error: Integrated dataset not found
```
**Solution**: Run feature engineering first:
```bash
python run_phase1.py --only-step features
```

**2. Memory Error**
```
MemoryError during graph construction
```
**Solution**: Reduce sample size in `config.py`:
```python
VIZ_CONFIG['max_nodes_to_display'] = 200
```

**3. Slow Execution**
Sentiment analysis is time-consuming. Skip it:
```bash
python run_phase1.py --skip-sentiment
```

**4. Import Errors**
Ensure you're running from the Knowledge_base directory:
```bash
cd Knowledge_base
python run_phase1.py
```

## Performance Notes

- **Full pipeline runtime**: ~15-30 minutes (without real sentiment analysis)
- **With real sentiment**: 2-4 hours (depends on network speed)
- **Memory usage**: ~2-4GB for full graphs
- **Disk space**: ~500MB for all outputs

## Next Steps: Phase 2

Phase 1 provides the knowledge representations. Phase 2 will:

1. Build portfolio construction models using the KGs
2. Implement stock selection algorithms
3. Perform portfolio optimization
4. Generate explainable AI (natural language explanations)
5. Conduct extrinsic evaluation (performance metrics)

## Citation

If you use this code for research, please cite:

```
@inproceedings{fund_manager_kg_2025,
  title={Knowledge Representation for Imitating Fund Manager Decisions:
         A Temporal and Causal Graph Approach},
  author={[Your Name]},
  booktitle={[Conference/Journal]},
  year={2025}
}
```

## License

[Specify your license]

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: [your email]

---

**Last Updated**: 2025-01-05
**Version**: 1.0.0
