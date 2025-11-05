"""
Configuration file for Phase 1 Knowledge Graph Construction
"""
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "outputs"
GRAPHS_DIR = OUTPUT_DIR / "graphs"
METRICS_DIR = OUTPUT_DIR / "metrics"
REPORTS_DIR = OUTPUT_DIR / "reports"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR,
                  OUTPUT_DIR, GRAPHS_DIR, METRICS_DIR, REPORTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data paths
MASTER_CSV_PATH = BASE_DIR / "fund_portfolio_master_smart.csv"

# Date range
START_DATE = "2022-01-31"
END_DATE = "2025-11-30"

# Fundamental data configuration
FUNDAMENTAL_METRICS = [
    "trailingPE",           # P/E ratio
    "priceToBook",          # P/B ratio
    "returnOnEquity",       # ROE
    "revenueGrowth",        # Revenue growth
    "debtToEquity",         # Debt/Equity
    "profitMargins",        # Profit margin
    "marketCap",            # Market capitalization
    "beta"                  # Beta (volatility)
]

# Macroeconomic indicators
MACRO_INDICATORS = {
    "NIFTY50": "^NSEI",           # NIFTY 50 Index
    "INDIA_VIX": "^INDIAVIX",     # India VIX
    "USD_INR": "INR=X",           # USD/INR exchange rate
    "10Y_BOND": "^TNX"            # 10-year Treasury (US as proxy)
}

# Sector mapping (standard sectors)
STANDARD_SECTORS = [
    "Automobile", "Banking", "Capital Goods", "Chemicals", "Consumer Durables",
    "Energy", "Financial Services", "FMCG", "Healthcare", "Infrastructure",
    "IT", "Media", "Metals", "Oil & Gas", "Pharma", "Power", "Realty",
    "Telecom", "Textiles", "Other"
]

# Sentiment analysis configuration
SENTIMENT_CONFIG = {
    "model_name": "yiyanghkust/finbert-tone",
    "batch_size": 16,
    "max_length": 512,
    "news_sources": ["google_news"],
    "lookback_days": 30,  # Aggregate news from past 30 days
    "sector_level": True,  # True = sector-level, False = stock-level
    "cache_enabled": True
}

# Knowledge graph configuration
TEMPORAL_KG_CONFIG = {
    "change_threshold": 0.01,  # Minimum 1% change to create edge
    "relationship_types": [
        "HOLDS",           # Fund holds stock at time T
        "INCREASED",       # Fund increased position
        "DECREASED",       # Fund decreased position
        "ENTERED",         # New position
        "EXITED",          # Position closed
        "BELONGS_TO_SECTOR"  # Stock belongs to sector
    ]
}

CAUSAL_KG_CONFIG = {
    "granger_max_lag": 3,        # Test up to 3-month lag
    "granger_significance": 0.05, # 5% significance level
    "correlation_threshold": 0.3, # Minimum correlation for edge
    "min_observations": 12,       # Minimum 12 months of data
    "relationship_types": [
        "CAUSES",           # Direct causal relationship
        "INFLUENCES",       # Strong correlation
        "PRECEDES",         # Temporal precedence
        "CORRELATED_WITH"   # Correlation relationship
    ]
}

# Evaluation metrics configuration
EVALUATION_CONFIG = {
    "structural_metrics": [
        "node_count", "edge_count", "density", "average_degree",
        "degree_distribution", "clustering_coefficient", "connected_components"
    ],
    "semantic_metrics": [
        "modularity", "community_structure", "sector_purity",
        "temporal_coherence"
    ],
    "consistency_metrics": [
        "cardinality_violations", "temporal_violations",
        "causal_cycles", "orphan_nodes"
    ],
    "inferential_metrics": [
        "path_length_distribution", "reachability",
        "query_response_time"
    ]
}

# Visualization configuration
VIZ_CONFIG = {
    "figure_size": (16, 12),
    "node_size_range": (100, 3000),
    "edge_width_range": (0.5, 5),
    "color_scheme": "Set3",
    "layout_algorithm": "spring",  # spring, kamada_kawai, circular
    "max_nodes_to_display": 500,  # For performance
    "dpi": 300
}

# Optimization settings
OPTIMIZATION = {
    "use_cache": True,
    "parallel_processing": True,
    "n_jobs": -1,  # Use all CPU cores
    "chunk_size": 100,
    "verbose": True
}

# Neo4j configuration (optional)
NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "password",  # Change this!
    "database": "fundmanager_kg"
}

print(f"Configuration loaded. Master CSV: {MASTER_CSV_PATH}")
print(f"Date range: {START_DATE} to {END_DATE}")
print(f"Output directory: {OUTPUT_DIR}")
