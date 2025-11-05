# Real Data Pipeline Guide

## Overview

This guide explains how to run the complete Phase 1 Knowledge Graph construction pipeline using **REAL DATA ONLY** - no synthetic data, no assumptions.

The pipeline collects:
1. **Fundamental Data**: Stock financials from Yahoo Finance (P/E, ROE, Market Cap, etc.)
2. **Macroeconomic Data**: Indian market indices, volatility, currency, commodities
3. **Sentiment Analysis**: FinBERT sentiment analysis on sector-level news

---

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `yfinance` - For financial data
- `pandas`, `numpy` - Data processing
- `transformers`, `torch` - FinBERT sentiment analysis
- `feedparser` - News collection
- `networkx` - Graph construction
- `statsmodels` - Granger causality
- `matplotlib`, `seaborn`, `plotly` - Visualization
- `tqdm` - Progress bars

### 2. Verify Data Files

Ensure you have:
- `fund_portfolio_master_smart.csv` - Portfolio data
- `data/BhavCopy_BSE_CM_*.CSV` - BSE stock data
- `data/BhavCopy_NSE_CM_*.csv` - NSE stock data

---

## Step-by-Step Execution

### Step 1: Create ISIN to Yahoo Finance Ticker Mapping

**Purpose**: Map ISINs from portfolio to Yahoo Finance tickers for data fetching.

```bash
python create_isin_ticker_mapping.py
```

**Output**: `data/isin_ticker_mapping.json`

**What it does**:
- Reads BSE and NSE BhavCopy files
- Extracts ISIN and ticker symbols
- Creates mapping (prioritizes NSE .NS over BSE .BO)
- Shows coverage statistics

**Expected output**:
```
✓ Added 2,500+ NSE mappings
✓ Added 500+ BSE mappings (not on NSE)
✓ Total mappings created: 3,000+
✓ Coverage: 85-95%
```

---

### Step 2: Run Complete Pipeline

**Option A: Run Everything**

```bash
python run_phase1.py
```

This executes all 6 steps:
1. Data collection (fundamentals, macro, sentiment)
2. Feature engineering
3. Knowledge graph construction
4. Evaluation
5. Visualization
6. Report generation

**Option B: Run Specific Steps**

```bash
# Step 1: Data collection only
python run_phase1.py --only-step data

# Step 2: Feature engineering only
python run_phase1.py --only-step features

# Step 3: KG construction only
python run_phase1.py --only-step kg

# Step 4: Evaluation only
python run_phase1.py --only-step eval

# Step 5: Visualization only
python run_phase1.py --only-step viz

# Step 6: Report generation only
python run_phase1.py --only-step report
```

**Option C: Skip Time-Consuming Steps**

```bash
# Skip sentiment analysis (saves 1-2 hours)
python run_phase1.py --skip-sentiment

# Skip visualizations
python run_phase1.py --skip-viz

# Skip causal KG (if only need temporal KG)
python run_phase1.py --skip-causal
```

---

## Data Collection Details

### Fundamental Data Collection

**What it fetches**:
- `trailingPE` - Price-to-Earnings ratio
- `priceToBook` - Price-to-Book ratio
- `returnOnEquity` - Return on Equity
- `revenueGrowth` - Revenue growth rate
- `debtToEquity` - Debt-to-Equity ratio
- `profitMargins` - Profit margins
- `marketCap` - Market capitalization
- `beta` - Stock volatility

**How it works**:
1. Loads portfolio ISINs
2. Maps ISINs to Yahoo tickers using `isin_ticker_mapping.json`
3. Fetches current fundamental data for each stock
4. Replicates data across all dates in portfolio
5. Caches results to avoid re-fetching

**Time estimate**: 10-30 minutes (depends on number of stocks)

**Cache location**: `data/cache/`

---

### Macroeconomic Data Collection

**Indicators fetched** (from Yahoo Finance):

**Indian Market Indices**:
- NIFTY 50 (`^NSEI`)
- NIFTY Bank (`^NSEBANK`)
- NIFTY IT (`^CNXIT`)
- NIFTY Pharma (`^CNXPHARMA`)
- NIFTY Auto (`^CNXAUTO`)
- NIFTY FMCG (`^CNXFMCG`)
- NIFTY Metal (`^CNXMETAL`)
- NIFTY Realty (`^CNXREALTY`)
- NIFTY Energy (`^CNXENERGY`)
- NIFTY Midcap 50 (`^NSEMDCP50`)
- NIFTY Smallcap 50 (`NIFTYSMLCAP50.NS`)

**Volatility & Currency**:
- India VIX (`^INDIAVIX`)
- USD/INR (`INR=X`)

**Global Context**:
- S&P 500 (`^GSPC`)
- US Dollar Index (`DX-Y.NYB`)

**Commodities**:
- Gold (`GC=F`)
- Crude Oil (`CL=F`)
- Brent Crude (`BZ=F`)

**How it works**:
1. Fetches daily data for each indicator (2022-01-31 to 2025-11-30)
2. Resamples to monthly (last value of month)
3. Calculates returns and changes
4. Caches results

**Time estimate**: 2-5 minutes

---

### Sentiment Analysis

**What it does**:
- Fetches sector-level news from Google News RSS
- Analyzes sentiment using FinBERT (financial sentiment model)
- Aggregates to sector-level (not stock-level for efficiency)

**Sectors analyzed**: Banking, IT, Pharma, Energy, Auto, FMCG, Metals, Realty, etc.

**Time estimate**: 1-2 hours (can be skipped with `--skip-sentiment`)

---

## Pipeline Outputs

After running the pipeline, you'll find:

### Processed Data (`data/processed/`)
- `fundamental_data.csv` - Stock fundamentals
- `macro_data.csv` - Macroeconomic indicators
- `sentiment_data.csv` - Sector sentiment scores
- `integrated_dataset.csv` - All data merged

### Knowledge Graphs (`outputs/graphs/`)
- `temporal_kg.gpickle` - Temporal knowledge graph
- `causal_kg.gpickle` - Causal knowledge graph
- `integrated_kg.gpickle` - Combined graph

### Evaluation Metrics (`outputs/metrics/`)
- `Temporal_KG_evaluation_results.json`
- `Causal_KG_evaluation_results.json`

### Visualizations (`outputs/figures/`)
- Degree distributions
- Node/edge type distributions
- Interactive network visualizations (HTML)
- Temporal evolution plots

### Reports (`outputs/reports/`)
- `phase1_evaluation_report.md` - Comprehensive evaluation report
- `evaluation_summary.csv` - Metrics summary

---

## Performance & Timing

**Full pipeline (with sentiment)**:
- Data collection: 1-2 hours
- Feature engineering: 2-5 minutes
- KG construction: 5-10 minutes
- Evaluation: 2-3 minutes
- Visualization: 3-5 minutes
- **Total: ~2-3 hours**

**Full pipeline (without sentiment)**:
- Data collection: 15-30 minutes
- Feature engineering: 2-5 minutes
- KG construction: 5-10 minutes
- Evaluation: 2-3 minutes
- Visualization: 3-5 minutes
- **Total: ~30-45 minutes**

**Subsequent runs** (with cache):
- Most data is cached
- **Total: ~10-15 minutes**

---

## Troubleshooting

### Issue: "No ticker mapping available"

**Solution**: Run the mapping creation script first:
```bash
python create_isin_ticker_mapping.py
```

---

### Issue: "Failed to fetch macroeconomic data"

**Possible causes**:
1. No internet connection
2. Yahoo Finance is down
3. Invalid ticker symbols

**Solutions**:
- Check internet connection
- Try again later
- Verify tickers in `phase1_kg/config.py` (MACRO_INDICATORS)

---

### Issue: "Rate limit exceeded" or "Too many requests"

**Solution**: Yahoo Finance has rate limits. The code includes delays (0.2s per stock), but you may need to:
1. Wait 5-10 minutes
2. Run in smaller batches
3. Use cached data if available

---

### Issue: Memory error during graph construction

**Solution**: Reduce visualization sample size in `phase1_kg/config.py`:
```python
VIZ_CONFIG['max_nodes_to_display'] = 200  # Reduce from 500
```

---

### Issue: Some stocks missing fundamental data

**Expected behavior**: Not all stocks have complete data on Yahoo Finance. The pipeline will:
- Skip stocks without data
- Report success/failure counts
- Continue with available data

**Coverage typically**: 70-85% of stocks have data

---

## Data Quality Checks

The pipeline performs automatic quality checks:

1. **Coverage verification**: Shows % of ISINs with ticker mappings
2. **Success/failure tracking**: Reports how many stocks/indicators fetched successfully
3. **Missing data handling**: Uses left joins to preserve all portfolio records
4. **Validation**: Checks for empty dataframes and warns user

---

## Caching Strategy

All data collection is cached to avoid redundant API calls:

- **Cache location**: `data/cache/`
- **Cache keys**: Based on date range and parameters
- **Cache invalidation**: Automatic (based on parameters)
- **Clear cache**: Delete files in `data/cache/` to force re-fetch

---

## Configuration

Edit `phase1_kg/config.py` to customize:

```python
# Date range
START_DATE = "2022-01-31"
END_DATE = "2025-11-30"

# Fundamental metrics to collect
FUNDAMENTAL_METRICS = [
    "trailingPE", "priceToBook", "returnOnEquity",
    "revenueGrowth", "debtToEquity", "profitMargins",
    "marketCap", "beta"
]

# Macroeconomic indicators (add/remove as needed)
MACRO_INDICATORS = {
    "NIFTY50": "^NSEI",
    "INDIA_VIX": "^INDIAVIX",
    # ... add more
}

# Sentiment analysis settings
SENTIMENT_CONFIG = {
    "model_name": "yiyanghkust/finbert-tone",
    "batch_size": 16,
    "max_length": 512,
    "sector_level": True,
    "cache_enabled": True
}
```

---

## Best Practices

1. **Run mapping first**: Always create ISIN-ticker mapping before data collection
2. **Use cache**: Don't clear cache unless necessary
3. **Skip sentiment initially**: For faster testing, skip sentiment and add later
4. **Monitor progress**: Watch console output for errors/warnings
5. **Check outputs**: Verify CSV files after each step
6. **Network stability**: Ensure stable internet connection for data fetching

---

## Next Steps

After successful pipeline execution:

1. **Review evaluation report**: `outputs/reports/phase1_evaluation_report.md`
2. **Examine visualizations**: `outputs/figures/`
3. **Analyze graphs**: Load `.gpickle` files with NetworkX
4. **Proceed to Phase 2**: Portfolio construction and prediction

---

## Example: Complete Workflow

```bash
# 1. Create ticker mapping
python create_isin_ticker_mapping.py

# 2. Run full pipeline (skip sentiment for speed)
python run_phase1.py --skip-sentiment

# 3. Check outputs
ls -lh data/processed/
ls -lh outputs/graphs/
ls -lh outputs/figures/

# 4. Review report
cat outputs/reports/phase1_evaluation_report.md

# 5. Later: Add sentiment analysis
python run_phase1.py --only-step data --skip-fundamentals --skip-macro

# 6. Rebuild KG with sentiment
python run_phase1.py --only-step features
python run_phase1.py --only-step kg
```

---

## Support

For issues or questions:
1. Check this guide first
2. Review error messages carefully
3. Verify prerequisites and data files
4. Check internet connection and API access
5. Review cache and clear if needed

---

**Last Updated**: 2025-11-05
**Version**: 2.0.0 (Real Data Implementation)
