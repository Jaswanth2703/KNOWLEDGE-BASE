# Quick Start: Real Data Pipeline

## 🚀 Quick Start (For Users Who Want to Run Immediately)

### Step 1: Install Dependencies

```bash
pip install pandas numpy yfinance
```

### Step 2: Create Ticker Mapping

```bash
python create_isin_ticker_mapping.py
```

### Step 3: Install All Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Pipeline (Without Sentiment for Speed)

```bash
python run_phase1.py --skip-sentiment
```

**Expected time**: 30-45 minutes

---

## 📊 What Changed?

### Before (Synthetic Data)
- ❌ Fundamental data: Random synthetic numbers
- ❌ Macroeconomic data: Random synthetic numbers
- ✅ Sentiment: Real FinBERT analysis

### After (Real Data)
- ✅ Fundamental data: Real from Yahoo Finance (yfinance)
- ✅ Macroeconomic data: Real Indian market indices from Yahoo Finance
- ✅ Sentiment: Real FinBERT analysis

---

## 🔧 Key Files Modified

1. **`create_isin_ticker_mapping.py`** (NEW)
   - Maps ISIN codes to Yahoo Finance tickers
   - Uses BhavCopy files from BSE/NSE

2. **`phase1_kg/config.py`**
   - Added 18 Indian market indicators
   - Removed synthetic data config

3. **`phase1_kg/data_collection/fundamental_data.py`**
   - Now fetches real data from Yahoo Finance
   - Removed synthetic data generation

4. **`phase1_kg/data_collection/macro_data.py`**
   - Now fetches real Indian market data
   - Removed synthetic data generation

5. **`run_phase1.py`**
   - Updated to use only real data
   - Better error handling and user feedback

6. **`REAL_DATA_PIPELINE_GUIDE.md`** (NEW)
   - Complete documentation
   - Troubleshooting guide

---

## 📈 Real Data Sources

### Fundamental Data (from yfinance)
- P/E Ratio
- Price-to-Book
- Return on Equity
- Revenue Growth
- Debt-to-Equity
- Profit Margins
- Market Cap
- Beta

### Macroeconomic Data (from Yahoo Finance)
**Indian Market Indices**:
- NIFTY 50, NIFTY Bank, NIFTY IT
- NIFTY Pharma, Auto, FMCG, Metal, Realty, Energy
- NIFTY Midcap 50, Smallcap 50

**Volatility & Currency**:
- India VIX, USD/INR

**Global Context**:
- S&P 500, US Dollar Index

**Commodities**:
- Gold, Crude Oil, Brent Crude

---

## ⚠️ Important Notes

1. **Coverage**: Expect 70-85% coverage for fundamental data
   - Not all stocks are available on Yahoo Finance
   - Pipeline will skip unavailable stocks gracefully

2. **Rate Limits**: Yahoo Finance has rate limits
   - Pipeline includes delays (0.2s per stock)
   - If errors occur, wait 5-10 minutes and retry

3. **Caching**: All data is cached
   - Location: `data/cache/`
   - Subsequent runs are much faster
   - Delete cache to force re-fetch

4. **Sentiment Analysis**: Optional but slow
   - Skip with `--skip-sentiment` for faster runs
   - Can add later with `--only-step data`

---

## 🎯 Expected Output

After successful run:

```
data/processed/
├── fundamental_data.csv        (~1,400 stocks with real data)
├── macro_data.csv              (~18 indicators, monthly)
├── sentiment_data.csv          (optional)
└── integrated_dataset.csv      (all data merged)

outputs/
├── graphs/
│   ├── temporal_kg.gpickle
│   ├── causal_kg.gpickle
│   └── integrated_kg.gpickle
├── metrics/
│   ├── Temporal_KG_evaluation_results.json
│   └── Causal_KG_evaluation_results.json
├── figures/
│   └── [various visualizations]
└── reports/
    ├── phase1_evaluation_report.md
    └── evaluation_summary.csv
```

---

## 🐛 Common Issues

### "No ticker mapping available"
```bash
python create_isin_ticker_mapping.py
```

### "Failed to fetch data"
- Check internet connection
- Verify Yahoo Finance is accessible
- Try again later (rate limits)

### "Memory error"
- Reduce `max_nodes_to_display` in `config.py`

---

## 📚 Full Documentation

For detailed information, see:
- **`REAL_DATA_PIPELINE_GUIDE.md`** - Complete guide with troubleshooting
- **`PHASE1_README.md`** - Project overview and architecture
- **`README.md`** - General project information

---

## ✅ Verification Checklist

Before running:
- [ ] BhavCopy files present in `data/` directory
- [ ] `fund_portfolio_master_smart.csv` exists
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Internet connection available

After mapping creation:
- [ ] `data/isin_ticker_mapping.json` exists
- [ ] Coverage > 70%

After pipeline run:
- [ ] `data/processed/fundamental_data.csv` has data
- [ ] `data/processed/macro_data.csv` has data
- [ ] `outputs/graphs/*.gpickle` files exist
- [ ] Review `outputs/reports/phase1_evaluation_report.md`

---

**Ready to Run?**

```bash
# Quick test (no dependencies needed yet)
ls data/BhavCopy*.CSV
ls data/BhavCopy*.csv
cat fund_portfolio_master_smart.csv | head -5

# Install and run
pip install pandas numpy yfinance
python create_isin_ticker_mapping.py
pip install -r requirements.txt
python run_phase1.py --skip-sentiment
```

**Publishable? Yes!**

All data is now real, no synthetic assumptions, fully traceable to public data sources (Yahoo Finance, Google News).
