# Knowledge Graph Methodology Explained

## 1. How to Check if Intrinsic Metrics are Good

### Metric Score Interpretation

| Score Range | Quality Level | Meaning |
|------------|---------------|---------|
| 0.90 - 1.00 | Excellent | Production-ready, high confidence for downstream tasks |
| 0.80 - 0.89 | Good | Acceptable for research, minor improvements possible |
| 0.70 - 0.79 | Fair | Usable but needs improvement before deployment |
| 0.60 - 0.69 | Poor | Significant issues, may impact downstream performance |
| < 0.60 | Critical | Not suitable for downstream tasks, requires fixes |

### What Each Metric Tells You

**1. Structural Completeness (0-1 score)**
- **What it measures**: Are all required nodes and edges present?
- **Good score**: > 0.85
- **Why it matters**: Missing data = incomplete decisions in downstream tasks
- **Example**: If completeness = 0.92, it means 92% of expected portfolio holdings are captured

**2. Consistency & Integrity (0-1 score)**
- **What it measures**: Are there logical errors or contradictions?
- **Good score**: > 0.95 (higher is critical here)
- **Why it matters**: Inconsistent data will cause wrong predictions
- **Example**: Fund can't simultaneously ENTER and EXIT same stock on same date

**3. Semantic Coherence (0-1 score)**
- **What it measures**: Do related entities cluster together meaningfully?
- **Good score**: > 0.75
- **Why it matters**: Good clustering = better pattern recognition
- **Example**: Tech stocks should group together, not scattered randomly

**4. Informativeness (0-1 score)**
- **What it measures**: How much useful information does each node/edge carry?
- **Good score**: > 0.80
- **Why it matters**: Rich metadata = better decision explanations
- **Example**: Edge with weight, date, market_value is more informative than just weight

**5. Inferential Utility (0-1 score)**
- **What it measures**: Can you answer complex queries from the graph?
- **Good score**: > 0.70
- **Why it matters**: This directly predicts downstream task success
- **Example**: Can you find "Which funds increased tech stocks during market crash?"

### Overall Assessment Rule

```
if ALL metrics > 0.80:
    ✓ Downstream tasks will perform well
    ✓ Safe to proceed to Phase 2

elif MOST metrics > 0.70 AND consistency > 0.90:
    ⚠ Proceed with caution
    ⚠ Document known limitations

else:
    ✗ Fix data quality issues first
    ✗ Downstream tasks will have poor performance
```

**Critical Rule**: Consistency must ALWAYS be > 0.90, even if other metrics are lower.

---

## 2. How Knowledge Graphs Will Be Used in Downstream Tasks (Phase 2)

### Overview: From KGs to Portfolio Construction

```
Phase 1 (Current):                    Phase 2 (Next):
┌─────────────────┐                  ┌──────────────────────┐
│ Build KGs       │                  │ Portfolio Imitation  │
│ • Temporal KG   │───────────────>  │ • Query KGs          │
│ • Causal KG     │                  │ • Find patterns      │
│ • Integration   │                  │ • Make decisions     │
└─────────────────┘                  └──────────────────────┘
        │                                      │
        │                                      ▼
        ▼                              ┌──────────────────────┐
┌─────────────────┐                  │ Explainable AI       │
│ Evaluate (5     │                  │ • Natural language   │
│  metrics)       │                  │ • Decision reasons   │
└─────────────────┘                  └──────────────────────┘
```

### What We Extract from Each KG

#### From Temporal KG: **WHAT** and **WHEN**

**Use Case 1: Find Similar Past Decisions**
```
Query: "Fund X wants to decide about Stock Y in current month"

Extract from Temporal KG:
1. What did Fund X do with Stock Y historically?
   → Holdings over time, increases/decreases

2. What did similar funds do with Stock Y?
   → Find funds with similar portfolios
   → See their actions on Stock Y

3. What did Fund X do with similar stocks?
   → Same sector, same characteristics
   → Identify patterns (e.g., "always exits banking stocks in Q4")
```

**Use Case 2: Identify Temporal Patterns**
```
Query: "What actions do top-performing funds take during market crashes?"

Extract from Temporal KG:
1. Filter time periods where NIFTY dropped > 10%
2. Find funds with positive returns in those periods
3. Extract their INCREASED/ENTERED edges
4. Identify common stocks/sectors
→ Pattern: "Defensive funds increase pharma/FMCG during crashes"
```

**Use Case 3: Portfolio Evolution**
```
Query: "How should our portfolio evolve month-to-month?"

Extract from Temporal KG:
1. Get current holdings (HOLDS edges for latest month)
2. Compare to holdings 1 month ago
3. Calculate churn rate (% of portfolio changed)
4. Find optimal churn rate from historical data
→ Decision: "Change 15% of portfolio monthly (historical average)"
```

#### From Causal KG: **WHY**

**Use Case 1: Find Decision Drivers**
```
Query: "Why are funds increasing IT sector stocks?"

Extract from Causal KG:
1. Find causal edges pointing to SECTOR_IT
2. Check macro factors:
   → USD_INR increase → IT exports more profitable
   → NIFTY_IT rising → momentum effect
   → Positive sentiment → market confidence

→ Reason: "3 causal factors support IT sector increase"
```

**Use Case 2: Predict Impact of Macro Changes**
```
Query: "Interest rates are rising. What will funds do?"

Extract from Causal KG:
1. Find all edges from INTEREST_RATE node
2. Identify affected sectors:
   → INFLUENCES → SECTOR_Banking (positive, lag=1)
   → INFLUENCES → SECTOR_Real_Estate (negative, lag=2)

→ Prediction: "Increase banking stocks next month, reduce real estate in 2 months"
```

**Use Case 3: Risk Assessment**
```
Query: "Is increasing Pharma stocks risky now?"

Extract from Causal KG:
1. Find causal predecessors of SECTOR_Pharma
2. Check current values of those factors:
   → USD_INR: Rising (favorable)
   → Sentiment: Neutral (no signal)
   → NIFTY_PHARMA: Declining (unfavorable)

→ Risk: "Mixed signals, 1 favorable + 1 unfavorable = moderate risk"
```

#### From Integrated KG: **WHAT + WHEN + WHY**

**Use Case 1: Complete Decision Explanation**
```
Query: "Should we increase Stock ABC?"

Step 1: Check Temporal KG
→ "Top funds increased ABC 3 months ago by avg 2%"

Step 2: Check Causal KG
→ "ABC's sector driven by rising exports (USD_INR ↑) and positive sentiment"

Step 3: Use Integration Layer
→ Link temporal pattern to causal explanation
→ Generate: "Increase ABC by 2% because sector fundamentals support it
   (USD_INR rising, positive sentiment) and peer funds already positioned"
```

**Use Case 2: Anomaly Detection**
```
Query: "Fund X decreased Stock Y, but sector fundamentals are strong. Why?"

Step 1: Temporal KG shows DECREASED edge
Step 2: Causal KG shows sector has 4 positive factors
Step 3: Integration layer identifies mismatch

→ Output: "Anomaly detected. Possible reasons:
   1. Stock-specific issue (not in our KG)
   2. Fund strategy change
   3. Data quality issue
   → Flag for manual review"
```

---

## 3. What the Integration Layer Does

### Purpose: Bridge Temporal Events with Causal Explanations

The integration layer creates **cross-reference edges** that link:
- Temporal events (e.g., "Fund A increased Stock B on 2024-12")
- Causal factors (e.g., "USD_INR rising influences IT sector")

### How It Works (Step-by-Step)

**Step 1: Identify Shared Entities**

```python
Temporal KG has:        Causal KG has:
- Stock: INE123456      - Stock: INE123456 (same)
- SECTOR_IT             - SECTOR_IT (same)
- Fund_ABC              - (not present)
- TIME_2024-12          - (not present)
```

Shared entities = {Stocks, Sectors}

**Step 2: Create Cross-References**

For each portfolio change in Temporal KG:
```python
Event: Fund_ABC → INCREASED → Stock_XYZ (Date: 2024-12)

1. Get stock's sector from Temporal KG
   → Stock_XYZ belongs to SECTOR_IT

2. Find causal factors for SECTOR_IT in Causal KG
   → USD_INR INFLUENCES SECTOR_IT (strength=0.65, lag=1)
   → Sentiment INFLUENCES SECTOR_IT (strength=0.42, lag=0)

3. Create cross-reference edges
   → USD_INR → EXPLAINED_BY → "Fund_ABC-Stock_XYZ-INCREASED-2024-12"
   → Sentiment → EXPLAINED_BY → "Fund_ABC-Stock_XYZ-INCREASED-2024-12"
```

**Step 3: Query Integrated Graph**

```python
query_decision_explanation(
    fund_name="Fund_ABC",
    stock_isin="INE123456",
    date="2024-12"
)

Returns:
{
    'decision': 'INCREASED',
    'temporal_context': {
        'weight_change': 0.02,
        'prev_weight': 0.03,
        'new_weight': 0.05
    },
    'causal_factors': [
        {
            'factor': 'USD_INR',
            'relationship': 'INFLUENCES',
            'strength': 0.65,
            'lag': 1
        },
        {
            'factor': 'Sentiment_IT',
            'relationship': 'INFLUENCES',
            'strength': 0.42,
            'lag': 0
        }
    ]
}
```

### Why Integration Layer Matters

**Without Integration:**
- Temporal KG: "Fund increased Stock X by 2%"
- Causal KG: "USD_INR influences IT sector"
- **Problem**: No connection between them!

**With Integration:**
- "Fund increased Stock X by 2% **because** USD_INR is rising (strength=0.65), which influences IT sector"
- **Benefit**: Complete, explainable decision

### Real-World Example

```
Scenario: Advisee needs to explain to investor why they're buying Infosys stock

Without KG Integration:
"We're buying Infosys."
→ Investor: "Why?"
→ Advisee: "Uh... good fundamentals?"
→ Investor: "Be specific!"

With KG Integration:
query_decision_explanation("My_Fund", "INE009A01021", "2024-12")

→ "We're buying Infosys (2% position) because:
   1. Top-performing funds increased IT stocks last month (Temporal KG)
   2. USD/INR exchange rate rising benefits IT exporters (Causal KG, strength=0.65)
   3. IT sector sentiment turned positive (Causal KG, strength=0.42)
   4. Similar funds (Fund_X, Fund_Y) already positioned in Infosys (Temporal KG)

   Expected impact: Position will benefit if USD/INR continues rising (lag=1 month)"
```

---

## 4. Methodology Flow Summary

### Temporal KG Construction
```
Input: Portfolio holdings data (22 funds, 45 months, 1,762 stocks)
  ↓
Add Nodes: Funds (22) + Stocks (1,762) + Sectors (20) + Time (45)
  ↓
Add HOLDS edges: Fund → Stock with (date, weight, market_value)
  ↓
Add CHANGE edges: INCREASED/DECREASED/ENTERED/EXITED
  ↓
Output: MultiDiGraph with temporal relationships
```

### Causal KG Construction
```
Input: Macro data + Fundamentals + Sentiment
  ↓
Granger Causality Testing (max_lag=3, p_value < 0.05)
  ↓
Create nodes: Macro factors + Sectors
  ↓
Add INFLUENCES edges with (strength, lag, p_value)
  ↓
DAG Validation: Remove cycles, keep strongest edges
  ↓
Output: DiGraph with causal relationships
```

### Integration
```
Input: Temporal KG + Causal KG
  ↓
Identify shared entities (Stocks, Sectors)
  ↓
For each portfolio change:
  - Get stock's sector
  - Find causal factors for sector
  - Create EXPLAINED_BY edges
  ↓
Output: Unified graph with cross-references
```

### Evaluation (5 Metrics)
```
Input: All 3 KGs
  ↓
Structural Completeness: Check node/edge coverage
Consistency & Integrity: Validate logical constraints
Semantic Coherence: Measure community structure
Informativeness: Count attributes per node/edge
Inferential Utility: Test query success rate
  ↓
Output: 5 scores (0-1) indicating quality
```

---

## 5. Quick Reference: Threshold Values

| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| Structural Completeness | 0.80 | 0.85 | 0.90+ |
| Consistency & Integrity | 0.90 | 0.95 | 0.98+ |
| Semantic Coherence | 0.70 | 0.75 | 0.85+ |
| Informativeness | 0.75 | 0.80 | 0.90+ |
| Inferential Utility | 0.65 | 0.70 | 0.80+ |

**Overall System Quality:**
- Sum of all 5 metrics > 4.0 = Good system
- Sum of all 5 metrics > 4.5 = Excellent system
- Consistency must ALWAYS be > 0.90 regardless of others

---

## 6. Expected Outputs for Your Project

Based on your data (22 funds, 1,762 stocks, 45 months):

**Temporal KG:**
- Nodes: ~1,850 (22 funds + 1,762 stocks + 20 sectors + 45 time periods)
- Edges: ~140,000 (HOLDS edges dominate)
- Expected completeness: 0.85-0.92

**Causal KG:**
- Nodes: ~40-60 (20 macro factors + 20-40 sectors/stocks)
- Edges: ~80-150 (after p-value filtering and DAG enforcement)
- Expected causal relationships: 15-25% of tested pairs

**Integrated KG:**
- Nodes: ~1,890 (combined + cross-reference nodes)
- Edges: ~140,200 (combined + cross-reference edges)
- Cross-references: ~1,000-5,000 (depends on portfolio change frequency)

**Evaluation Scores (Expected):**
- Structural: 0.87 (some missing fundamental data expected)
- Consistency: 0.94 (good data quality from fund reports)
- Coherence: 0.78 (moderate sector clustering)
- Informativeness: 0.82 (rich temporal metadata)
- Inferential: 0.73 (good query coverage)

If your scores are significantly different, check:
- Low completeness → missing data in collection
- Low consistency → check data preprocessing
- Low coherence → sector assignments may be incorrect
- Low informativeness → check if all edge attributes are populated
- Low inferential → query functions may need debugging
