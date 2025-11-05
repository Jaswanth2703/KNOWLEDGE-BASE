# Smart Fund Extractor - User Guide

## What's New? 🎯

**NO MORE HARDCODING!** The new system automatically detects Small Cap and Mid Cap sheets using intelligent fuzzy matching.

## Key Features

### 1. Intelligent Sheet Detection
- **Fuzzy Matching**: Detects sheets even with inconsistent naming
  - "MC" → Mid Cap ✓
  - "MIDCAP" → Mid Cap ✓
  - "TATA SMALL CAP FUND" → Small Cap ✓
  - "SMALLCAP" → Small Cap ✓

- **Auto-Exclusion**: Automatically skips irrelevant sheets
  - Large Cap ✗
  - Flexi Cap ✗
  - Index sheets ✗
  - Multi Cap ✗
  - Balanced/Hybrid ✗

### 2. Flexible Directory Structure
Handles both structures:
```
Fund Name/
  ├── 2022/
  │   ├── file1.xlsx
  │   └── file2.xlsx
  └── 2023/
      └── file3.xlsx
```
OR
```
Fund Name/
  ├── file1.xlsx
  ├── file2.xlsx
  └── file3.xlsx
```

### 3. Comprehensive Logging
Three log files generated:
1. **extraction_log.json** - Overall statistics
2. **sheet_detection_log.json** - Which sheets were detected and why
3. **Console output** - Real-time progress

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (Default Path)
```python
python smart_fund_extractor.py
```

### Custom Path
Edit line 415 in `smart_fund_extractor.py`:
```python
BASE_PATH = r"YOUR_PATH_HERE"
```

### Adjust Confidence Threshold
Lower threshold = more aggressive matching (may include false positives)
Higher threshold = stricter matching (may miss some sheets)

Edit line 418:
```python
extractor = SmartExtractor(BASE_PATH, confidence_threshold=60)  # Default: 60
```

## Output Files

### 1. fund_portfolio_master_smart.csv
Main output with columns:
- `Fund_Name`: Name of the fund
- `Fund_Type`: "small_cap" or "mid_cap"
- `Date`: Date extracted from filename
- `ISIN`: Stock ISIN code
- `Stock_Name`: Company name
- `Quantity`: Number of shares/units held
- `Market_Value`: Market value (typically in Rs. lakhs)
- `Portfolio_Weight`: Weight in portfolio (%)
- `Sheet_Name`: Source sheet name
- `Detection_Confidence`: How confident we are (0-100)

### 2. extraction_log.json
```json
{
  "files_processed": 150,
  "sheets_extracted": 200,
  "records_extracted": 5000,
  "fund_breakdown": {
    "Fund A": 1200,
    "Fund B": 800
  },
  "errors": []
}
```

### 3. sheet_detection_log.json
Shows which sheets were detected from each file:
```json
[
  {
    "file": "FundA_Sep2024.xlsx",
    "total_sheets": 5,
    "detected_sheets": 2,
    "sheets": [
      {
        "sheet_name": "SMALLCAP",
        "fund_type": "small_cap",
        "confidence": 100,
        "reason": "Exact match: 'small cap' in 'SMALLCAP'"
      }
    ]
  }
]
```

## How It Works

### Step 1: Sheet Detection
For each Excel file:
1. Read all sheet names
2. Skip first sheet if it looks like index/TOC
3. For each remaining sheet:
   - Check exclusion list (large cap, etc.)
   - Try exact keyword match
   - Try fuzzy match with confidence scoring
   - Return matched sheets with confidence scores

### Step 2: Data Extraction
For each detected sheet:
1. Auto-detect ISIN column
2. Auto-detect Stock Name column
3. Auto-detect Portfolio Weight column
4. Auto-detect Quantity column
5. Auto-detect Market Value column
6. Extract all rows with valid ISINs
7. Tag with fund name and type

### Step 3: Consolidation
1. Combine all extracted data
2. Remove duplicates
3. Clean and standardize
4. Generate reports

## Customization

### Add More Fund Types
Edit `SheetDetector.TARGET_KEYWORDS` (line 26):
```python
TARGET_KEYWORDS = {
    'small_cap': ['small cap', 'smallcap', 'sc fund'],
    'mid_cap': ['mid cap', 'midcap', 'mc fund'],
    'large_cap': ['large cap', 'largecap', 'lc fund']  # Add this
}
```

And update exclusion list accordingly.

### Change Keywords
Modify the keyword lists to match your Excel files:
```python
TARGET_KEYWORDS = {
    'small_cap': ['small', 'sc', 'scf', 'small cap fund'],
    'mid_cap': ['mid', 'mc', 'mcf', 'mid cap fund']
}
```

### Adjust Fuzzy Threshold
```python
# More lenient (may get false positives)
extractor = SmartExtractor(BASE_PATH, confidence_threshold=50)

# Stricter (may miss some sheets)
extractor = SmartExtractor(BASE_PATH, confidence_threshold=80)
```

## Troubleshooting

### Problem: Some sheets not detected
**Solution**: Lower confidence threshold or add more keywords

### Problem: Wrong sheets detected
**Solution**:
1. Increase confidence threshold
2. Add sheet names to exclusion list
3. Review sheet_detection_log.json to see what was detected

### Problem: Date extraction fails
**Solution**: Check filename format. Update `extract_date_from_filename()` with your pattern

### Problem: ISIN column not found
**Solution**: Sheets may have non-standard format. Check the Excel file structure

## Validation Workflow

1. Run extractor
2. Check console output for real-time progress
3. Review `sheet_detection_log.json` to verify correct sheets were selected
4. Check `extraction_log.json` for overall statistics
5. Inspect `fund_portfolio_master_smart.csv` for sample data
6. Filter by `Detection_Confidence` to identify low-confidence extractions

## Example Console Output

```
================================================================================
SMART FUND EXTRACTOR - AUTO-DETECTION MODE
================================================================================
Base Path: C:\Users\...\Jaswanth
Discovered 37 fund folders
Target: Small Cap & Mid Cap funds only
================================================================================

================================================================================
Processing: DSP Small Cap Fund
================================================================================
Found 36 Excel files
  Processing: DSP_Sep2024.xlsx
    ✓ Extracted 45 records
  Processing: DSP_Oct2024.xlsx
    ✓ Extracted 47 records

  ✅ Total for DSP Small Cap Fund: 1,234 records

================================================================================
EXTRACTION REPORT
================================================================================
Files Processed: 150
Files Failed: 5
Sheets Extracted: 200
Total Records: 50,000

FUND BREAKDOWN
--------------------------------------------------------------------------------
DSP Small Cap Fund: 1,234 records
HDFC Small Cap Fund: 2,100 records
...
```

## Advanced: Programmatic Usage

```python
from smart_fund_extractor import SmartExtractor

# Create extractor
extractor = SmartExtractor(
    base_path="path/to/funds",
    confidence_threshold=70
)

# Process specific fund
fund_df = extractor.process_fund("DSP Small Cap Fund", "path/to/fund")

# Or process all
master_df = extractor.process_all_funds()

# Get logs
print(extractor.logger.generate_report())
detection_log = extractor.detector.get_detection_log()
```

## Comparison: Old vs New

| Feature | Old (Hardcoded) | New (Smart) |
|---------|----------------|-------------|
| Sheet mapping | Manual for each fund | Automatic |
| New fund | Update code | Just add folder |
| Sheet name changes | Update code | Auto-adapts |
| Handles inconsistency | No | Yes |
| Validation | None | Full logs |
| Confidence scoring | No | Yes |
| Flexible structure | No | Yes |
| Maintenance | High | Low |

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Test run**: Start with one fund folder
3. **Review logs**: Check sheet_detection_log.json
4. **Adjust threshold**: If needed based on results
5. **Full run**: Process all 37 funds
6. **Validate output**: Review CSV and confidence scores
