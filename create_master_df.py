import pandas as pd
import os
from datetime import datetime
import re
import warnings

# Suppress openpyxl warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# Fund-Sheet Mapping
FUND_SHEET_MAPPING = {
    "Baroda BNP Paribas Midcap Fund": ["MC"],
    "DSP Small Cap Fund": ["MIDCAP", "SMALLCAP"],
    "Edelweiss Small Cap Fund": ["EESMCF"],
    "HDFC Small Cap Fund": ["HDFCSMALLF"],
    "HSBC Small Cap Fund": [0],  # First sheet
    "Invesco India Midcap Fund": ["Midcap"],
    "LIC MF Midcap Fund": [0],
    "Quant Small Cap Fund": [0],
    "Tata Small Cap Fund": ["TATA MID CAP GROWTH FUND", "TATA SMALL CAP FUND"],
}

BASE_PATH = r"C:\Users\koden\Desktop\Mimic Fund\MF Portfolio (Sept2022 to Sep2025)\Jaswanth"

def is_valid_isin(value):
    """Check if value is a valid Indian equity ISIN"""
    if pd.isna(value):
        return False
    pattern = r'^INE[A-Z0-9]{9}$'
    return bool(re.match(pattern, str(value).strip()))

# Removed Excel date extraction - using filename only

def extract_date_from_filename(filename):
    """Extract date from filename - HARDCODED patterns"""
    filename_lower = filename.lower()
    
    # Month mapping
    months = {
        'jan': '01', 'january': '01',
        'feb': '02', 'february': '02',
        'mar': '03', 'march': '03',
        'apr': '04', 'april': '04',
        'may': '05',
        'jun': '06', 'june': '06',
        'jul': '07', 'july': '07',
        'aug': '08', 'august': '08',
        'sep': '09', 'sept': '09', 'september': '09',
        'oct': '10', 'october': '10',
        'nov': '11', 'november': '11',
        'dec': '12', 'december': '12'
    }
    
    # Extract year (look for 4-digit year)
    year_match = re.search(r'(2022|2023|2024|2025)', filename)
    if not year_match:
        return None
    year = year_match.group(1)
    
    # Extract month
    month = None
    for month_name, month_num in months.items():
        if month_name in filename_lower:
            month = month_num
            break
    
    if not month:
        return None
    
    # Extract day (look for day number near month/year)
    day_patterns = [
        r'(\d{1,2})[-\s_]?' + re.escape(list(months.keys())[list(months.values()).index(month)]),
        r'(\d{1,2})[-\s_]?' + month,
        r'(\d{1,2})[-\s_](\d{1,2})[-\s_]' + year,
    ]
    
    day = None
    for pattern in day_patterns:
        day_match = re.search(pattern, filename_lower)
        if day_match:
            day = day_match.group(1).zfill(2)
            break
    
    # Default to last day of month if day not found
    if not day:
        last_days = {
            '01': '31', '02': '28', '03': '31', '04': '30', '05': '31', '06': '30',
            '07': '31', '08': '31', '09': '30', '10': '31', '11': '30', '12': '31'
        }
        day = last_days[month]
    
    # Construct date
    date_str = f"{year}-{month}-{day}"
    try:
        return pd.to_datetime(date_str)
    except:
        return None

def find_isin_column(df):
    """Find the column containing ISIN codes"""
    for col_idx in range(df.shape[1]):
        sample = df.iloc[:100, col_idx].astype(str)
        isin_count = sum(sample.str.match(r'^INE[A-Z0-9]{9}$', na=False))
        if isin_count >= 3:
            return col_idx
    return None

def find_column_by_keywords(df, keywords, start_row=0, end_row=20):
    """Find column index by searching for keywords in header area"""
    for col_idx in range(df.shape[1]):
        sample = df.iloc[start_row:end_row, col_idx].astype(str).str.lower()
        for cell in sample:
            if any(kw in cell for kw in keywords):
                return col_idx
    return None

def find_header_row(df, isin_col):
    """Find the row that contains column headers (row before first ISIN)"""
    for row_idx in range(min(20, len(df))):
        if is_valid_isin(df.iloc[row_idx, isin_col]):
            # Header is likely in one of the previous rows
            for check_row in range(max(0, row_idx - 5), row_idx):
                cell_value = str(df.iloc[check_row, isin_col]).lower()
                if 'isin' in cell_value:
                    return check_row
            # If no explicit "ISIN" found, assume row just before data
            return row_idx - 1
    return None

def extract_stocks_from_sheet_simple(file_path, sheet_identifier, fund_name, file_date):
    """Extract stock data from a single sheet - simplified with date passed in"""
    try:
        # Handle sheet by index or name
        if isinstance(sheet_identifier, int):
            excel_file = pd.ExcelFile(file_path)
            if sheet_identifier >= len(excel_file.sheet_names):
                print(f"    ⚠️  Sheet index {sheet_identifier} out of range")
                return pd.DataFrame()
            sheet_name = excel_file.sheet_names[sheet_identifier]
        else:
            sheet_name = sheet_identifier
        
        # Read entire sheet without header assumption
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        # Find ISIN column
        isin_col = find_isin_column(df)
        if isin_col is None:
            print(f"    ⚠️  No ISIN column found in sheet: {sheet_name}")
            return pd.DataFrame()
        
        # Find header row
        header_row = find_header_row(df, isin_col)
        
        # Find Name column
        name_keywords = ['name', 'instrument', 'security', 'company', 'stock']
        name_col = find_column_by_keywords(df, name_keywords, 0, 20)
        
        # Find Weight/% column
        weight_keywords = ['% to nav', '% nav', 'percentage', 'weight', 'asset']
        weight_col = find_column_by_keywords(df, weight_keywords, 0, 20)
        
        # Filter rows with valid ISIN
        mask = df.iloc[:, isin_col].apply(is_valid_isin)
        stock_rows = df[mask].copy()
        
        if len(stock_rows) == 0:
            print(f"    ⚠️  No valid stocks found in sheet: {sheet_name}")
            return pd.DataFrame()
        
        # Extract stock names (handle missing name column)
        if name_col is not None:
            stock_names = stock_rows.iloc[:, name_col].values
        else:
            # Try to find name column near ISIN (usually adjacent)
            potential_name_cols = [isin_col - 1, isin_col + 1, isin_col - 2]
            stock_names = None
            for col in potential_name_cols:
                if 0 <= col < df.shape[1]:
                    # Check if this column has text (not numbers)
                    sample = stock_rows.iloc[:5, col].astype(str)
                    if not sample.str.match(r'^\d+\.?\d*

def process_fund(fund_name, fund_path):
    """Process all files for a single fund"""
    all_data = []
    
    print(f"\n{'='*60}")
    print(f"Processing: {fund_name}")
    print(f"{'='*60}")
    
    # Get sheet mapping
    sheet_identifiers = FUND_SHEET_MAPPING[fund_name]
    
    # Iterate through year folders
    for year_folder in ['2022', '2023', '2024', '2025']:
        year_path = os.path.join(fund_path, year_folder)
        
        if not os.path.exists(year_path):
            continue
        
        # Get all Excel files in year folder
        files = [f for f in os.listdir(year_path) 
                if f.endswith(('.xlsx', '.xls', '.xlsb'))]
        
        print(f"\n  Year {year_folder}: Found {len(files)} files")
        
        for file in files:
            file_path = os.path.join(year_path, file)
            
            # Extract date from filename ONLY (hardcoded approach)
            file_date = extract_date_from_filename(file)
            
            if file_date is None:
                print(f"  ⚠️  Skipping {file} - could not parse date from filename")
                continue
            
            print(f"  Processing: {file} [{file_date.strftime('%Y-%m-%d')}]")
            
            # Extract from each specified sheet
            sheets_data = []
            
            for sheet_id in sheet_identifiers:
                df_sheet = extract_stocks_from_sheet_simple(file_path, sheet_id, fund_name, file_date)
                
                if not df_sheet.empty:
                    sheets_data.append(df_sheet)
            
            # Add all extracted data from this file
            if sheets_data:
                all_data.extend(sheets_data)
            else:
                print(f"    ⚠️  No data extracted from any sheet")
    
    # Combine all data for this fund
    if all_data:
        fund_df = pd.concat(all_data, ignore_index=True)
        print(f"\n  ✅ Total extracted for {fund_name}: {len(fund_df)} records")
        return fund_df
    else:
        print(f"\n  ❌ No data extracted for {fund_name}")
        return pd.DataFrame()

def create_master_dataframe():
    """Process all funds and create master dataframe"""
    all_funds_data = []
    
    for fund_name in FUND_SHEET_MAPPING.keys():
        fund_path = os.path.join(BASE_PATH, fund_name)
        
        if not os.path.exists(fund_path):
            print(f"\n⚠️  Path not found: {fund_path}")
            continue
        
        fund_df = process_fund(fund_name, fund_path)
        if not fund_df.empty:
            all_funds_data.append(fund_df)
    
    # Combine all funds
    if all_funds_data:
        master_df = pd.concat(all_funds_data, ignore_index=True)
        
        # Clean and standardize
        master_df['Date'] = pd.to_datetime(master_df['Date'])
        master_df['Portfolio_Weight'] = pd.to_numeric(master_df['Portfolio_Weight'], errors='coerce')
        
        # Remove duplicates (same fund, date, ISIN)
        master_df = master_df.drop_duplicates(subset=['Fund_Name', 'Date', 'ISIN'])
        
        # Sort by date and fund
        master_df = master_df.sort_values(['Date', 'Fund_Name', 'ISIN']).reset_index(drop=True)
        
        print(f"\n{'='*60}")
        print(f"MASTER DATAFRAME CREATED")
        print(f"{'='*60}")
        print(f"Total Records: {len(master_df)}")
        print(f"Date Range: {master_df['Date'].min()} to {master_df['Date'].max()}")
        print(f"Unique Funds: {master_df['Fund_Name'].nunique()}")
        print(f"Unique Stocks (ISIN): {master_df['ISIN'].nunique()}")
        print(f"\nFunds breakdown:")
        print(master_df['Fund_Name'].value_counts())
        
        return master_df
    else:
        print("\n❌ No data extracted from any fund!")
        return pd.DataFrame()

# Run the extraction
if __name__ == "__main__":
    master_df = create_master_dataframe()
    
    # Save to CSV
    if not master_df.empty:
        output_file = "fund_portfolio_master.csv"
        master_df.to_csv(output_file, index=False)
        print(f"\n✅ Saved to: {output_file}")
        
        # Display sample
        print(f"\n{'='*60}")
        print("SAMPLE DATA:")
        print(f"{'='*60}")
        print(master_df.head(20))
        
        # Show summary statistics
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS:")
        print(f"{'='*60}")
        print(f"\nRecords per year:")
        master_df['Year'] = master_df['Date'].dt.year
        print(master_df['Year'].value_counts().sort_index())
        
        print(f"\nMissing values:")
        print(master_df.isnull().sum())
).any():
                        stock_names = stock_rows.iloc[:, col].values
                        break
            if stock_names is None:
                stock_names = ["Unknown"] * len(stock_rows)
        
        # Extract weights
        if weight_col is not None:
            weights = stock_rows.iloc[:, weight_col].values
        else:
            weights = [None] * len(stock_rows)
        
        # Build result dataframe
        result = pd.DataFrame({
            'Fund_Name': fund_name,
            'Date': file_date,
            'ISIN': stock_rows.iloc[:, isin_col].values,
            'Stock_Name': stock_names,
            'Portfolio_Weight': weights,
            'Selected': 1
        })
        
        print(f"    ✓ Extracted {len(result)} stocks from sheet: {sheet_name}")
        return result
        
    except Exception as e:
        print(f"    ✗ Error processing sheet {sheet_identifier}: {e}")
        return pd.DataFrame()

def process_fund(fund_name, fund_path):
    """Process all files for a single fund"""
    all_data = []
    
    print(f"\n{'='*60}")
    print(f"Processing: {fund_name}")
    print(f"{'='*60}")
    
    # Get sheet mapping
    sheet_identifiers = FUND_SHEET_MAPPING[fund_name]
    
    # Iterate through year folders
    for year_folder in ['2022', '2023', '2024', '2025']:
        year_path = os.path.join(fund_path, year_folder)
        
        if not os.path.exists(year_path):
            continue
        
        # Get all Excel files in year folder
        files = [f for f in os.listdir(year_path) 
                if f.endswith(('.xlsx', '.xls', '.xlsb'))]
        
        print(f"\n  Year {year_folder}: Found {len(files)} files")
        
        for file in files:
            file_path = os.path.join(year_path, file)
            
            # Extract date from filename ONLY (hardcoded approach)
            file_date = extract_date_from_filename(file)
            
            if file_date is None:
                print(f"  ⚠️  Skipping {file} - could not parse date from filename")
                continue
            
            print(f"  Processing: {file} [{file_date.strftime('%Y-%m-%d')}]")
            
            # Extract from each specified sheet
            sheets_data = []
            
            for sheet_id in sheet_identifiers:
                df_sheet = extract_stocks_from_sheet_simple(file_path, sheet_id, fund_name, file_date)
                
                if not df_sheet.empty:
                    sheets_data.append(df_sheet)
            
            # Add all extracted data from this file
            if sheets_data:
                all_data.extend(sheets_data)
            else:
                print(f"    ⚠️  No data extracted from any sheet")
    
    # Combine all data for this fund
    if all_data:
        fund_df = pd.concat(all_data, ignore_index=True)
        print(f"\n  ✅ Total extracted for {fund_name}: {len(fund_df)} records")
        return fund_df
    else:
        print(f"\n  ❌ No data extracted for {fund_name}")
        return pd.DataFrame()

def create_master_dataframe():
    """Process all funds and create master dataframe"""
    all_funds_data = []
    
    for fund_name in FUND_SHEET_MAPPING.keys():
        fund_path = os.path.join(BASE_PATH, fund_name)
        
        if not os.path.exists(fund_path):
            print(f"\n⚠️  Path not found: {fund_path}")
            continue
        
        fund_df = process_fund(fund_name, fund_path)
        if not fund_df.empty:
            all_funds_data.append(fund_df)
    
    # Combine all funds
    if all_funds_data:
        master_df = pd.concat(all_funds_data, ignore_index=True)
        
        # Clean and standardize
        master_df['Date'] = pd.to_datetime(master_df['Date'])
        master_df['Portfolio_Weight'] = pd.to_numeric(master_df['Portfolio_Weight'], errors='coerce')
        
        # Remove duplicates (same fund, date, ISIN)
        master_df = master_df.drop_duplicates(subset=['Fund_Name', 'Date', 'ISIN'])
        
        # Sort by date and fund
        master_df = master_df.sort_values(['Date', 'Fund_Name', 'ISIN']).reset_index(drop=True)
        
        print(f"\n{'='*60}")
        print(f"MASTER DATAFRAME CREATED")
        print(f"{'='*60}")
        print(f"Total Records: {len(master_df)}")
        print(f"Date Range: {master_df['Date'].min()} to {master_df['Date'].max()}")
        print(f"Unique Funds: {master_df['Fund_Name'].nunique()}")
        print(f"Unique Stocks (ISIN): {master_df['ISIN'].nunique()}")
        print(f"\nFunds breakdown:")
        print(master_df['Fund_Name'].value_counts())
        
        return master_df
    else:
        print("\n❌ No data extracted from any fund!")
        return pd.DataFrame()

# Run the extraction
if __name__ == "__main__":
    master_df = create_master_dataframe()
    
    # Save to CSV
    if not master_df.empty:
        output_file = "fund_portfolio_master.csv"
        master_df.to_csv(output_file, index=False)
        print(f"\n✅ Saved to: {output_file}")
        
        # Display sample
        print(f"\n{'='*60}")
        print("SAMPLE DATA:")
        print(f"{'='*60}")
        print(master_df.head(20))
        
        # Show summary statistics
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS:")
        print(f"{'='*60}")
        print(f"\nRecords per year:")
        master_df['Year'] = master_df['Date'].dt.year
        print(master_df['Year'].value_counts().sort_index())
        
        print(f"\nMissing values:")
        print(master_df.isnull().sum())