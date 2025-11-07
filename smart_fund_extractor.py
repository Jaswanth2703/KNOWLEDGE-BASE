"""
Smart Mutual Fund Portfolio Extractor
Auto-detects Small Cap and Mid Cap sheets using intelligent fuzzy matching
"""

import pandas as pd
import os
from datetime import datetime
import re
import warnings
from fuzzywuzzy import fuzz
from collections import defaultdict
import json
import yfinance as yf
import time

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')


class SheetDetector:
    """Intelligent sheet detection using fuzzy matching and keyword analysis"""

    # Target fund types we want
    TARGET_KEYWORDS = {
        'small_cap': ['small cap', 'smallcap', 'small-cap', 'sc fund', 'small'],
        'mid_cap': ['mid cap', 'midcap', 'mid-cap', 'mc fund', 'mid']
    }

    # Fund types to explicitly exclude
    EXCLUDE_KEYWORDS = [
        'large cap', 'largecap', 'large-cap',
        'flexi cap', 'flexicap', 'flexi-cap',
        'multi cap', 'multicap', 'multi-cap',
        'index', 'balanced', 'hybrid',
        'debt', 'liquid', 'gilt',
        'toc', 'table of content', 'contents',
        'summary', 'overview'
    ]

    def __init__(self, confidence_threshold=60):
        """
        Args:
            confidence_threshold: Minimum fuzzy match score (0-100) to consider a match
        """
        self.confidence_threshold = confidence_threshold
        self.detection_log = []

    def detect_relevant_sheets(self, file_path):
        """
        Analyze Excel file and return relevant sheet names/indices

        Returns:
            List of dicts with: {
                'sheet_name': str,
                'sheet_index': int,
                'fund_type': 'small_cap' or 'mid_cap',
                'confidence': int (0-100),
                'reason': str
            }
        """
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names

            relevant_sheets = []

            for idx, sheet_name in enumerate(sheet_names):
                # Skip first sheet if it looks like TOC/Index
                if idx == 0 and self._is_toc_sheet(sheet_name):
                    continue

                # Check if sheet should be excluded
                if self._should_exclude(sheet_name):
                    continue

                # Check if sheet matches our target types
                match_result = self._match_sheet(sheet_name)

                if match_result:
                    match_result['sheet_index'] = idx
                    relevant_sheets.append(match_result)

            # Log detection results
            self.detection_log.append({
                'file': os.path.basename(file_path),
                'total_sheets': len(sheet_names),
                'detected_sheets': len(relevant_sheets),
                'sheets': relevant_sheets
            })

            return relevant_sheets

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return []

    def _is_toc_sheet(self, sheet_name):
        """Check if sheet is likely a TOC/Index"""
        toc_indicators = ['index', 'toc', 'content', 'summary', 'sheet1', 'overview']
        sheet_lower = sheet_name.lower().strip()
        return any(ind in sheet_lower for ind in toc_indicators)

    def _should_exclude(self, sheet_name):
        """Check if sheet should be excluded"""
        sheet_lower = sheet_name.lower().strip()

        for exclude_kw in self.EXCLUDE_KEYWORDS:
            if exclude_kw in sheet_lower:
                return True

        return False

    def _match_sheet(self, sheet_name):
        """
        Match sheet name against target fund types

        Returns:
            Dict with match info or None if no match
        """
        sheet_lower = sheet_name.lower().strip()
        best_match = None
        best_score = 0

        for fund_type, keywords in self.TARGET_KEYWORDS.items():
            for keyword in keywords:
                # Exact substring match
                if keyword in sheet_lower:
                    score = 100
                    reason = f"Exact match: '{keyword}' in '{sheet_name}'"

                    if score > best_score:
                        best_score = score
                        best_match = {
                            'sheet_name': sheet_name,
                            'fund_type': fund_type,
                            'confidence': score,
                            'reason': reason
                        }

                # Fuzzy match
                else:
                    score = fuzz.partial_ratio(keyword, sheet_lower)

                    if score >= self.confidence_threshold and score > best_score:
                        best_score = score
                        best_match = {
                            'sheet_name': sheet_name,
                            'fund_type': fund_type,
                            'confidence': score,
                            'reason': f"Fuzzy match: '{keyword}' ~ '{sheet_name}' ({score}%)"
                        }

        return best_match

    def get_detection_log(self):
        """Return the detection log"""
        return self.detection_log


class ExtractionLogger:
    """Tracks extraction progress and generates reports"""

    def __init__(self):
        self.log = {
            'files_processed': 0,
            'files_failed': 0,
            'sheets_extracted': 0,
            'records_extracted': 0,
            'fund_breakdown': defaultdict(int),
            'errors': [],
            'file_details': []
        }

    def log_file(self, file_path, sheets_detected, records_extracted, success=True):
        """Log processing of a single file"""
        if success:
            self.log['files_processed'] += 1
            self.log['sheets_extracted'] += len(sheets_detected)
            self.log['records_extracted'] += records_extracted
        else:
            self.log['files_failed'] += 1

        self.log['file_details'].append({
            'file': os.path.basename(file_path),
            'success': success,
            'sheets_detected': sheets_detected,
            'records': records_extracted
        })

    def log_error(self, file_path, error_msg):
        """Log an error"""
        self.log['errors'].append({
            'file': os.path.basename(file_path),
            'error': error_msg
        })

    def increment_fund(self, fund_name, count):
        """Increment record count for a fund"""
        self.log['fund_breakdown'][fund_name] += count

    def generate_report(self):
        """Generate a formatted extraction report"""
        report = []
        report.append("\n" + "="*80)
        report.append("EXTRACTION REPORT")
        report.append("="*80)
        report.append(f"\nFiles Processed: {self.log['files_processed']}")
        report.append(f"Files Failed: {self.log['files_failed']}")
        report.append(f"Sheets Extracted: {self.log['sheets_extracted']}")
        report.append(f"Total Records: {self.log['records_extracted']}")

        report.append("\n" + "-"*80)
        report.append("FUND BREAKDOWN")
        report.append("-"*80)
        for fund, count in sorted(self.log['fund_breakdown'].items()):
            report.append(f"{fund}: {count} records")

        if self.log['errors']:
            report.append("\n" + "-"*80)
            report.append("ERRORS")
            report.append("-"*80)
            for error in self.log['errors']:
                report.append(f"{error['file']}: {error['error']}")

        report.append("\n" + "-"*80)
        report.append("DETAILED FILE LOG")
        report.append("-"*80)
        for detail in self.log['file_details']:
            if detail['success']:
                sheets_str = ", ".join([f"{s['sheet_name']} ({s['fund_type']})"
                                       for s in detail['sheets_detected']])
                report.append(f"✓ {detail['file']}: {detail['records']} records from [{sheets_str}]")
            else:
                report.append(f"✗ {detail['file']}: FAILED")

        return "\n".join(report)

    def save_detailed_log(self, output_path):
        """Save detailed JSON log"""
        with open(output_path, 'w') as f:
            json.dump(self.log, f, indent=2, default=str)


class SmartExtractor:
    """Main extraction engine with intelligent sheet detection"""

    def __init__(self, base_path, confidence_threshold=60):
        self.base_path = base_path
        self.detector = SheetDetector(confidence_threshold)
        self.logger = ExtractionLogger()

    def is_valid_isin(self, value):
        """Check if value is a valid Indian equity ISIN"""
        if pd.isna(value):
            return False
        pattern = r'^INE[A-Z0-9]{9}$'
        return bool(re.match(pattern, str(value).strip()))

    def is_valid_stock_name(self, name):
        """
        Check if a stock name is valid (not an ISIN code or empty)
        Valid names should contain company name indicators like Limited, Ltd, Inc, Bank, Fund, etc.
        """
        if name is None or pd.isna(name):
            return False

        name_str = str(name).strip()

        # Check if it's empty or just whitespace
        if not name_str or len(name_str) == 0:
            return False

        # Check if it's an ISIN code (starts with INE or similar patterns)
        if name_str.startswith('INE') or name_str.startswith('IEX'):
            return False

        # Check if it's just a pipe or special character
        if name_str in ['|', '-', ' ', 'nan', 'NaN', 'nan', 'Unknown', 'N/A']:
            return False

        # Valid if contains common company name indicators
        valid_keywords = [
            'limited', 'ltd', 'inc', 'corp', 'corporation',
            'bank', 'financial', 'group', 'company',
            'industries', 'services', 'systems',
            'pharma', 'laboratory', 'labs',
            'energy', 'power', 'oil',
            'tech', 'technology', 'infosys', 'tcs', 'wipro',
            'reliance', 'hdfc', 'icici', 'axis',
            'motors', 'auto', 'steel', 'cement',
            'textiles', 'chemicals', 'agro',
            'infrastructure', 'realty', 'estate',
            'media', 'broadcasting'
        ]

        name_lower = name_str.lower()
        if any(kw in name_lower for kw in valid_keywords):
            return True

        # If it has reasonable length and no numeric-only pattern, might be valid
        if len(name_str) > 5 and not name_str.replace('.', '').replace(',', '').isnumeric():
            return True

        return False

    def get_stock_info_from_yahoo(self, ticker):
        """
        Get stock name and sector from Yahoo Finance

        Returns:
            Dict with 'name' and 'sector' keys, or None if fetch fails
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info:
                return None

            # Try to get proper name
            name = info.get('longName') or info.get('shortName') or info.get('companyName')
            sector = info.get('sector', 'Other')

            # Handle None/missing
            if not name:
                name = None
            if not sector:
                sector = 'Other'

            return {
                'name': name,
                'sector': sector
            }

        except Exception as e:
            return None

    def enrich_master_dataframe(self, master_df):
        """
        Enrich master dataframe by:
        1. Validating stock names
        2. Fetching proper names from Yahoo Finance for invalid ones
        3. Adding sector information

        Returns:
            Enriched dataframe with valid Stock_Name and Sector columns
        """
        print("\n" + "="*80)
        print("ENRICHING MASTER DATAFRAME WITH SECTOR AND VALID STOCK NAMES")
        print("="*80)

        # Add sector column if not exists
        if 'Sector' not in master_df.columns:
            master_df['Sector'] = 'Other'

        # Count invalid names
        invalid_mask = ~master_df['Stock_Name'].apply(self.is_valid_stock_name)
        invalid_count = invalid_mask.sum()
        print(f"\nFound {invalid_count} invalid stock names ({invalid_count/len(master_df)*100:.1f}%)")

        if invalid_count == 0:
            print("✓ All stock names are valid!")
            return master_df

        # Get unique ISINs with invalid names
        invalid_isins = master_df[invalid_mask]['ISIN'].unique()
        print(f"Unique ISINs with invalid names: {len(invalid_isins)}")

        # Load ticker mapping if available
        ticker_mapping_path = os.path.join(os.path.dirname(__file__), 'data', 'isin_ticker_mapping.json')
        ticker_mapping = {}

        if os.path.exists(ticker_mapping_path):
            with open(ticker_mapping_path, 'r') as f:
                ticker_mapping = json.load(f)
            print(f"Loaded ticker mapping for {len(ticker_mapping)} ISINs")
        else:
            print("⚠ No ticker mapping found. Will skip Yahoo Finance enrichment.")
            return master_df

        # Enrich invalid names
        print(f"\nFetching stock info from Yahoo Finance for {len(invalid_isins)} ISINs...")
        enrichment_data = {}
        success_count = 0
        fail_count = 0

        for idx, isin in enumerate(invalid_isins):
            if isin not in ticker_mapping:
                fail_count += 1
                continue

            ticker = ticker_mapping[isin]

            # Fetch from Yahoo Finance
            info = self.get_stock_info_from_yahoo(ticker)

            if info and info['name']:
                enrichment_data[isin] = info
                success_count += 1
                print(f"  [{idx+1}/{len(invalid_isins)}] {isin} -> {info['name'][:40]} ({info['sector']})")
            else:
                fail_count += 1
                print(f"  [{idx+1}/{len(invalid_isins)}] {isin} -> FAILED")

            # Rate limiting
            time.sleep(0.05)

        print(f"\n✓ Fetched info for {success_count} stocks")
        print(f"✗ Failed to fetch: {fail_count} stocks")

        # Update dataframe with enriched data
        print(f"\nUpdating dataframe...")
        for isin, info in enrichment_data.items():
            # Update stock names
            mask = master_df['ISIN'] == isin
            master_df.loc[mask, 'Stock_Name'] = info['name']
            # Update sectors
            master_df.loc[mask, 'Sector'] = info['sector']

        # Count remaining invalid
        still_invalid = ~master_df['Stock_Name'].apply(self.is_valid_stock_name)
        print(f"\n✓ Valid stock names now: {(~still_invalid).sum()} ({(~still_invalid).sum()/len(master_df)*100:.1f}%)")
        print(f"⚠ Still invalid: {still_invalid.sum()} ({still_invalid.sum()/len(master_df)*100:.1f}%)")

        return master_df

    def extract_date_from_filename(self, filename):
        """Extract date from filename"""
        filename_lower = filename.lower()

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

        # Extract year
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

        # Default to last day of month
        last_days = {
            '01': '31', '02': '28', '03': '31', '04': '30', '05': '31', '06': '30',
            '07': '31', '08': '31', '09': '30', '10': '31', '11': '30', '12': '31'
        }
        day = last_days[month]

        date_str = f"{year}-{month}-{day}"
        try:
            return pd.to_datetime(date_str)
        except:
            return None

    def find_isin_column(self, df):
        """Find the column containing ISIN codes"""
        for col_idx in range(df.shape[1]):
            sample = df.iloc[:100, col_idx].astype(str)
            isin_count = sum(sample.str.match(r'^INE[A-Z0-9]{9}$', na=False))
            if isin_count >= 3:
                return col_idx
        return None

    def find_column_by_keywords(self, df, keywords, start_row=0, end_row=20):
        """Find column index by searching for keywords"""
        for col_idx in range(df.shape[1]):
            sample = df.iloc[start_row:end_row, col_idx].astype(str).str.lower()
            for cell in sample:
                if any(kw in cell for kw in keywords):
                    return col_idx
        return None

    def find_header_row(self, df, isin_col):
        """Find the row that contains column headers"""
        for row_idx in range(min(20, len(df))):
            if self.is_valid_isin(df.iloc[row_idx, isin_col]):
                for check_row in range(max(0, row_idx - 5), row_idx):
                    cell_value = str(df.iloc[check_row, isin_col]).lower()
                    if 'isin' in cell_value:
                        return check_row
                return row_idx - 1
        return None

    def extract_from_sheet(self, file_path, sheet_info, file_date, fund_name):
        """Extract stock data from a single sheet"""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_info['sheet_name'], header=None)

            # Find ISIN column
            isin_col = self.find_isin_column(df)
            if isin_col is None:
                return pd.DataFrame()

            # Find Name column
            name_keywords = ['name', 'instrument', 'security', 'company', 'stock']
            name_col = self.find_column_by_keywords(df, name_keywords, 0, 20)

            # Find Weight column
            weight_keywords = ['% to nav', '% nav', 'percentage', 'weight', 'asset']
            weight_col = self.find_column_by_keywords(df, weight_keywords, 0, 20)

            # Find Quantity column
            quantity_keywords = ['quantity', 'no. of shares', 'shares', 'units', 'holdings', 'nos']
            quantity_col = self.find_column_by_keywords(df, quantity_keywords, 0, 20)

            # Find Market Value column
            market_value_keywords = ['market value', 'value', 'amount', 'mkt val', 'lakhs', 'crores', 'market cap']
            market_value_col = self.find_column_by_keywords(df, market_value_keywords, 0, 20)

            # Filter rows with valid ISIN
            mask = df.iloc[:, isin_col].apply(self.is_valid_isin)
            stock_rows = df[mask].copy()

            if len(stock_rows) == 0:
                return pd.DataFrame()

            # Extract stock names
            if name_col is not None:
                stock_names = stock_rows.iloc[:, name_col].values
            else:
                potential_name_cols = [isin_col - 1, isin_col + 1, isin_col - 2]
                stock_names = None
                for col in potential_name_cols:
                    if 0 <= col < df.shape[1]:
                        sample = stock_rows.iloc[:5, col].astype(str)
                        if not sample.str.match(r'^\d+\.?\d*$').any():
                            stock_names = stock_rows.iloc[:, col].values
                            break
                if stock_names is None:
                    stock_names = ["Unknown"] * len(stock_rows)

            # Extract weights
            if weight_col is not None:
                weights = stock_rows.iloc[:, weight_col].values
            else:
                weights = [None] * len(stock_rows)

            # Extract quantities
            if quantity_col is not None:
                quantities = stock_rows.iloc[:, quantity_col].values
            else:
                quantities = [None] * len(stock_rows)

            # Extract market values
            if market_value_col is not None:
                market_values = stock_rows.iloc[:, market_value_col].values
            else:
                market_values = [None] * len(stock_rows)

            # Build result dataframe
            result = pd.DataFrame({
                'Fund_Name': fund_name,
                'Fund_Type': sheet_info['fund_type'],
                'Date': file_date,
                'ISIN': stock_rows.iloc[:, isin_col].values,
                'Stock_Name': stock_names,
                'Quantity': quantities,
                'Market_Value': market_values,
                'Portfolio_Weight': weights,
                'Sheet_Name': sheet_info['sheet_name'],
                'Detection_Confidence': sheet_info['confidence']
            })

            return result

        except Exception as e:
            self.logger.log_error(file_path, f"Sheet {sheet_info['sheet_name']}: {e}")
            return pd.DataFrame()

    def process_file(self, file_path, fund_name):
        """Process a single Excel file with auto-detection"""
        # Extract date from filename
        file_date = self.extract_date_from_filename(os.path.basename(file_path))

        if file_date is None:
            self.logger.log_error(file_path, "Could not parse date from filename")
            return pd.DataFrame()

        # Detect relevant sheets
        relevant_sheets = self.detector.detect_relevant_sheets(file_path)

        if not relevant_sheets:
            self.logger.log_file(file_path, [], 0, success=True)
            return pd.DataFrame()

        # Extract from each detected sheet
        all_data = []
        for sheet_info in relevant_sheets:
            df = self.extract_from_sheet(file_path, sheet_info, file_date, fund_name)
            if not df.empty:
                all_data.append(df)

        # Combine results
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            self.logger.log_file(file_path, relevant_sheets, len(combined), success=True)
            self.logger.increment_fund(fund_name, len(combined))
            return combined
        else:
            self.logger.log_file(file_path, relevant_sheets, 0, success=True)
            return pd.DataFrame()

    def find_all_excel_files(self, fund_path):
        """
        Find all Excel files in fund directory
        Handles multiple year folder structures:
        - 2022, 2023, 2024, 2025
        - year-1, year-2, year-3
        - Flat (no year folders)
        - Recursively finds all Excel files
        """
        excel_files = []

        def collect_excel_files(directory):
            """Recursively collect Excel files"""
            try:
                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)

                    if os.path.isfile(item_path):
                        if item.endswith(('.xlsx', '.xls', '.xlsb')):
                            excel_files.append(item_path)
                    elif os.path.isdir(item_path):
                        # Recurse into subdirectories
                        collect_excel_files(item_path)
            except PermissionError:
                pass  # Skip directories we can't access

        collect_excel_files(fund_path)
        return excel_files

    def process_fund(self, fund_name, fund_path):
        """Process all files for a single fund"""
        print(f"\n{'='*80}")
        print(f"Processing: {fund_name}")
        print(f"{'='*80}")

        # Find all Excel files
        excel_files = self.find_all_excel_files(fund_path)

        print(f"Found {len(excel_files)} Excel files")

        all_data = []

        for file_path in excel_files:
            print(f"  Processing: {os.path.basename(file_path)}")
            df = self.process_file(file_path, fund_name)
            if not df.empty:
                all_data.append(df)
                print(f"    ✓ Extracted {len(df)} records")
            else:
                print(f"    - No relevant data")

        if all_data:
            fund_df = pd.concat(all_data, ignore_index=True)
            print(f"\n  ✅ Total for {fund_name}: {len(fund_df)} records")
            return fund_df
        else:
            print(f"\n  ⚠️  No data extracted for {fund_name}")
            return pd.DataFrame()

    def process_all_funds(self):
        """
        Process all fund folders in base path
        Auto-discovers fund folders across multiple parent directories
        Structure: Base/Parent/Fund/Year/Files.xlsx
        """
        all_funds_data = []
        all_funds_list = []

        # Discover all fund folders
        # First check if base path contains parent folders (Jaswanth, Mayur, etc.)
        # or fund folders directly

        parent_folders = []
        for item in os.listdir(self.base_path):
            item_path = os.path.join(self.base_path, item)
            if os.path.isdir(item_path):
                parent_folders.append((item, item_path))

        # For each parent folder, get fund folders
        for parent_name, parent_path in parent_folders:
            try:
                for fund_name in os.listdir(parent_path):
                    fund_path = os.path.join(parent_path, fund_name)
                    if os.path.isdir(fund_path):
                        # Create unique fund identifier: Parent/FundName
                        full_fund_name = f"{parent_name}/{fund_name}"
                        all_funds_list.append((full_fund_name, fund_path))
            except PermissionError:
                continue

        print(f"\n{'='*80}")
        print(f"SMART FUND EXTRACTOR - AUTO-DETECTION MODE")
        print(f"{'='*80}")
        print(f"Base Path: {self.base_path}")
        print(f"Discovered {len(all_funds_list)} fund folders across {len(parent_folders)} parent directories")
        print(f"Target: Small Cap & Mid Cap funds only")
        print(f"{'='*80}")

        for fund_name, fund_path in all_funds_list:
            fund_df = self.process_fund(fund_name, fund_path)

            if not fund_df.empty:
                all_funds_data.append(fund_df)

        # Combine all data
        if all_funds_data:
            master_df = pd.concat(all_funds_data, ignore_index=True)

            # Clean and standardize
            master_df['Date'] = pd.to_datetime(master_df['Date'])
            master_df['Portfolio_Weight'] = pd.to_numeric(master_df['Portfolio_Weight'], errors='coerce')
            master_df['Quantity'] = pd.to_numeric(master_df['Quantity'], errors='coerce')
            master_df['Market_Value'] = pd.to_numeric(master_df['Market_Value'], errors='coerce')

            # Remove duplicates
            master_df = master_df.drop_duplicates(subset=['Fund_Name', 'Date', 'ISIN'])

            # Sort
            master_df = master_df.sort_values(['Date', 'Fund_Name', 'ISIN']).reset_index(drop=True)

            return master_df
        else:
            return pd.DataFrame()


def main():
    """Main execution"""
    BASE_PATH = r"C:\Users\koden\Desktop\Knowledge_base\MF Portfolio (Sept2022 to Sep2025)"

    # Create extractor with confidence threshold
    extractor = SmartExtractor(BASE_PATH, confidence_threshold=60)

    # Process all funds
    master_df = extractor.process_all_funds()

    # Generate and print report
    print(extractor.logger.generate_report())

    # Enrich master dataframe with valid stock names and sectors
    if not master_df.empty:
        master_df = extractor.enrich_master_dataframe(master_df)

    # Save results
    if not master_df.empty:
        # Save master CSV
        output_file = "fund_portfolio_master_smart.csv"
        master_df.to_csv(output_file, index=False)
        print(f"\n✅ Master data saved to: {output_file}")

        # Save detailed log
        log_file = "extraction_log.json"
        extractor.logger.save_detailed_log(log_file)
        print(f"✅ Detailed log saved to: {log_file}")

        # Save sheet detection log
        detection_file = "sheet_detection_log.json"
        with open(detection_file, 'w') as f:
            json.dump(extractor.detector.get_detection_log(), f, indent=2)
        print(f"✅ Sheet detection log saved to: {detection_file}")

        # Display summary
        print(f"\n{'='*80}")
        print("MASTER DATAFRAME SUMMARY")
        print(f"{'='*80}")
        print(f"Total Records: {len(master_df)}")
        print(f"Date Range: {master_df['Date'].min()} to {master_df['Date'].max()}")
        print(f"Unique Funds: {master_df['Fund_Name'].nunique()}")
        print(f"Unique Stocks (ISIN): {master_df['ISIN'].nunique()}")

        print(f"\n{'='*80}")
        print("FUND TYPE BREAKDOWN")
        print(f"{'='*80}")
        print(master_df['Fund_Type'].value_counts())

        print(f"\n{'='*80}")
        print("SAMPLE DATA")
        print(f"{'='*80}")
        print(master_df.head(20).to_string())

    else:
        print("\n❌ No data extracted!")


if __name__ == "__main__":
    main()
