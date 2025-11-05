"""
Quick test script to see how sheet detection works
Run this first to validate detection before full extraction
"""

from smart_fund_extractor import SheetDetector
import os
import sys

def test_single_file(file_path):
    """Test sheet detection on a single Excel file"""
    print(f"\n{'='*80}")
    print(f"Testing: {os.path.basename(file_path)}")
    print(f"{'='*80}")

    detector = SheetDetector(confidence_threshold=60)
    relevant_sheets = detector.detect_relevant_sheets(file_path)

    if relevant_sheets:
        print(f"\n✅ Found {len(relevant_sheets)} relevant sheet(s):\n")
        for i, sheet in enumerate(relevant_sheets, 1):
            print(f"{i}. Sheet: '{sheet['sheet_name']}'")
            print(f"   Type: {sheet['fund_type']}")
            print(f"   Confidence: {sheet['confidence']}%")
            print(f"   Reason: {sheet['reason']}")
            print()
    else:
        print("\n❌ No relevant Small Cap or Mid Cap sheets detected")

    return relevant_sheets

def test_fund_folder(fund_path, max_files=5):
    """Test sheet detection on multiple files in a folder"""
    print(f"\n{'='*80}")
    print(f"Testing folder: {os.path.basename(fund_path)}")
    print(f"{'='*80}")

    # Find Excel files (check both year structure and flat)
    excel_files = []

    # Try year folders first
    for year in ['2022', '2023', '2024', '2025']:
        year_path = os.path.join(fund_path, year)
        if os.path.exists(year_path):
            for file in os.listdir(year_path):
                if file.endswith(('.xlsx', '.xls', '.xlsb')):
                    excel_files.append(os.path.join(year_path, file))

    # If no year folders, check root
    if not excel_files and os.path.exists(fund_path):
        for file in os.listdir(fund_path):
            if file.endswith(('.xlsx', '.xls', '.xlsb')):
                excel_files.append(os.path.join(fund_path, file))

    if not excel_files:
        print("No Excel files found!")
        return

    print(f"Found {len(excel_files)} Excel files")
    print(f"Testing first {min(max_files, len(excel_files))} files...\n")

    detector = SheetDetector(confidence_threshold=60)

    summary = {
        'files_with_sheets': 0,
        'files_without_sheets': 0,
        'total_sheets_detected': 0
    }

    for i, file_path in enumerate(excel_files[:max_files], 1):
        print(f"\n{i}. {os.path.basename(file_path)}")
        print("-" * 60)

        relevant_sheets = detector.detect_relevant_sheets(file_path)

        if relevant_sheets:
            summary['files_with_sheets'] += 1
            summary['total_sheets_detected'] += len(relevant_sheets)

            for sheet in relevant_sheets:
                print(f"  ✓ {sheet['sheet_name']} → {sheet['fund_type']} ({sheet['confidence']}%)")
        else:
            summary['files_without_sheets'] += 1
            print("  - No relevant sheets")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Files with relevant sheets: {summary['files_with_sheets']}")
    print(f"Files without relevant sheets: {summary['files_without_sheets']}")
    print(f"Total sheets detected: {summary['total_sheets_detected']}")


def interactive_test():
    """Interactive testing mode"""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                  SMART SHEET DETECTION - TEST MODE                         ║
╚════════════════════════════════════════════════════════════════════════════╝

This script helps you test the sheet detection logic before running the full
extraction. Use this to:
  1. Verify correct sheets are being detected
  2. Check confidence scores
  3. Identify any false positives/negatives
  4. Adjust threshold if needed

""")

    while True:
        print("\nOptions:")
        print("1. Test single Excel file")
        print("2. Test entire fund folder")
        print("3. Exit")

        choice = input("\nEnter choice (1-3): ").strip()

        if choice == '1':
            file_path = input("\nEnter full path to Excel file: ").strip().strip('"')
            if os.path.exists(file_path):
                test_single_file(file_path)
            else:
                print(f"❌ File not found: {file_path}")

        elif choice == '2':
            folder_path = input("\nEnter full path to fund folder: ").strip().strip('"')
            if os.path.exists(folder_path):
                max_files = input("How many files to test? (default 5): ").strip()
                max_files = int(max_files) if max_files.isdigit() else 5
                test_fund_folder(folder_path, max_files)
            else:
                print(f"❌ Folder not found: {folder_path}")

        elif choice == '3':
            print("\nExiting...")
            break

        else:
            print("Invalid choice!")


def quick_test():
    """Quick test with hardcoded paths - modify as needed"""
    BASE_PATH = r"C:\Users\koden\Desktop\Mimic Fund\MF Portfolio (Sept2022 to Sep2025)\Jaswanth"

    # Test a few known funds
    test_funds = [
        "DSP Small Cap Fund",
        "HDFC Small Cap Fund",
        "Tata Small Cap Fund"
    ]

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                         QUICK TEST MODE                                    ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

    for fund_name in test_funds:
        fund_path = os.path.join(BASE_PATH, fund_name)
        if os.path.exists(fund_path):
            test_fund_folder(fund_path, max_files=3)
        else:
            print(f"\n⚠️  Folder not found: {fund_name}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line argument provided
        path = sys.argv[1]
        if os.path.isfile(path):
            test_single_file(path)
        elif os.path.isdir(path):
            test_fund_folder(path)
        else:
            print(f"❌ Path not found: {path}")
    else:
        # Interactive mode
        print("\nNo arguments provided.")
        print("Options:")
        print("1. Interactive mode")
        print("2. Quick test (uses hardcoded paths)")

        choice = input("\nEnter choice (1-2): ").strip()

        if choice == '1':
            interactive_test()
        elif choice == '2':
            quick_test()
        else:
            print("\nUsage:")
            print("  python test_detection.py                    # Interactive mode")
            print("  python test_detection.py <file.xlsx>        # Test single file")
            print("  python test_detection.py <folder_path>      # Test folder")
