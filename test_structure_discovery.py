"""
Quick test to verify structure discovery is working correctly
"""

import os

BASE_PATH = r"C:\Users\koden\Desktop\Knowledge_base\MF Portfolio (Sept2022 to Sep2025)"

def discover_structure():
    """Discover all funds and Excel files"""

    all_funds_list = []
    total_excel_files = 0

    # Discover parent folders
    parent_folders = []
    for item in os.listdir(BASE_PATH):
        item_path = os.path.join(BASE_PATH, item)
        if os.path.isdir(item_path):
            parent_folders.append((item, item_path))

    print(f"Found {len(parent_folders)} parent directories:")
    for parent_name, _ in parent_folders:
        print(f"  - {parent_name}")

    # For each parent folder, get fund folders
    for parent_name, parent_path in parent_folders:
        try:
            fund_count = 0
            for fund_name in os.listdir(parent_path):
                fund_path = os.path.join(parent_path, fund_name)
                if os.path.isdir(fund_path):
                    full_fund_name = f"{parent_name}/{fund_name}"

                    # Count Excel files recursively
                    excel_count = count_excel_files(fund_path)

                    all_funds_list.append({
                        'parent': parent_name,
                        'fund': fund_name,
                        'full_name': full_fund_name,
                        'path': fund_path,
                        'excel_files': excel_count
                    })

                    fund_count += 1
                    total_excel_files += excel_count

            print(f"\n{parent_name}: {fund_count} funds")
        except PermissionError:
            print(f"\n{parent_name}: Permission denied")
            continue

    # Summary
    print(f"\n{'='*80}")
    print(f"DISCOVERY SUMMARY")
    print(f"{'='*80}")
    print(f"Total Funds: {len(all_funds_list)}")
    print(f"Total Excel Files: {total_excel_files}")

    # Detailed breakdown
    print(f"\n{'='*80}")
    print(f"FUND BREAKDOWN")
    print(f"{'='*80}")

    for fund_info in all_funds_list:
        print(f"{fund_info['full_name']}: {fund_info['excel_files']} Excel files")

def count_excel_files(directory):
    """Count Excel files recursively"""
    count = 0
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)

            if os.path.isfile(item_path):
                if item.endswith(('.xlsx', '.xls', '.xlsb')):
                    count += 1
            elif os.path.isdir(item_path):
                count += count_excel_files(item_path)
    except PermissionError:
        pass

    return count

if __name__ == "__main__":
    discover_structure()
