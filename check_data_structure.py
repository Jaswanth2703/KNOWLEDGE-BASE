import pandas as pd

# Read the master CSV
df = pd.read_csv(r"C:\Users\koden\Desktop\Knowledge_base\fund_portfolio_master_smart.csv")

print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Total records: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nDate range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Total unique dates: {df['Date'].nunique()}")
print(f"Total unique funds: {df['Fund_Name'].nunique()}")
print(f"Total unique stocks (ISIN): {df['ISIN'].nunique()}")

print("\n" + "="*80)
print("UNIQUE FUNDS")
print("="*80)
print(df['Fund_Name'].unique())

print("\n" + "="*80)
print("DATE DISTRIBUTION")
print("="*80)
print(df.groupby('Date').size().sort_index())

print("\n" + "="*80)
print("SAMPLE DATA STRUCTURE")
print("="*80)
print(df.head(10))

print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
print(df.isnull().sum())

print("\n" + "="*80)
print("DATA TYPES")
print("="*80)
print(df.dtypes)
