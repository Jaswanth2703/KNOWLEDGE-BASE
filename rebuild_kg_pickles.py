"""
Rebuild KG pickle files with current pandas version
This is faster than re-running the entire pipeline
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "phase1_kg"))

print("Rebuilding KG pickle files...")
print("This will take ~2-3 minutes")

# Import the KG classes
from phase1_kg.knowledge_graphs.temporal_kg import TemporalKG
from phase1_kg.knowledge_graphs.causal_kg import CausalKG
from phase1_kg.config import PROCESSED_DATA_DIR
import pandas as pd

# Load integrated dataset
print("\n1. Loading integrated dataset...")
df = pd.read_csv(PROCESSED_DATA_DIR / "integrated_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])
print(f"   Loaded {len(df):,} records")

# Build Temporal KG
print("\n2. Building Temporal KG...")
tkg = TemporalKG()
tkg.build_from_dataframe(df)
tkg.save()
print("   ✓ Temporal KG saved")

# Build Causal KG
print("\n3. Building Causal KG...")
ckg = CausalKG()
ckg.build_from_dataframe(df)
ckg.save()
print("   ✓ Causal KG saved")

print("\n" + "="*80)
print("KG PICKLE FILES REBUILT SUCCESSFULLY!")
print("="*80)
print("\nYou can now run queries without pickle compatibility issues.")
