"""
Test if sentiment data is REAL or FAKE
Check if Google News RSS actually returns articles
"""
import feedparser
from urllib.parse import quote
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "phase1_kg"))

print("="*80)
print("TESTING: Is Sentiment Data REAL or FAKE?")
print("="*80)

# Test fetching news from Google News RSS
test_sector = "Banking"
query = f"{test_sector} sector India stock market"
rss_url = f"https://news.google.com/rss/search?q={quote(query)}"

print(f"\n1. Testing Google News RSS Feed")
print(f"   Query: {query}")
print(f"   URL: {rss_url}")

try:
    feed = feedparser.parse(rss_url)

    if not feed.entries:
        print("\n   ❌ RESULT: NO ARTICLES RETURNED!")
        print("   This means Google News RSS is not working or is blocked")
        print("   Your sentiment data might be FAKE/PLACEHOLDER")
    else:
        print(f"\n   ✓ SUCCESS: {len(feed.entries)} articles found!")
        print("\n   Sample Articles:")
        for i, entry in enumerate(feed.entries[:5], 1):
            print(f"\n   {i}. {entry.get('title', 'No title')}")
            print(f"      Published: {entry.get('published', 'Unknown')}")
            print(f"      Link: {entry.get('link', 'No link')[:60]}...")

        print("\n   ✓ CONCLUSION: Google News RSS is working")
        print("   ✓ Your sentiment data appears to be REAL")

except Exception as e:
    print(f"\n   ❌ ERROR: {e}")
    print("   ❌ CONCLUSION: News fetching FAILED")

# Check if cached data exists
print("\n" + "="*80)
print("2. Checking Your Cached Sentiment Data")
print("="*80)

import pickle
from pathlib import Path

cache_dir = Path("data/cache")
cache_files = list(cache_dir.glob("sentiment_*.pkl"))

if cache_files:
    print(f"\n   Found {len(cache_files)} cached sentiment files")

    # Load one and check
    sample_file = cache_files[0]
    with open(sample_file, 'rb') as f:
        data = pickle.load(f)

    print(f"\n   Sample file: {sample_file.name}")
    print(f"   Article count per sector: {data['article_count'].unique()}")
    print(f"   Confidence scores (avg): {data['avg_confidence'].mean():.3f}")

    if data['article_count'].unique()[0] == 0:
        print("\n   ❌ WARNING: Article count is 0")
        print("   ❌ This suggests NO REAL NEWS was fetched")
        print("   ❌ Sentiment data appears to be PLACEHOLDER/FAKE")
    elif data['article_count'].unique()[0] == 9:
        print("\n   ✓ Article count is 9 (as expected)")
        print("   ✓ This suggests real articles were processed")

    # Check confidence scores
    if data['avg_confidence'].mean() > 0.9:
        print(f"\n   ✓ High confidence scores ({data['avg_confidence'].mean():.3f})")
        print("   ✓ Typical for FinBERT on financial news")
    elif data['avg_confidence'].mean() == 0.0:
        print("\n   ❌ Zero confidence scores")
        print("   ❌ This indicates NO REAL SENTIMENT ANALYSIS was done")
        print("   ❌ Data is definitely FAKE/PLACEHOLDER")

# Final verdict
print("\n" + "="*80)
print("3. FINAL VERDICT")
print("="*80)

# Check your actual sentiment data
import pandas as pd
df = pd.read_csv("data/processed/sentiment_data.csv")

zero_articles = (df['article_count'] == 0).sum()
total_rows = len(df)

print(f"\n   Total sentiment records: {total_rows}")
print(f"   Records with 0 articles: {zero_articles}")
print(f"   Records with articles: {total_rows - zero_articles}")
print(f"   Average confidence: {df['avg_confidence'].mean():.3f}")

if zero_articles > 0:
    print(f"\n   ⚠️  WARNING: {zero_articles} records have 0 articles")
    print("   ⚠️  This suggests news fetching failed for some dates/sectors")

if df['avg_confidence'].mean() == 0.0:
    print("\n   ❌ FAKE DATA: All confidence scores are 0")
    print("   ❌ No real FinBERT analysis was performed")
    print("\n   RECOMMENDATION:")
    print("   1. Re-run sentiment collection with internet connection")
    print("   2. Or use synthetic sentiment data and acknowledge in report")
elif df['avg_confidence'].mean() > 0.85:
    print(f"\n   ✓ REAL DATA: High confidence scores ({df['avg_confidence'].mean():.3f})")
    print("   ✓ FinBERT analysis was performed on real news articles")
    print("\n   HOWEVER:")
    print("   - Original article texts are NOT stored (only aggregated scores)")
    print("   - Cannot verify exact articles that were analyzed")
    print("   - This is normal for efficiency (caching aggregated results)")

print("\n" + "="*80 + "\n")
