"""
Optimized News Sentiment Analysis using FinBERT
Sector-level aggregation for computational efficiency
"""
import feedparser
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from urllib.parse import quote
from typing import List, Dict, Tuple
from tqdm import tqdm
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import SENTIMENT_CONFIG, CACHE_DIR, STANDARD_SECTORS
from utils.helpers import save_to_cache, load_from_cache, log_step


class SentimentAnalyzer:
    """
    Optimized sentiment analyzer using FinBERT
    - Sector-level analysis (not individual stocks)
    - Batch processing for efficiency
    - Caching to avoid re-computation
    """

    def __init__(self, config: Dict = None):
        self.config = config or SENTIMENT_CONFIG
        log_step("Initializing FinBERT Sentiment Analyzer")

        # Load FinBERT model and tokenizer
        model_name = self.config['model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode

        self.labels = ['Positive', 'Negative', 'Neutral']
        self.batch_size = self.config['batch_size']
        self.max_length = self.config['max_length']
        self.cache_enabled = self.config['cache_enabled']

        print(f"✓ FinBERT model loaded: {model_name}")
        print(f"✓ Batch size: {self.batch_size}")

    def fetch_sector_news(self, sector: str, num_articles: int = 10) -> List[Dict]:
        """
        Fetch news articles for a specific sector

        Args:
            sector: Sector name (e.g., 'Banking', 'IT', 'Pharma')
            num_articles: Number of articles to fetch

        Returns:
            List of article dictionaries
        """
        # Create sector-specific queries
        queries = [
            f"{sector} sector India stock market",
            f"{sector} India news",
            f"{sector} stocks India"
        ]

        all_articles = []
        for query in queries:
            rss_url = f"https://news.google.com/rss/search?q={quote(query)}"
            try:
                feed = feedparser.parse(rss_url)
                articles = feed.entries[:num_articles // len(queries)]

                for item in articles:
                    all_articles.append({
                        "title": item.get('title', ''),
                        "link": item.get('link', ''),
                        "published": item.get('published', ''),
                        "sector": sector
                    })
            except Exception as e:
                print(f"Error fetching news for {sector}: {e}")

        return all_articles

    def analyze_sentiment_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Analyze sentiment for a batch of texts using FinBERT

        Args:
            texts: List of text strings

        Returns:
            List of (sentiment, confidence) tuples
        """
        if not texts:
            return []

        results = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length
            )

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).numpy()

            # Extract sentiment and confidence
            for probs in probabilities:
                max_idx = np.argmax(probs)
                sentiment = self.labels[max_idx]
                confidence = probs[max_idx]
                results.append((sentiment, confidence))

        return results

    def aggregate_sector_sentiment(
        self,
        sectors: List[str],
        date: str,
        articles_per_sector: int = 10
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores for multiple sectors

        Args:
            sectors: List of sector names
            date: Date string for caching
            articles_per_sector: Number of articles per sector

        Returns:
            DataFrame with sector sentiment scores
        """
        cache_key = f"sentiment_{date}_{articles_per_sector}"

        # Try loading from cache
        if self.cache_enabled:
            cached_data = load_from_cache(cache_key, CACHE_DIR)
            if cached_data is not None:
                return cached_data

        log_step(f"Fetching and analyzing sentiment for {len(sectors)} sectors")

        sector_sentiments = []

        for sector in tqdm(sectors, desc="Processing sectors"):
            # Fetch news
            articles = self.fetch_sector_news(sector, articles_per_sector)

            if not articles:
                # No articles found
                sector_sentiments.append({
                    'sector': sector,
                    'date': date,
                    'positive_pct': 0.33,
                    'negative_pct': 0.33,
                    'neutral_pct': 0.34,
                    'avg_confidence': 0.0,
                    'sentiment_score': 0.0,
                    'article_count': 0
                })
                continue

            # Extract titles for sentiment analysis
            titles = [article['title'] for article in articles]

            # Analyze sentiments
            sentiments = self.analyze_sentiment_batch(titles)

            # Aggregate results
            sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
            confidences = []

            for sentiment, confidence in sentiments:
                sentiment_counts[sentiment] += 1
                confidences.append(confidence)

            total = len(sentiments)
            avg_confidence = np.mean(confidences) if confidences else 0.0

            # Calculate percentages
            pos_pct = sentiment_counts['Positive'] / total
            neg_pct = sentiment_counts['Negative'] / total
            neu_pct = sentiment_counts['Neutral'] / total

            # Calculate weighted sentiment score (-1 to +1)
            sentiment_score = pos_pct - neg_pct

            sector_sentiments.append({
                'sector': sector,
                'date': date,
                'positive_pct': pos_pct,
                'negative_pct': neg_pct,
                'neutral_pct': neu_pct,
                'avg_confidence': avg_confidence,
                'sentiment_score': sentiment_score,
                'article_count': total
            })

        df = pd.DataFrame(sector_sentiments)

        # Cache results
        if self.cache_enabled:
            save_to_cache(df, cache_key, CACHE_DIR)

        return df

    def collect_historical_sentiment(
        self,
        sectors: List[str],
        start_date: str,
        end_date: str,
        frequency: str = 'M'
    ) -> pd.DataFrame:
        """
        Collect historical sentiment data for sectors

        Args:
            sectors: List of sectors
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: 'M' for monthly, 'Q' for quarterly

        Returns:
            DataFrame with historical sentiment data
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)

        all_sentiments = []

        for date in tqdm(date_range, desc="Collecting historical sentiment"):
            date_str = date.strftime('%Y-%m-%d')
            print(f"\nProcessing date: {date_str}")

            df_sentiment = self.aggregate_sector_sentiment(
                sectors=sectors,
                date=date_str,
                articles_per_sector=10
            )

            all_sentiments.append(df_sentiment)

        # Combine all dataframes
        df_combined = pd.concat(all_sentiments, ignore_index=True)
        df_combined['date'] = pd.to_datetime(df_combined['date'])

        return df_combined


def main():
    """Test sentiment analysis"""
    log_step("Testing Sentiment Analyzer")

    # Initialize analyzer
    analyzer = SentimentAnalyzer()

    # Test with a few sectors
    test_sectors = ["Banking", "IT", "Pharma", "Energy"]
    test_date = "2024-12-31"

    # Get sentiment
    df_sentiment = analyzer.aggregate_sector_sentiment(
        sectors=test_sectors,
        date=test_date,
        articles_per_sector=5
    )

    print("\n" + "="*80)
    print("SECTOR SENTIMENT RESULTS")
    print("="*80)
    print(df_sentiment)

    # Save test results
    from config import PROCESSED_DATA_DIR
    output_path = PROCESSED_DATA_DIR / "test_sentiment.csv"
    df_sentiment.to_csv(output_path, index=False)
    print(f"\n✓ Test results saved to: {output_path}")


if __name__ == "__main__":
    main()
