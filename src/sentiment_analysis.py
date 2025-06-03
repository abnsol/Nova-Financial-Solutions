from textblob import TextBlob
import nltk
import pandas as pd
from typing import Dict, Any

nltk.download('punkt')

def calculate_sentiment(text: str) -> Dict[str, Any]:
    """Calculate sentiment metrics for a given text"""
    blob = TextBlob(str(text))
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'word_count': len(blob.words)
    }

def add_sentiment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment analysis columns to dataframe"""
    sentiment_df = df['headline'].apply(
        lambda x: pd.Series(calculate_sentiment(x))
    )
    return pd.concat([df, sentiment_df], axis=1)

def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment scores by day"""
    return df.groupby('date_only').agg({
        'polarity': ['mean', 'std', 'count'],
        'subjectivity': 'mean',
        'daily_return': 'last'
    }).reset_index()