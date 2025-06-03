import pandas as pd
from typing import Dict

def load_and_clean_news_data(filepath: str) -> pd.DataFrame:
    """Load and clean news data"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
    df = df.dropna(subset=['date'])
    df['date'] = df['date'].dt.normalize()  # Remove time component
    return df

def load_stock_data(ticker: str, filepath: str) -> pd.DataFrame:
    """Load and clean stock data for a specific ticker"""
    df = pd.read_csv(f"{filepath}/{ticker}_historical_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['daily_return'] = df['Close'].pct_change()
    return df

def merge_datasets(news_df: pd.DataFrame, stock_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Merge news and stock data for a specific ticker"""
    ticker_news = news_df[news_df['stock'] == ticker].copy()
    merged = pd.merge(
        ticker_news,
        stock_df,
        left_on='date',
        right_on='Date',
        how='inner'
    )
    return merged