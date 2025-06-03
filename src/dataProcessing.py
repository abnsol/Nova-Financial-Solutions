import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_and_clean_news_data(filepath):
    """Load and clean news data"""
    df = pd.read_csv(filepath)
    
    # Clean data
    df['headline'] = df['headline'].fillna("")
    df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
    df = df.dropna(subset=['date'])
    
    # Extract date only (without time) for alignment with stock data
    df['date_only'] = df['date'].dt.date
    
    return df

def load_stock_data(ticker):
    """Load historical stock data for a given ticker"""
    filepath = f"../data/{ticker}_historical_data.csv"
    stock_df = pd.read_csv(filepath)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df['date_only'] = stock_df['Date'].dt.date
    
    # Calculate daily returns
    stock_df['daily_return'] = stock_df['Close'].pct_change()
    
    return stock_df

def merge_news_stock_data(news_df, stock_df, ticker):
    """Merge news and stock data for a specific ticker"""
    # Filter news for this ticker
    ticker_news = news_df[news_df['stock'] == ticker].copy()
    
    # Merge with stock data
    merged_df = pd.merge(
        ticker_news,
        stock_df,
        how='left',
        left_on='date_only',
        right_on='date_only'
    )
    
    # Forward fill missing stock data (for weekends/holidays)
    merged_df['daily_return'] = merged_df['daily_return'].fillna(0)
    
    return merged_df.dropna(subset=['daily_return'])