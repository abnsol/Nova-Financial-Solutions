# src/correlation_analyzer.py

import pandas as pd
import numpy as np
import os

class CorrelationAnalyzer:
    """
    A class to analyze the correlation between news sentiment and stock movements.
    """
    def __init__(self):
        pass

    def align_data_by_date(self, news_df: pd.DataFrame, stock_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aligns news sentiment data and stock price data by date.
        Assumes 'date' in news_df and index (date) in stock_df are datetime objects.
        Normalizes news date to day for aggregation.
        """
        if news_df.empty or stock_df.empty:
            raise ValueError("Input DataFrames cannot be empty for alignment.")

        # Ensure news_df 'date' is just date for aggregation
        news_df['publication_day'] = news_df['date'].dt.normalize() # Normalize to just the date part

        # Aggregate sentiment scores per day
        daily_sentiment = news_df.groupby('publication_day')['sentiment'].mean().reset_index()
        daily_sentiment.rename(columns={'publication_day': 'Date', 'sentiment': 'Avg_Sentiment'}, inplace=True)
        daily_sentiment.set_index('Date', inplace=True)

        # Calculate Daily Stock Returns
        # Ensure stock_df index is named 'Date'
        stock_df.index.name = 'Date'
        stock_df['Daily_Return'] = stock_df['Close'].pct_change() #

        # Merge the two datasets on their Date index
        # Using an inner join to ensure only dates common to both exist
        merged_df = pd.merge(stock_df[['Close', 'Daily_Return']], daily_sentiment,
                             left_index=True, right_index=True, how='inner')
        
        if merged_df.empty:
            print("Warning: No overlapping dates found between news and stock data after merging.")
        
        print("News and stock data aligned by date.")
        return merged_df

    def calculate_pearson_correlation(self, merged_df: pd.DataFrame) -> float:
        """
        Calculates the Pearson correlation coefficient between average daily sentiment
        and daily stock returns.
        """
        if merged_df.empty or 'Avg_Sentiment' not in merged_df.columns or 'Daily_Return' not in merged_df.columns:
            raise ValueError("Merged DataFrame is missing required columns for correlation calculation.")
        
        # Drop NaN values introduced by pct_change() or missing sentiment
        correlation_data = merged_df[['Avg_Sentiment', 'Daily_Return']].dropna()

        if correlation_data.empty:
            print("Warning: No valid data points to calculate correlation after dropping NaNs.")
            return np.nan # Return NaN if no data for correlation

        correlation = correlation_data['Avg_Sentiment'].corr(correlation_data['Daily_Return'], method='pearson')
        print(f"Pearson Correlation (Avg Sentiment vs. Daily Return): {correlation:.4f}")
        return correlation

    def visualize_correlation(self, merged_df: pd.DataFrame, correlation_value: float, stock_symbol: str = "Stock"):
        """
        Visualizes the correlation using a scatter plot and time series plots.
        """
        if merged_df.empty or 'Avg_Sentiment' not in merged_df.columns or 'Daily_Return' not in merged_df.columns:
            print("Cannot plot correlation: Merged DataFrame is empty or missing columns.")
            return

        plt.figure(figsize=(16, 7))

        # Scatter plot of sentiment vs. daily returns
        plt.subplot(1, 2, 1)
        sns.scatterplot(x='Avg_Sentiment', y='Daily_Return', data=merged_df)
        plt.title(f'Sentiment vs. Daily Returns for {stock_symbol}\nPearson Corr: {correlation_value:.4f}')
        plt.xlabel('Average Daily Sentiment')
        plt.ylabel('Daily Stock Return (%)')
        plt.grid(True)

        # Time series plot of sentiment and returns
        plt.subplot(1, 2, 2)
        plt.plot(merged_df.index, merged_df['Avg_Sentiment'], label='Average Daily Sentiment', color='blue', alpha=0.7)
        plt.plot(merged_df.index, merged_df['Daily_Return'], label='Daily Stock Return (%)', color='orange', alpha=0.7)
        plt.title(f'Time Series: Sentiment and Returns for {stock_symbol}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        print("Correlation visualizations generated.")