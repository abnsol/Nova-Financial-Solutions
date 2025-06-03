import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

def set_visualization_defaults():
    """Set consistent visualization settings"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12

def plot_sentiment_vs_returns(agg_df, ticker):
    """Plot sentiment vs returns with regression line"""
    set_visualization_defaults()
    
    plt.figure(figsize=(10, 6))
    sns.regplot(
        x=agg_df[('polarity', 'mean')],
        y=agg_df['daily_return'],
        scatter_kws={'alpha': 0.6},
        line_kws={'color': 'red'}
    )
    plt.title(f"Sentiment vs. Daily Returns for {ticker}")
    plt.xlabel("Average Daily Sentiment Polarity")
    plt.ylabel("Daily Return")
    plt.tight_layout()
    return plt

def plot_correlation_heatmap(corr_df):
    """Plot heatmap of correlation results across tickers"""
    set_visualization_defaults()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        corr_df.set_index('ticker'),
        annot=True,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1
    )
    plt.title("Correlation Between Sentiment and Returns by Ticker")
    plt.tight_layout()
    return plt