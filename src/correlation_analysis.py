import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class CorrelationAnalyzer:
    def __init__(self, merged_df: pd.DataFrame):
        self.df = merged_df
        
    def calculate_daily_sentiment(self) -> pd.DataFrame:
        """Calculate average daily sentiment scores"""
        daily_sentiment = self.df.groupby('Date').agg({
            'textblob_polarity': 'mean',
            'vader_compound': 'mean',
            'daily_return': 'first'
        }).reset_index()
        return daily_sentiment
    
    def analyze_correlations(self) -> dict:
        """Calculate correlation between sentiment and returns"""
        daily_df = self.calculate_daily_sentiment()
        
        pearson_textblob = stats.pearsonr(
            daily_df['textblob_polarity'],
            daily_df['daily_return']
        )
        
        pearson_vader = stats.pearsonr(
            daily_df['vader_compound'],
            daily_df['daily_return']
        )
        
        return {
            'textblob_pearson': pearson_textblob,
            'vader_pearson': pearson_vader,
            'daily_sentiment': daily_df
        }
    
    def plot_correlations(self, save_path: str = None):
        """Visualize sentiment-return relationships"""
        daily_df = self.calculate_daily_sentiment()
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        sns.regplot(
            x='textblob_polarity',
            y='daily_return',
            data=daily_df,
            scatter_kws={'alpha': 0.5}
        )
        plt.title('TextBlob Polarity vs Daily Returns')
        
        plt.subplot(1, 2, 2)
        sns.regplot(
            x='vader_compound',
            y='daily_return',
            data=daily_df,
            scatter_kws={'alpha': 0.5}
        )
        plt.title('VADER Compound Score vs Daily Returns')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()