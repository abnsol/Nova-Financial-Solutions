import pandas as pd
from scipy.stats import pearsonr
import numpy as np

def calculate_correlations(agg_df: pd.DataFrame) -> dict:
    """Calculate various correlation metrics"""
    results = {}
    
    # Pearson correlation between mean polarity and returns
    corr, p_value = pearsonr(
        agg_df[('polarity', 'mean')], 
        agg_df['daily_return']
    )
    results['polarity_return_corr'] = corr
    results['polarity_return_p_value'] = p_value
    
    # Correlation between sentiment count and absolute returns
    corr, p_value = pearsonr(
        agg_df[('polarity', 'count')],
        agg_df['daily_return'].abs()
    )
    results['volume_volatility_corr'] = corr
    results['volume_volatility_p_value'] = p_value
    
    return results

def analyze_all_tickers(news_df, tickers):
    """Run correlation analysis for all tickers"""
    results = []
    
    for ticker in tickers:
        try:
            stock_df = load_stock_data(ticker)
            merged_df = merge_news_stock_data(news_df, stock_df, ticker)
            sentiment_df = add_sentiment_columns(merged_df)
            agg_df = aggregate_daily_sentiment(sentiment_df)
            
            corr_results = calculate_correlations(agg_df)
            corr_results['ticker'] = ticker
            results.append(corr_results)
            
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
    
    return pd.DataFrame(results)