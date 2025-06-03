from textblob import TextBlob
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download(['vader_lexicon', 'punkt'])

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment using multiple methods"""
        blob = TextBlob(str(text))
        vader_scores = self.sia.polarity_scores(str(text))
        
        return {
            'textblob_polarity': blob.sentiment.polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity,
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu']
        }
    
    def add_sentiment_features(self, df: pd.DataFrame, text_col: str = 'headline') -> pd.DataFrame:
        """Add sentiment features to dataframe"""
        sentiment_df = df[text_col].apply(lambda x: pd.Series(self.analyze_sentiment(x)))
        return pd.concat([df, sentiment_df], axis=1)