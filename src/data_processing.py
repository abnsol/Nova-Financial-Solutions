import pandas as pd
import numpy as np
from textblob import TextBlob
import swifter # Ensure swifter is installed: pip install swifter
import nltk # Ensure nltk is installed: pip install nltk
import os

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

class FinancialNewsProcessor:
    """
    A class to process financial news data, including loading,
    sentiment analysis, and basic feature engineering.
    """
    def __init__(self, data_path: str):
        """
        Initializes the processor with the path to the raw news data.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        self.data_path = data_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads the raw analyst ratings CSV into a DataFrame.
        Handles potential errors during loading.
        """
        try:
            self.df = pd.read_csv(self.data_path)
            print("News data loaded successfully.")
            return self.df
        except Exception as e:
            print(f"Error loading data from {self.data_path}: {e}")
            raise # Re-raise the exception after logging

    def preprocess_headlines(self) -> pd.DataFrame:
        """
        Fills NaN headlines and calculates headline length.
        Requires data to be loaded.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.df['headline'] = self.df['headline'].fillna("") # Replace NaNs with empty strings
        self.df['headline_length'] = self.df['headline'].apply(len)
        print("Headlines preprocessed: NaNs handled, length calculated.")
        return self.df

    def calculate_sentiment(self) -> pd.DataFrame:
        """
        Calculates sentiment polarity for each headline using TextBlob.
        Requires 'headline' column to be preprocessed.
        """
        if self.df is None or 'headline' not in self.df.columns:
            raise ValueError("Data not loaded or 'headline' column missing. Call load_data() and preprocess_headlines() first.")
        # Using swifter for potential performance gain on larger datasets
        self.df['sentiment'] = self.df['headline'].swifter.apply(
            lambda text: TextBlob(str(text)).sentiment.polarity
        )
        print("Sentiment polarity calculated.")
        return self.df

    def parse_dates(self) -> pd.DataFrame:
        """
        Parses the 'date' column into datetime objects and handles invalid dates.
        Drops rows where date parsing fails.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        initial_rows = len(self.df)
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce', format='mixed')
        self.df = self.df.dropna(subset=['date'])
        if len(self.df) < initial_rows:
            print(f"Dropped {initial_rows - len(self.df)} rows due to invalid date parsing.")
        print("Dates parsed successfully.")
        return self.df

# Example of basic error handling:
def run_news_processing(data_path: str):
    """Orchestrates the news data processing."""
    try:
        processor = FinancialNewsProcessor(data_path)
        processor.load_data()
        processor.preprocess_headlines()
        processor.calculate_sentiment()
        processor.parse_dates()
        print("News data processing complete.")
        return processor.df
    except (FileNotFoundError, ValueError) as e:
        print(f"Critical error during news processing: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during news processing: {e}")
        return None