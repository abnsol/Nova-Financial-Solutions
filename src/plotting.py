# src/plotting.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class NewsDataPlotter:
    """
    A class to generate visualizations for financial news data.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the plotter with the DataFrame containing news data.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("DataFrame must not be empty or invalid.")
        self.df = df

    def plot_headline_length_distribution(self):
        """
        Plots the distribution of headline lengths.
        """
        if 'headline_length' not in self.df.columns:
            raise ValueError("'headline_length' column not found. Run preprocess_headlines() first.")
        plt.figure(figsize=(10, 4))
        sns.histplot(self.df['headline_length'], bins=50)
        plt.title("Distribution of Headline Lengths")
        plt.xlabel("Headline Length")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
        print("Headline length distribution plotted.")

    def plot_sentiment_distribution(self):
        """
        Plots the distribution of sentiment polarity scores.
        """
        if 'sentiment' not in self.df.columns:
            raise ValueError("'sentiment' column not found. Run calculate_sentiment() first.")
        plt.figure(figsize=(10, 4))
        sns.histplot(self.df['sentiment'], bins=50, kde=True)
        plt.title("Sentiment Polarity Distribution")
        plt.xlabel("Sentiment Polarity")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
        print("Sentiment polarity distribution plotted.")

    def plot_top_publishers(self, n=10):
        """
        Plots the top N publishers by article count.
        """
        if 'publisher' not in self.df.columns:
            raise ValueError("'publisher' column not found.")
        plt.figure(figsize=(12, 5))
        self.df['publisher'].value_counts().nlargest(n).plot(kind='bar')
        plt.title(f"Top {n} Publishers by Article Count")
        plt.xlabel("Publisher")
        plt.ylabel("Number of Articles")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        print(f"Top {n} publishers plotted.")

    def plot_articles_per_day(self):
        """
        Plots the number of articles published per day.
        Requires 'date' column to be parsed to datetime and set as index.
        """
        if 'date' not in self.df.columns or not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            raise ValueError("'date' column is not in datetime format or missing. Ensure dates are parsed.")
        
        # Ensure 'date' is the index for resampling, then reset if needed later
        temp_df = self.df.set_index('date')
        plt.figure(figsize=(14, 5))
        temp_df['headline'].resample('D').count().plot(title='Articles per Day')
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.tight_layout()
        plt.show()
        print("Articles per day plotted.")