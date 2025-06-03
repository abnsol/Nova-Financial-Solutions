# src/stock_plotter.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class StockDataPlotter:
    """
    A class to generate visualizations for stock data and technical indicators.
    """
    def __init__(self, df: pd.DataFrame, symbol: str):
        """
        Initializes the plotter with the DataFrame containing stock data and its symbol.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("DataFrame must not be empty or invalid.")
        self.df = df
        self.symbol = symbol

    def plot_price_and_moving_averages(self):
        """
        Plots stock closing price along with Simple Moving Averages.
        """
        if not all(col in self.df.columns for col in ['Close', 'SMA_10', 'SMA_20']):
            raise ValueError("Missing 'Close', 'SMA_10', or 'SMA_20' columns for plotting.")
        
        plt.figure(figsize=(15, 7))
        plt.plot(self.df['Close'], label='Close Price', alpha=0.8)
        plt.plot(self.df['SMA_10'], label='SMA 10', alpha=0.7)
        plt.plot(self.df['SMA_20'], label='SMA 20', alpha=0.7)
        plt.title(f'{self.symbol} Stock Price with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        print(f"Plotted price and moving averages for {self.symbol}.")

    def plot_rsi(self):
        """
        Plots the Relative Strength Index (RSI).
        """
        if 'RSI' not in self.df.columns:
            raise ValueError("Missing 'RSI' column for plotting.")
        
        plt.figure(figsize=(15, 5))
        plt.plot(self.df['RSI'], label='RSI', color='purple')
        plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        plt.title(f'{self.symbol} Relative Strength Index (RSI)')
        plt.xlabel('Date')
        plt.ylabel('RSI Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        print(f"Plotted RSI for {self.symbol}.")

    def plot_macd(self):
        """
        Plots the Moving Average Convergence Divergence (MACD) and Signal Line.
        """
        if not all(col in self.df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
            raise ValueError("Missing MACD components for plotting.")
        
        plt.figure(figsize=(15, 7))
        plt.plot(self.df['MACD'], label='MACD Line', color='blue')
        plt.plot(self.df['MACD_Signal'], label='Signal Line', color='red')
        plt.bar(self.df.index, self.df['MACD_Hist'], label='MACD Histogram', color='gray', alpha=0.5)
        plt.title(f'{self.symbol} Moving Average Convergence Divergence (MACD)')
        plt.xlabel('Date')
        plt.ylabel('MACD Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        print(f"Plotted MACD for {self.symbol}.")

    def plot_daily_returns_and_volatility(self):
        """
        Plots daily returns and rolling volatility (if calculated).
        Requires 'Daily_Return' and 'Daily_Vol' (from pynance metrics)
        """
        # Ensure daily returns are calculated before plotting
        if 'Daily_Return' not in self.df.columns:
            self.df['Daily_Return'] = self.df['Close'].pct_change()

        plt.figure(figsize=(15, 8))

        # Plot Daily Returns
        ax1 = plt.subplot(211)
        ax1.plot(self.df['Daily_Return'], label='Daily Returns', color='gray', alpha=0.7)
        ax1.set_title(f'{self.symbol} Daily Returns')
        ax1.set_ylabel('Return (%)')
        ax1.grid(True)
        ax1.legend()

        # Plot Daily Volatility (if available from PyNance-like calculation)
        if 'Daily_Vol' in self.df.columns:
            ax2 = plt.subplot(212, sharex=ax1)
            ax2.plot(self.df['Daily_Vol'], label='Rolling 20-Day Volatility (Annualized)', color='orange')
            ax2.set_title(f'{self.symbol} Rolling Volatility')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Volatility')
            ax2.grid(True)
            ax2.legend()
        
        plt.tight_layout()
        plt.show()
        print(f"Plotted daily returns and volatility for {self.symbol}.")