# src/financial_indicators.py

import pandas as pd
import talib # Ensure TA-Lib is installed: pip install TA-Lib
import pynance as pn # Ensure PyNance is installed: pip install pynance
import os

class StockAnalyzer:
    """
    A class to load stock historical data and compute technical indicators.
    """
    def __init__(self, data_dir: str):
        """
        Initializes the StockAnalyzer with the directory containing stock CSVs.
        """
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found at: {data_dir}")
        self.data_dir = data_dir
        self.stock_data = {} # To store data for multiple stocks

    def load_stock_data(self, symbol: str) -> pd.DataFrame:
        """
        Loads historical stock data for a given symbol from a CSV file.
        Assumes file name format like 'AAPL_historical_data.csv'.
        """
        file_path = os.path.join(self.data_dir, f"{symbol}_historical_data.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Stock data file not found for {symbol} at: {file_path}")
        try:
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            # Ensure required columns are present [cite: 44]
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns in {symbol} data. Expected: {required_cols}")
            # Ensure numeric types for calculations
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=required_cols) # Drop rows where essential numeric data failed
            self.stock_data[symbol] = df
            print(f"Loaded historical data for {symbol} successfully.")
            return df
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            raise # Re-raise the exception after logging

    def calculate_technical_indicators(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates common technical indicators (MA, RSI, MACD) using TA-Lib.
        Requires 'Open', 'High', 'Low', 'Close', 'Volume' columns. [cite: 44]
        """
        if df.empty:
            raise ValueError(f"DataFrame for {symbol} is empty. Cannot calculate indicators.")

        # Ensure correct column names for TA-Lib
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values

        # Moving Averages (MA) - e.g., Simple Moving Average (SMA) [cite: 25]
        df['SMA_10'] = talib.SMA(close, timeperiod=10)
        df['SMA_20'] = talib.SMA(close, timeperiod=20)

        # Relative Strength Index (RSI) [cite: 25]
        df['RSI'] = talib.RSI(close, timeperiod=14)

        # Moving Average Convergence Divergence (MACD) [cite: 25]
        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macdsignal
        df['MACD_Hist'] = macdhist

        print(f"Calculated technical indicators for {symbol}.")
        return df

    def calculate_pynance_metrics(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Uses PyNance for additional financial metrics.
        Note: PyNance usually works on pn.DataFrame which extends pandas.
        """
        if df.empty:
            raise ValueError(f"DataFrame for {symbol} is empty. Cannot calculate PyNance metrics.")

        try:
            # Convert to PyNance DataFrame for easier metric calculation
            pn_df = pn.DataFrame(df)
            
            # Let's add a simple cumulative return using PyNance's capabilities if it were a full pn.DataFrame
            # If pn.DataFrame is just for convenience methods, you could do:
            df['Cumulative_Return'] = (1 + df['Close'].pct_change()).cumprod() - 1
            df['Daily_Vol'] = pn_df['Close'].rolling(window=20).std() * (252**0.5) # Example: annualized rolling volatility
            
            print(f"Calculated PyNance metrics for {symbol}.")
            return df
        except Exception as e:
            print(f"Error calculating PyNance metrics for {symbol}: {e}")
            raise # Re-raise the exception after logging