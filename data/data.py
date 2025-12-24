import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MarketDataLoader:
    """Load and prepare market data for HMM regime detection."""

    def __init__(self, ticker='SPY', start_date=None, end_date=None):
        """
        Args:
            ticker: Stock/ETF ticker symbol
            start_date: Start date (string 'YYYY-MM-DD' or datetime)
            end_date: End date (string 'YYYY-MM-DD' or datetime)
        """
        self.ticker = ticker
        self.start_date = start_date or (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
    
    def download_data(self):
        """Download historical data from Yahoo Finance."""
        print(f"Downloading {self.ticker} from {self.start_date} to {self.end_date}")
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        print(f"Downloaded {len(self.data)} data points")
        return self.data
    
    def calculate_returns(self, price_col='Close'):
        """Calculate log returns."""
        if self.data is None:
            raise ValueError("No data loaded. Call download_data() first.")
        
        returns = np.log(self.data[price_col] / self.data[price_col].shift(1))
        return returns.dropna()
    
    def calculate_features(self, window=20):
        """
        Calculate features for regime detection.
        
        Args:
            window: Rolling window size for volatility calculation
            
        Returns:
            DataFrame with features
        """

        if self.data is None:
            raise ValueError("No data loaded. Call download_data() first.")
        
        #Create a new DataFrame to store features
        features = pd.DataFrame(index=self.data.index) 

        #1. Log returns to measure relative price changes
        #log return = log(today's close / yesterday's close)
        features['returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))

        #2. Rolling volatility - standard deviation of returns over a moving window
        #Higher volatility often indicates more turbulent market conditions
        features['volatility'] = features['returns'].rolling(window=window).std()

        #3. Volume features for tracking absolute volume and relative changes in trading volume
        if 'Volume' in self.data.columns:
            features['volume'] = self.data['Volume']
            #Percentage change in volume compared to previous day
            features['volume_change'] = features['volume'].pct_change()
        
        #4. Moving average crossover signal for comparing short-term and long-term trends
        #Short-term MA reacts faster to price changes
        features['ma_short'] = self.data['Close'].rolling(window=window).mean()

        #Long-term MA reacts slower, using twice the short-term window
        features['ma_long'] = self.data['Close'].rolling(window=window*2).mean()

        #Difference between short-term and long-term MA can indicate trend direction
        features['ma_diff'] = features['ma_short'] - features['ma_long']

        return features.dropna()
    
    def format_hmm_data(self, features=['returns', 'volatility'], window=20):
        """
        Get data formatted for HMM training.
        
        Args:
            features: List of feature names to include
            window: Rolling window for calculations
            
        Returns:
            numpy array of shape (n_samples, n_features)
        """

        feature_df = self.calculate_features(window=window)

        #Select requested features
        selected_features = feature_df[features].dropna()

        return selected_features.values, selected_features.index




