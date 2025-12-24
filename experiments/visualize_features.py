import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from data.data import MarketDataLoader

def test_data_loader():
    """Test the data loader and display basic info."""
    print("Data loader test:")
    print("-" * 50)

    #Initialize loader
    loader = MarketDataLoader(ticker='SPY', start_date='2019-01-01')

    #Download data
    data = loader.download_data()
    print(f"\nColumns: {list(data.columns)}")
    print(f"\nFirst few rows:\n{data.head()}")

    #Calculate features
    features = loader.calculate_features(window=20)
    print(f"\nFeature columns: {list(features.columns)}")

    #Format HMM data
    values, index = loader.format_hmm_data(features=['returns', 'volatility'])
    print(f"\n\nHMM-ready data shape: {values.shape}")

    return loader, features, values, index

def visualize_features(loader, features):
    """Create visualizations of the features."""
    print("Visualizations:")
    print("-" * 50)

    sns.set_style("darkgrid")

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(f'{loader.ticker} Feature Analysis for HMM Regime Detection', fontsize=16, fontweight='bold')


    #1. Price and Moving Averages
    ax1 = axes[0]
    ax1.plot(loader.data.index, loader.data['Close'], label='Close Price', linewidth=1.5)
    ax1.plot(features.index, features['ma_short'], label='MA Short (20d)', alpha=0.7, linewidth=1)
    ax1.plot(features.index, features['ma_long'], label='MA Long (40d)', alpha=0.7, linewidth=1)
    ax1.set_title('Price with Moving Averages')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    #2. Returns
    ax2 = axes[1]
    ax2.plot(features.index, features['returns'], label='Log Returns', linewidth=0.8, alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.fill_between(features.index, features['returns'], 0, where=(features['returns'] > 0), alpha=0.3, color='green', label='Positive')
    ax2.fill_between(features.index, features['returns'], 0, where=(features['returns'] <= 0), alpha=0.3, color='red', label='Negative')
    ax2.set_title('Daily Log Returns')
    ax2.set_ylabel('Returns')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    #3. Volatility
    ax3 = axes[2]
    ax3.plot(features.index, features['volatility'], label='Rolling Volatility (20d)', color='purple', linewidth=1.5)
    ax3.fill_between(features.index, features['volatility'], alpha=0.3, color='purple')
    ax3.set_title('Rolling Volatility (Regime Indicator)')
    ax3.set_ylabel('Volatility')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    #4. Volume (if available)
    ax4 = axes[3]
    if 'volume' in features.columns:
        ax4.bar(features.index, features['volume'], alpha=0.6, width=1, color='blue')
        ax4.set_title('Trading Volume')
        ax4.set_ylabel('Volume')
    else:
        ax4.text(0.5, 0.5, 'Volume data not available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Trading Volume')
    ax4.set_xlabel('Date')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    #Create correlation heatmap
    fig2, ax = plt.subplots(figsize=(10, 8))
    correlation_features = features[['returns', 'volatility', 'ma_diff']].dropna()
    corr_matrix = correlation_features.corr()

    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold') 
    plt.tight_layout()

    #Distribution plots
    fig3, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig3.suptitle('Feature Distributions', fontsize=14, fontweight='bold')

    #Returns distribution
    axes[0].hist(features['returns'].dropna(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(features['returns'].mean(), color='red', linestyle='--', label=f'Mean: {features["returns"].mean():.4f}')
    axes[0].set_xlabel('Returns')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Returns Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    #Volatility distribution
    axes[1].hist(features['volatility'].dropna(), bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1].axvline(features['volatility'].mean(), color='red', linestyle='--', label=f'Mean: {features["volatility"].mean():.4f}')
    axes[1].set_xlabel('Volatility')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Volatility Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    # Run tests and visualizations
    loader, features, values, index = test_data_loader()
    visualize_features(loader, features)


    

    




    



    



    



