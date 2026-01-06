import numpy as np
from src.hmm import GaussianHMM
from src.evaluation import analyze_regimes, plot_regimes_on_price, compute_aic, compute_bic
from data.data import MarketDataLoader

def run_analysis(ticker='SPY'):
    #1. Load Data
    print(f"Fetching {ticker} Market Data")
    loader = MarketDataLoader(ticker=ticker, start_date='2018-01-01', end_date='2024-01-01')
    loader.download_data()
    X, dates = loader.format_hmm_data(features=['returns', 'volatility'])

    #2. Fit Sticky HMM
    print("Training Sticky HMM")
    model = GaussianHMM(n_states=3, sticky_param=0.5, random_state=42)
    model.fit(X, n_restarts=5)

    #3. Predict and Analyze
    states = model.predict(X)
    probs = model.predict_probabilities(X)
    stats = analyze_regimes(model, X, dates, feature_names=['returns', 'volatility'])

    #Annualize stats for the report
    stats['Annual Return'] = stats['returns_mean'] * 252
    stats['Annual Volatility'] = stats['returns_std'] * np.sqrt(252)

    #4. Map Regimes to Interpretations
    sorted_idx = stats.sort_values('Annual Return').index.tolist()
    names = {sorted_idx[0]: "Bear/Crisis", sorted_idx[1]: "Sideways", sorted_idx[2]: "Bull Market"}
    regime_labels = [names[i] for i in range(3)]

    #5. Visualizations
    prices = loader.data['Close'].values.flatten()
    n_dropped = len(prices) - len(X)
    prices = prices[n_dropped:]
    fig1 = plot_regimes_on_price(dates, states, prices, title=f'{ticker} Regime Analysis', regime_names=regime_labels)
    fig1.savefig('regime_overlay.png')

    print("---Model Performance Summary---")
    print(f"AIC: {compute_aic(model, X):.2f} | BIC: {compute_bic(model, X):.2f}")
    print("\nRegime Statistics:")
    print(stats[['regime', 'percentage', 'Annual Return', 'Annual Volatility']])

if __name__ == "__main__":
    run_analysis()



