import numpy as np
import matplotlib.pyplot as plt
from data.data import MarketDataLoader
from src.hmm import GaussianHMM

from src.evaluation import (
    compute_aic,
    compute_bic,
    compare_models,
    analyze_regimes,
    get_regime_transitions,
    plot_regimes_on_price,
    plot_regime_probabilities,
)

#Test 1: Data loading
print("HMM for Regime Detection Test Script")
print("-" * 70)
print()
print("Test 1: Data Loading")
print("-" * 70)

loader = MarketDataLoader(ticker='SPY', start_date='2020-01-01', end_date='2024-12-31')
loader.download_data()
X, dates = loader.format_hmm_data(features=['returns', 'volatility'])

print(f"Loaded {len(X)} samples from {dates[0]} to {dates[-1]}")
print(f"Data shape: {X.shape}")

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
dates_train, dates_test = dates[:split], dates[split:]
print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

#Test 2: Train Standard HMM
print("-" * 70)
print("Test 2: Training Standard HMM")
print("-" * 70)

model_standard = GaussianHMM(n_states=3, n_iter=50, random_state=42)
model_standard.fit(X_train)

print(f"Converged: {model_standard.converged}")
print(f"Iterations: {len(model_standard.log_likelihood_history)}")
print(f"Final log-likelihood: {model_standard.log_likelihood_history[-1]:.4f}")

#Test 3: Train Sticky HMM
print("-" * 70)
print("Test 3: Training Sticky HMM")
print("-" * 70)

model_sticky = GaussianHMM(n_states=3, sticky_param=0.5, n_iter=50, random_state=42)
model_sticky.fit(X_train)

print(f"Converged: {model_sticky.converged}")
print(f"Iterations: {len(model_sticky.log_likelihood_history)}")
print(f"Final log-likelihood: {model_sticky.log_likelihood_history[-1]:.4f}")

#Test 4: Random Restarts
print("-" * 70)
print("Test 4: Random Restarts")
print("-" * 70)

model_restarts = GaussianHMM(n_states=3, sticky_param=0.5, n_iter=50, random_state=42)
model_restarts.fit(X_train, n_restarts=3)

print(f"Final log-likelihood from 3 restarts: {model_restarts.log_likelihood_history[-1]:.4f}")

#Test 5: Model Comparison
print("-" * 70)
print("Test 5: Model Comparison")
print("-"*70)

models = {
    '2-state': GaussianHMM(n_states=2, n_iter=50, random_state=42).fit(X_train),
    '3-state': model_standard,
    '3-state-sticky': model_sticky,
    '4-state': GaussianHMM(n_states=4, n_iter=50, random_state=42).fit(X_train)
}

comparison = compare_models(models, X_train, X_test)

print("\n" + comparison.to_string(index=False))

best_model_name = comparison.iloc[0]['model']
best_model = models[best_model_name]
print(f"\n Best model: {best_model_name} (lowest BIC)")

#Test 6: Regime Analysis
print("-" * 70)
print("Test 6: Regime analysis")
print("-"*70)
states = best_model.predict(X_train)
regime_stats = analyze_regimes(best_model, X_train, dates_train, feature_names=['returns', 'volatility'])

print("\nRegime Statistics:")
print(regime_stats.to_string(index=False))

#Test 7: Transition Analysis
print("-" * 70)
print("Test 7: Transition Analysis")
print("-"*70)

transitions = get_regime_transitions(states)
print("\nTransition Counts:")
print(transitions)

print("\nTransition Probabilities (from model):")
for i in range(best_model.n_states):
    print(f"State {i}: {best_model.trans_mat[i]}")

#Test 8: Predictions
print("-" * 70)
print("Test 8: Predictions")
print("-"*70)

#Hard decoding (Viterbi)
states_test = best_model.predict(X_test)
print(f" Predicted states for {len(X_test)} test samples")
print(f" State distribution: {np.bincount(states_test)}")

#Soft decoding (probabilities)
probs_test = best_model.predict_probabilities(X_test)
print(f" Predicted probabilities shape: {probs_test.shape}")
print(f" Average max probability: {np.mean(np.max(probs_test, axis=1)):.3f}")

#Test 9: Visualization
print("-" * 70)
print("Test 9: Visualizations")
print("-"*70)

#Use full dataset for visualizations
states_full = best_model.predict(X)
probs_full = best_model.predict_probabilities(X)

prices_df = loader.data.loc[dates, 'Close']
prices = prices_df.values.flatten()

#Plot 1: Regimes on price
plot_regimes_on_price(
    dates, states_full, prices,
    title=f'{best_model_name.upper()} Regime Detection on SPY',
    regime_names=['Regime 0', 'Regime 1', 'Regime 2'],
)
#plt.close()

#Plot 2: Regime Probabilities
plot_regime_probabilities(
    dates, probs_full,
    title='Regime Probabilities Over Time',
    regime_names=['Regime 0', 'Regime 1', 'Regime 2'],
)
#plt.close()

plt.show()

#Test 10: Model Scoring
print("-" * 70)
print("Test 10: Model Scoring")
print("-"*70)

train_score = best_model.score(X_train)
test_score = best_model.score(X_test)
train_aic = compute_aic(best_model, X_train)
train_bic = compute_bic(best_model, X_train)

print(f"Train log-likelihood: {train_score:.4f}")
print(f"Test log-likelihood:  {test_score:.4f}")
print(f"AIC:                  {train_aic:.4f}")
print(f"BIC:                  {train_bic:.4f}")






