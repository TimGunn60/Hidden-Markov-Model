import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.data import MarketDataLoader
from src.hmm import GaussianHMM
from src.evaluation import (
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
    compute_var,
    compute_cvar,
    backtest_regime_strategy,
    compare_strategies,
    plot_backtest_results,
    plot_regime_performance,
)

print("Backtesting Test Script")
print("-" * 70)
print()

#Test 1: Data loading/preparation
print("Test 1: Data Loading")
print("-" * 70)
loader = MarketDataLoader(ticker='SPY', start_date='2018-01-01', end_date='2024-12-31')
loader.download_data()

#Get features for HMM
X, dates = loader.format_hmm_data(features=['returns', 'volatility'])

#Get returns for backtesting (calculate from Close prices)
returns = loader.data.loc[dates, 'Close'].pct_change().fillna(0).values.flatten()

#Get prices for visualization
prices_df = loader.data.loc[dates, 'Close']
prices = prices_df.values.flatten()

print(f"Loaded {len(X)} samples from {dates[0]} to {dates[-1]}")
print()

#Train/test split
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
dates_train, dates_test = dates[:split], dates[split:]
returns_train, returns_test = returns[:split], returns[split:]

print(f"Train period: {dates_train[0]} to {dates_train[-1]} ({len(X_train)} samples)")
print(f"Test period:  {dates_test[0]} to {dates_test[-1]} ({len(X_test)} samples)")
print()

#Test 2: Training the HMM
print("-" * 70)
print("Test 2: Training Standard HMM")
print("-" * 70)

# Train a 3-state HMM
model = GaussianHMM(n_states=3, sticky_param=0.5, n_iter=200, random_state=42)
model.fit(X_train, n_restarts=5)

print(f"Model converged: {model.converged}")
print(f"Final log-likelihood: {model.log_likelihood_history[-1]:.4f}")
print(f"Number of states: {model.n_states}")
print()

#Analyze regimes to understand them
states_train = model.predict(X_train)
regime_info = pd.DataFrame({
    'regime': range(model.n_states),
    'count': [np.sum(states_train == i) for i in range(model.n_states)],
    'avg_return': [returns_train[states_train == i].mean() for i in range(model.n_states)],
    'avg_volatility': [X_train[states_train == i, 1].mean() for i in range(model.n_states)]
})
regime_info['percentage'] = regime_info['count'] / len(states_train) * 100

print("Regime Characteristics:")
print(regime_info.to_string(index=False))
print()

#Test 3: Risk Metrics on Buy-and-Hold
print("-" * 70)
print("Test 3: Risk Metrics on Buy-and-Hold Strategy")
print("-" * 70)

cumulative_returns = np.cumprod(1 + returns_train) - 1

print(f"Total Return:           {cumulative_returns[-1]:.2%}")
print(f"Annualized Return:      {np.mean(returns_train) * 252:.2%}")
print(f"Annualized Volatility:  {np.std(returns_train) * np.sqrt(252):.2%}")
print(f"Sharpe Ratio:           {compute_sharpe_ratio(returns_train):.3f}")
print(f"Sortino Ratio:          {compute_sortino_ratio(returns_train):.3f}")
print(f"Maximum Drawdown:       {compute_max_drawdown(cumulative_returns):.2%}")
print(f"Calmar Ratio:           {compute_calmar_ratio(returns_train):.3f}")
print(f"VaR (95%):              {compute_var(returns_train, 0.95):.2%}")
print(f"CVaR (95%):             {compute_cvar(returns_train, 0.95):.2%}")
print()

#Test 4: Simple Long-Only Strategy (Regime 0)
print("-" * 70)
print("Test 4: Simple Long-Only Strategy (Best Regime)")
print("-" * 70)

#Strategy: Long when in regime with highest average return, cash otherwise
best_regime = regime_info.loc[regime_info['avg_return'].idxmax(), 'regime']
print(f"Identified Regime {int(best_regime)} as best performing")
print(f"Average return in this regime: {regime_info.loc[regime_info['regime'] == best_regime, 'avg_return'].values[0]:.4%}")
print()

#Define positions: 1.0 (long) in best regime, 0.0 (cash) otherwise
regime_positions_simple = {i: 1.0 if i == best_regime else 0.0 for i in range(model.n_states)}

results_simple = backtest_regime_strategy(
    model=model,
    X=X_train,
    returns=returns_train,
    dates=dates_train,
    regime_positions=regime_positions_simple,
    transaction_cost=0.001
)

print("Strategy Performance:")
print(f"Total Return:           {results_simple['total_return']:.2%}")
print(f"Annualized Return:      {results_simple['annualized_return']:.2%}")
print(f"Annualized Volatility:  {results_simple['annualized_volatility']:.2%}")
print(f"Sharpe Ratio:           {results_simple['sharpe_ratio']:.3f}")
print(f"Sortino Ratio:          {results_simple['sortino_ratio']:.3f}")
print(f"Maximum Drawdown:       {results_simple['max_drawdown']:.2%}")
print(f"Calmar Ratio:           {results_simple['calmar_ratio']:.3f}")
print(f"Number of Trades:       {results_simple['n_trades']}")
print(f"Win Rate:               {results_simple['win_rate']:.2%}")
print()

#Test 5: Multi-Regime Strategy (Long/Cash/Short)
print("-" * 70)
print("Test 5: Multi-Regime Strategy (Long/Cash/Short)")
print("-" * 70)

#Sort regimes by average return
sorted_regimes = regime_info.sort_values('avg_return', ascending=False)['regime'].values

#Assign positions: Best=Long, Middle=Cash, Worst=Short
regime_positions_multi = {}
if len(sorted_regimes) >= 3:
    regime_positions_multi[sorted_regimes[0]] = 1.0   #Best: Long
    regime_positions_multi[sorted_regimes[1]] = 0.0   #Middle: Cash
    regime_positions_multi[sorted_regimes[2]] = -0.5  #Worst: Moderate Short
elif len(sorted_regimes) == 2:
    regime_positions_multi[sorted_regimes[0]] = 1.0   #Best: Long
    regime_positions_multi[sorted_regimes[1]] = 0.0   #Worst: Cash

print("Position mapping:")
for regime, position in regime_positions_multi.items():
    pos_str = "LONG" if position > 0 else "SHORT" if position < 0 else "CASH"
    print(f"  Regime {int(regime)}: {position:+.1f} ({pos_str})")
print()

results_multi = backtest_regime_strategy(
    model=model,
    X=X_train,
    returns=returns_train,
    dates=dates_train,
    regime_positions=regime_positions_multi,
    transaction_cost=0.001
)

print("Strategy Performance:")
print(f"Total Return:           {results_multi['total_return']:.2%}")
print(f"Annualized Return:      {results_multi['annualized_return']:.2%}")
print(f"Annualized Volatility:  {results_multi['annualized_volatility']:.2%}")
print(f"Sharpe Ratio:           {results_multi['sharpe_ratio']:.3f}")
print(f"Sortino Ratio:          {results_multi['sortino_ratio']:.3f}")
print(f"Maximum Drawdown:       {results_multi['max_drawdown']:.2%}")
print(f"Calmar Ratio:           {results_multi['calmar_ratio']:.3f}")
print(f"Number of Trades:       {results_multi['n_trades']}")
print(f"Win Rate:               {results_multi['win_rate']:.2%}")
print()

#Test 6: Conservative Strategy (Volatility-Based)
print("-" * 70)
print("Test 6: Conservative Strategy (Low Volatility = Long)")
print("-" * 70)

#Strategy: Long in lowest volatility regime, cash otherwise
lowest_vol_regime = regime_info.loc[regime_info['avg_volatility'].idxmin(), 'regime']
print(f"Identified Regime {int(lowest_vol_regime)} as lowest volatility")
print(f"Average volatility in this regime: {regime_info.loc[regime_info['regime'] == lowest_vol_regime, 'avg_volatility'].values[0]:.4f}")
print()

regime_positions_conservative = {i: 1.0 if i == lowest_vol_regime else 0.0 for i in range(model.n_states)}

results_conservative = backtest_regime_strategy(
    model=model,
    X=X_train,
    returns=returns_train,
    dates=dates_train,
    regime_positions=regime_positions_conservative,
    transaction_cost=0.001
)

print("Strategy Performance:")
print(f"Total Return:           {results_conservative['total_return']:.2%}")
print(f"Sharpe Ratio:           {results_conservative['sharpe_ratio']:.3f}")
print(f"Maximum Drawdown:       {results_conservative['max_drawdown']:.2%}")
print(f"Number of Trades:       {results_conservative['n_trades']}")
print()

#Test 7: Strategy Comparison
print("-" * 70)
print("Test 7: Strategy Comparison")
print("-" * 70)

# Create buy-and-hold baseline
results_buyhold = backtest_regime_strategy(
    model=model,
    X=X_train,
    returns=returns_train,
    dates=dates_train,
    regime_positions={i: 1.0 for i in range(model.n_states)},  #Always long
    transaction_cost=0.0  #No trades
)

strategies = {
    'Buy & Hold': results_buyhold,
    'Long Best Regime': results_simple,
    'Long/Cash/Short': results_multi,
    'Low Volatility': results_conservative,
}

comparison = compare_strategies(strategies)
print("\nStrategy Comparison:")
print(comparison.to_string(index=False))
print()

#Test 8: Out of Sample Testing
print("-" * 70)
print("Test 8: Out of Sample Testing")
print("-" * 70)

#Test best strategy on hold-out test set
best_strategy_name = comparison.iloc[0]['strategy']
print(f"Testing '{best_strategy_name}' on out-of-sample data:")
print()

#Get the regime positions from the best strategy
if best_strategy_name == 'Long Best Regime':
    test_positions = regime_positions_simple
elif best_strategy_name == 'Long/Cash/Short':
    test_positions = regime_positions_multi
elif best_strategy_name == 'Low Volatility':
    test_positions = regime_positions_conservative
else:
    test_positions = {i: 1.0 for i in range(model.n_states)}

results_test = backtest_regime_strategy(
    model=model,
    X=X_test,
    returns=returns_test,
    dates=dates_test,
    regime_positions=test_positions,
    transaction_cost=0.001
)

results_test_buyhold = backtest_regime_strategy(
    model=model,
    X=X_test,
    returns=returns_test,
    dates=dates_test,
    regime_positions={i: 1.0 for i in range(model.n_states)},
    transaction_cost=0.0
)

print("Out-of-Sample Performance:")
print(f"Strategy:      {results_test['total_return']:>8.2%}  (Sharpe: {results_test['sharpe_ratio']:.3f})")
print(f"Buy & Hold:    {results_test_buyhold['total_return']:>8.2%}  (Sharpe: {results_test_buyhold['sharpe_ratio']:.3f})")
print(f"Outperformance: {(results_test['total_return'] - results_test_buyhold['total_return']):>7.2%}")
print()

#Test 9: Performance By Regime
print("-" * 70)
print("Test 9: Performance by Regime Analysis")
print("-" * 70)

regime_perf, fig_regime = plot_regime_performance(results_multi, returns_train)
fig_regime.savefig('regime_market_performance.png', dpi=300, bbox_inches='tight')

print("\nPerformance Statistics by Regime:")
print(regime_perf.to_string(index=False))
print()

#Test 10: Visualizations
print("-" * 70)
print("Test 10: Creating Visualizations")
print("-" * 70)

#Plot 1: Backtest results for best strategy
fig1 = plot_backtest_results(
    results_multi,
    title='Multi-Regime Strategy: Long/Cash/Short'
)

#Save Fig 1
fig1.savefig('multi_regime_backtest.png', dpi=300, bbox_inches='tight')

#Plot 2: Compare all strategies
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

#Cumulative returns comparison
for name, results in strategies.items():
    axes[0, 0].plot(results['dates'], results['cumulative_returns'], label=name, linewidth=2, alpha=0.8)
axes[0, 0].set_title('Cumulative Returns Comparison', fontweight='bold')
axes[0, 0].set_ylabel('Cumulative Return', fontweight='bold')
axes[0, 0].legend(loc='best', framealpha=0.9)
axes[0, 0].grid(True, alpha=0.3)

#Sharpe ratios
sharpes = [results['sharpe_ratio'] for results in strategies.values()]
axes[0, 1].bar(range(len(strategies)), sharpes, color='steelblue', alpha=0.7)
axes[0, 1].set_xticks(range(len(strategies)))
axes[0, 1].set_xticklabels(strategies.keys(), rotation=45, ha='right')
axes[0, 1].set_title('Sharpe Ratio Comparison', fontweight='bold')
axes[0, 1].set_ylabel('Sharpe Ratio', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

#Max drawdowns
drawdowns = [results['max_drawdown'] for results in strategies.values()]
axes[1, 0].bar(range(len(strategies)), drawdowns, color='red', alpha=0.7)
axes[1, 0].set_xticks(range(len(strategies)))
axes[1, 0].set_xticklabels(strategies.keys(), rotation=45, ha='right')
axes[1, 0].set_title('Maximum Drawdown Comparison', fontweight='bold')
axes[1, 0].set_ylabel('Max Drawdown', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

#Number of trades
trades = [results['n_trades'] for results in strategies.values()]
axes[1, 1].bar(range(len(strategies)), trades, color='green', alpha=0.7)
axes[1, 1].set_xticks(range(len(strategies)))
axes[1, 1].set_xticklabels(strategies.keys(), rotation=45, ha='right')
axes[1, 1].set_title('Number of Trades', fontweight='bold')
axes[1, 1].set_ylabel('Trades', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig2.savefig('strategy_comparison_grid.png', dpi=300, bbox_inches='tight')

#Performance Summary 
print("-" * 70)
print("Summary")
print("-" * 70)
print()
print(f"Best Training Strategy: {comparison.iloc[0]['strategy']}")
print(f"Sharpe Ratio: {comparison.iloc[0]['sharpe_ratio']:.3f}")
print(f"Total Return: {comparison.iloc[0]['total_return']:.2%}")
print(f"Max Drawdown: {comparison.iloc[0]['max_drawdown']:.2%}")
print()
print(f"Out-of-Sample Performance:")
print(f"Total Return: {results_test['total_return']:.2%}")
print(f"Sharpe Ratio: {results_test['sharpe_ratio']:.3f}")
print(f"vs Buy & Hold: {(results_test['total_return'] - results_test_buyhold['total_return']):+.2%}")
print()










