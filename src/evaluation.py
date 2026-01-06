"""
Evaluation methods for Hidden Markov Models in regime detection.

This module provides tools for:
- Model selection (AIC/BIC)
- Regime analysis and interpretation
- Transition dynamics
- Visualization
- Model comparison

Contains reusable functions that work with any trained HMM model.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)

def compute_aic(model, X):
    """
    Computes Akaike Information Criterion.
    
    Balances model fit quality against complexity to prevent overfitting.
    More states lead to better fit, but worse generalization.
    
    AIC = 2k - 2ln(L)
    where k = number of parameters, L = likelihood
    
    Lower AIC = better model
    
    Parameters
    ----------
    model : GaussianHMM
        Trained HMM model
    X : array-like, shape (n_samples, n_features)
        Observations
    
    Returns
    -------
    aic : float
        AIC score 
    """
    n_features = X.shape[1]
    n_states = model.n_states

    #Count free parameters in the model
    #Transition matrix (rows must sum to 1)
    n_transition_params = n_states * (n_states - 1)

    #Initial probabilities (must sum to 1)
    n_initial_params = n_states - 1

    #Emission means
    n_mean_params = n_states * n_features

    #Emission covariances
    if model.covariance_type == 'full':
        #Symmetric covariance
        n_covar_params = n_states * n_features * (n_features + 1) // 2
    elif model.covariance_type == 'diag':
        #Diagonal covariance:
        n_covar_params = n_states * n_features
    else:
        n_covar_params = n_states * n_features

    #Total parameters
    k = n_transition_params + n_initial_params + n_mean_params + n_covar_params

    log_likelihood = model.score(X)
    aic = 2 * k - 2 * log_likelihood

    return aic

def compute_bic(model, X):
    """
    Compute Bayesian Information Criterion.
    
    Similar to AIC, but penalizes complexity more heavily.
    Better for model selection when you want to avoid overfitting.
    
    BIC = k*ln(n) - 2ln(L)
    where k = parameters, n = sample size, L = likelihood
    
    Lower BIC = better model
    
    Parameters
    ----------
    model : GaussianHMM
        Trained HMM model
    X : array-like, shape (n_samples, n_features)
        Observations
    
    Returns
    -------
    bic : float
        BIC score 
    """
    n_samples, n_features = X.shape
    n_states = model.n_states
    
    #Count parameters 
    n_transition_params = n_states * (n_states - 1)
    n_initial_params = n_states - 1
    n_mean_params = n_states * n_features

    if model.covariance_type == 'full':
        n_covar_params = n_states * n_features * (n_features + 1) // 2
    elif model.covariance_type == 'diag':
        n_covar_params = n_states * n_features
    else:
        n_covar_params = n_states * n_features
    
    #Total parameters
    k = n_transition_params + n_initial_params + n_mean_params + n_covar_params

    log_likelihood = model.score(X)

    bic = k * np.log(n_samples) - 2 * log_likelihood
    
    return bic

def compare_models(models_dict, X_train, X_test=None):
    """
    Compares multiple HMM models using various metrics.
    
    Parameters
    ----------
    models_dict : dict
        Dictionary mapping model names to trained GaussianHMM models
        Example: {'2-state': model1, '3-state': model2}
    X_train : array-like
        Training data
    X_test : array-like, optional
        Test data for validation scores
    
    Returns
    -------
    results : pd.DataFrame
        Comparison table sorted by BIC 
    """
    results = []

    for name, model in models_dict.items():
        result = {
            'model': name,
            'n_states': model.n_states,
            'train_log_likelihood': model.score(X_train),
            'aic': compute_aic(model, X_train),
            'bic': compute_bic(model, X_train),
            'converged': model.converged,
            'n_iterations': len(model.log_likelihood_history)
        }

        if X_test is not None:
            result['test_loglik'] = model.score(X_test)

        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('bic')

    return results_df

def analyze_regimes(model, X, dates, feature_names=None):
    """
    Analyzes characteristics of discovered regimes.
    Tells you what the states from HMMM outputs actually mean (i.e. whether its a bull market or bear marker). 
    Computes statistical profiles for each regime such as how often it occurs, when it occured, etc.

    
    Parameters
    ----------
    model : GaussianHMM
        Trained HMM model
    X : array-like, shape (n_samples, n_features)
        Observations
    dates : array-like
        Date index for observations
    feature_names : list of str, optional
        Names of features (e.g., ['returns', 'volatility'])
    
    Returns
    -------
    regime_stats : pd.DataFrame
        Statistics for each regime including counts, percentages, and
        feature statistics (mean, std, min, max)
    """
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    states = model.predict(X)

    stats = []

    for state in range(model.n_states):
        mask = (states == state)
        regime_data = X[mask]

        if len(regime_data) == 0:
            continue

        stat = {
            'regime': state,
            'count': np.sum(mask),
            'percentage': np.sum(mask) / len(states) * 100
        }

        #Add statistics for each feature
        for i, feat_name in enumerate(feature_names):
            stat[f'{feat_name}_mean'] = regime_data[:, i].mean()
            stat[f'{feat_name}_std'] = regime_data[:, i].std()
            stat[f'{feat_name}_min'] = regime_data[:, i].min()
            stat[f'{feat_name}_max'] = regime_data[:, i].max()
        
        stats.append(stat)

    regime_stats = pd.DataFrame(stats)

    return regime_stats

def plot_regimes_on_price(dates, states, prices, title='Regime Detection', figsize=(14, 6), regime_names=None):
    """
    Visualizes regime switches overlaid on price chart.
    
    Parameters
    ----------
    dates : array-like
        Date index
    states : array-like
        Predicted state sequence from model.predict()
    prices : array-like
        Price series to plot
    title : str
        Plot title
    figsize : tuple
        Figure size
    regime_names : list of str, optional
        Names for regimes such as "bull" or "bear"
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    n_states = len(np.unique(states))

    if regime_names is None:
        regime_names = [f'Regime {i}' for i in range(n_states)]

    #Create color palette
    colors = plt.cm.Set3(np.linspace(0, 1, n_states))

    fig, ax = plt.subplots(figsize=figsize)

    #Plot price line
    ax.plot(dates, prices, linewidth=2, color='black', alpha=0.7, label='Price', zorder=2)

    #Shade background by regime
    #Find continuous segments of each regime
    for state in range(n_states):
        mask = (states == state)

        #Find where regime starts and stops
        regime_changes = np.diff(np.concatenate(([False], mask, [False])).astype(int))
        starts = np.where(regime_changes == 1)[0]
        ends = np.where(regime_changes == -1)[0]

        #Shade each segment
        for start, end in zip(starts, ends):
            ax.axvspan(dates[start], dates[end-1], alpha=0.3, color=colors[state], label=regime_names[state] if start == starts[0] else "", zorder=1)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    #Remove duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', framealpha=0.9)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig

def plot_regime_probabilities(dates, probs, title='Regime Probabilities Over Time',figsize=(14, 6), regime_names=None):
    """
    Plot state probabilities over time from soft decoding.
    
    Shows uncertainty in regime classification. Sometimes the model
    is very confident (100% bull), and sometimes it is uncertain (50% bull, 50% bear).
    This is valuable information that Viterbi hard decoding loses.
    
    Parameters
    ----------
    dates : array-like
        Date index
    proba : array-like, shape (n_samples, n_states)
        State probabilities from model.predict_proba()
    title : str
        Plot title
    figsize : tuple
        Figure size
    regime_names : list of str, optional
        Names for regimes
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_states = probs.shape[1]

    if regime_names is None:
        regime_names = [f'Regime {i}' for i in range(n_states)]

    fig, ax = plt.subplots(figsize=figsize)

    #Stacked area plot
    ax.stackplot(dates, probs.T, labels=regime_names, alpha=0.8)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def get_regime_transitions(states):
    """
    Counts transitions between regimes.
    
    Parameters
    ----------
    states : array-like
        Predicted state sequence from model.predict()
    
    Returns
    -------
    transition_counts : pd.DataFrame
        Matrix where entry [i,j] = number of transitions from state i to j
    """
    n_states = len(np.unique(states))
    transition_matrix = np.zeros((n_states, n_states), dtype=int)

    #Count each transition
    for t in range(len(states) - 1):
        from_state = states[t]
        to_state = states[t + 1]
        transition_matrix[from_state, to_state] += 1
        

    transition_count_df = pd.DataFrame(
        transition_matrix,
        index=[f'From_{i}' for i in range(n_states)],
        columns=[f'To_{i}' for i in range(n_states)]
    )

    return transition_count_df

def compute_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Compute annualized Sharpe ratio.
    
    Parameters
    ----------
    returns : array-like
        Period returns (not cumulative)
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Trading periods per year (252 for daily, 12 for monthly)
    
    Returns
    -------
    sharpe : float
    """
    excess_returns = returns - risk_free_rate / periods_per_year

    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
    return sharpe

def compute_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Compute annualized Sortino ratio (penalizes downside volatility only).
    """

    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    downside_std = np.std(downside_returns)
    sortino = np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)
    return sortino

def compute_max_drawdown(cumulative_returns):
    """
    Compute maximum drawdown from peak.
    
    Parameters
    ----------
    cumulative_returns : array-like
        Cumulative returns series
    
    Returns
    -------
    max_dd : float
        Maximum drawdown (positive number -> 0.25 = 25% drawdown)
    """

    cumulative_wealth = (1 + cumulative_returns)
    running_max = np.maximum.accumulate(cumulative_wealth)
    drawdown = (cumulative_wealth - running_max) / running_max
    max_dd = np.abs(np.min(drawdown))
    return max_dd

def compute_calmar_ratio(returns, periods_per_year=252):
    """
    Compute Calmar ratio (annualized return / max drawdown).
    """

    cumulative_returns = np.cumprod(1 + returns) - 1
    
    max_dd = compute_max_drawdown(cumulative_returns)

    if max_dd <= 0:
        return 0.0
    
    annualized_return = np.mean(returns) * periods_per_year
    calmar = annualized_return / max_dd

    return calmar

def compute_var(returns, confidence_level=0.95):
    """
    Compute Value at Risk at given confidence level.
    
    Parameters
    ----------
    returns : array-like
        Period returns
    confidence_level : float
        Confidence level (e.g. 0.95 for 95%)
    
    Returns
    -------
    var : float
        VaR (positive number representing loss)
    """
    var = np.abs(np.percentile(returns, (1 - confidence_level) * 100))
    return var

def compute_cvar(returns, confidence_level=0.95):
    """
    Compute Conditional Value at Risk (Expected Shortfall).
    Average loss beyond VaR threshold.
    """
    var_threshold = -compute_var(returns, confidence_level)
    cvar = np.abs(np.mean(returns[returns <= var_threshold]))
    return cvar

def backtest_regime_strategy(model, X, returns, dates, regime_positions=None,transaction_cost=0.001):
    """
    Backtest a regime-based trading strategy.
    
    Parameters
    ----------
    model : GaussianHMM
        Trained HMM model
    X : array-like, shape (n_samples, n_features)
        Features for regime detection
    returns : array-like, shape (n_samples,)
        Asset returns to trade on
    dates : array-like
        Date index
    regime_positions : dict, optional
        Dictionary mapping regime number to position
        Example: {0: 1.0, 1: 0.0, 2: -1.0} for long/cash/short
        Default: regime 0 = long, others = cash
    transaction_cost : float
        Cost per trade as fraction (e.g. 0.001 = 0.1%)
    
    Returns
    -------
    results : dict
        Dictionary containing strategy returns, positions, metrics
    """
    states = model.predict(X)
    n_states = model.n_states

    #Default positions: regime 0 long, others cash
    if regime_positions is None:
        regime_positions = {i: 1.0 if i == 0 else 0.0 for i in range(n_states)}

    #Create position series based on predicted regimes
    positions = np.array([regime_positions.get(s, 0.0) for s in states])

    #Calculate strategy returns
    strategy_returns = positions * returns

    #Apply transaction costs when position changes
    position_changes = np.abs(np.diff(positions, prepend=positions[0]))
    transaction_costs = position_changes * transaction_cost
    strategy_returns = strategy_returns - transaction_costs

    #Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + strategy_returns) - 1
    buy_hold_cumulative = np.cumprod(1 + returns) - 1

    #Filter for active trading days to get an accurate win rate
    active_days_mask = positions != 0
    active_returns = strategy_returns[active_days_mask]

    #Calculate win rate only on days with market exposure
    if len(active_returns) > 0:
        win_rate = np.sum(active_returns > 0) / len(active_returns)
    else:
        win_rate = 0.0

    #Compute metrics
    results = {
        'dates': dates,
        'states': states,
        'positions': positions,
        'strategy_returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'buy_hold_cumulative': buy_hold_cumulative,
        'total_return': cumulative_returns[-1],
        'annualized_return': np.mean(strategy_returns) * 252,
        'annualized_volatility': np.std(strategy_returns) * np.sqrt(252),
        'sharpe_ratio': compute_sharpe_ratio(strategy_returns),
        'sortino_ratio': compute_sortino_ratio(strategy_returns),
        'max_drawdown': compute_max_drawdown(cumulative_returns),
        'calmar_ratio': compute_calmar_ratio(strategy_returns),
        'var_95': compute_var(strategy_returns, 0.95),
        'cvar_95': compute_cvar(strategy_returns, 0.95),
        'n_trades': np.sum(position_changes > 0),
        'win_rate': win_rate
    }
    
    return results

def compare_strategies(results_dict):
    """
    Compare multiple backtest results.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping strategy names to backtest results
        Example: {'2-state': results1, '3-state': results2}
    
    Returns
    -------
    comparison : pd.DataFrame
        Comparison table with key metrics
    """
    comparison = []

    for name, results in results_dict.items():
        comparison.append({
            'strategy': name,
            'total_return': results['total_return'],
            'annualized_return': results['annualized_return'],
            'annualized_volatility': results['annualized_volatility'],
            'sharpe_ratio': results['sharpe_ratio'],
            'sortino_ratio': results['sortino_ratio'],
            'max_drawdown': results['max_drawdown'],
            'calmar_ratio': results['calmar_ratio'],
            'var_95': results['var_95'],
            'cvar_95': results['cvar_95'],
            'n_trades': results['n_trades'],
            'win_rate': results['win_rate']
        })
    
    df = pd.DataFrame(comparison)
    df = df.sort_values('sharpe_ratio', ascending=False)

    return df

def plot_backtest_results(results, title='Strategy Performance', figsize=(14, 10)):
    """
    Comprehensive visualization of backtest results.
    
    Parameters
    ----------
    results : dict
        Results from backtest_regime_strategy()
    title : str
        Plot title
    figsize : tuple
        Figure size
    """    

    fig, axes = plt.subplots(3, 1, figsize=figsize)
    dates = results['dates']

    #Panel 1: Cumulative returns
    axes[0].plot(dates, results['cumulative_returns'], label='Strategy', linewidth=2, color='blue')
    axes[0].plot(dates, results['buy_hold_cumulative'], label='Buy & Hold', linewidth=2, color='gray', alpha=0.7)
    axes[0].set_ylabel('Cumulative Return', fontweight='bold')
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    #Panel 2: Positions by regime
    colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(results['states']))))
    for state in np.unique(results['states']):
        mask = results['states'] == state
        axes[1].scatter(dates[mask], results['positions'][mask], label=f'Regime {state}', alpha=0.6, color=colors[state])
    axes[1].set_ylabel('Position', fontweight='bold')
    axes[1].set_ylim([-1.2, 1.2])
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    #Panel 3: Rolling Sharpe ratio
    window = min(60, len(results['strategy_returns']) // 5)
    rolling_sharpe = pd.Series(results['strategy_returns']).rolling(window).apply(
        lambda x: compute_sharpe_ratio(x.values) if len(x) == window else np.nan
    )
    axes[2].plot(dates, rolling_sharpe, linewidth=2, color='green')
    axes[2].set_xlabel('Date', fontweight='bold')
    axes[2].set_ylabel(f'Rolling Sharpe ({window}d)', fontweight='bold')
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_regime_performance(results, asset_returns, figsize=(12, 6)):
    """
    Analyze performance by regime.
    
    Parameters
    ----------
    results : dict
        Results from backtest_regime_strategy()
    figsize : tuple
        Figure size
    
    Returns
    -------
    regime_stats : pd.DataFrame
        Performance statistics by regime
    """

    states = results['states']

    regime_stats = []

    for state in np.unique(states):
        mask = states == state
        regime_market_returns = asset_returns[mask]
        
        regime_stats.append({
            'regime': state,
            'count': np.sum(mask),
            'mean_return': np.mean(regime_market_returns),
            'volatility': np.std(regime_market_returns),
            'sharpe': compute_sharpe_ratio(regime_market_returns),
            'win_rate': np.sum(regime_market_returns > 0) / len(regime_market_returns),
            'best_return': np.max(regime_market_returns),
            'worst_return': np.min(regime_market_returns)
        })

    
    df = pd.DataFrame(regime_stats)

    #Visualization
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    #Mean returns by regime
    axes[0].bar(df['regime'], df['mean_return'], color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Regime', fontweight='bold')
    axes[0].set_ylabel('Mean Return', fontweight='bold')
    axes[0].set_title('Average Returns by Regime', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    #Sharpe ratio by regime
    axes[1].bar(df['regime'], df['sharpe'], color='green', alpha=0.7)
    axes[1].set_xlabel('Regime', fontweight='bold')
    axes[1].set_ylabel('Sharpe Ratio', fontweight='bold')
    axes[1].set_title('Sharpe Ratio by Regime', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    return df, fig








