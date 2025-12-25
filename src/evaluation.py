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
from typing import List, Dict, Tuple, Optional

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
            'converged': model.converged_,
            'n_iterations': len(model.log_likelihood_history_)
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










