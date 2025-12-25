# Sticky Hidden Markov Model for Financial Regime Detection

## Overview
This project implements a **custom Hidden Markov Model (HMM)** from scratch to identify **latent market regimes** in financial time series. 
The model is designed to uncover persistent market regimes with distinct risk and return characteristics, enabling better analysis of market behavior across time. 

Key features include log-space EM training, sticky transitions for regime persistence, and random restarts to avoid poor local optima.

---

## Model Features

- Custom HMM implementation (no `hmmlearn`)
- Log-space forward–backward and Viterbi algorithms
- Gaussian emissions with covariance regularization
- Sticky transition matrix to encourage regime persistence
- Random restarts with model selection via log-likelihood
- Feature standardization for stable training

---

## Training & Evaluation

The model is trained using the EM (Baum–Welch) algorithm in log-space.  
Since regimes are unlabeled, evaluation focuses on:

- Log-likelihood (primary objective)
- Regime persistence (average duration, self-transition probability)
- Economic interpretability (per-regime return, volatility, drawdowns)

---


