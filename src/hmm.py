import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

class GaussianHMM:
    """
    Hidden Markov Model with Gaussian emissions for regime detection.
    
    Uses the Baum-Welch (EM) algorithm for training and Viterbi algorithm
    for decoding the most likely state sequence.
    
    Parameters
    ----------
    n_states : int, default=3
        Number of hidden states (regimes)
    n_iter : int, default=100
        Maximum number of iterations for Baum-Welch algorithm
    tol : float, default=1e-4
        Convergence threshold for log-likelihood improvement
    covariance_type : str, default='full'
        Type of covariance parameters ('full', 'diag')
    min_covar : float, default=1e-3
        Minimum value for covariance regularization
    sticky_param : float, default=0.0
        Sticky HMM parameter to encourage regime persistence.
        Higher values such as 0.5-1.0 make states more persistent.
    Set to 0.0 for standard HMM.
    random_state : int, default=42
        Random seed for reproducibility
    """

    def __init__(self, n_states=3, n_iter=100, tol=1e-4,  covariance_type='full', min_covar=1e-3, sticky_param=0.0, random_state=42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.sticky_param = sticky_param
        self.random_state = random_state

        #Model parameters (to be learned)
        self.trans_mat = None  # Transition matrix 
        self.means = None     # Emission means for each state
        self.covars = None    # Emission covariances for each state
        self.startprobs = None # Initial state distribution

        #Standardization parameters
        self.feature_mean = None
        self.feature_std = None

        #Training history
        self.log_likelihood_history = []
        self.converged = False

        np.random.seed(random_state)

    def standardize_features(self, X, fit=False):
        """Standardize features to zero mean and unit variance."""
        if fit:
             self.feature_mean = np.mean(X, axis=0)
             self.feature_std = np.std(X, axis=0) + 1e-8 
        
        return (X - self.feature_mean) / self.feature_std
    
    def unstandardize_means(self):
        """Convert standardized means back to original scale."""
        return self.means * self.feature_std + self.feature_mean

    def initialize_parameters(self, X):
        """Initialize HMM parameters randomly with proper regularization.."""
        n_samples, n_features = X.shape

        #Initialize transition matrix with Dirichlet and smoothing
        alpha = np.ones(self.n_states) * 0.5
        self.trans_mat = np.random.dirichlet(alpha, size=self.n_states)

        #Ensure no exact zeros by adding a small epsilon
        self.trans_mat = np.maximum(self.trans_mat, 1e-10)
        self.trans_mat /= self.trans_mat.sum(axis=1, keepdims=True)

        #Initialize start probabilities with Dirichlet and smoothing
        self.startprobs = np.random.dirichlet(alpha)
        self.startprobs = np.maximum(self.startprobs, 1e-10)
        self.startprobs /= self.startprobs.sum()

        #Initialize means using k-means
        #Meant to spread out the initial means better than random selection
        indices = [np.random.randint(n_samples)]
        for _ in range(1, self.n_states):
            #Select points far from existing means
            distances = np.array([np.min([np.sum((X[i] - X[j])**2) for j in indices]) for i in range(n_samples)])
            probs = distances / distances.sum()
            indices.append(np.random.choice(n_samples, p=probs))
        
        self.means = X[indices].copy()
        
        #Initialize covariances with proper regularization
        if self.covariance_type == 'full':
            #Start with scaled identity matrices
            scale = np.var(X, axis=0).mean()
            self.covars = np.array([np.eye(n_features) * scale for _ in range(self.n_states)])
        elif self.covariance_type == 'diag':
            #Diagonal covariances
            self.covars = np.tile(np.var(X, axis=0), (self.n_states, 1))

        #Apply minimum covariance regularization
        self.regularize_covariances()
    
    def regularize_covariances(self):
        """Apply strong regularization to covariance matrices."""

        if self.covariance_type == 'full':
            for i in range(self.n_states):
                #Add regularization to diagonal
                self.covars[i] += np.eye(self.covars[i].shape[0]) * self.min_covar

                #Ensure positive definiteness
                eigvals = np.linalg.eigvalsh(self.covars[i])
                if np.min(eigvals) < self.min_covar:
                    #Add more to diagonal if still not positive definite
                    self.covars[i] += np.eye(self.covars[i].shape[0]) * (self.min_covar - np.min(eigvals))
        elif self.covariance_type == 'diag':
            self.covars = np.maximum(self.covars, self.min_covar)


    def compute_log_emission_probs(self, X):
        """
        Compute log emission probabilities for numerical stability.
        
        Returns log P(X_t | state=i) for all states and time steps.
        """
        n_samples = X.shape[0]
        log_emission_probs = np.zeros((self.n_states, n_samples))


        for state in range(self.n_states):
            try:
                if self.covariance_type == 'full':
                    log_emission_probs[state] = multivariate_normal.logpdf(X, mean=self.means[state], cov=self.covars[state])
                elif self.covariance_type == 'diag':
                    #For diagonal covariance, compute manually
                    diff = X - self.means[state]
                    log_emission_probs[state] = -0.5 * (
                        np.sum(np.log(2 * np.pi * self.covars[state])) +
                        np.sum((diff ** 2) / self.covars[state], axis=1)
                    )
            except:
                log_emission_probs[state] = 1e-10

        return log_emission_probs
    
    def forward_log(self, log_emission_probs):
        """
        Forward algorithm - compute forward probabilities in log space for stability.
        
        Computes log(alpha[t,i]) = log P(X_0...X_t, state_t=i)
        """

        n_samples = log_emission_probs.shape[1]
        log_alpha = np.zeros((n_samples, self.n_states))

        #Initialization: log(alpha[0,i]) = log(pi[i]) + log(B[i, obs[0]])
        log_alpha[0] = np.log(self.startprobs) + log_emission_probs[:, 0]

        #Pre-compute log transition matrix
        log_trans_mat = np.log(self.trans_mat)

        #Induction
        for t in range(1, n_samples):
            for j in range(self.n_states):
                #log(sum(alpha[t-1,i] * A[i,j])) = logsumexp(log_alpha[t-1] + log_A[:,j])
                log_alpha[t, j] = logsumexp(log_alpha[t-1] + log_trans_mat[:, j]) + log_emission_probs[j, t]

        return log_alpha
    
    def backward_log(self, log_emission_probs):
        """
        Backward algorithm - compute backward probabilities in log space for stability.
        
        Computes log(beta[t,i]) = log P(X_t+1...X_T | state_t=i)
        """
        n_samples = log_emission_probs.shape[1]
        log_beta = np.zeros((n_samples, self.n_states))

        #Initialization: log(beta[T-1, i]) = log(1) = 0 for all states
        log_beta[-1] = 0.0

        #Pre-compute log transition matrix
        log_trans_mat = np.log(self.trans_mat)

        #Induction (backward from T-2 to 0)
        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                #log(sum(A[i,j] * B[j,obs[t+1]] * beta[t+1,j]))
                log_beta[t, i] = logsumexp(log_trans_mat[i] + log_emission_probs[:, t+1] + log_beta[t+1])

        return log_beta
    
    def compute_gamma_xi_log(self, log_emission_probs):
        """
        Compute posterior probabilities in log space (E-step of Baum-Welch).
        
        Returns gamma and xi in probability space (not log).
        """

        n_samples = log_emission_probs.shape[1]

        #Compute forward and backward in log space
        log_alpha = self.forward_log(log_emission_probs)
        log_beta = self.backward_log(log_emission_probs)

        #Compute log P(observations | model) for normalization
        log_prob_obs = logsumexp(log_alpha[-1])

        #Compute gamma: P(state_t = i | X, model)
        log_gamma = log_alpha + log_beta - log_prob_obs
        gamma = np.exp(log_gamma)

        gamma /= gamma.sum(axis=1, keepdims=True)


        #Compute xi: P(state_t = i, state_t+1 = j | X, model)
        log_trans_mat = np.log(self.trans_mat)
        xi = np.zeros((n_samples - 1, self.n_states, self.n_states))

        for t in range(n_samples - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    log_xi = (log_alpha[t, i] + log_trans_mat[i, j] + log_emission_probs[j, t+1] + log_beta[t+1, j] - log_prob_obs)
                    xi[t, i, j] = np.exp(log_xi)
        
        #Normalize xi
        xi_sum = xi.sum(axis=(1, 2), keepdims=True)
        xi_sum[xi_sum == 0] = 1.0 
        xi /= xi_sum

        return gamma, xi
    
    def update_parameters(self, X, gamma, xi):
        """
        Update HMM parameters (M-step of Baum-Welch) with smoothing and sticky bias.

        """

        #Update initial state probabilities with Laplace smoothing
        self.startprobs = gamma[0] + 1e-10
        self.startprobs /= self.startprobs.sum()


        #Update transition matrix with Laplace smoothing
        numerator = np.sum(xi, axis=0) + 1e-10
        denominator = np.sum(gamma[:-1], axis=0, keepdims=True).T + 1e-10 * self.n_states
        self.trans_mat = numerator / denominator

        #Apply sticky HMM bias to encourage regime persistence
        if self.sticky_param > 0:
            for i in range(self.n_states):
                self.trans_mat[i, i] += self.sticky_param

        #Normalize to ensure rows sum to 1
        self.trans_mat /= self.trans_mat.sum(axis=1, keepdims=True)

        #Update means and covariances 
        for state in range(self.n_states):
            gamma_sum = np.sum(gamma[:, state]) + 1e-10

            # Weighted mean
            self.means[state] = np.sum(gamma[:, state][:, None] * X, axis=0) / gamma_sum

            #Weighted covariance
            diff = X - self.means[state]

            if self.covariance_type == 'full':
                self.covars[state] = np.dot((gamma[:, state][:, None] * diff).T, diff) / gamma_sum
            elif self.covariance_type == 'diag':
                self.covars_[state] = np.sum(gamma[:, state][:, None] * (diff ** 2), axis=0) / gamma_sum


        #Add regularization
        self._regularize_covariances()

    def compute_log_likelihood(self, log_alpha):
        """Compute log-likelihood of the observations."""
        return logsumexp(log_alpha[-1])

    def fit(self, X, n_restarts=1):
        """
        Train the HMM using the Baum-Welch (EM) algorithm.

        Multiple restarts help avoid local optima in the likelihood surface.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Contains training data such as returns and volatility
        
        Returns
        -------
        self : object
            Returns self for method chaining
        """
        X = np.atleast_2d(X)
    
        # Standardize features (do this once, before restarts)
        X_std = self.standardize_features(X, fit=True)
    
        # Track best model across restarts
        best_log_likelihood = -np.inf
        best_params = None

        for restart in range(n_restarts):
            #Change random seed for each restart
            np.random.seed(self.random_state + restart)

            #Initialize parameters randomly
            self.initialize_parameters(X_std)

            #Run Baum-Welch iterations for this restart
            prev_log_likelihood = -np.inf
            restart_history = []

            for i in range(self.n_iter):
                #E-step: Compute emission probabilities in log space
                log_emission_probs = self.compute_log_emission_probs(X_std)

                # E-step: Compute gamma and xi
                gamma, xi = self.compute_gamma_xi_log(log_emission_probs)

                #M-step: Update parameters
                self.update_parameters(X_std, gamma, xi)

                #Compute log-likelihood for convergence check
                log_alpha = self.forward_log(log_emission_probs)
                log_likelihood = self.compute_log_likelihood(log_alpha)
                restart_history.append(log_likelihood)

                #Check convergence
                improvement = log_likelihood - prev_log_likelihood

                if i > 0 and abs(improvement) < self.tol:
                    self.converged = True
                    break

                prev_log_likelihood = log_likelihood

            if not self.converged:
                print("Baum-Welch did not converge. Consider increasing n_iter")

            #Check if this restart produced a better model
            final_log_likelihood = restart_history[-1]

            if final_log_likelihood > best_log_likelihood:
                best_log_likelihood = final_log_likelihood
                # Save the best parameters
                best_params = {
                'transmat': self.trans_mat.copy(),
                'means': self.means.copy(),
                'covars': self.covars.copy(),
                'startprob': self.startprobs.copy(),
                'history': restart_history.copy(),
                'converged': self.converged
                }

        # Restore the best parameters from all restarts
        if best_params is not None:
            self.trans_mat = best_params['transmat']
            self.means = best_params['means']
            self.covars = best_params['covars']
            self.startprobs = best_params['startprob']
            self.log_likelihood_history_ = best_params['history']
            self.converged = best_params['converged']

        return self
    

    def predict(self, X):
        """
        Predict the most likely state sequence using Viterbi algorithm.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Observations to decode
        
        Returns
        -------
        states : array, shape (n_samples,)
            Most likely state sequence
        """
        X = np.atleast_2d(X)
        X_std = self.standardize_features(X, fit=False)
        n_samples = X_std.shape[0]

        #Compute log emission probabilities
        log_emission_probs = self.compute_log_emission_probs(X_std)
        log_trans_mat = np.log(self.trans_mat)
        log_startprobs = np.log(self.startprobs)

        #Viterbi algorithm
        #delta[t, i] = max probability of any path ending in state i at time t
        log_delta = np.zeros((n_samples, self.n_states))
        
        #psi[t, i] = most likely previous state leading to state i at time t
        psi = np.zeros((n_samples, self.n_states), dtype=int)

        #Initialization
        log_delta[0] = log_startprobs + log_emission_probs[:, 0]

        #Recursion
        for t in range(1, n_samples):
            for j in range(self.n_states):
                temp = log_delta[t-1] + log_trans_mat[:, j]
                psi[t, j] = np.argmax(temp)
                log_delta[t, j] = np.max(temp) * log_emission_probs[j, t]
        
        #Termination - backtrack to find most likely path
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(log_delta[-1])

        for t in range(n_samples - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states


    def predict_probabilities(self, X):
        """
        Predict state probabilities for each time step.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Observations
        
        Returns
        -------
        state_probs : array, shape (n_samples, n_states)
            State probabilities for each time step
        """

        X = np.atleast_2d(X)
        X_std = self._standardize_features(X, fit=False)
        log_emission_probs = self.compute_log_emission_probs(X_std)
        state_probs, _ = self.compute_gamma_xi_log(log_emission_probs)

        return state_probs
    
    def score(self, X):
        """
        Compute the log-likelihood of observations under the model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Observations
        
        Returns
        -------
        log_likelihood : float
            Log-likelihood of the observations
        """
        X = np.atleast_2d(X)
        X_std = self._standardize_features(X, fit=False)
        log_emission_probs = self.compute_emission_probs(X_std)
        log_alpha = self.forward(log_emission_probs)
        return self.compute_log_likelihood(log_alpha)




