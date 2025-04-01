functions {
  // Include necessary functions
  #include alr.stan
  #include component_product.stan
}

data {
  int<lower=1> T;                       // Number of time periods
  int<lower=2> C;                       // Number of categories
  array[T] simplex[C] Y;                // Response array
  int<lower=1, upper=C> ref;            // ALR reference element of simplex

  int<lower=0> N;                       // Total number of covariates
  array[C - 1] int<lower=0> K;          // Number of covariates by component
  array[T] vector[N] X;                 // Covariates across all components

  int<lower=1> K_phi;                   // Number of population effects for phi
  matrix[T, K_phi] X_phi;               // Design matrix for phi

  int<lower=0> P;                       // Number of auto-regressive lags
  int<lower=0> Q;                       // Number of moving average lags

  int<lower=0> T_new;                   // Number of time periods to forecast
  array[T_new] vector[N] X_new;         // New covariates across all components
  matrix[T_new, K_phi] X_phi_new;       // Design matrix for forecasting phi

  int prior_only;                       // Whether to ignore the likelihood (1)
}

transformed data {
  int M = max(P, Q);
  array[T] vector[C - 1] alr_Y;

  for (t in 1:T) {
    alr_Y[t] = alr(Y[t], ref);
  }

  // Compute the total number of elements in A and B matrices (including diagonals)
  int num_A = P * (C - 1) * (C - 1);  // Including diagonals
  int num_B = Q * (C - 1) * (C - 1);  // Including diagonals
}

parameters {
  vector[N] beta;                   // Coefficients for covariates

  // VAR coefficients with horseshoe priors
  array[P] matrix[C - 1, C - 1] A;  // VAR coefficients
  array[Q] matrix[C - 1, C - 1] B;  // VMA coefficients

  // Horseshoe prior parameters
  real<lower=0> tau;               // Global shrinkage parameter for beta
  vector<lower=0>[N] lambda_beta;       // Local shrinkage parameters for beta

  real<lower=0> tau_A;                  // Global shrinkage parameter for A
  vector<lower=0>[num_A] lambda_A;      // Local shrinkage parameters for A

  real<lower=0> tau_B;                  // Global shrinkage parameter for B
  vector<lower=0>[num_B] lambda_B;      // Local shrinkage parameters for B

  vector[K_phi] beta_phi;               // Coefficients for phi covariates
}

transformed parameters {
  vector[T] phi = X_phi * beta_phi;

  array[T] vector[C - 1] Xbeta = component_product(X, beta, K);
  array[T] vector[C - 1] eta;
  array[T] vector[C] alpha;

  // Initialize eta and alpha for t = 1 to M
  for (t in 1:M) {
    eta[t] = alr_Y[t];
    alpha[t] = exp(phi[t]) * alrinv(eta[t], ref);
  }

  // Compute eta and alpha for t = M+1 to T
  for (t in (M + 1):T) {
    vector[C - 1] ar = rep_vector(0, C - 1);
    vector[C - 1] ma = rep_vector(0, C - 1);

    for (p in 1:P) {
      ar += A[p] * (alr_Y[t - p] - Xbeta[t - p]);
    }

    for (q in 1:Q) {
      ma += B[q] * (alr_Y[t - q] - eta[t - q]);
    }

    eta[t] = ar + ma + Xbeta[t];
    alpha[t] = exp(phi[t]) * alrinv(eta[t], ref);
  }
}

model {
  // Horseshoe prior hyperparameters
  tau ~ cauchy(0, 1);
  lambda_beta ~ cauchy(0, 1);

  tau_A ~ cauchy(0, 1);
  lambda_A ~ cauchy(0, 1);

  tau_B ~ cauchy(0, 1);
  lambda_B ~ cauchy(0, 1);

  // Prior for beta with horseshoe prior
  beta ~ normal(0, tau * lambda_beta);

  // Prior for A matrices with horseshoe prior
  {
    int idx = 1;
    for (p in 1:P) {
      for (i in 1:(C - 1)) {
        for (j in 1:(C - 1)) {
          A[p][i, j] ~ normal(0, tau_A * lambda_A[idx]);
          idx += 1;
        }
      }
    }
  }

  // Prior for B matrices with horseshoe prior
  {
    int idx = 1;
    for (q in 1:Q) {
      for (i in 1:(C - 1)) {
        for (j in 1:(C - 1)) {
          B[q][i, j] ~ normal(0, tau_B * lambda_B[idx]);
          idx += 1;
        }
      }
    }
  }

  // Prior for beta_phi
  beta_phi[1] ~ normal(7, 1.5);  // Intercept term
  if (K_phi > 1) {
    beta_phi[2:K_phi] ~ normal(0, 0.1);
  }

  // Likelihood
  if (!prior_only) {
    for (t in (M + 1):T) {
      Y[t] ~ dirichlet(alpha[t]);
    }
  }
}

generated quantities {
  array[T_new] simplex[C] Y_hat;    // Predictions for new data
  vector[T - M] log_lik;
  array[T] vector[C] alpha_hat;    // Expected values for training data

  // Generate fitted values for training data
  for (t in 1:T) {
    alpha_hat[t] = alpha[t];
  }

  // Log-likelihood contributions
  for (t in (M + 1):T) {
    log_lik[t - M] = dirichlet_lpdf(Y[t] | alpha[t]);
  }

  // Predictions for out-of-sample data
  if (T_new > 0) {
    array[T_new] vector[C - 1] Xbeta_new = component_product(X_new, beta, K);
    array[T_new] vector[C - 1] alr_Y_hat;
    array[T_new] vector[C - 1] eta_new;
    vector[T_new] phi_new = X_phi_new * beta_phi;
    array[T_new] vector[C] alpha_new;

    for (t in 1:T_new) {
      vector[C - 1] ar = rep_vector(0, C - 1);
      vector[C - 1] ma = rep_vector(0, C - 1);

      for (p in 1:P) {
        vector[C - 1] alr_Y_lag;
        vector[C - 1] Xbeta_lag;

        if (t < p + 1) {
          alr_Y_lag = alr_Y[T + t - p];
          Xbeta_lag = Xbeta[T + t - p];
        } else {
          alr_Y_lag = alr_Y_hat[t - p];
          Xbeta_lag = Xbeta_new[t - p];
        }

        ar += A[p] * (alr_Y_lag - Xbeta_lag);
      }

      for (q in 1:Q) {
        vector[C - 1] alr_Y_lag;
        vector[C - 1] eta_lag;

        if (t < q + 1) {
          alr_Y_lag = alr_Y[T + t - q];
          eta_lag = eta[T + t - q];
        } else {
          alr_Y_lag = alr_Y_hat[t - q];
          eta_lag = eta_new[t - q];
        }

        ma += B[q] * (alr_Y_lag - eta_lag);
      }

      eta_new[t] = ar + ma + Xbeta_new[t];
      alpha_new[t] = exp(phi_new[t]) * alrinv(eta_new[t], ref);

      Y_hat[t] = dirichlet_rng(alpha_new[t]);
      alr_Y_hat[t] = alr(Y_hat[t], ref);
    }
  }
}
