functions {
  // Include necessary functions
  #include alr.stan
  #include component_product.stan

  // Approximate Dirac delta function
  real dirac_delta(real x) {
    return normal_lpdf(x | 0, 1e-4);
  }
}

data {
  int<lower=1> T;
  int<lower=2> C;
  array[T] simplex[C] Y;
  int<lower=1, upper=C> ref;

  int<lower=0> N;
  array[C - 1] int<lower=0> K;
  array[T] vector[N] X;

  int<lower=1> K_phi;
  matrix[T, K_phi] X_phi;

  int<lower=0> P;
  int<lower=0> Q;

  int<lower=0> T_new;
  array[T_new] vector[N] X_new;
  matrix[T_new, K_phi] X_phi_new;

  int prior_only;
}

transformed data {
  int M = max(P, Q);
  array[T] vector[C - 1] alr_Y;

  for (t in 1:T) {
    alr_Y[t] = alr(Y[t], ref);
  }

  // Compute the total number of elements in A and B matrices
  int num_A = P * (C - 1) * (C - 1);
  int num_B = Q * (C - 1) * (C - 1);
}

parameters {
  vector[N] beta;                   // Coefficients for covariates

  array[P] matrix[C - 1, C - 1] A;  // VAR coefficients
  array[Q] matrix[C - 1, C - 1] B;  // VMA coefficients

  // Inclusion indicators
  vector<lower=0, upper=1>[N] theta_beta;

  vector<lower=0, upper=1>[num_A] theta_A;
  vector<lower=0, upper=1>[num_B] theta_B;

  vector[K_phi] beta_phi;
}

transformed parameters {
  // Compute phi
  vector[T] phi = X_phi * beta_phi;

  // Initialize transformed parameters
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
  // Hyperparameters for inclusion probabilities
  theta_beta ~ beta(1, 1);  // Uniform prior
  theta_A ~ beta(1, 1);
  theta_B ~ beta(1, 1);

  // Spike-and-slab prior for beta
  for (n in 1:N) {
    target += log_mix(theta_beta[n],
                      normal_lpdf(beta[n] | 0, 1),     // Slab component
                      dirac_delta(beta[n]));           // Spike at zero
  }

  // Spike-and-slab prior for A matrices
  {
    int idx = 1;
    for (p in 1:P) {
      for (i in 1:(C - 1)) {
        for (j in 1:(C - 1)) {
          target += log_mix(theta_A[idx],
                            normal_lpdf(A[p][i, j] | 0, 1),  // Slab
                            dirac_delta(A[p][i, j]));        // Spike
          idx += 1;
        }
      }
    }
  }

  // Spike-and-slab prior for B matrices
  {
    int idx = 1;
    for (q in 1:Q) {
      for (i in 1:(C - 1)) {
        for (j in 1:(C - 1)) {
          target += log_mix(theta_B[idx],
                            normal_lpdf(B[q][i, j] | 0, 1),  // Slab
                            dirac_delta(B[q][i, j]));        // Spike
          idx += 1;
        }
      }
    }
  }

  // Prior for beta_phi
  beta_phi[1] ~ normal(7, 1.5);
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
    // Simulate Y_hat_train[t] from the Dirichlet distribution
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
