functions {
  // Include necessary functions
  #include alr.stan
  #include component_product.stan
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

  // Hyperparameters for Laplace prior
  real<lower=0> b_beta;
  real<lower=0> b_A;
  real<lower=0> b_B;
}

transformed data {
  int M = max(P, Q);
  array[T] vector[C - 1] alr_Y;

  for (t in 1:T) {
    alr_Y[t] = alr(Y[t], ref);
  }

  int num_A = P * (C - 1) * (C - 1);
  int num_B = Q * (C - 1) * (C - 1);
}

parameters {
  vector[N] beta;
  array[P] matrix[C - 1, C - 1] A;
  array[Q] matrix[C - 1, C - 1] B;

  // Auxiliary variables for Laplace prior
  vector<lower=0>[N] lambda_beta;
  vector<lower=0>[num_A] lambda_A;
  vector<lower=0>[num_B] lambda_B;

  vector[K_phi] beta_phi;
}

transformed parameters {
  // Same as before
  vector[T] phi = X_phi * beta_phi;

  array[T] vector[C - 1] Xbeta = component_product(X, beta, K);
  array[T] vector[C - 1] eta;
  array[T] vector[C] alpha;

  // Initialize eta and alpha
  for (t in 1:M) {
    eta[t] = alr_Y[t];
    alpha[t] = exp(phi[t]) * alrinv(eta[t], ref);
  }

  // Compute eta and alpha
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
  // Hyperpriors for lambda (scale parameters)
  lambda_beta ~ exponential(1 / b_beta);
  lambda_A ~ exponential(1 / b_A);
  lambda_B ~ exponential(1 / b_B);

  // Laplace prior for beta
  beta ~ normal(0, lambda_beta);

  // Laplace prior for A matrices
  {
    int idx = 1;
    for (p in 1:P) {
      for (i in 1:(C - 1)) {
        for (j in 1:(C - 1)) {
          A[p][i, j] ~ normal(0, lambda_A[idx]);
          idx += 1;
        }
      }
    }
  }

  // Laplace prior for B matrices
  {
    int idx = 1;
    for (q in 1:Q) {
      for (i in 1:(C - 1)) {
        for (j in 1:(C - 1)) {
          B[q][i, j] ~ normal(0, lambda_B[idx]);
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
  array[T] simplex[C] Y_hat_train;  // Fitted values for training data
  array[T_new] simplex[C] Y_hat;    // Predictions for new data
  vector[T - M] log_lik;
  array[T] vector[C] alpha_hat;     // Expected values for training data

  // Generate fitted values for training data
  for (t in 1:T) {
    alpha_hat[t] = alpha[t];
    Y_hat_train[t] = dirichlet_rng(alpha[t]);
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

    // Initialize alr_Y_hat and eta_new
    alr_Y_hat = rep_array(rep_vector(0, C - 1), T_new);
    eta_new = rep_array(rep_vector(0, C - 1), T_new);

    // Store last P observed values from training data
    array[P] vector[C - 1] alr_Y_prev;
    array[P] vector[C - 1] eta_prev;

    for (p in 1:P) {
      alr_Y_prev[p] = alr_Y[T - P + p];
      eta_prev[p] = eta[T - P + p];
    }

    // Precompute starting index for Xbeta
    int start_idx_Xbeta = T - P;

    for (t in 1:T_new) {
      vector[C - 1] ar = rep_vector(0, C - 1);
      vector[C - 1] ma = rep_vector(0, C - 1);

      // AR terms
      for (p in 1:P) {
        vector[C - 1] alr_Y_lag;
        vector[C - 1] Xbeta_lag;
        int idx_ar;

        if (t - p <= 0) {
          idx_ar = P + t - p;
          alr_Y_lag = alr_Y_prev[idx_ar];
          Xbeta_lag = Xbeta[start_idx_Xbeta + t - p];
        } else {
          idx_ar = t - p;
          alr_Y_lag = alr_Y_hat[idx_ar];
          Xbeta_lag = Xbeta_new[idx_ar];
        }

        ar += A[p] * (alr_Y_lag - Xbeta_lag);
      }

      // MA terms
      for (q in 1:Q) {
        vector[C - 1] alr_Y_lag;
        vector[C - 1] eta_lag;
        int idx_ma;

        if (t - q <= 0) {
          idx_ma = P + t - q;
          alr_Y_lag = alr_Y_prev[idx_ma];
          eta_lag = eta_prev[idx_ma];
        } else {
          idx_ma = t - q;
          alr_Y_lag = alr_Y_hat[idx_ma];
          eta_lag = eta_new[idx_ma];
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
