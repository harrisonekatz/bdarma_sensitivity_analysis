# Load necessary libraries
library(MASS)       # For multivariate normal distribution
library(MCMCpack)   # For Dirichlet distribution
library(rstan)      # For Stan models
library(loo)        # For computing WAIC and LOO-CV
library(posterior)  # For summarizing posterior samples
library(dplyr)      # For data manipulation
library(tidyr)      # For data reshaping
library(purrr)
# Set seed for reproducibility
set.seed(123)

# Number of simulations
num_simulations <- 50

load(file = "coefficient_matrices.RData")

# Number of time points and components
T_total <- 100  # Total time points
T_train <- 80  # Training time points
T_test <- T_total - T_train  # Test time points
C <- 6         # Number of categories/components
T_test
# Define the priors to be used
priors <- c("informative", "horseshoe", "laplace", "spike_slab", "hierarchical")

#priors <- c("hierarchical")

# Initialize lists to store results
forecast_results <- list()
coverage_results <- list()
parameter_results <- list()

# Define the alr and inverse alr functions
alr <- function(x) {
  log(x[-length(x)] / x[length(x)])
}

inv_alr <- function(eta) {
  exp_eta <- exp(eta)
  denom <- 1 + sum(exp_eta)
  x <- c(exp_eta / denom, 1 / denom)
  return(x)
}

load("coefficient_matrices.RData")

# Function to generate data
generate_data <- function(T, C) {
  # Initialize matrices to store simulated data
  y <- matrix(NA, nrow = T_total, ncol = C)
  eta <- matrix(NA, nrow = T_total, ncol = C - 1)

  # Define AR and MA coefficient matrices
  P <- 2  # Order of AR
  Q <- 1  # Order of MA



  B1
  A1
  A2
  A3 <- matrix(data=rep(0,25),ncol=5)
  A4 <- matrix(data=rep(0,25),ncol=5)
  B2 <- matrix(data=rep(0,25),ncol=5)

  # Covariate coefficients (beta)
  beta <- c(0.1, -0.05, 0.03, -0.02, 0.04)  # Length C - 1
  beta
  # Create covariate vector X_t (e.g., sine wave)
  X_t <- matrix(data=c(rep(1,5*T_total)),ncol=5) # Period of 50

  # Set scale parameter phi
  phi <- rep(500, T_total)

  # Initial values for y
  y[1, ] <- rdirichlet(1, rep(1, C))
  y[2, ] <- rdirichlet(1, rep(1, C))

  # Compute initial eta values
  eta[1, ] <- alr(y[1, ])
  eta[2, ] <- alr(y[2, ])

  # Simulate data
  for (t in 3:T_total) {
    covariate_effect <- beta * X_t[t]  # beta is length C - 1
    covariate_effect_lag1 <- beta * X_t[t - 1]
    covariate_effect_lag2 <- beta * X_t[t - 2]

    # Compute AR terms
    AR_term <- A1 %*% (alr(y[t - 1, ]) - covariate_effect_lag1)+A2 %*% (alr(y[t - 2, ]) - covariate_effect_lag2)


    # Compute MA term
    MA_term <- B1 %*% ( (alr(y[t - 1, ]) )-eta[t - 1, ])

    # Update eta_t
    eta[t, ] <- AR_term + MA_term + covariate_effect

    # Map eta_t back to composition y_t
    mu_t <- inv_alr(eta[t, ])

    # Generate y_t from Dirichlet distribution
    y[t, ] <- rdirichlet(1, phi[t] * mu_t)
  }

  # Return the data and true parameters
  return(list(y = y, beta = beta, A1 = A1, A2 = A2, B1 = B1,A3=A3,A4=A4,B2=B2))
}

# Function to fit the model
fit_model <- function(y_train, prior_type, stan_model_file, y_test, true_params, X_t_train, X_t_test) {
  # Define the reference category for ALR transformation
  ref_cat <- ncol(y_train)


  # Number of components
  C <- ncol(y_train)

  # Order of AR and MA terms
  P <- 4
  Q <- 2

  # Covariate design
  C <- 6          # Number of categories/components
  N <- C - 1
  K <- rep(N, C - 1)  # Number of covariates per component (K is vector of ones)

  # Prepare covariate matrices as lists of vectors

  # Create X_train and X_test
  X_train <- purrr::map(
    seq_len(T_train),
    ~ rep(X_t_train[.x], N)  # Each element is a vector of length N
  )

  X_test <- purrr::map(
    seq_len(T_test),
    ~ rep(X_t_test[.x], N)
  )

  str(X_train)
  str(X_test)



  print(N)

  # Phi covariates (intercept only)
  K_phi <- 1
  X_phi_train <- matrix(1, nrow = T_train, ncol = K_phi)
  X_phi_test <- matrix(1, nrow = T_test, ncol = K_phi)

  # Convert y_train to a list of simplex vectors
  Y_train <- lapply(1:T_train, function(i) y_train[i, ])

  length(X_train[[1]])
  purrr::map_int(X_train[[1]], length)

  stan_data <- list(
    T = T_train,
    C = C,
    Y = Y_train,            # List of simplex vectors
    ref = ref_cat,
    N = length(X_train[[1]]),                  # N = C - 1 = 5
    K = purrr::map_int(X_train[[1]], length),          # K is a vector of ones of length N
    X = X_train,            # List of vectors of length N
    K_phi = K_phi,
    X_phi = X_phi_train,
    P = P,
    Q = Q,
    T_new = T_test,
    X_new = X_test,         # List of vectors of length N
    X_phi_new = X_phi_test,
    prior_only = 0,
    beta_sd = 1       # Include beta_sd if required by the prior
  )

  # Include additional data for specific priors
  if (prior_type == "noninformative" || prior_type == "informative") {
    if (prior_type == "noninformative") {
      beta_sd <- 1
    } else {
      beta_sd <- 0.1
    }
    stan_data$beta_sd <- beta_sd
  }

  if (prior_type == "laplace") {
    stan_data$b_beta <- 1.0
    stan_data$b_A <- 1.0
    stan_data$b_B <- 1.0
  }


  str(stan_data)
  # Fit the model using Stan

  mod <- cmdstanr::cmdstan_model(stan_model_file)



  fit_bdarma <- mod$sample(
    data = stan_data,
    seed = 1234,
    init = .25,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 500,
    iter_sampling = 750,
    max_treedepth = 11,
    adapt_delta = .85
  )


  # Extract posterior samples
  posterior_df <- fit_bdarma$draws(format = "df")


  ds_var <- "t"
  compositional_var <- "component"
  beta_draws <- posterior_df %>%
    select(matches("^beta")) %>%                # Select all columns starting with 'beta'
    select(-matches("^beta_"))

  # Refine B_draws to only include 'B' without including 'beta'
  B_draws <- posterior_df %>%
    select(matches("^B")) %>%                   # Select all columns starting with 'B'
    select(-matches("^beta"))

  A_draws  <- posterior_df %>%
    select(matches("^A")) %>%  select(-matches("^alpha"))


  #B_draws  <- posterior_df %>% dplyr::select(starts_with("B"))
  Y_hat <- fit_bdarma$draws("Y_hat", format = "draws_df")


  Y_hat_long_raw_bdarma <-
    Y_hat |>
    dplyr::select(.draw, starts_with("Y_hat")) |>
    tidyr::pivot_longer(starts_with("Y_hat"))

  Y_hat_long_bdarma <-
    Y_hat_long_raw_bdarma |>
    dplyr::summarize(
      mean = mean(value, na.rm = TRUE),
      qlo = quantile(value, probs = 0.05, na.rm = TRUE, names = FALSE),
      qhi = quantile(value, probs = 0.95, na.rm = TRUE, names = FALSE),
      .by = name
    ) |>
    dplyr::mutate(idx = stringr::str_extract(name, "(?<=\\[).+?(?=\\])")) |>
    tidyr::separate(idx, c("t", compositional_var), sep = ",") |>
    dplyr::select(all_of(c(ds_var, compositional_var)), mean, qlo, qhi)

  Y_hat_long_bdarma_sorted <- Y_hat_long_bdarma %>%
    mutate(t = as.numeric(t)) %>%  # Convert t to numeric for correct ordering
    arrange(t)

  Y_hat_wide <- Y_hat_long_bdarma_sorted %>%
    select(t, component, mean) %>%  # Keep only the relevant columns
    pivot_wider(names_from = component, values_from = mean) %>%  # Pivot to wide format
    arrange(t) %>%  # Ensure rows are ordered by time step
    select(-t)  # Remove the 't' column for matrix format matching

  # Convert to a matrix if desired
  y_test_compatible <- as.matrix(Y_hat_wide)

  # Compute point forecasts (posterior mean)
  forecast_mean <-y_test_compatible

  # Compute forecast performance metrics
  rmse_components <- sqrt(colMeans((forecast_mean - y_test)^2))
  overall_rmse <- sqrt(mean((forecast_mean - y_test)^2))

  # Compute coverage probabilities for parameters
  # For beta
  beta_samples <-beta_draws  # Dimensions: iterations x C - 1
  beta_lower <- apply(beta_samples, 2, quantile, probs = 0.025)
  beta_upper <- apply(beta_samples, 2, quantile, probs = 0.975)
  beta_true <- true_params$beta
  coverage_beta <- (beta_true >= beta_lower) & (beta_true <= beta_upper)

  # For A1
  head(A_draws)

  A1_samples <- A_draws %>%
    select(matches("^A\\[1"))  # Extracts columns starting with 'A[1,1,' to capture only A1 elements across different iterations.

  # Print the result to verify
  head(A1_samples)

  A1_lower <- apply(A1_samples, 2, quantile, probs = 0.025)
  A1_upper <- apply(A1_samples, 2, quantile, probs = 0.975)
  A1_true <- true_params$A1
  coverage_A1 <- (A1_true >= A1_lower) & (A1_true <= A1_upper)

  # For A2
  colnames(A_draws)
  A2_samples <- A_draws %>%
    select(matches("^A\\[2"))


  A2_lower <- apply(A2_samples, 2, quantile, probs = 0.025)
  A2_upper <- apply(A2_samples, 2, quantile, probs = 0.975)
  A2_true <- true_params$A2
  coverage_A2 <- (A2_true >= A2_lower) & (A2_true <= A2_upper)

  # For B1


  B1_samples <- B_draws %>%
    select(matches("^B\\[1"))# B[1] corresponds to B1



  B1_lower <- apply(B1_samples, 2, quantile, probs = 0.025)
  B1_upper <- apply(B1_samples,2, quantile, probs = 0.975)
  B1_true <- true_params$B1
  coverage_B1 <- (B1_true >= B1_lower) & (B1_true <= B1_upper)

  # In the fit_model function, after extracting posterior samples

  # For A3
  A3_samples <- A_draws %>%
    select(matches("^A\\[3,")) %>%
    as.matrix()
  A3_lower <- apply(A3_samples, 2, quantile, probs = 0.025)
  A3_upper <- apply(A3_samples, 2, quantile, probs = 0.975)
  A3_true <- as.vector(true_params$A3)
  coverage_A3 <- (A3_true >= A3_lower) & (A3_true <= A3_upper)

  # For A4
  A4_samples <- A_draws %>%
    select(matches("^A\\[4,")) %>%
    as.matrix()
  A4_lower <- apply(A4_samples, 2, quantile, probs = 0.025)
  A4_upper <- apply(A4_samples, 2, quantile, probs = 0.975)
  A4_true <- as.vector(true_params$A4)
  coverage_A4 <- (A4_true >= A4_lower) & (A4_true <= A4_upper)

  # For B2
  B2_samples <- B_draws %>%
    select(matches("^B\\[2,")) %>%
    as.matrix()
  B2_lower <- apply(B2_samples, 2, quantile, probs = 0.025)
  B2_upper <- apply(B2_samples, 2, quantile, probs = 0.975)
  B2_true <- as.vector(true_params$B2)
  coverage_B2 <- (B2_true >= B2_lower) & (B2_true <= B2_upper)




  coverage <- list(
    beta = mean(coverage_beta),
    A1 = mean(coverage_A1),
    A2 = mean(coverage_A2),
    A3 = mean(coverage_A3),
    A4 = mean(coverage_A4),
    B1 = mean(coverage_B1),
    B2 = mean(coverage_B2)
  )

  # Calculate coverage probabilities
  beta_estimates <- colMeans(beta_draws)
  A1_estimates <- colMeans(A1_samples)
  A2_estimates <- colMeans(A2_samples)
  A3_estimates <- colMeans(A3_samples)
  A4_estimates <- colMeans(A4_samples)


  B1_estimates <- colMeans(B1_samples)
  B2_estimates <- colMeans(B2_samples)


  ### bias time

  bias_beta <- beta_estimates - true_params$beta
  bias_A1 <- A1_estimates - as.vector(true_params$A1)
  bias_A2 <- A2_estimates - as.vector(true_params$A2)
  bias_A3 <- A3_estimates - as.vector(true_params$A3)
  bias_A4 <- A4_estimates - as.vector(true_params$A4)
  bias_B1 <- B1_estimates - as.vector(true_params$B1)
  bias_B2 <- B2_estimates - as.vector(true_params$B2)


  # Compute RMSE for parameters
  rmse_beta <- sqrt(mean(bias_beta^2))
  rmse_A1 <- sqrt(mean(bias_A1^2))
  rmse_A2 <- sqrt(mean(bias_A2^2))
  rmse_A3 <- sqrt(mean(bias_A3^2))
  rmse_A4 <- sqrt(mean(bias_A4^2))
  rmse_B1 <- sqrt(mean(bias_B1^2))
  rmse_B2 <- sqrt(mean(bias_B2^2))




  ci_length_beta <- beta_upper - beta_lower
  ci_length_A1 <- A1_upper - A1_lower
  ci_length_A2 <- A2_upper - A2_lower
  ci_length_A3 <- A3_upper - A3_lower
  ci_length_A4 <- A4_upper - A4_lower
  ci_length_B1 <- B1_upper - B1_lower
  ci_length_B2 <- B2_upper - B2_lower



  # Return results
  return(list(
    forecast_metrics = list(
      rmse_components = rmse_components,
      overall_rmse = overall_rmse
    ),
    coverage = list(
      beta = coverage_beta,
      A1 = coverage_A1,
      A2 = coverage_A2,
      A3 = coverage_A3,
      A4 = coverage_A4,
      B1 = coverage_B1,
      B2 = coverage_B2
    ),
    parameter_metrics = list(
      bias_beta = bias_beta,
      bias_A1 = bias_A1,
      bias_A2= bias_A2,
      bias_A3= bias_A3,
      bias_A4= bias_A4,
      bias_B1= bias_B1,
      bias_B2= bias_B2,
      rmse_beta = rmse_beta,
      rmse_A1 = rmse_A1,
      rmse_A2 = rmse_A2,
      rmse_A3 = rmse_A3,
      rmse_A4 = rmse_A4,
      rmse_B1 = rmse_B1,
      rmse_B2 = rmse_B2,
      ci_length_beta = ci_length_beta,
      ci_length_A1 = ci_length_A1,
      ci_length_A2 = ci_length_A2,
      ci_length_A3 = ci_length_A3,
      ci_length_A4 = ci_length_A4,
      ci_length_B1 = ci_length_B1,
      ci_length_B2 = ci_length_B2
    )
  ))
}

# Main simulation loop
for (sim in 1:num_simulations) {
  cat("Running simulation", sim, "\n")

  # Generate data
  data_gen <- generate_data(T = T_total, C = C)
  y <- data_gen$y
  true_params <- list(beta = data_gen$beta, A1 = data_gen$A1, A2 = data_gen$A2, B1 = data_gen$B1,B2=data_gen$B2,A3=data_gen$A3,A4=data_gen$A4)

  # Covariate data
  X_t <- matrix(data=c(rep(1,5*T_total)),ncol=5) # Period of 50  # Period of 50

  # Split data into training and test sets
  y_train <- y[1:T_train, ]
  y_test <- y[(T_train + 1):T_total, ]

  X_t_train <- X_t[1:T_train]
  X_t_test <- X_t[(T_train + 1):T_total]

  forecast_results[[sim]] <- list()
  coverage_results[[sim]] <- list()
  parameter_results[[sim]] <- list()

  # Loop over priors
  for (prior in priors) {
    cat("  Fitting model with prior:", prior, "\n")

    # Specify the Stan model file for the prior
    stan_model_file <- paste0("stan/bdarma_", prior, ".stan")

    # Check if the Stan model file exists
    if (!file.exists(stan_model_file)) {
      stop(paste("Stan model file not found:", stan_model_file))
    }

    # Fit the model
    results <- fit_model(
      y_train = y_train,
      prior_type = prior,
      stan_model_file = stan_model_file,
      y_test = y_test,
      true_params = true_params,
      X_t_train = X_t_train,
      X_t_test = X_t_test
    )

    # Store the results
    forecast_results[[sim]][[prior]] <- results$forecast_metrics
    coverage_results[[sim]][[prior]] <- results$coverage
    parameter_results[[sim]][[prior]] <- results$parameter_metrics
  }
  print(sim)
}
sim


# After simulations, aggregate results
# Initialize data frames to store aggregated results
forecast_summary <- data.frame()
coverage_summary <- data.frame()

parameter_summary <- data.frame()

# Loop over priors to aggregate results
for (prior in priors) {


  # Extract RMSE across simulations
  rmse_list <- lapply(forecast_results, function(res) res[[prior]]$overall_rmse)
  rmse_values <- unlist(rmse_list)

  # Compute mean and standard deviation of RMSE
  mean_rmse <- mean(rmse_values, na.rm = TRUE)
  sd_rmse <- sd(rmse_values, na.rm = TRUE)

  # Store in forecast summary
  forecast_summary <- rbind(forecast_summary, data.frame(
    Prior = prior,
    Mean_RMSE = mean_rmse,
    SD_RMSE = sd_rmse
  ))

  # Extract coverage probabilities across simulations
  coverage_list <- lapply(coverage_results, function(res) res[[prior]])
  # coverage_df <- do.call(rbind, lapply(coverage_list, function(cov) {
  #   data.frame(beta = cov$beta, A1 = cov$A1, A2 = cov$A2,A3 = cov$A3,A4 = cov$A4, B1 = cov$B1,B2 = cov$B2)
  # }))

  coverage_df <- do.call(rbind, lapply(coverage_list, function(cov) {
    # Flatten matrices and vectors
    beta_vec <- as.vector(cov$beta)
    A1_vec <- as.vector(cov$A1)
    A2_vec <- as.vector(cov$A2)
    A3_vec <- as.vector(cov$A3)
    A4_vec <- as.vector(cov$A4)
    B1_vec <- as.vector(cov$B1)
    B2_vec <- as.vector(cov$B2)

    # Find the maximum length
    max_length <- max(length(beta_vec), length(A1_vec), length(A2_vec),
                      length(A3_vec), length(A4_vec), length(B1_vec), length(B2_vec))

    # Ensure all vectors are of the same length by padding with NAs if necessary
    pad_vector <- function(x, length_out) {
      length(x) <- length_out
      return(x)
    }

    beta_vec <- pad_vector(beta_vec, max_length)
    A1_vec <- pad_vector(A1_vec, max_length)
    A2_vec <- pad_vector(A2_vec, max_length)
    A3_vec <- pad_vector(A3_vec, max_length)
    A4_vec <- pad_vector(A4_vec, max_length)
    B1_vec <- pad_vector(B1_vec, max_length)
    B2_vec <- pad_vector(B2_vec, max_length)

    # Create the data frame
    data.frame(beta = beta_vec, A1 = A1_vec, A2 = A2_vec,
               A3 = A3_vec, A4 = A4_vec, B1 = B1_vec, B2 = B2_vec)
  }))


  # Compute mean coverage for each parameter group
  mean_coverage <- colMeans(coverage_df, na.rm = TRUE)

  coverage_summary <- rbind(coverage_summary, data.frame(
    Prior = prior,
    Beta_Coverage = unname(mean_coverage["beta"]),
    A1_Coverage = unname(mean_coverage["A1"]),
    A2_Coverage = unname(mean_coverage["A2"]),
    A3_Coverage = unname(mean_coverage["A3"]),
    A4_Coverage = unname(mean_coverage["A4"]),
    B1_Coverage = unname(mean_coverage["B1"]),
    B2_Coverage = unname(mean_coverage["B2"])
  ))





  parameters_list <- lapply(parameter_results, function(res) res[[prior]])


  mean_bias_beta <- colMeans(do.call(rbind, lapply(parameters_list, function(x) x$bias_beta)))

  # Calculate mean bias_A1 across all simulations
  mean_bias_A1 <- colMeans(do.call(rbind, lapply(parameters_list, function(x) x$bias_A1)))
  mean_bias_A2 <- colMeans(do.call(rbind, lapply(parameters_list, function(x) x$bias_A2)))
  mean_bias_A3 <- colMeans(do.call(rbind, lapply(parameters_list, function(x) x$bias_A3)))
  mean_bias_A4 <- colMeans(do.call(rbind, lapply(parameters_list, function(x) x$bias_A4)))
  mean_bias_B1 <- colMeans(do.call(rbind, lapply(parameters_list, function(x) x$bias_B1)))
  mean_bias_B2 <- colMeans(do.call(rbind, lapply(parameters_list, function(x) x$bias_B2)))

  # Calculate mean rmse_beta across all simulations

  rmse_beta_values <- map_dbl(parameters_list, ~ pluck(.x, "rmse_beta", .default = NA_real_))
  mean_rmse_beta <- mean(rmse_beta_values, na.rm = TRUE)

  parameters_list
  # Calculate mean rmse_A1 across all simulations

  mean_rmse_A1 <- map_dbl(parameters_list, ~ pluck(.x, "rmse_A1", .default = NA_real_))
  mean_rmse_A1 <- mean(mean_rmse_A1, na.rm = TRUE)

  mean_rmse_A2 <- map_dbl(parameters_list, ~ pluck(.x, "rmse_A2", .default = NA_real_))
  mean_rmse_A2 <- mean(mean_rmse_A2, na.rm = TRUE)

  mean_rmse_A3 <- map_dbl(parameters_list, ~ pluck(.x, "rmse_A3", .default = NA_real_))
  mean_rmse_A3 <- mean(mean_rmse_A3, na.rm = TRUE)

  mean_rmse_A4 <- map_dbl(parameters_list, ~ pluck(.x, "rmse_A4", .default = NA_real_))
  mean_rmse_A4 <- mean(mean_rmse_A4, na.rm = TRUE)

  mean_rmse_B1 <- map_dbl(parameters_list, ~ pluck(.x, "rmse_B1", .default = NA_real_))
  mean_rmse_B1 <- mean(mean_rmse_B1, na.rm = TRUE)


  mean_rmse_B2 <- map_dbl(parameters_list, ~ pluck(.x, "rmse_B2", .default = NA_real_))
  mean_rmse_B2 <- mean(mean_rmse_B2, na.rm = TRUE)

  mean_ci_length_beta <- colMeans(do.call(rbind, lapply(parameters_list, function(x) x$ci_length_beta)),na.rm = T)

  # Calculate mean ci_length_A1 across all simulations
  mean_ci_length_A1 <- colMeans(do.call(rbind, lapply(parameters_list, function(x) x$ci_length_A1)),na.rm = T)
  mean_ci_length_A2 <- colMeans(do.call(rbind, lapply(parameters_list, function(x) x$ci_length_A2)),na.rm = T)
  mean_ci_length_A3 <- colMeans(do.call(rbind, lapply(parameters_list, function(x) x$ci_length_A3)),na.rm = T)
  mean_ci_length_A4 <- colMeans(do.call(rbind, lapply(parameters_list, function(x) x$ci_length_A4)),na.rm = T)
  mean_ci_length_B1 <- colMeans(do.call(rbind, lapply(parameters_list, function(x) x$ci_length_B1)),na.rm = T)
  mean_ci_length_B2 <- colMeans(do.call(rbind, lapply(parameters_list, function(x) x$ci_length_B2)),na.rm = T)




  parameter_summary <- rbind(parameter_summary, data.frame(
    Prior = prior,
    Mean_Bias_Beta = mean(mean_bias_beta),
    Mean_Bias_A1 = mean(mean_bias_A1),
    Mean_Bias_A2 = mean(mean_bias_A2),
    Mean_Bias_A3 = mean(mean_bias_A3),
    Mean_Bias_A4 = mean(mean_bias_A4),
    Mean_Bias_B1 = mean(mean_bias_B1),
    Mean_Bias_B2 = mean(mean_bias_B2),
    Mean_RMSE_Beta = mean(mean_rmse_beta),
    Mean_RMSE_A1 = mean(mean_rmse_A1),
    Mean_RMSE_A2 = mean(mean_rmse_A2),
    Mean_RMSE_A3 = mean(mean_rmse_A3),
    Mean_RMSE_A4 = mean(mean_rmse_A4),
    Mean_RMSE_B1 = mean(mean_rmse_B1),
    Mean_RMSE_B2 = mean(mean_rmse_B2),
    Mean_CI_Length_Beta = mean(mean_ci_length_beta),
    Mean_CI_Length_A1 = mean(mean_ci_length_A1),
    Mean_CI_Length_A2 = mean(mean_ci_length_A2),
    Mean_CI_Length_A3 = mean(mean_ci_length_A3),
    Mean_CI_Length_A4 = mean(mean_ci_length_A4),
    Mean_CI_Length_B1 = mean(mean_ci_length_B1),
    Mean_CI_Length_B2 = mean(mean_ci_length_B2)

  ))

}



# Print forecast performance summary
cat("\nForecast Performance Summary:\n")
print(forecast_summary)

# Print coverage probability summary
cat("\nCoverage Probability Summary:\n")
print(coverage_summary)

print(parameter_summary)

