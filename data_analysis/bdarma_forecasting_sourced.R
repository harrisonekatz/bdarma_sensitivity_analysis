################################################################################
# bdarma_forecasting.R
# Full script to run multiple priors for a BD-ARMA model, saving partial results
################################################################################

# ==============================================================================
# 1) Load all required packages
# ==============================================================================
library(checkmate)
install.packages("corrplot")

library(corrplot)
library(DBI)
library(dbplyr)
library(dplyr)
library(forcats)
library(ggplot2)
library(here)
library(lubridate)
library(readr)
library(rlang)
library(RPresto)
library(scales)
library(tidyr)
library(purrr)
Sys.setenv(GITHUB_PAT="")
# we recommend running this is a fresh R session or restarting your current session
install.packages("cmdstanr", repos = c('https://stan-dev.r-universe.dev', getOption("repos")))
library(cmdstanr)  # Make sure cmdstanr is installed and properly set up
install_cmdstan(cores = 2)
# ==============================================================================
# 2) Source helper scripts (adjust file paths if needed)
# ==============================================================================
proj_dir <- here::here()


# ==============================================================================
# 3) Identify user (using some function defined in cli.R, or just skip)
# ==============================================================================

# ==============================================================================
# 4) Define the priors and other loop settings
# ==============================================================================
priors <- c("informative", "horseshoe", "laplace", "spike_slab", "hierarchical")
start_idx <- 1
end_idx   <- length(priors)

# ==============================================================================
# 5) Create directory for partial results (if it doesn’t exist)
# ==============================================================================
save_dir <- file.path(proj_dir, "saved_output")
if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)

# ==============================================================================
# 6) Load & prepare data that do NOT depend on the prior
#    (We do this once, outside the loop, to avoid duplication)
# ==============================================================================
message("Reading 'sp500_stocks_w_sector.csv'...")
stocks <- readr::read_csv("~/monthly_production/sp500_stocks_w_sector.csv")

# Example test subset for quick debugging (optional):
# stocks <- stocks %>% filter(Date > "2022-01-01")

################################################################################
# A) CREATE A LIST OF TRADING DAYS, REMOVE WEEKENDS & HOLIDAYS
################################################################################
all_days <- seq.Date(
  from = as.Date("2021-01-01"),
  to   = as.Date("2023-12-31"),
  by   = "day"
)

holidays <- c(
  "2015-01-01", "2015-01-19", "2015-02-16", "2015-04-03", "2015-05-25", 
  "2015-07-03", "2015-09-07", "2015-11-26", "2015-12-25",
  "2016-01-01", "2016-01-18", "2016-02-15", "2016-03-25", "2016-05-30", 
  "2016-07-04", "2016-09-05", "2016-11-24", "2016-12-26",
  "2017-01-02", "2017-01-16", "2017-02-20", "2017-04-14", "2017-05-29", 
  "2017-07-04", "2017-09-04", "2017-11-23", "2017-12-25",
  "2018-01-01", "2018-01-15", "2018-02-19", "2018-03-30", "2018-05-28", 
  "2018-07-04", "2018-09-03", "2018-11-22", "2018-12-25",
  "2019-01-01", "2019-01-21", "2019-02-18", "2019-04-19", "2019-05-27", 
  "2019-07-04", "2019-09-02", "2019-11-28", "2019-12-25",
  "2020-01-01", "2020-01-20", "2020-02-17", "2020-04-10", "2020-05-25", 
  "2020-07-03", "2020-09-07", "2020-11-26", "2020-12-25",
  "2021-01-01", "2021-01-18", "2021-02-15", "2021-04-02", "2021-05-31", 
  "2021-07-05", "2021-09-06", "2021-11-25", "2021-12-24",
  "2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30", "2022-06-20", 
  "2022-07-04", "2022-09-05", "2022-11-24", "2022-12-26",
  "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29", 
  "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25",
  "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", 
  "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25"
)

trading_days <- all_days[
  !weekdays(all_days) %in% c("Saturday", "Sunday") & 
    !all_days %in% holidays
]

################################################################################
# B) SUBSET AND PREPARE THE STOCKS DATA
################################################################################
# In your example, you also do:
#    sp_test <- stocks %>% filter(Date >= "2023-07-01", Date < "2024-01-01") ...
#    stocks <- stocks %>% filter(Date > "2020-12-31")
#
# The logic below follows your original snippet but can be adapted.

sp_test <- stocks %>%
  filter(Date >= as.Date("2023-07-01"), Date < as.Date("2024-01-01")) %>%
  mutate(forecast = "actuals")

stocks <- stocks %>%
  filter(Date > as.Date("2020-12-31"))

################################################################################
# C) BUILD A CALENDAR OF TRADING DAYS, ADD FOURIER TERMS
################################################################################
trading_calendar <- data.frame(date = trading_days) %>%
  mutate(
    # Monday=1, Tuesday=2, ..., Sunday=7
    day_of_week = wday(date, week_start = 1),
    year        = lubridate::year(date)
  ) %>%
  group_by(year) %>%
  mutate(
    trading_day_of_year = row_number()
  ) %>%
  ungroup()

# If you want a 5-day cycle (Mon-Fri), keep the divisor = 5
trading_calendar <- trading_calendar %>%
  mutate(
    sin_week_k1 = sin(2 * pi * day_of_week / 5),
    cos_week_k1 = cos(2 * pi * day_of_week / 5),
    sin_week_k2 = sin(2 * pi * 2 * day_of_week / 5),
    cos_week_k2 = cos(2 * pi * 2 * day_of_week / 5)
  )

# Increase yearly seasonal terms (K=3 means up to 3 harmonics)
K_year <- 5
for(k in seq_len(K_year)) {
  trading_calendar[[paste0("sin_year_k", k)]] <-
    sin(2 * pi * k * trading_calendar$trading_day_of_year / 252)
  trading_calendar[[paste0("cos_year_k", k)]] <-
    cos(2 * pi * k * trading_calendar$trading_day_of_year / 252)
}

################################################################################
# D) SPLIT CALENDAR INTO TRAINING & TESTING
################################################################################
trading_calendar_training <- trading_calendar %>%
  filter(date >= "2019-01-01", date < as.Date("2024-01-01"))

trading_calendar_testing <- trading_calendar %>%
  filter(date >= "2023-07-01", date < as.Date("2024-01-01"))

################################################################################
# E) JOIN FOURIER TERMS BACK ONTO YOUR STOCKS DATA
################################################################################
stocks_with_fourier <- stocks %>%
  left_join(trading_calendar, by = c("Date" = "date"))

# Create wide version by Sector
stocks_wide <- stocks_with_fourier %>%
  dplyr::select(
    Date, Sector, prop,
    starts_with("sin"), starts_with("cos")
  ) %>%
  pivot_wider(
    names_from  = Sector,
    values_from = prop
  )

# Build Y_ (list of vectors, each date is a row, with sector proportions)
Y_ <- stocks %>%
  group_by(Date) %>%
  arrange(Sector) %>%
  summarize(prop = list(prop)) %>%
  pull(prop) %>%
  lapply(as.numeric)

# Build Fourier terms as a matrix for each date
fourier_terms <- stocks_wide %>%
  dplyr::select(starts_with("sin"), starts_with("cos")) %>%
  as.matrix()

# Build similar for the testing portion
fourier_terms_new <- trading_calendar_testing %>%
  dplyr::select(starts_with("sin"), starts_with("cos")) %>%
  as.matrix()

# Example: you have a boolean vector for adding trend or not
has_trend <- rep(c(TRUE, FALSE), times = c(10, 0))  # Adjust as needed

# Build X: a list of length T (each day), each containing sub-list for each sector
X_component <- map(
  seq(1, length.out = length(Y_)),
  ~ {
    map(
      has_trend,
      function(cond) {
        if (cond) {
          # Include trend
          trend <- .x
        } else {
          # No trend
          trend <- NULL
        }
        # Return (1 + Fourier terms + maybe trend)
        c(1, as.numeric(fourier_terms[.x, ]))
      }
    )
  }
)

# Flatten each day's sub-list into a numeric vector
X <- map(X_component, flatten_dbl)

# Also build a combined phi design X_phi (example if your model uses it)
X_phi <- cbind(1, seq_along(Y_), fourier_terms)
X_phi <- as.matrix(X_phi)

# For new data (test)
X_new_component <- map(
  seq(1, length.out = nrow(trading_calendar_testing)),  # e.g. 126
  ~ {
    map(
      has_trend,
      function(cond) {
        if (cond) {
          trend <- .x
        } else {
          trend <- NULL
        }
        c(1, as.numeric(fourier_terms_new[.x, ]))
      }
    )
  }
)
X_new <- map(X_new_component, flatten_dbl)

t_abs <- seq(length(Y_) + 1, length.out = nrow(trading_calendar_testing))
X_phi_new <- cbind(1, seq_along(t_abs), fourier_terms_new)
X_phi_new <- as.matrix(X_phi_new)
colnames(X_phi_new) <- NULL

################################################################################
# F) Prepare Stan data template (common to all priors, but we may tweak inside loop)
################################################################################
sector_ordered <- colnames(stocks_wide)[-c(1:15)]  # sector names, if needed
sector_ordered
stan_data <- list(
  T       = length(Y_),
  C       = length(Y_[[1]]),
  Y       = Y_,
  N       = length(X[[1]]),
  K       = map_int(X_component[[1]], length),
  X       = X,
  K_phi   = ncol(X_phi),
  X_phi   = X_phi,
  ref     = 1,
  P       = 10,
  Q       = 0,
  T_new   = nrow(trading_calendar_testing),
  X_new   = X_new,
  X_phi_new = X_phi_new,
  prior_only = 0,
  beta_sd  = 1.0
)

message("Data prepared. Ready to loop over priors...")

# ==============================================================================
# 7) Loop over each prior, run the model if no partial file exists, save results
# ==============================================================================
all_results <- list()

prior_subset <- priors[3]

for (i in 1:1) {
  
  
  prior <- prior_subset[i]
  message("\n--- Processing prior: ", prior, " ---\n")
  
  # RDS file for partial results
  rds_file <- file.path(save_dir, paste0("bdarma_results_", prior, ".rds"))
  
  # If partial results exist, skip
  # if (file.exists(rds_file)) {
  #   message("File already exists for prior '", prior, "': ", rds_file)
  #   message("Skipping model run. Loading existing results from disk.")
  #   all_results[[prior]] <- readRDS(rds_file)
  #   next
  # }
  
  # ----------------------------------------------------------------
  # 7.1) Possibly tweak stan_data for this prior
  # ----------------------------------------------------------------
  # Example: If you have prior-specific data additions:
  if (prior == "laplace") {
    stan_data$b_beta <- 1.0
    stan_data$b_A    <- 1.0
    stan_data$b_B    <- 1.0
  } else {
    # If other priors also need specific data modifications, do them here
    stan_data$b_beta <- 0.5  # or some default if not needed
    stan_data$b_A    <- 0.5
    stan_data$b_B    <- 0.5
  }
  
  # ----------------------------------------------------------------
  # 7.2) Compile Stan model for this prior
  # ----------------------------------------------------------------
  # Assumes .stan files are named e.g. bdarma_informative.stan, bdarma_horseshoe.stan, etc.
  stan_model_file <- file.path("stan", paste0("bdarma_", prior, ".stan"))
  if (!file.exists(stan_model_file)) {
    stop("Stan model file not found: ", stan_model_file)
  }
  stan_model_file
  mod <- cmdstanr::cmdstan_model(stan_model_file)
  
  # ----------------------------------------------------------------
  # 7.3) Run MCMC sampling
  # ----------------------------------------------------------------
  fit <- mod$sample(
    data            = stan_data,
    seed            = 123,
    chains          = 6,
    iter_sampling   = 1250,
    iter_warmup     = 1250,
    parallel_chains = 8,
    max_treedepth   = 11,
    adapt_delta     = 0.88,
    init=.3
  )
  
  # ----------------------------------------------------------------
  # 7.4) Post-process the model output
  # ----------------------------------------------------------------
  # Example: Extract Y_hat
  Y_hat_bdarma <- fit$draws("Y_hat", format = "draws_df")
  
  # Convert to long
  Y_hat_long_raw_bdarma <- Y_hat_bdarma %>%
    dplyr::select(.draw, starts_with("Y_hat")) %>%
    pivot_longer(starts_with("Y_hat"))
  
  # Summarize
  Y_hat_long_bdarma <- Y_hat_long_raw_bdarma %>%
    summarize(
      mean = mean(value, na.rm = TRUE),
      qlo  = quantile(value, probs = 0.05, na.rm = TRUE, names = FALSE),
      qhi  = quantile(value, probs = 0.95, na.rm = TRUE, names = FALSE),
      .by  = name
    ) %>%
    mutate(idx = stringr::str_extract(name, "(?<=\\[).+?(?=\\])")) %>%
    separate(idx, c("t", "Sector"), sep = ",") %>%
    mutate(
      t = as.integer(t),
      Sector = as.integer(Sector),
      Sector = sector_ordered[Sector]  # map integer back to sector name
    )
  
  # Create a date reference for the test period
  date_df <- stocks %>%
    mutate(Date = as.Date(Date)) %>%
    filter(Date > as.Date("2018-12-31")) %>%
    distinct(Date) %>%
    arrange(Date) %>%
    mutate(
      DayOfWeek = wday(Date, week_start = 1),
      DayOfYear = yday(Date),
      Time      = row_number()
    )
  
  data_df_training <- date_df %>%
    filter(Date < as.Date("2023-07-01"))
  
  data_df_test <- date_df %>%
    filter(Date >= as.Date("2023-07-01"))
  
  data_df_test$t <- 1:nrow(data_df_test)
  data_df_test_merge <- data_df_test %>% dplyr::select(t, Date)
  
  # Merge predictions with test date
  Y_hat_long_bdarma <- full_join(data_df_test_merge, Y_hat_long_bdarma, by = "t") %>%
    dplyr::select(Date, mean, qlo, qhi, Sector) %>%
    mutate(forecast = "forecast")
  
  # Compare forecast vs actual
  sp_data_test <- stocks %>%
    mutate(Date = as.Date(Date)) %>%
    filter(Date > as.Date("2021-12-31"), Date < as.Date("2024-01-01")) %>%
    dplyr::select(Date, Sector, mean = prop) %>%
    mutate(forecast = "actuals")
  
  # Combine
  Y_hat_long_bdarma_comp <- full_join(Y_hat_long_bdarma, sp_data_test, 
                                      by = c("Date", "Sector", "forecast", "mean")) %>%
    distinct()
  
  # Possibly fix the above join if columns mismatch. Another approach:
  # Y_hat_long_bdarma_comp <- full_join(Y_hat_long_bdarma, sp_data_test,
  #   by = c("Date", "Sector", "forecast")
  # ) %>%
  # rename(pred_mean = mean.x, actual_mean = mean.y) ...
  #
  # (But we'll keep it close to your original script.)
  
  # Compute error metrics
  yhat_wide <- Y_hat_long_bdarma_comp %>%
    filter(!is.na(forecast)) %>%  # remove weird merges
    dplyr::select(Date, Sector, forecast, mean) %>%
    distinct() %>%
    pivot_wider(names_from = forecast, values_from = mean) %>%
    filter(!is.na(actuals), !is.na(forecast)) %>%
    mutate(
      error          = forecast - actuals,
      abs_error      = abs(error),
      squared_error  = error^2,
      ape           = abs_error / ifelse(abs(actuals) < 1e-12, NA, abs(actuals))
    )
  
  # Summarize errors
  error_summary <- yhat_wide %>%
    group_by(Sector) %>%
    summarize(
      MSE  = mean(squared_error, na.rm = TRUE),
      RMSE = sqrt(MSE),
      MAE  = mean(abs_error, na.rm = TRUE),
      MAPE = 100 * mean(ape, na.rm = TRUE),
      .groups = "drop"
    )
  
  # Overall error
  overall_error <- yhat_wide %>%
    summarize(
      MSE  = mean(squared_error, na.rm = TRUE),
      RMSE = sqrt(MSE),
      MAE  = mean(abs_error, na.rm = TRUE),
      MAPE = 100 * mean(ape, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(Sector = "ALL")
  
  error_summary <- bind_rows(error_summary, overall_error)
  
  # Parameter summaries
  param_names_of_interest <- c("beta", "A", "B", "phi")
  existing_params <- intersect(param_names_of_interest, fit$metadata()$stan_variables)
  param_summary   <- NULL
  if (length(existing_params) > 0) {
    param_summary <- fit$summary(variables = existing_params)%>%
      dplyr::select(variable, mean, sd, q5 = q5, q95 = q95)
  }
  
  # Store final results in all_results
  all_results[[prior]] <- list(
    fit           = fit,
    predictions   = Y_hat_long_bdarma_comp,
    error_summary = error_summary,
    param_summary = param_summary
  )
  
  # Save partial results
  saveRDS(all_results[[prior]], rds_file)
  message("Successfully saved results for prior '", prior, "' at: ", rds_file)
}



all_results <- list()

for (prior in priors) {
  rds_file <- file.path(save_dir, paste0("bdarma_results_", prior, ".rds"))
  
  if (file.exists(rds_file)) {
    message("Loading existing RDS file for prior: ", prior)
    all_results[[prior]] <- readRDS(rds_file)
  } else {
    message("No RDS file found for prior: ", prior, " (maybe it never finished).")
  }
}
# ==============================================================================
# 8) Combine everything once the loop is done
# ==============================================================================
combined_predictions <- map2_dfr(
  .x = all_results,
  .y = names(all_results),
  ~ .x$predictions %>% mutate(model = .y)
)

combined_errors <- map2_dfr(
  .x = all_results,
  .y = names(all_results),
  ~ .x$error_summary %>% mutate(model = .y)
)

combined_param_summaries <- map2_dfr(
  .x = all_results,
  .y = names(all_results),
  ~ {
    if (!is.null(.x$param_summary)) {
      .x$param_summary %>% mutate(model = .y)
    } else {
      NULL
    }
  }
)

# ==============================================================================
# 9) (Optional) Save these combined objects as RDS
# ==============================================================================
saveRDS(combined_predictions, file.path(save_dir, "combined_predictions.rds"))
saveRDS(combined_errors,      file.path(save_dir, "combined_errors.rds"))
saveRDS(combined_param_summaries, file.path(save_dir, "combined_param_summaries.rds"))

# ==============================================================================
# 10) Example final plots
# ==============================================================================
message("\nCreating example plots...")

head(combined_predictions)






# (A) Plot forecast vs actual for each prior
p1 <- ggplot(combined_predictions, aes(x = Date, y = mean, color = forecast, group = forecast)) +
  geom_line(size = 1) +
  facet_wrap(~ Sector + model, scales = "free_y") +
  theme_minimal() +
  labs(
    title = "Forecast vs Actuals by Sector & Prior",
    x     = "Date",
    y     = "Value",
    color = "Type",
    fill  = "Type"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title  = element_text(hjust = 0.5, size = 16, face = "bold")
  )
print(p1)


p1 <- ggplot(combined_predictions %>% filter(Date>as.Date("2022-12-31")), 
             aes(x = Date, y = mean, color = forecast, group = forecast)) +
  geom_line(size = 1) +
  facet_grid(model ~ Sector, scales = "free_y",
             labeller = labeller(
               Sector = label_wrap_gen(width = 10),  # Adjust width as needed
               model  = label_wrap_gen(width = 10)
             )) +
  scale_x_date(
    date_breaks = "3 month",      # Adjust the break interval as needed
    date_labels = "%b %Y"         # Format labels to "Jul 2023", etc.
  ) +
  coord_cartesian(clip = "off") + 
  theme_minimal() +
  labs(
    title = "Forecast and Actuals by Sector & Prior",
    x     = "Date",
    y     = "Value",
    color = "Type",
    fill  = "Type"
  ) +
  theme(
    axis.text.x   = element_text(angle = 45, hjust = 1),
    plot.title    = element_text(hjust = 0.5, size = 16, face = "bold"),
    plot.margin   = margin(10, 10, 10, 10),   # adjust margins as needed
    strip.text.x  = element_text(margin = margin(t = 5, b = 5))  # add margin around facet labels
  )
print(p1)


unique(combined_errors$model)

# (B) Bar plot of RMSE by prior and sector
p2 <- ggplot(combined_errors %>% filter(Sector!="ALL"), aes(x = model, y = RMSE, fill = model)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  facet_wrap(~ Sector, scales = "free_y") +
  theme_minimal() +
  labs(
    title = "RMSE by Prior & Sector",
    x     = "Prior",
    y     = "RMSE"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(p2)

# (C) Parameter summaries (posterior means, if available)
if (nrow(combined_param_summaries) > 0) {
  p3 <- ggplot(combined_param_summaries, aes(x = model, y = mean, color = model)) +
    geom_point(position = position_jitter(width = 0.1), size = 2) +
    facet_wrap(~ variable, scales = "free_y") +
    theme_minimal() +
    labs(
      title = "Parameter Estimates (Posterior Means) by Prior",
      x     = "Prior",
      y     = "Posterior Mean"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  print(p3)
} else {
  message("No parameter summaries to plot.")
}

message("\nAll done! Results are available in:")
message("  all_results         (list)")
message("  combined_predictions (data.frame)")
message("  combined_errors      (data.frame)")
message("  combined_param_summaries (data.frame)")

################################################################################
# END OF SCRIPT
################################################################################
