################################################################################
# pomp Tutorial: SE2I2R model with Negative Binomial observations
# Country: Germany | First wave COVID-19 (March–June 2020)
#
# Pipeline: Data → POMP object → MLE → MCMC
################################################################################

# 1. load packages ------------------------------------------------------------------
library(tidyverse)
library(lubridate)
library(pomp)       
library(nloptr)     
library(BayesianTools)
library(MCMCvis)
library(coda)


# 2. load data ----------------------------------------------------------------------
# Load the data for Germany
dat <- readRDS("_data/dat_tutorial.rds")
ct_nm <- "Germany"

# Plot data
ggplot(dat, aes(x = date, y = confirmed)) +
  geom_line() +
  labs(title = paste("Daily COVID-19 cases –", ct_nm),
       x = NULL, y = "Confirmed cases")

ggplot(dat, aes(x = date, y = stringency_index)) +
  geom_line(colour = "steelblue") +
  labs(title = paste("NPI stringency index –", ct_nm),
       x = NULL, y = "Stringency (0–100)")

# POMP requires columns: observation times, observed state, and covariates.
dat_pomp <- dat |>
  arrange(day) |>
  select(day, CC_obs = confirmed, theta = stringency_index)


# 3. C-Snippets  -------------------------------------------------------
# Each snippet is a small piece of C code that pomp compiles and links.
# Variables that appear as state names / parameter names are automatically
# available inside the snippet.

# ── 3a. Global utility functions (shared across all snippets) -----------------
globs <- Csnippet("
  #include <math.h>

  /* Discrete-time transition probability: p = 1 - exp(-rate * dt) */
  static double p_trans(double r, double dt) {
    return 1.0 - exp(-r * dt);
  }

  /* Time-of-week reporting multiplier, mean = 1 over any 7-day cycle.
     A_w = 0  ->  constant 1  (no weekly effect)
     A_w = 1  ->  oscillates between 0 and pi/2  */
  static double f_week(double A_w, double phi_w, double t) {
    return (1.0 - A_w) + M_PI / 2.0 * A_w * fabs(cos(M_PI * (t / 7.0 - phi_w)));
  }
")

# ── 3b. Initial conditions ----------------------------------------------------
# A small fraction of the population starts exposed (E1); everyone else is
# susceptible. Re is initialised to R0 because S ≈ N at t0.
rinit <- Csnippet("
  S  = nearbyint(N * (1.0 - e10));
  E1 = nearbyint(N * e10);
  E2 = 0;
  I1 = 0;
  I2 = 0;
  R  = 0;
  Q  = 0;
  CC = 0;
  Re = R0;
")

# ── 3c. Deterministic skeleton -------------------------
# Discrete-time map with step size delta_t = 1 day.
# Two-stage Erlang latency (E1→E2→I) and infectious period (I1→I2→R) give
# Erlang-2 distributed sojourn times, relaxing the memoryless (exponential)
# assumption.
#
# NPI modifier:
#  theta = stringency index in [0, 100]
# Reporting queue:
#   New infectious cases (leaving E2) enter Q; they exit at rate r_Q and are
#   counted in CC (accumulated each day, then reset by pomp).
#
# 'D' prefix (DS, DE1, …) is pomp's notation for the next-step value.
skel <- Csnippet("
  double lambda_t, r_t;
  double n_S_E1, n_E1_E2, n_E2_I1, n_I1_I2, n_I2_R, n_Q_CC;
  double delta_t = 1.0;

  r_t      = fmin2(1.0, b_theta * theta / 100.0);
  lambda_t = R0 * gamma * (1.0 - r_t) * (I1 + I2) / N;

  n_S_E1  = S  * p_trans(lambda_t,      delta_t);
  n_E1_E2 = E1 * p_trans(2.0 * sigma,   delta_t);
  n_E2_I1 = E2 * p_trans(2.0 * sigma,   delta_t);
  n_I1_I2 = I1 * p_trans(2.0 * gamma,   delta_t);
  n_I2_R  = I2 * p_trans(2.0 * gamma,   delta_t);
  n_Q_CC  = Q  * p_trans(r_Q,           delta_t);

  DS  = S  - n_S_E1;
  DE1 = E1 + n_S_E1  - n_E1_E2;
  DE2 = E2 + n_E1_E2 - n_E2_I1;
  DI1 = I1 + n_E2_I1 - n_I1_I2;
  DI2 = I2 + n_I1_I2 - n_I2_R;
  DR  = R  + n_I2_R;
  DQ  = Q  + n_E2_I1 - n_Q_CC;
  DCC = CC + n_Q_CC;
  DRe = R0 * (1.0 - r_t) * S / N;
")

# ── 3d. Stochastic Euler simulator (rprocess) ---------------------------------
# Identical structure to the skeleton, but flows are Binomial draws.
rsim <- Csnippet("
  double lambda_t, r_t;
  double n_S_E1, n_E1_E2, n_E2_I1, n_I1_I2, n_I2_R, n_Q_CC;

  r_t      = fmin2(1.0, b_theta * theta / 100.0);
  lambda_t = R0 * gamma * (1.0 - r_t) * (I1 + I2) / N;

  n_S_E1  = rbinom(S,  p_trans(lambda_t,      dt));
  n_E1_E2 = rbinom(E1, p_trans(2.0 * sigma,   dt));
  n_E2_I1 = rbinom(E2, p_trans(2.0 * sigma,   dt));
  n_I1_I2 = rbinom(I1, p_trans(2.0 * gamma,   dt));
  n_I2_R  = rbinom(I2, p_trans(2.0 * gamma,   dt));
  n_Q_CC  = rbinom(Q,  p_trans(r_Q,            dt));

  S  += -n_S_E1;
  E1 += n_S_E1  - n_E1_E2;
  E2 += n_E1_E2 - n_E2_I1;
  I1 += n_E2_I1 - n_I1_I2;
  I2 += n_I1_I2 - n_I2_R;
  R  += n_I2_R;
  Q  += n_E2_I1 - n_Q_CC;
  CC += n_Q_CC;
  Re  = R0 * (1.0 - r_t) * S / N;
")

# ── 3e. Negative-Binomial observation density (dmeasure) ----------------------
# Observed count CC_obs ~ NegBinom(mu = rho_t * CC,  size = 1/rho_D)
# Variance = mu + mu^2 * rho_D.  As rho_D -> 0 this converges to Poisson.
# rho_t = rho_M * f_week(A_w, phi_w, t)  captures both mean under-reporting
# and systematic day-of-week variation.
dmeas <- Csnippet("
  double fCC, rho_t;
  rho_t = rho_M * f_week(A_w, phi_w, t);
  fCC   = ISNA(CC_obs) ? 0.0 :
            dnbinom_mu(nearbyint(CC_obs), 1.0 / rho_D, rho_t * CC, 1);
  lik   = (give_log) ? fCC : exp(fCC);
")

# ── 3f. Negative-Binomial observation sampler (rmeasure) ----------------------
rmeas <- Csnippet("
  double rho_t = rho_M * f_week(A_w, phi_w, t);
  CC_obs = rnbinom_mu(1.0 / rho_D, rho_t * CC);
")


# 4. BUILD THE POMP OBJECT -----------------------------------------------------
# Fixed epidemiological parameters (not estimated):
#   sigma: discrete-time latency rate  (D_E = 4 days)
#   gamma: discrete-time recovery rate (D_I = 5 days)
#   rho_M: mean reporting probability for Germany (literature estimate)
#   N    : population size from data
D_E   <- 4
D_I   <- 5
sigma <- -log(1 - 2 / D_E) / 2
gamma <- -log(1 - 2 / D_I) / 2
rho_M <- 0.25   
N     <- dat$population[1]

parms_init <- c(
  R0      = 3.5,    # Basic reproduction number
  sigma   = sigma,  # Fixed
  gamma   = gamma,  # Fixed
  b_theta = 1.0,    # Sensitivity of transmission to NPI stringency
  e10     = 1e-5,   # Initial exposed fraction
  r_Q     = 0.15,   # Reporting delay rate (~7-day mean delay)
  rho_M   = rho_M,  # Fixed 
  rho_D   = 0.05,   # NegBinom overdispersion
  A_w     = 0.4,    # Day-of-week amplitude
  phi_w   = 0.5,    # Day-of-week phase
  N       = N       # Fixed
)

mod <- pomp(
  data       = dat_pomp[, c("day", "CC_obs")],
  times      = "day",
  t0         = min(dat_pomp$day),
  obsnames   = "CC_obs",
  covar      = covariate_table(dat_pomp[, c("day", "theta")], times = "day"),
  statenames = c("S", "E1", "E2", "I1", "I2", "R", "Q", "CC", "Re"),
  accumvars  = "CC",   # CC is reset to 0 before each accumulation step
  paramnames = names(parms_init),
  params     = parms_init,
  globals    = globs,
  rinit      = rinit,
  skeleton   = pomp::map(f = skel, delta.t = 1),
  rprocess   = euler(step.fun = rsim, delta.t = 1),
  dmeasure   = dmeas,
  rmeasure   = rmeas,
  # Optimisation works on the transformed scale; pomp back-transforms automatically.
  partrans   = parameter_trans(
    log   = c("R0", "gamma", "b_theta", "r_Q", "rho_D"),
    logit = c("e10", "rho_M", "A_w", "phi_w")
  )
)

# Sanity check: simulate and plot a single trajectory
sim0 <- simulate(mod, nsim = 1, format = "data.frame")
ggplot(sim0, aes(x = day, y = CC_obs)) +
  geom_line() +
  labs(title = "Single simulated trajectory (initial parameters)",
       x = "Day (0 = 2020-03-01)", y = "Simulated daily cases")


# 5. Maximum Likelihood Estimation -----------------------
# traj_objfun() returns the negative log-likelihood based on the deterministic
# skeleton (trajectory matching).
pars_est_nm <- c("R0", "b_theta", "r_Q", "e10", "A_w", "phi_w", "rho_D")

minusLL <- traj_objfun(data = mod, est = pars_est_nm, partrans = mod@partrans)

# -- 5a. Multi-start optimisation using a Sobol sequence ----------------------
set.seed(4821)
start_range <- list(
  R0      = c(1,    8),
  b_theta = c(0,    2),
  r_Q     = c(0.01, 1),
  rho_D   = c(0.01, 0.5),
  e10     = c(1e-6, 1e-3),
  A_w     = c(0,    1),
  phi_w   = c(0,    1)
)

start_parms <- sobol_design(
  lower = sapply(start_range, `[`, 1)[pars_est_nm],
  upper = sapply(start_range, `[`, 2)[pars_est_nm],
  nseq  = 10   # change if you want
)

# Optimise from each starting point sequentially 
fits_list <- vector("list", nrow(start_parms))
for (i in seq_len(nrow(start_parms))) {
  x0 <- as.numeric(start_parms[i, pars_est_nm])
  coef(mod, pars_est_nm) <- x0
  x0_trans <- coef(mod, pars_est_nm, transform = TRUE)

  fit <- tryCatch(
    nloptr(
      x0     = x0_trans,
      eval_f = minusLL,
      opts   = list(
        algorithm  = "NLOPT_LN_SBPLX",   # Derivative-free Nelder-Mead variant
        maxtime    = 30,                  # seconds per starting point
        maxeval    = -1,
        xtol_rel   = -1,
        print_level = 0
      )
    ),
    error = function(e) NULL
  )

  if (!is.null(fit)) {
    coef(mod, pars_est_nm, transform = TRUE) <- fit$solution
    fits_list[[i]] <- data.frame(
      as.list(coef(mod, pars_est_nm)),
      ll        = -fit$objective,
      conv_code = fit$status
    )
  }
}

fits_mle <- bind_rows(fits_list) |>
  filter(is.finite(ll)) |>
  arrange(desc(ll))

# -- 5b. Inspect MLE results --------------------------------------------------
cat("Top 5 MLE solutions:\n")
head(fits_mle, 5) |> print()

cat(sprintf("\nBest log-likelihood: %.2f\n", fits_mle$ll[1]))
cat(sprintf("AIC: %.2f\n", 2 * length(pars_est_nm) - 2 * fits_mle$ll[1]))

# 99% profile-likelihood confidence region 
mle_ci <- fits_mle |>
  filter(2 * abs(ll - max(ll)) <= qchisq(0.99, df = length(pars_est_nm)))
cat(sprintf("\nParameter sets in 99%% joint CI: %d\n", nrow(mle_ci)))
summary(mle_ci[, pars_est_nm]) |> print()

# -- 5c. Apply MLE to the model and plot fitted trajectory --------------------
mle_best <- setNames(as.numeric(fits_mle[1, pars_est_nm]), pars_est_nm)
coef(mod, pars_est_nm) <- mle_best

traj_mle <- trajectory(mod, format = "data.frame")

dat_obs <- dat |>
  select(day, date, confirmed) |>
  filter(!is.na(confirmed))

ggplot() +
  geom_point(data = dat_obs, aes(x = day, y = confirmed),
             size = 0.8, alpha = 0.7) +
  geom_line(data = traj_mle, aes(x = day, y = CC),
            colour = "firebrick", linewidth = 1) +
  labs(title = paste("MLE fit –", ct_nm, "| Negative Binomial"),
       x = "Day (0 = 2020-03-01)",
       y = "Daily cases",
       caption = sprintf("Log-likelihood = %.1f", fits_mle$ll[1]))


# 6. Bayesian MCMC ---------------------------------------------

# -- 6a. Log-likelihood on the natural scale ------------------
# BayesianTools works on the untransformed scale; set partrans = NULL so pomp
# does not apply its internal transformations.
minusLL_nat <- traj_objfun(data = mod, est = pars_est_nm, partrans = NULL)
LL_nat      <- function(x) -minusLL_nat(x)

# -- 6b. Priors ---------------------------------------------------------------
log_prior <- function(par) {
  names(par) <- pars_est_nm
  d <- c(
    dexp( par["R0"]      - 1,  rate = 1, log = TRUE),  # R0 > 1
    dexp( par["b_theta"],      rate = 1, log = TRUE),
    dexp( par["r_Q"],          rate = 1, log = TRUE),
    dbeta(par["e10"],  1, 100, log = TRUE),             # concentrated near 0
    dbeta(par["A_w"],  1,   1, log = TRUE),             # uniform
    dbeta(par["phi_w"],1,   1, log = TRUE),
    dexp( par["rho_D"],        rate = 1, log = TRUE)
  )
  sum(d)
}

samp_prior <- function(n = 1) {
  m           <- matrix(NA, nrow = n, ncol = length(pars_est_nm))
  colnames(m) <- pars_est_nm #MLE
  m[, "R0"]      <- 1 + rexp(n, 1)
  m[, "b_theta"] <- rexp(n, 1)
  m[, "r_Q"]     <- rexp(n, 1)
  m[, "e10"]     <- rbeta(n, 1, 100)
  m[, "A_w"]     <- rbeta(n, 1, 1)
  m[, "phi_w"]   <- rbeta(n, 1, 1)
  m[, "rho_D"]   <- rexp(n, 1)
  m
}

# Hard bounds for each parameter 
lower_bounds <- c(R0 = 1,    b_theta = 0, r_Q = 0,    e10 = 0,
                  A_w = 0,   phi_w   = 0, rho_D = 0)
upper_bounds <- c(R0 = 100,  b_theta = 100, r_Q = 100, e10 = 1,
                  A_w = 1,   phi_w   = 1,   rho_D = 100)

prior_obj <- createPrior(
  density = log_prior,
  sampler = samp_prior,
  lower   = lower_bounds[pars_est_nm],
  upper   = upper_bounds[pars_est_nm]
)

# -- 6c. BayesianTools setup --------------------------------------------------
bayes_setup <- createBayesianSetup(
  likelihood = LL_nat,
  prior      = prior_obj,
  names      = pars_est_nm,
  best       = mle_best
)

# Matrix seeded from the 99% MLE confidence region
Z_mat <- as.matrix(mle_ci[, pars_est_nm])

n_chains <- 3       # change if you want
n_iter   <- 20000    # change if you want

settings <- list(
  iterations = n_chains * n_iter,
  startValue = replicate(n_chains, mle_best) |> t(),
  Z          = Z_mat
)

# -- 6d. Run MCMC -------------------------------------------------------------
fit_mcmc <- runMCMC(
  bayesianSetup = bayes_setup,
  sampler       = "DEzs",
  settings      = settings
)

# -- 6e. Convergence diagnostics -----------------------------------------------
# Keep only the last 50% of each chain as post-burn-in samples, then thin.
n_stored  <- nrow(fit_mcmc$chain[[1]])
n_burnin  <- floor(n_stored / 2)
post_coda <- getSample(fit_mcmc, coda = TRUE, start = n_burnin + 1, thin = 5)

# R-hat
gelman_diag <- gelman.diag(post_coda, multivariate = FALSE)
print(gelman_diag)

# Effective sample sizes
effectiveSize(post_coda) |> round() |> print()

# -- 6f. Posterior summary and trace plots ------------------------------------
MCMCvis::MCMCtrace(
  object   = post_coda,
  pdf      = FALSE,   
  ind      = TRUE,
  Rhat     = TRUE,
  n.eff    = TRUE,
  iter     = n_stored / 2
)

# MAP and 95% HPD intervals
map_est <- MAP(fit_mcmc)
hpd_tab <- MCMCvis::MCMCsummary(
  object   = post_coda,
  hpd_prob = 0.95,
  probs    = c(0.025, 0.975)
)
cat("\nPosterior summary (95% HPD):\n")
print(signif(hpd_tab[, c("mean", "2.5%", "97.5%")], 3))

# -- 6g. Posterior predictive check ------------------------------------------
# Draw 200 parameter vectors from the posterior and simulate the observation
# model to obtain a predictive envelope.
post_samp <- getSample(fit_mcmc, start = n_burnin + 1, thin = 5)
n_pp      <- min(200, nrow(post_samp))
set.seed(9274)
idx_pp    <- sample(nrow(post_samp), n_pp)

sim_list <- lapply(idx_pp, function(i) {
  coef(mod, pars_est_nm) <- post_samp[i, pars_est_nm]
  simulate(mod, nsim = 1, format = "data.frame") |>
    mutate(rep = i)
})

sim_pp <- bind_rows(sim_list) |>
  filter(!is.na(CC_obs))

# 95% simulation envelope
pp_env <- sim_pp |>
  group_by(day) |>
  summarise(
    lo  = quantile(CC_obs, 0.025),
    mid = quantile(CC_obs, 0.500),
    hi  = quantile(CC_obs, 0.975),
    .groups = "drop"
  )

ggplot() +
  geom_ribbon(data = pp_env, aes(x = day, ymin = lo, ymax = hi),
              fill = "steelblue", alpha = 0.3) +
  geom_line(data = pp_env, aes(x = day, y = mid),
            colour = "steelblue", linewidth = 0.8) +
  geom_point(data = dat_obs, aes(x = day, y = confirmed),
             size = 0.9, alpha = 0.8) +
  labs(title = paste("Posterior predictive check –", ct_nm,
                     "| Negative Binomial"),
       x = "Day (0 = 2020-03-01)",
       y = "Daily cases",
       caption = "Ribbon: 95% posterior predictive interval")

################################################################################
# END
################################################################################
