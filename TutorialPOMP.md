# Epidemic Inference with `pomp`

## SE₂I₂R + Negative Binomial · Germany · COVID-19 Wave 1

> This tutorial fits a mechanistic transmission model to daily COVID-19 case counts in Germany (March–June 2020) using the R package `pomp`. It covers building the model, maximum likelihood estimation, and Bayesian MCMC.
>
> **Code:** [`tutorial_negbinom_germany.R`](tutorial_negbinom_germany.R)

------------------------------------------------------------------------

## The POMP Framework

A **Partially Observed Markov Process** (POMP) has two distinct layers:

1.  **A latent process** $\{X_t\}$ — the true epidemic state (e.g. the number of people in each compartment) that evolves as a Markov chain.

2.  **An observation process** $\{Y_t\}$ — the data we actually observe (e.g. confirmed case counts), which are noisy, incomplete functions of the true state. Observations are assumed conditionally independent given the latent state.

T**he process model and the observation model are independent specifications.** You can change the transmission dynamics without touching the observation model, and vice versa.

### What `pomp` does

The R package `pomp` provides the scaffolding for specifying and fitting POMP models.

| Layer | What you specify | `pomp` term |
|------------------------|------------------------|------------------------|
| **Latent dynamics** | State transitions — your compartmental model | `rprocess` / `skeleton` |
| **Observation model** | How observed data relate to the latent state | `dmeasure` / `rmeasure` |
| **Inference** | MLE, particle filtering, MCMC | `traj_objfun`, `pfilter`, etc. |

- **`rprocess`** — a stochastic simulator. Given the current state $X_t$, it draws the next state from $p(X_{t+1} \mid X_t, \theta)$. This is your compartmental model written with Binomial draws over flows.

- **`skeleton`** — the deterministic of the same dynamics. Used for fast trajectory matching (MLE without stochastic replicates).

- **`dmeasure`** — evaluates $\log p(Y_t \mid X_t, \theta)$, the log-likelihood contribution of one observed data point given the latent state.

- **`rmeasure`** — draws a simulated observation $\tilde{Y}_t \sim p(Y_t \mid X_t, \theta)$. Used for posterior predictive checks.

------------------------------------------------------------------------

## 1. The Epidemiological Model

### Compartmental Structure

The process model is an SE₂I₂R with a **two-stage (Erlang-2) latent and infectious period** and a reporting queue:

```         
S → E₁ → E₂ → I₁ → I₂ → R
               ↓
               Q → CC   (observed)
```

| Compartment | Description |
|------------------------------------|------------------------------------|
| S | Susceptible |
| E₁, E₂ | Exposed — two-stage Erlang latency (mean D_E = 4 days) |
| I₁, I₂ | Infectious — two-stage Erlang infectious period (mean D_I = 5 days) |
| R | Removed / recovered |
| Q | Reporting queue (geometric delay, exit rate r_Q) |
| CC | Daily cases exiting Q — **this is what we observe** |

### Discrete-Time Dynamics

The model runs in **discrete time** with step Δt = 1 day. The probability of leaving a compartment with exit rate r in one step is:

$$p(r) = 1 - e^{-r \, \Delta t}$$

The **force of infection** at time t is:

$$\lambda_t = R_0 \cdot \gamma \cdot (1 - r_t) \cdot \frac{I_1 + I_2}{N}$$

where the **NPI modifier** $r_t$ reduces transmission according to the Oxford Stringency Index $θ ∈ [0, 100]$:

$$r_t = \min\!\left(1,\; b_\theta \cdot \frac{\theta}{100}\right)$$

The parameter $b_θ$ controls how sensitive transmission is to the stringency index. At $b_θ= 1$ and $θ = 100$ (maximum stringency), transmission is fully suppressed. The **effective reproduction number** is tracked as:

$$R_{e,t} = R_0 \cdot (1 - r_t) \cdot \frac{S_t}{N}$$

### Reporting Model

Only a fraction $ρ_M$ of true cases are confirmed. Reporting also shows a systematic day-of-week effect — laboratory processing rates drop over weekends. The effective reporting probability at time t is:

$$\rho_t = \rho_M \cdot f_\text{week}(A_w, \phi_w, t)$$

where the day-of-week multiplier (normalised to mean 1 over any 7-day cycle) is:

$$f_\text{week}(A_w, \phi_w, t) = (1 - A_w) + \frac{\pi}{2} A_w \left|\cos\!\left(\pi\!\left(\frac{t}{7} - \phi_w\right)\right)\right|$$

Setting $A_w = 0$ gives constant reporting; larger $A_w$ produces stronger weekly oscillation. The phase $φ_w$ shifts the weekly peak.

### Observation Model — Negative Binomial

The observed daily case count $CC_{obs}$ is modelled as:

$$CC_{obs} \mid \text{CC} \sim \text{NegBinom}\!\left(\mu = \rho_t \cdot \text{CC},\;\; \text{size} = \frac{1}{\rho_D}\right)$$

with variance:

$$\text{Var}(CC_{obs}) = \mu + \mu^2 \cdot \rho_D$$

As $ρ_D → 0$ this converges to Poisson. $ρ_D$ is estimated rather than fixed, allowing the degree of overdispersion to be inferred from the data. Missing observations (NA) contribute zero to the log-likelihood.

------------------------------------------------------------------------

## 2. Parameters

### Fixed Parameters

These parameters are fixed at values informed by the literature and are not estimated.

| Parameter | Value                 | Rationale                                 |
|------------------|-------------------|------------------------------------|
| $D_E$     | 4 days                | Mean latent period                        |
| $D_I$     | 5 days                | Mean infectious period                    |
| σ         | $−log(1 − 2/D_E) / 2$ | Discrete-time rate for two-stage latency  |
| γ         | $−log(1 − 2/D_I) / 2$ | Discrete-time rate for two-stage recovery |
| $ρ_M$     | 0.25                  | Mean case detection fraction              |
| N         | From data             | Population size                           |

The rates σ and γ are derived analytically so that the mean sojourn time in the Erlang-2 chain equals $D_E$ and $D_I$ respectively.

### Estimated Parameters

| Parameter | Prior | Meaning |
|------------------------|------------------------|------------------------|
| R₀ | 1 + Exp(1) | Basic reproduction number (constrained \> 1) |
| $b_θ$ | Exp(1) | Sensitivity of transmission to NPI stringency |
| $r_Q$ | Exp(1) | Reporting delay rate (geometric queue exit probability) |
| e₁₀ | Beta(1, 100) | Initial exposed fraction (concentrated near 0) |
| $A_w$ | Uniform(0, 1) | Day-of-week reporting amplitude |
| $φ_w$ | Uniform(0, 1) | Day-of-week reporting phase |
| $ρ_D$ | Exp(1) | Negative Binomial overdispersion |

Priors enforce hard parameter constraints (e.g. R₀ \> 1) while placing little information within the plausible range.

------------------------------------------------------------------------

## 3. Implementation in `pomp`

`pomp` allows you to write the process model, observation model, and initial conditions as **C snippets** — fragments of C code passed as `Csnippet()` objects in R.

- You write only the *body* of the function.
- All state variables (e.g. `S`, `E1`, `I1`), parameters (e.g. `R0`, `sigma`), covariates (e.g. `theta`), and the time step (`dt`) are **automatically available by name** inside every snippet. You only need to declare intermediate variables that you create yourself.

The reason `pomp` uses C rather than R is speed: compiled C runs orders of magnitude faster than interpreted R.

### Global Utility Functions

Two helper functions are defined once and shared across all snippets via the `globals` argument. Anything defined here is visible inside every snippet.

``` c
/* Converts a continuous-time rate r into a discrete-time step probability.
   Derived from the survival function of an exponential: p = 1 - exp(-r * dt).
   With dt = 1 day and r = 1/D_I, this gives the daily probability of leaving
   the infectious compartment. */
static double p_trans(double r, double dt) {
    return 1.0 - exp(-r * dt);
}

/* Day-of-week multiplier for the reporting probability.
   Returns a value that averages to 1 over any 7-day cycle.
   A_w = 0 gives flat (constant) reporting; A_w close to 1 produces a strong
   weekend dip. phi_w shifts which day of the week is the peak. */
static double f_week(double A_w, double phi_w, double t) {
    return (1.0 - A_w) + M_PI / 2.0 * A_w * fabs(cos(M_PI * (t / 7.0 - phi_w)));
}
```

### Initial Conditions (`rinit`)

`pomp` calls this snippet once to set up the state vector at time $t_0$, before any simulation begins. A small fraction `e10` of the population starts in E₁ (exposed); everyone else is susceptible. `nearbyint()` rounds to the nearest integer to keep compartment sizes whole numbers.

``` c
S  = nearbyint(N * (1.0 - e10));   // almost the entire population susceptible
E1 = nearbyint(N * e10);           // small seed of exposed individuals
E2 = 0;  I1 = 0;  I2 = 0;  R = 0;  Q = 0;  CC = 0;
Re = R0;                           // Rₑ ≈ R₀ when S ≈ N
```

### Deterministic Skeleton (`skeleton`)

The **skeleton** is the deterministic version of the model. `pomp` uses this for trajectory matching during MLE (Section 5): it integrates the skeleton forward and scores the resulting trajectory against the data using `dmeasure`.

For discrete-time maps, `pomp` uses a `D`-prefix convention: `DS` means "the value of S at the next time step". Intermediate `n_X_Y` variables hold the *expected* number of people flowing from compartment X to Y in one step.

``` c
// Reduce transmission by the NPI stringency index (capped at 1 = full suppression)
r_t      = fmin2(1.0, b_theta * theta / 100.0);

// Force of infection: per-capita probability of a susceptible being infected this step
lambda_t = R0 * gamma * (1.0 - r_t) * (I1 + I2) / N;

// Expected flows between compartments this step (deterministic proportions)
n_S_E1  = S  * p_trans(lambda_t,  delta_t);   // new exposures
n_E1_E2 = E1 * p_trans(2.0*sigma, delta_t);   // progression through latency (stage 1→2)
n_E2_I1 = E2 * p_trans(2.0*sigma, delta_t);   // end of latency, become infectious
n_I1_I2 = I1 * p_trans(2.0*gamma, delta_t);   // progression through infectious period (1→2)
n_I2_R  = I2 * p_trans(2.0*gamma, delta_t);   // recovery
n_Q_CC  = Q  * p_trans(r_Q,       delta_t);   // cases exiting the reporting queue

// Next-step values: current value + inflows - outflows
DS  = S  - n_S_E1;
DE1 = E1 + n_S_E1  - n_E1_E2;
DE2 = E2 + n_E1_E2 - n_E2_I1;
DI1 = I1 + n_E2_I1 - n_I1_I2;
DI2 = I2 + n_I1_I2 - n_I2_R;
DR  = R  + n_I2_R;
DQ  = Q  + n_E2_I1 - n_Q_CC;
DCC = CC + n_Q_CC;                            // daily reported cases (reset each step by accumvars)
DRe = R0 * (1.0 - r_t) * S / N;              // tracked effective reproduction number
```

### Stochastic Simulator (`rprocess`)

This snippet has the same structure as the skeleton, but each flow is a **Binomial draw** rather than a deterministic expected value. Where the skeleton computes "on average, `S * p_trans(...)` people leave S this step", the stochastic version draws an actual integer from `rbinom(n, p)` — the number of "successes" out of `n` independent trials each with probability `p`.

``` c
// rbinom(n, p): how many of n individuals actually make the transition this step
n_S_E1  = rbinom(S,  p_trans(lambda_t,  dt));
n_E1_E2 = rbinom(E1, p_trans(2.0*sigma, dt));
n_E2_I1 = rbinom(E2, p_trans(2.0*sigma, dt));
n_I1_I2 = rbinom(I1, p_trans(2.0*gamma, dt));
n_I2_R  = rbinom(I2, p_trans(2.0*gamma, dt));
n_Q_CC  = rbinom(Q,  p_trans(r_Q,       dt));
// state updates are identical to the skeleton
```

This version is used for posterior predictive simulations. It could also replace the skeleton inside `pfilter()` for a fully stochastic particle-filter likelihood.

### Observation Density (`dmeasure`)

This snippet computes the log-likelihood contribution of a single observed data point. `pomp` calls it once per time step, passing the current latent state (`CC` — today's cases exiting the reporting queue) alongside the corresponding observation (`CC_obs` — the reported count in the data).

``` c
// Effective daily reporting probability: mean rate adjusted for day-of-week
double rho_t = rho_M * f_week(A_w, phi_w, t);

// Log-likelihood under NegBinom(mean = rho_t * CC, overdispersion = rho_D)
// ISNA() handles missing observations: an NA contributes 0 to the log-likelihood
double fCC = ISNA(CC_obs) ? 0.0 :
               dnbinom_mu(nearbyint(CC_obs), 1.0/rho_D, rho_t*CC, 1);

// pomp expects the result stored in `lik`; give_log controls log vs raw probability
lik = (give_log) ? fCC : exp(fCC);
```

`dnbinom_mu(x, size, mu, log)` is R's Negative Binomial density parameterised by mean `mu` and dispersion `size = 1/rho_D`. The trailing `1` requests the log-density. `CC` is the latent daily count produced by the process model; `CC_obs` is the observed count from the surveillance data.

------------------------------------------------------------------------

## 4. Building the `pomp` Object

Once all snippets are defined, everything is assembled into a single `pomp` object:

``` r
mod <- pomp(
  data       = dat_pomp[, c("day", "CC_obs")],
  times      = "day",
  t0         = min(dat_pomp$day),
  obsnames   = "CC_obs",
  covar      = covariate_table(dat_pomp[, c("day", "theta")], times = "day"),
  statenames = c("S", "E1", "E2", "I1", "I2", "R", "Q", "CC", "Re"),
  accumvars  = "CC",
  paramnames = names(parms_init),
  params     = parms_init,
  globals    = globs,
  rinit      = rinit,
  skeleton   = pomp::map(f = skel, delta.t = 1),
  rprocess   = euler(step.fun = rsim, delta.t = 1),
  dmeasure   = dmeas,
  rmeasure   = rmeas,
  partrans   = parameter_trans(
    log   = c("R0", "gamma", "b_theta", "r_Q", "rho_D"),
    logit = c("e10", "rho_M", "A_w", "phi_w")
  )
)
```

**`accumvars = "CC"`** — `pomp` resets CC to 0 before each time step's accumulation, so CC always holds the *daily* increment (not a running total). Without this, CC would grow monotonically and the observation model would be comparing against cumulative counts.

**`covariate_table`** — The NPI stringency index θ is time-varying. Wrapping it in `covariate_table()` lets `pomp` interpolate it to any requested time and make it available as the variable `theta` inside every C snippet.

**`partrans`** — applies transformations (e.g. log) and `pomp` back-transforms them automatically before evaluating the model.

------------------------------------------------------------------------

## 5. Maximum Likelihood Estimation

MLE here uses **trajectory matching**: the deterministic skeleton is integrated forward, and the resulting trajectory is scored against the data under the observation model. This is substantially faster than particle-filter-based likelihood approximation (`pfilter`) because no stochastic replicates are needed.

``` r
minusLL <- traj_objfun(data = mod, est = pars_est_nm, partrans = mod@partrans)
```

`traj_objfun()` returns a function that takes a parameter vector and returns the negative log-likelihood. `partrans = mod@partrans` tells it to accept inputs on the transformed (unconstrained) scale.

### Multi-Start Strategy

1.  Draw a **Sobol sequence** of starting points over broad parameter ranges.
2.  Optimise from each point using **the subplex algorithm**, a derivative-free Nelder-Mead variant.
3.  Collect all finite solutions, sort by log-likelihood, and keep the best.

``` r
start_parms <- sobol_design(lower = ..., upper = ..., nseq = 200)

for (i in seq_len(nrow(start_parms))) {
  coef(mod, pars_est_nm) <- start_parms[i, ]
  x0_trans <- coef(mod, pars_est_nm, transform = TRUE)

  fit <- nloptr(
    x0     = x0_trans,
    eval_f = minusLL,
    opts   = list(algorithm = "NLOPT_LN_SBPLX", maxtime = 30)
  )
  # back-transform and store ...
}
```

### 99% Profile-Likelihood Confidence Region

By Wilks' theorem, the set of parameter vectors satisfying

$$-2(\ell - \ell_\max) \leq \chi^2_{0.99,\, k}$$

forms an approximate **99% joint confidence region**. This region is also used to warm-start the MCMC sampler.

``` r
mle_ci <- fits_mle |>
  filter(2 * abs(ll - max(ll)) <= qchisq(0.99, df = length(pars_est_nm)))
```

------------------------------------------------------------------------

## 6. Bayesian MCMC

MCMC uses the same trajectory-matching log-likelihood plus an explicit prior, and the **DEzs** sampler (Differential Evolution with snooker update) from the `BayesianTools` package. DEzs is a population-based sampler that self-tunes its proposal distribution from the chain history.

### Warm Start

All chains start at the MLE. The DEzs *archive* (Z matrix) is seeded from the parameter sets in the 99% MLE confidence region, which approximates the posterior geometry and accelerates mixing.

``` r
bayes_setup <- createBayesianSetup(
  likelihood = LL_nat,    # log-likelihood on the natural (untransformed) scale
  prior      = prior_obj,
  names      = pars_est_nm
)

settings <- list(
  iterations = n_chains * n_iter,
  startValue = replicate(n_chains, mle_best) |> t(),
  Z          = as.matrix(mle_ci[, pars_est_nm])   # seed archive from MLE CI
)

fit_mcmc <- runMCMC(bayesianSetup = bayes_setup,
                    sampler       = "DEzs",
                    settings      = settings)
```

### Convergence Diagnostics

``` r
post_coda  <- getSample(fit_mcmc, coda = TRUE, start = n_burnin + 1, thin = 5)
gelman.diag(post_coda)    # R-hat
effectiveSize(post_coda)  # n_eff
```

### Posterior Predictive Check

Parameter vectors are drawn from the posterior and used to simulate the stochastic model. The resulting 95% simulation envelope is compared against the observed data to assess whether the model reproduces the observed case dynamics.

------------------------------------------------------------------------

## References

- King, A.A., Nguyen, D., Ionides, E.L. (2016). Statistical inference for partially observed Markov processes via the R package pomp. *JASA* 111, 552–563.
- ter Braak, C.J.F., Vrugt, J.A. (2008). Differential Evolution Markov Chain with snooker updater and fewer chains. *Statistics and Computing* 18, 435–446.
