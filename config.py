"""Settings of the experiments of Section 5 of the note, of Section S4 of its supplementary material, and of their
repetition with stratified resampling."""
from models import gordon_salmond_smith_model, linear_gaussian_model

# entropy of the root seed sequence; the seed sequence of each batch is derived from it with the
# spawn key (index of the experiment, number of particles, index of the batch)
SEED = 20261026

NUMBERS_OF_PARTICLES = (2, 4, 8, 16, 32, 64)

# each experiment and number of particles uses N_BATCHES * BATCH_SIZE data sets
N_BATCHES = 2000
BATCH_SIZE = 500

# selection rules of all the experiments, keys of csmc.ALGORITHM_1_SELECTORS and csmc.ALGORITHM_S1_SELECTORS; ratios
# are taken relative to BASELINE
RULES = ("BS", "FM-BS", "M-BS", "CM-BS")
BASELINE = "BS"

# linear-Gaussian (LG) model: x_1 ~ N(0, sigma^2 / (1 - rho^2) I_d), x_t = rho x_{t-1} + sigma eps_t,
# y_t = x_t + sigma eta_t
LG_RHO = 0.9
LG_SIGMA = 1.0
LG_T = 200
LG_DIMENSIONS = (1, 3)

# nonlinear (NL) model of Gordon, Salmond and Smith (1993), with the constants of Andrieu, Doucet and
# Holenstein (2010, Section 3.1): x_1 ~ N(0, 5), x_t = x_{t-1} / 2 + 25 x_{t-1} / (1 + x_{t-1}^2)
# + 8 cos(1.2 t) + sigma_V eps_t, y_t = x_t^2 / 20 + sigma_W eta_t
NL_INITIAL_VARIANCE = 5.0
NL_TRANSITION_PARAMS = (0.5, 25.0, 8.0, 1.2)
NL_TRANSITION_VARIANCE = 10.0
NL_OBSERVATION_PARAMS = (1.0 / 20.0,)
NL_OBSERVATION_VARIANCES = (10.0, 1.0)
NL_T = 100

# experiments of the note, with the bootstrap filter: name -> (T, d, model, label)
EXPERIMENTS = {
    **{f"lg_d{d}": (LG_T, d, linear_gaussian_model(LG_RHO, LG_SIGMA), rf"LG, $d = {d}$") for d in LG_DIMENSIONS},
    **{f"nl_w{v:g}": (NL_T, 1,
                      gordon_salmond_smith_model(NL_INITIAL_VARIANCE, NL_TRANSITION_PARAMS, NL_TRANSITION_VARIANCE,
                                                 NL_OBSERVATION_PARAMS, v),
                      rf"NL, $\sigma_W^2 = {v:g}$")
       for v in NL_OBSERVATION_VARIANCES},
}

# experiment of the supplementary material, with potentials which depend on the previous state: the LG model with the
# locally optimal proposal, name -> (T, d, model, label)
LOCALLY_OPTIMAL_EXPERIMENTS = {f"lg_d{d}_locally_optimal": (LG_T, d, linear_gaussian_model(LG_RHO, LG_SIGMA),
                                                            rf"LG, $d = {d}$")
                               for d in LG_DIMENSIONS}

# repetition of these experiments with stratified resampling (Kitagawa, 1996) instead of multinomial resampling, with
# Algorithm S1 of the supplementary material: name -> (T, d, model, label)
STRATIFIED_EXPERIMENTS = {f"{name}_stratified": setting for name, setting in EXPERIMENTS.items()}
STRATIFIED_LOCALLY_OPTIMAL_EXPERIMENTS = {f"{name}_stratified": (T, d, model, f"{label}, locally optimal")
                                          for name, (T, d, model, label) in LOCALLY_OPTIMAL_EXPERIMENTS.items()}
