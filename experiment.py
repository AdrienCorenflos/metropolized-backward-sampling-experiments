"""Experiments of Section 5 of the note and of Section S4 of its supplementary material, and their repetition with
stratified resampling: one iteration of conditional SMC from stationarity.

For each experiment of the groups GROUPS, with the rules config.RULES, and each number of particles N, data sets
(x_{1:T}, y_{1:T}) are drawn from the model, so that x_{1:T} is a draw from the smoothing distribution given
y_{1:T}. For each data set, the forward pass of the group is run once with the reference trajectory x_{1:T}, which
gives the indices b_1, ..., b_T of the reference, and each rule is applied to its particles, in the order of the
rules, which gives the indices l_1, ..., l_T of the output trajectory x'_{1:T}. The update indicators 1{l_t != b_t},
which record whether the index selected at time t differs from the index of the reference, the squared jumps
||x'_t - x_t||^2, and the squared norms ||x'_t||^2 and ||x_t||^2 are averaged over the data sets of each batch, and
the batch means are saved to results/<experiment>_N<N>.npz. Under stationarity, ||x'_t||^2 and ||x_t||^2 have the same
expectation.

Usage: python3 experiment.py <number of processes> <indices of configurations in CONFIGURATIONS>
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

from config import (BATCH_SIZE, EXPERIMENTS, LOCALLY_OPTIMAL_EXPERIMENTS, N_BATCHES, NUMBERS_OF_PARTICLES, RULES, SEED,
                    STRATIFIED_EXPERIMENTS, STRATIFIED_LOCALLY_OPTIMAL_EXPERIMENTS)
from csmc import (ALGORITHM_1_SELECTORS, ALGORITHM_S1_SELECTORS, conditional_forward_pass, conditional_forward_passes,
                  locally_optimal_ancestor_sampling_forward_pass,
                  locally_optimal_stratified_ancestor_sampling_forward_pass, simulate_data_sets,
                  stratified_ancestor_sampling_forward_pass)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# groups of experiments of config, with their forward pass and the selections of csmc of the rules: the bootstrap
# filter with Algorithm 1 (Section 5 of the note), and, with Algorithm S1 of its supplementary material, the locally
# optimal proposal (Section S4) and the two filters with stratified resampling
GROUPS = ((EXPERIMENTS, conditional_forward_pass, ALGORITHM_1_SELECTORS),
          (LOCALLY_OPTIMAL_EXPERIMENTS, locally_optimal_ancestor_sampling_forward_pass, ALGORITHM_S1_SELECTORS),
          (STRATIFIED_EXPERIMENTS, stratified_ancestor_sampling_forward_pass, ALGORITHM_S1_SELECTORS),
          (STRATIFIED_LOCALLY_OPTIMAL_EXPERIMENTS, locally_optimal_stratified_ancestor_sampling_forward_pass,
           ALGORITHM_S1_SELECTORS))
# all the experiments: name -> (T, d, model, label, forward pass, selection rules as a dictionary rule -> function of
# csmc), in the order of GROUPS, so that the seeds of an experiment do not depend on the later experiments
ALL_EXPERIMENTS = {name: (*setting, forward_pass, {rule: selectors[rule] for rule in RULES})
                   for experiments, forward_pass, selectors in GROUPS for name, setting in experiments.items()}
# configurations: (name of the experiment, number of particles)
CONFIGURATIONS = [(name, N) for name in ALL_EXPERIMENTS for N in NUMBERS_OF_PARTICLES]


def configuration_indices(experiments):
    """Indices in CONFIGURATIONS of the configurations of the given experiments.

    Parameters
    ----------
    experiments : dict
        Experiments, with the names of ALL_EXPERIMENTS as keys.

    Returns
    -------
    list of int
        Indices of the configurations whose experiment is a key of experiments.
    """
    return [index for index, (name, _) in enumerate(CONFIGURATIONS) if name in experiments]


def run_batch(job):
    """Batch means of the update indicators, the squared jumps, and the squared norms, for one batch of
    data sets.

    Parameters
    ----------
    job : tuple
        (name of the experiment, number of particles N, index of the batch).

    Returns
    -------
    job : tuple
        The argument.
    update_means : ndarray, shape (number of rules of the experiment, T)
        For each rule and time step t, the average over the data sets of 1{l_t != b_t}.
    jump_means : ndarray, shape (number of rules of the experiment, T)
        For each rule and time step t, the average over the data sets of ||x'_t - x_t||^2.
    square_means : ndarray, shape (number of rules of the experiment, T)
        For each rule and time step t, the average over the data sets of ||x'_t||^2.
    reference_square_means : ndarray, shape (T,)
        For each time step t, the average over the data sets of ||x_t||^2.
    """
    name, N, batch = job
    T, d, model, _, forward_pass, rules = ALL_EXPERIMENTS[name]
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(list(ALL_EXPERIMENTS).index(name), N, batch)))
    x, y = simulate_data_sets(BATCH_SIZE, T, d, model, rng)
    forward = conditional_forward_passes(forward_pass, x, y, N, model, rng)
    particles, reference_indices = forward[0], forward[4]
    data_sets, times = np.arange(BATCH_SIZE)[:, None], np.arange(T)[None, :]
    update_means = np.empty((len(rules), T))
    jump_means = np.empty((len(rules), T))
    square_means = np.empty((len(rules), T))
    for r, selector in enumerate(rules.values()):
        indices = selector(*forward, model, rng)
        output = particles[data_sets, times, indices]
        update_means[r] = np.mean(indices != reference_indices, axis=0)
        jump_means[r] = np.mean(np.sum((output - x) ** 2, axis=2), axis=0)
        square_means[r] = np.mean(np.sum(output ** 2, axis=2), axis=0)
    return job, update_means, jump_means, square_means, np.mean(np.sum(x ** 2, axis=2), axis=0)


def run_configurations(configurations, n_processes):
    """Run all the batches of the given configurations with one pool of processes, and save the batch
    means of each configuration to results/<name>_N<N>.npz.

    Parameters
    ----------
    configurations : list of tuple
        Configurations (name of the experiment, number of particles N), elements of CONFIGURATIONS.
    n_processes : int
        Number of processes which run the batches.
    """
    statistics = ("update", "jump", "square")
    batch_means = {(statistic, name, N): np.empty((N_BATCHES, len(ALL_EXPERIMENTS[name][5]), ALL_EXPERIMENTS[name][0]))
                   for statistic in statistics for name, N in configurations}
    reference_square = {(name, N): np.empty((N_BATCHES, ALL_EXPERIMENTS[name][0])) for name, N in configurations}
    jobs = [(name, N, batch) for name, N in configurations for batch in range(N_BATCHES)]
    with Pool(n_processes) as pool:
        for (name, N, batch), *means, reference_square_means in pool.imap_unordered(run_batch, jobs):
            for statistic, statistic_means in zip(statistics, means):
                batch_means[statistic, name, N][batch] = statistic_means
            reference_square[name, N][batch] = reference_square_means
    os.makedirs(RESULTS, exist_ok=True)
    for name, N in configurations:
        np.savez(os.path.join(RESULTS, f"{name}_N{N}.npz"), rules=np.array(list(ALL_EXPERIMENTS[name][5])),
                 **{f"{statistic}_batch_means": batch_means[statistic, name, N] for statistic in statistics},
                 reference_square_batch_means=reference_square[name, N])


if __name__ == "__main__":
    run_configurations([CONFIGURATIONS[int(index)] for index in sys.argv[2:]], int(sys.argv[1]))
