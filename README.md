# Metropolized backward sampling: experiments

This repository contains the code of the experiments of the paper *Metropolized backward sampling for conditional
sequential Monte Carlo* (Adrien Corenflos) and of its supplementary material. In each experiment, one iteration of
conditional sequential Monte Carlo (CSMC) is run from stationarity with the selection rules BS (backward sampling),
FM-BS (forced move plus backward sampling), M-BS (Metropolized backward sampling), and CM-BS (coupled Metropolized
backward sampling), on a linear-Gaussian model or on the nonlinear model of Gordon, Salmond, and Smith (1993). The
update rates and the expected squared jumped distances (ESJD) are estimated at each time step.

## Contents

- `csmc.py`: Algorithm 1 of the paper and Algorithm S1 of its supplementary material, in numba, for state-space models
  with Gaussian noise. In a forward pass (`bootstrap_forward_pass`, or `locally_optimal_forward_pass` for the
  locally optimal proposal, whose potentials depend on the previous state), the ancestor of each state of the
  reference is drawn with an ancestor kernel (`identity_kernel` for Algorithm 1, `independent_draw` for the ancestor
  sampling of Algorithm S1). The index of the reference and the ancestor indices are then drawn with a conditional
  resampling scheme: multinomial (`conditional_multinomial_resampling`, as in the `particles` package, with the reference at index 0) or
  stratified (`conditional_stratified_resampling`, Algorithm 6 of Finke et al., 2026). The kernels of BS, FM-BS, and
  M-BS start from the ancestors after the forward pass of Algorithm 1 (`select`), or from the indices of the reference
  (`select_from_reference`). The kernels of CM-BS, forced moves on the lineage of the reference and the coupling
  kernel C_{p,q} off it, are given by `select_coupled`.
- `models.py`: the linear-Gaussian model and the nonlinear model of Gordon, Salmond, and Smith (1993).
- `config.py`: all the settings of the experiments (model constants, numbers of particles, numbers of data sets, and
  seed), in four groups of experiments.
- `experiment.py`: the experiments. For each experiment and number of particles, data sets (x, y) are drawn from
  the model. For each data set, the forward pass of the experiment is run once from x, with the four selection rules
  applied to its particles. The batch means of the update indicators, of the squared jumps, and of the squared norms
  of the reference and output states are saved to `results/`, which is not tracked by git.
- `estimates.py`: means and ratios of means, with their standard errors, from batch means.
- `figures.py`: the figures and the rows of the tables of the paper in `fig/`, and the estimates quoted in the paper,
  with the check of the orderings and of the stationarity for the experiments with Algorithm S1, in `report.txt`.
- `tests/`: unit tests (pytest), with `pytest.ini`, which puts this directory on the import path.
- `avon_env.sh`, `avon_setup.sh`, `avon.slurm`: the experiments on the Avon cluster of the University of Warwick, as
  Slurm arrays of one task per configuration.
- `requirements.txt`: the pinned Python packages.

## Experiments

The four groups of experiments of `config.py` are those of the paper:

| Group | Paper | Configurations | Results |
|---|---|---|---|
| `EXPERIMENTS` | Section 5, bootstrap filter, Algorithm 1 | 24 | Figure 1, Table S1 |
| `LOCALLY_OPTIMAL_EXPERIMENTS` | Section S4, locally optimal proposal, Algorithm S1 | 12 | Figure S1, Table S2 |
| `STRATIFIED_EXPERIMENTS` | Section S5, bootstrap filter, stratified resampling, Algorithm S1 | 24 | Table S3 |
| `STRATIFIED_LOCALLY_OPTIMAL_EXPERIMENTS` | Section S5, locally optimal proposal, stratified resampling, Algorithm S1 | 12 | Table S3 |

A configuration is an experiment and a number of particles N ∈ {2, 4, 8, 16, 32, 64}. Each configuration uses
10^6 data sets, in 2,000 batches of 500. The seed sequence of each batch is derived from `config.SEED` with the
index of the experiment, N, and the index of the batch, so that the results do not depend on the number of processes
or on the order in which the batches run.

## Reproducing the results

Install the pinned packages, with Python 3.12, in a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run the tests:

```
python3 -m pytest
```

Run the 72 configurations, here with 16 processes, and write the figures, the table rows, and the report:

```
python3 experiment.py 16 $(seq 0 71)
python3 figures.py > report.txt
```

The indices of the configurations of one group are given by `experiment.configuration_indices`, for example
`experiment.configuration_indices(config.STRATIFIED_EXPERIMENTS)`. The outputs of `figures.py` are Figure 1
(`fig/ratios.pdf`, with the legend samples of its caption in `fig/ratios_legend.tex`), Figure S1
(`fig/locally_optimal_ratios.pdf`), and the rows of Tables S1, S2, and S3 (`fig/table_rows.tex`,
`fig/locally_optimal_table_rows.tex`, and `fig/stratified_table_rows.tex`).

Only the Slurm jobs are run on Avon. The tests are run locally. Once, on Avon, clone this repository and run
`./avon_setup.sh` to create the virtual environment with uv. Then, for each group, from the root of the clone:

```
git pull
sbatch --array=0-23 avon.slurm EXPERIMENTS
sbatch --array=0-11 avon.slurm LOCALLY_OPTIMAL_EXPERIMENTS
sbatch --array=0-23 avon.slurm STRATIFIED_EXPERIMENTS
sbatch --array=0-11 avon.slurm STRATIFIED_LOCALLY_OPTIMAL_EXPERIMENTS
```

Each task fails unless the array has one task per configuration of the group. Copy the results back with rsync,
and run `figures.py` locally. On another Slurm cluster, change the account, the partition, and the modules of
`avon.slurm` and `avon_env.sh`.
