"""Figure 1 of the note, Figure S1 and the rows of the tables of its supplementary material (the averages over the
time steps of the experiments of the note, of those with the locally optimal proposal, and of both with stratified
resampling), from the results of experiment.py, and a report of the estimates quoted in the texts."""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from config import (BASELINE, EXPERIMENTS, LOCALLY_OPTIMAL_EXPERIMENTS, NUMBERS_OF_PARTICLES, RULES,
                    STRATIFIED_EXPERIMENTS, STRATIFIED_LOCALLY_OPTIMAL_EXPERIMENTS)
from estimates import mean_and_standard_error, ratio_and_standard_error

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "fig")

# statistics saved by experiment.py, and their names in the figure
STATISTICS = {"update": "update rates", "jump": "ESJD"}
# colours of the rules compared with the baseline in the figures: vermillion, bluish green, and blue of the palette of
# Okabe and Ito (2008), which remain distinct under the common colour-vision deficiencies
COLOURS = {"FM-BS": "#D55E00", "M-BS": "#009E73", "CM-BS": "#0072B2"}
# rules compared with the baseline in the figures, with their colours
COMPARED_RULES = {rule: COLOURS[rule] for rule in RULES if rule != BASELINE}
# figures: name of the file in fig/ -> experiments, one per column; the legend samples of the first, in
# fig/<name>_legend.tex, are copied into the preambles of the note and of its supplementary material
FIGURES = {"ratios": EXPERIMENTS, "locally_optimal_ratios": LOCALLY_OPTIMAL_EXPERIMENTS}
# tables: file name in fig/ -> experiments, one block of rows each
TABLES = {"table_rows.tex": EXPERIMENTS, "locally_optimal_table_rows.tex": LOCALLY_OPTIMAL_EXPERIMENTS,
          "stratified_table_rows.tex": {**STRATIFIED_EXPERIMENTS, **STRATIFIED_LOCALLY_OPTIMAL_EXPERIMENTS}}
# groups of experiments with Algorithm S1 of the supplementary material, for which Proposition S2 extends the
# orderings of Theorem 1 of the note
ALGORITHM_S1_GROUPS = (LOCALLY_OPTIMAL_EXPERIMENTS, STRATIFIED_EXPERIMENTS, STRATIFIED_LOCALLY_OPTIMAL_EXPERIMENTS)
# pairs (higher, lower) of rules whose differences of update rates and ESJD these orderings predict to be nonnegative
ORDERED_PAIRS = (("M-BS", "FM-BS"), ("CM-BS", "FM-BS"), ("FM-BS", "BS"))
# threshold of the z-scores counted in the check of the orderings
CHECK_Z = 3.0
# numbers of particles shown in the figure, and their line styles
FIGURE_PARTICLES = {2: "-", 8: ":"}
# half-width of the shaded bands, in standard errors
BAND_HALF_WIDTH = 2.0
# opacity of the shaded bands
BAND_ALPHA = 0.3
# figure size in inches; the text width of the [sts] layout is 528pt
FIGURE_SIZE = (7.3, 3.1)
# legend samples of the captions: TikZ styles of the matplotlib line styles, line width as in STYLE, and length
TIKZ_LINE_STYLES = {"-": "solid", ":": "densely dotted"}
LEGEND_LINE_WIDTH = "1pt"
LEGEND_LENGTH = "16pt"
# names of the digits, for the names of the LaTeX commands of the legend samples, which cannot contain digits
DIGIT_NAMES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
# format of the averages in the table: decimals for the update rates, significant digits for the ESJD
TABLE_FORMATS = {"update": ".3f", "jump": "#.3g"}
# number of last time steps at which the report gives the ratios
FINAL_STEPS = 4
STYLE = {"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.0, "mathtext.fontset": "cm"}


def load(name, N):
    """Batch means saved by experiment.py.

    Parameters
    ----------
    name : str
        Name of the experiment, a key of config.EXPERIMENTS or config.LOCALLY_OPTIMAL_EXPERIMENTS.
    N : int
        Number of particles.

    Returns
    -------
    batch_means : dict
        For each key of STATISTICS, the batch means, of shape (B, number of rules, T).
    rules : tuple of str
        Rules of the experiment, in the order of the second axis of the batch means.
    """
    results = np.load(os.path.join(RESULTS, f"{name}_N{N}.npz"))
    return {statistic: results[statistic + "_batch_means"] for statistic in STATISTICS}, tuple(results["rules"])


def ratios_to_baseline(batch_means, rules, rule):
    """Ratio to the baseline rule of the means of a statistic, and its standard error.

    Parameters
    ----------
    batch_means : ndarray, shape (B, len(rules), ...)
        Batch means of the statistic for all the rules.
    rules : tuple of str
        Rules, in the order of the second axis of batch_means.
    rule : str
        Element of rules.

    Returns
    -------
    ratio, standard_error : ndarray, shape (...)
        See estimates.ratio_and_standard_error.
    """
    return ratio_and_standard_error(batch_means[:, rules.index(rule)], batch_means[:, rules.index(BASELINE)])


def figure(experiments, compared_rules, file_name):
    """Ratios to the baseline rule of the update rates and of the ESJD at each time step, with bands
    of BAND_HALF_WIDTH standard errors.

    Parameters
    ----------
    experiments : dict
        Experiments of config, one per column.
    compared_rules : dict
        Rules compared with the baseline, and their colours.
    file_name : str
        Name of the file in fig/.
    """
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(len(STATISTICS), len(experiments), figsize=FIGURE_SIZE, sharex="col")
    for column, (name, (T, _, _, label)) in enumerate(experiments.items()):
        times = np.arange(1, T + 1)
        for N, line_style in FIGURE_PARTICLES.items():
            batch_means, rules = load(name, N)
            for row, statistic in enumerate(STATISTICS):
                for rule, colour in compared_rules.items():
                    ratio, standard_error = ratios_to_baseline(batch_means[statistic], rules, rule)
                    axes[row, column].fill_between(times, ratio - BAND_HALF_WIDTH * standard_error,
                                                   ratio + BAND_HALF_WIDTH * standard_error, color=colour,
                                                   alpha=BAND_ALPHA, linewidth=0)
                    axes[row, column].plot(times, ratio, color=colour, linestyle=line_style)
        axes[0, column].set_title(label)
        axes[-1, column].set_xlabel("time step $t$")
    for row, statistic_name in enumerate(STATISTICS.values()):
        axes[row, 0].set_ylabel(f"ratio of {statistic_name}")
    for ax in axes.flat:
        ax.axhline(1.0, color="0.4", linewidth=0.6)
        ax.grid(alpha=0.3, linewidth=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, file_name))
    plt.close(fig)


def legend_name(rule, N):
    """Name of the LaTeX command of the legend sample of a rule and a number of particles.

    Parameters
    ----------
    rule : str
        Name of the rule.
    N : int
        Number of particles.

    Returns
    -------
    str
        For example legMBSNtwo for the rule M-BS and N = 2.
    """
    return "leg" + rule.replace("-", "") + "N" + "".join(DIGIT_NAMES[int(digit)] for digit in str(N))


def legend_samples(compared_rules, file_name):
    """Robust LaTeX commands which draw the line of each rule and number of particles of a figure, for the
    legend in its caption, as in langevin_csmc.tex.

    Parameters
    ----------
    compared_rules : dict
        Rules compared with the baseline in the figure, and their colours.
    file_name : str
        Name of the file in fig/.
    """
    lines = ["% Generated by code/figures.py: legend samples of the lines of the figure, for use in its caption.",
             "% Requires \\RequirePackage{tikz}."]
    for rule, colour in compared_rules.items():
        red, green, blue = (int(round(255 * channel)) for channel in to_rgb(colour))
        tikz_colour = f"{{rgb,255:red,{red};green,{green};blue,{blue}}}"
        for N, line_style in FIGURE_PARTICLES.items():
            lines.append(f"\\DeclareRobustCommand{{\\{legend_name(rule, N)}}}{{\\tikz[baseline=-0.6ex]{{\\draw[color={tikz_colour}, "
                         f"line width={LEGEND_LINE_WIDTH}, {TIKZ_LINE_STYLES[line_style]}] (0,0) -- ({LEGEND_LENGTH},0);}}}}")
    with open(os.path.join(FIG, file_name), "w") as f:
        f.write("\n".join(lines) + "\n")


def rule_macro(rule):
    """LaTeX macro of a rule in the note, for example \\MBS{} for M-BS.

    Parameters
    ----------
    rule : str
        Name of the rule.

    Returns
    -------
    str
        The macro, with braces.
    """
    return "\\" + rule.replace("-", "") + "{}"


def table_rows(experiments, file_name):
    """Rows of a table: for each experiment, a block with one row per rule of config.RULES, which gives, for each
    statistic and number of particles, the average of the statistic over the time steps.

    Parameters
    ----------
    experiments : dict
        Experiments of config, one block of rows each.
    file_name : str
        Name of the file in fig/.
    """
    n_columns = 1 + len(STATISTICS) * len(NUMBERS_OF_PARTICLES)
    blocks = []
    for name, (_, _, _, label) in experiments.items():
        results = {N: load(name, N) for N in NUMBERS_OF_PARTICLES}
        rows = [rf"\multicolumn{{{n_columns}}}{{@{{}}l}}{{\emph{{{label}}}}} \\"]
        for rule in RULES:
            cells = [rule_macro(rule)]
            for statistic in STATISTICS:
                for N in NUMBERS_OF_PARTICLES:
                    batch_means, experiment_rules = results[N]
                    mean, _ = mean_and_standard_error(batch_means[statistic].mean(axis=2))
                    cells.append(f"{mean[experiment_rules.index(rule)]:{TABLE_FORMATS[statistic]}}")
            rows.append(" & ".join(cells) + r" \\")
        blocks.append("\n".join(rows))
    with open(os.path.join(FIG, file_name), "w") as f:
        f.write("\n\\addlinespace\n".join(blocks) + "\n")


def report(experiments):
    """Print the estimates quoted in the text: for each rule compared with the baseline, the ratio of
    the averages over the time steps, the ratios at the last FINAL_STEPS time steps, and the range of
    the ratios over the time steps; and the largest standard errors of the entries of the tables.

    Parameters
    ----------
    experiments : dict
        Experiments of config.
    """
    largest_standard_errors = {statistic: 0.0 for statistic in STATISTICS}
    for name in experiments:
        for N in NUMBERS_OF_PARTICLES:
            batch_means, rules = load(name, N)
            for statistic in STATISTICS:
                mean, standard_error = mean_and_standard_error(batch_means[statistic].mean(axis=2))
                largest_standard_errors[statistic] = max(largest_standard_errors[statistic],
                                                         (standard_error if statistic == "update"
                                                          else standard_error / mean).max())
                for rule in rules:
                    average, average_standard_error = ratios_to_baseline(batch_means[statistic].mean(axis=2), rules,
                                                                         rule)
                    ratio, _ = ratios_to_baseline(batch_means[statistic], rules, rule)
                    print(f"{name} N={N:2d} {statistic:6s} {rule}/{BASELINE}: time-averaged {average:.4f} "
                          f"({average_standard_error:.4f}), last {FINAL_STEPS} steps "
                          + np.array2string(ratio[-FINAL_STEPS:], precision=3)
                          + f", range [{ratio.min():.3f}, {ratio.max():.3f}]")
    print("largest standard error of the update rates and largest relative standard error of the ESJD in the tables:",
          largest_standard_errors)


def ordering_check(experiments):
    """Print the check of the orderings and of the stationarity for a group of experiments with Algorithm S1.

    For each pair (higher, lower) of ORDERED_PAIRS, the z-scores (mean over the batches divided by its standard
    error) of the batch differences higher - lower of the update rates and of the squared jumps, over all the
    experiments, numbers of particles, and time steps: the smallest z-score, its location, and the numbers of
    z-scores below -CHECK_Z and of comparisons. For each rule, the largest absolute z-score of the batch differences
    between the squared norms of the output and reference states, which have the same expectation under
    stationarity.

    Parameters
    ----------
    experiments : dict
        Experiments of config, a group of ALGORITHM_S1_GROUPS.
    """
    print("check of " + ", ".join(experiments) + ":")
    for higher, lower in ORDERED_PAIRS:
        z_scores = {}
        for name in experiments:
            for N in NUMBERS_OF_PARTICLES:
                batch_means, rules = load(name, N)
                for statistic in STATISTICS:
                    mean, standard_error = mean_and_standard_error(batch_means[statistic][:, rules.index(higher)]
                                                                   - batch_means[statistic][:, rules.index(lower)])
                    z_scores[name, N, statistic] = mean / standard_error
        (name, N, statistic), z = min(z_scores.items(), key=lambda item: item[1].min())
        n_below = sum(int(np.sum(z_values < -CHECK_Z)) for z_values in z_scores.values())
        n_comparisons = sum(z_values.size for z_values in z_scores.values())
        print(f"ordering {higher} - {lower}: smallest z-score {z.min():.2f} ({name}, N={N}, {statistic}, "
              f"t={z.argmin() + 1}), {n_below} z-scores below -{CHECK_Z:g} among {n_comparisons} comparisons")
    for rule in RULES:
        largest = 0.0
        for name in experiments:
            for N in NUMBERS_OF_PARTICLES:
                results = np.load(os.path.join(RESULTS, f"{name}_N{N}.npz"))
                rules = tuple(results["rules"])
                mean, standard_error = mean_and_standard_error(results["square_batch_means"][:, rules.index(rule)]
                                                               - results["reference_square_batch_means"])
                largest = max(largest, np.abs(mean / standard_error).max())
        print(f"stationarity {rule}: largest absolute z-score {largest:.2f} among "
              f"{len(NUMBERS_OF_PARTICLES) * sum(T for T, _, _, _ in experiments.values())} comparisons")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    for name, experiments in FIGURES.items():
        figure(experiments, COMPARED_RULES, name + ".pdf")
    legend_samples(COMPARED_RULES, next(iter(FIGURES)) + "_legend.tex")
    for file_name, experiments in TABLES.items():
        table_rows(experiments, file_name)
    for experiments in (EXPERIMENTS, *ALGORITHM_S1_GROUPS):
        report(experiments)
    for experiments in ALGORITHM_S1_GROUPS:
        ordering_check(experiments)
