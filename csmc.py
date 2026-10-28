"""Conditional sequential Monte Carlo (Algorithm 1 of the note) with the selection rules BS, FM-BS,
M-BS, and CM-BS, for state-space models with Gaussian noise,

    x_1 ~ N(0, s_1^2 I_d),
    x_t | x_{t-1} ~ N(f(x_{t-1}, t; theta_f), s_x^2 I_d),    t = 2, ..., T,
    y_t | x_t ~ N(h(x_t; theta_h), s_y^2 I_d),                t = 1, ..., T,

with the bootstrap filter: m_t is the transition density and g_t(x_t) is the density of y_t given
x_t. A model is a tuple (s_1, f, theta_f, s_x, h, theta_h, s_y), where f(x, t, theta_f) and
h(x, theta_h) are jitted functions which act on arrays of states whose last axis has length d.
For models with h(x; theta_h) = x, the particle filter with the locally optimal proposal is also
available. Its potentials depend on the previous state (Section S3 of the supplementary material of the note).

A forward pass draws, at each time t < T, the index c of the ancestor of the reference state x_{t+1} with an ancestor
kernel started from the index b_t of the reference, and then the index b_{t+1} of the reference and the ancestor
indices a_t with a conditional resampling scheme given that a_t^{b_{t+1}} = c:
- the forward pass of Algorithm 1: identity_kernel (c = b_t) and conditional_multinomial_resampling;
- the forward pass of Algorithm S1 of the supplementary material: independent_draw (ancestor sampling,
  c ~ B_t(. | x_{t+1})), and conditional_multinomial_resampling or conditional_stratified_resampling.
With multinomial resampling, the reference has index 0 at every time step (fixed indices, Section 2.1 of the note).
The selection rules then start from the indices of the reference, as in Algorithm 1.c to 1.e, or, for BS, FM-BS, and
M-BS after the forward pass of Algorithm 1, from the ancestors (the usual implementation of Section 2.1 of the note).

Time indices t = 1, ..., T are those of the note; arrays are indexed from 0, so that the state at
time t is x[t - 1]. All weights are handled through their logarithms.
"""
import numpy as np
from numba import njit


@njit
def log_sum_exp_excluding(log_w, i):
    """Logarithm of the sum of the exponentials of the entries of a vector, except the entry i.

    Parameters
    ----------
    log_w : ndarray, shape (n,)
        Entries in [-inf, inf).
    i : int
        Excluded index.

    Returns
    -------
    float
        log(sum_{k != i} exp(log_w[k])), which is -inf if all the entries other than i are -inf.
    """
    m = -np.inf
    for k in range(log_w.shape[0]):
        if k != i:
            m = max(m, log_w[k])
    if m == -np.inf:
        return -np.inf
    total = 0.0
    for k in range(log_w.shape[0]):
        if k != i:
            total += np.exp(log_w[k] - m)
    return m + np.log(total)


@njit
def sample_index(log_w, rng):
    """One draw from the categorical distribution on {0, ..., n - 1} with probabilities proportional
    to exp(log_w), by inversion.

    The draw is the smallest index k with u < c_k, where c_k is the sum of the weights of the indices
    0, ..., k and u is uniform on [0, c_{n-1}); the partial sums are accumulated in the same order as
    c_{n-1}, so that the search stops at an index k <= n - 1. An index with log-weight -inf does not
    increase the partial sum, and is therefore never drawn.

    Parameters
    ----------
    log_w : ndarray, shape (n,)
        Log-weights in [-inf, inf), at least one of which is finite.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    int
        Drawn index.
    """
    m = np.max(log_w)
    total = 0.0
    for k in range(log_w.shape[0]):
        total += np.exp(log_w[k] - m)
    u = rng.random() * total
    partial_sum = 0.0
    k = -1
    while u >= partial_sum:
        k += 1
        partial_sum += np.exp(log_w[k] - m)
    return k


@njit
def sample_index_excluding(log_w, i, rng):
    """One draw from the categorical distribution on the indices {0, ..., n - 1} other than i, with
    probabilities proportional to exp(log_w), by inversion, as in sample_index with the weight of i
    set to zero.

    Parameters
    ----------
    log_w : ndarray, shape (n,)
        Log-weights in [-inf, inf), at least one of which, other than the entry i, is finite.
    i : int
        Excluded index.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    int
        Drawn index, different from i.
    """
    m = -np.inf
    for k in range(log_w.shape[0]):
        if k != i:
            m = max(m, log_w[k])
    total = 0.0
    for k in range(log_w.shape[0]):
        if k != i:
            total += np.exp(log_w[k] - m)
    u = rng.random() * total
    partial_sum = 0.0
    k = -1
    while u >= partial_sum:
        k += 1
        if k != i:
            partial_sum += np.exp(log_w[k] - m)
    return k


@njit
def multinomial_resampling(log_w, size, rng):
    """Multinomial resampling: size independent draws from the categorical distribution on
    {0, ..., n - 1} with probabilities proportional to exp(log_w), returned in increasing order.

    As in the multinomial resampling of the particles package
    (https://github.com/nchopin/particles/blob/master/particles/resampling.py), the draws are given
    by a single pass of the inverse CDF algorithm through sorted uniform variates, which are the
    normalized cumulative sums of size + 1 independent standard exponential variates. Here the
    uniform variates are multiplied by the sum c_{n-1} of the unnormalized weights instead of
    normalizing the weights, and the partial sums are accumulated in the same order as c_{n-1}, so
    that the pass stops at an index k <= n - 1.

    Parameters
    ----------
    log_w : ndarray, shape (n,)
        Finite log-weights.
    size : int
        Number of draws.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    ndarray of int64, shape (size,)
        Drawn indices, in increasing order.
    """
    cumulative_exponentials = np.cumsum(rng.standard_exponential(size + 1))
    m = np.max(log_w)
    total = 0.0
    for k in range(log_w.shape[0]):
        total += np.exp(log_w[k] - m)
    indices = np.empty(size, dtype=np.int64)
    k = 0
    partial_sum = np.exp(log_w[0] - m)
    for draw in range(size):
        u = cumulative_exponentials[draw] / cumulative_exponentials[size] * total
        while u > partial_sum:
            k += 1
            partial_sum += np.exp(log_w[k] - m)
        indices[draw] = k
    return indices


@njit
def conditional_multinomial_resampling(log_w, c, rng):
    """Conditional multinomial resampling given that the ancestor of the particle of the reference is c, with the
    index of the reference fixed to 0: a^0 = c, and a^1, ..., a^{N-1} are drawn with multinomial_resampling.

    With multinomial resampling, the index of the reference is uniform and independent of the other variables, and
    the other ancestor indices are independent draws from the weights (Section S3.2 of the supplementary material of
    the note), so that the particles may be relabelled with the reference at index 0 and the other ancestor indices
    in increasing order.

    Parameters
    ----------
    log_w : ndarray, shape (N,)
        Finite log-weights of the particles at time t.
    c : int
        Index of the ancestor of the particle of the reference at time t + 1.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    reference_index : int
        Index b_{t+1} = 0 of the reference at time t + 1.
    ancestors : ndarray of int64, shape (N,)
        Ancestor indices a_t^0, ..., a_t^{N-1}.
    """
    N = log_w.shape[0]
    ancestors = np.empty(N, dtype=np.int64)
    ancestors[0] = c
    ancestors[1:] = multinomial_resampling(log_w, N - 1, rng)
    return 0, ancestors


@njit
def conditional_stratified_resampling(log_w, c, rng):
    """Conditional stratified resampling of Finke et al. (2026, Algorithm 6) given that the ancestor of the particle
    of the reference is c.

    With the cumulative weights F(i) = W^0 + ... + W^i, F(-1) = 0, stratified resampling (Kitagawa, 1996) sets
    a^m = F^{-1}((m + U_m) / N), m = 0, ..., N - 1, the smallest index i with (m + U_m) / N <= F(i), for independent
    uniforms U_m. Given that a^b = c for the index b of the reference, b is drawn with probabilities proportional to
    the lengths of the intersections of the strata [m / N, (m + 1) / N) with (F(c - 1), F(c)], a^b is set to c, which
    holds with probability 1 in Algorithm 6, and the other ancestor indices are drawn as in stratified resampling, by
    one pass of the inverse transform through the increasing points (m + U_m) / N (Finke et al., 2026, Algorithm 7).
    The weights are not normalized: the points are multiplied by the total weight, after the division by N, so that
    they do not exceed the last partial sum, which is accumulated in the same order as the total weight.

    Parameters
    ----------
    log_w : ndarray, shape (N,)
        Finite log-weights of the particles at time t.
    c : int
        Index of the ancestor of the particle of the reference at time t + 1.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    reference_index : int
        Index b_{t+1} of the reference at time t + 1.
    ancestors : ndarray of int64, shape (N,)
        Ancestor indices a_t^0, ..., a_t^{N-1}, in increasing order.
    """
    N = log_w.shape[0]
    m = np.max(log_w)
    partial_sums = np.empty(N)
    total = 0.0
    for i in range(N):
        total += np.exp(log_w[i] - m)
        partial_sums[i] = total
    lower = partial_sums[c - 1] if c > 0 else 0.0
    log_overlaps = np.empty(N)
    for stratum in range(N):
        log_overlaps[stratum] = np.log(max(0.0, min(partial_sums[c], (stratum + 1) / N * total)
                                           - max(lower, stratum / N * total)))
    reference_index = sample_index(log_overlaps, rng)
    ancestors = np.empty(N, dtype=np.int64)
    i = 0
    for stratum in range(N):
        if stratum == reference_index:
            ancestors[stratum] = c
        else:
            u = (stratum + rng.random()) / N * total
            while u > partial_sums[i]:
                i += 1
            ancestors[stratum] = i
    return reference_index, ancestors


@njit
def first_index(N, rng):
    """Index 0 of the reference at time 1, which may be fixed when the resampling scheme is multinomial (Section 2.1
    of the note).

    Parameters
    ----------
    N : int
        Number of particles.
    rng : numpy.random.Generator
        Random number generator, which is not used.

    Returns
    -------
    int
        0.
    """
    return 0


@njit
def uniform_index(N, rng):
    """Index of the reference at time 1, drawn uniformly on {0, ..., N - 1} as in Algorithm S1.a of the supplementary
    material of the note.

    Parameters
    ----------
    N : int
        Number of particles.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    int
        Drawn index.
    """
    return rng.integers(0, N)


@njit
def identity_kernel(log_w, i, rng):
    """One step from i of the identity kernel, the kernel K_{t,j,j} of AT and FM-AT in Section 2.1 of the note, which
    is the ancestor kernel of the forward pass of Algorithm 1.

    Parameters
    ----------
    log_w : ndarray, shape (n,)
        Log-weights, which are not used.
    i : int
        Current index.
    rng : numpy.random.Generator
        Random number generator, which is not used.

    Returns
    -------
    int
        i.
    """
    return i


@njit
def independent_draw(log_w, i, rng):
    """One step from i of the Gibbs kernel K(i, .) = p of Section 2.2 of the note, for the probabilities
    p proportional to exp(log_w): a draw from p, which does not depend on i.

    Parameters
    ----------
    log_w : ndarray, shape (n,)
        Log-weights in [-inf, inf), at least one of which is finite.
    i : int
        Current index.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    int
        New index.
    """
    return sample_index(log_w, rng)


@njit
def forced_move(log_w, i, rng):
    """One step from i of the forced-move kernel F_p of Liu (1996), Section 2.2 of the note, for
    the probabilities p proportional to exp(log_w).

    If p_i < 1, an index j != i is proposed with probability p_j / (1 - p_i) and accepted with probability
    min{1, (1 - p_i) / (1 - p_j)}. Up to the same additive constant, log(1 - p_i) and log(1 - p_j) are the
    log-sum-exps of log_w without its entries i and j. If p_i = 1, that is, if all the entries of log_w other
    than i are -inf, then F_p(i, .) is the point mass at i, as in Section 2.2 of the note.

    Parameters
    ----------
    log_w : ndarray, shape (n,)
        Log-weights in [-inf, inf), at least one of which is finite, n >= 2.
    i : int
        Current index.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    int
        New index.
    """
    log_rest = log_sum_exp_excluding(log_w, i)
    if log_rest == -np.inf:
        return i
    j = sample_index_excluding(log_w, i, rng)
    log_acceptance_ratio = log_rest - log_sum_exp_excluding(log_w, j)
    return j if np.log(rng.random()) < log_acceptance_ratio else i


@njit
def coupling_draw(log_p, log_q, i, rng):
    """One step from i of the Markov kernel C_{p,q} of Section 2.3 of the note, for the probabilities p and q
    proportional to exp(log_p) and exp(log_q): an index is drawn from q and, if it equals i, it is replaced by a
    forced move from i for the probabilities proportional to p_k q_k. If i is distributed according to p, the new
    index is distributed according to q (Lemma 2 of the note).

    Parameters
    ----------
    log_p : ndarray, shape (n,)
        Log-weights of p, in [-inf, inf).
    log_q : ndarray, shape (n,)
        Log-weights of q, in [-inf, inf), at least one of which is finite.
    i : int
        Current index.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    int
        New index.
    """
    k = sample_index(log_q, rng)
    return forced_move(log_p + log_q, i, rng) if k == i else k


# selection rules of the note whose kernels K_{t,j,j'}, j != j', are the backward-sampling probabilities B_t(. | j'),
# as the pairs of kernels (K_T, K_{t,j,j}), acting as (log_w, i, rng) -> new index: backward sampling (BS), forced
# move plus backward sampling (FM-BS), and Metropolized backward sampling (M-BS); coupled Metropolized backward
# sampling (CM-BS) is given by select_coupled
SELECTION_RULES = {
    "BS": (independent_draw, independent_draw),
    "FM-BS": (forced_move, independent_draw),
    "M-BS": (forced_move, forced_move),
}


@njit
def log_unnormalized_gaussian_density(x, mean, sd):
    """Logarithm of the density of N(mean, sd^2 I_d) at x, up to the additive constant
    -(d / 2) log(2 pi sd^2), which cancels in all the normalized weights and acceptance ratios.

    Parameters
    ----------
    x : ndarray, shape (d,)
        Point.
    mean : ndarray, shape (d,)
        Mean.
    sd : float
        Standard deviation.

    Returns
    -------
    float
        -||x - mean||^2 / (2 sd^2).
    """
    squared_distance = 0.0
    for k in range(x.shape[0]):
        squared_distance += (x[k] - mean[k]) ** 2
    return -0.5 * squared_distance / sd ** 2


@njit
def simulate(T, d, model, rng):
    """Draw the states and the observations of the state-space model.

    Parameters
    ----------
    T : int
        Number of time steps.
    d : int
        Dimension of the states and of the observations.
    model : tuple
        (s_1, f, theta_f, s_x, h, theta_h, s_y), see the module docstring.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    x : ndarray, shape (T, d)
        States x_1, ..., x_T.
    y : ndarray, shape (T, d)
        Observations y_1, ..., y_T.
    """
    initial_sd, transition_mean, transition_params, transition_sd, observation_mean, observation_params, observation_sd = model
    x = np.empty((T, d))
    x[0] = initial_sd * rng.standard_normal(d)
    for t in range(1, T):
        x[t] = transition_mean(x[t - 1], t + 1, transition_params) + transition_sd * rng.standard_normal(d)
    y = observation_mean(x, observation_params) + observation_sd * rng.standard_normal((T, d))
    return x, y


@njit
def log_backward_weights_to(log_potentials, transition_means, transition_sd, t, x_next, out):
    """Logarithms of the backward-sampling probabilities B_t(. | x_next) of (2) in the note, and of (S3) in its
    supplementary material, up to an additive constant: log W_t^i + log m_{t+1}(x_t^i, x_next), i = 1, ..., N, where
    log W_t^i is log_potentials[t - 1, i] up to an additive constant.

    Parameters
    ----------
    log_potentials, transition_means : ndarray
        Output of a forward pass.
    transition_sd : float
        Standard deviation s_x of the transition kernels.
    t : int
        Array index t - 1 of the time step t of the note, 0 <= t - 1 <= T - 2.
    x_next : ndarray, shape (d,)
        State at time t + 1.
    out : ndarray, shape (N,)
        Output array.
    """
    for i in range(out.shape[0]):
        out[i] = log_potentials[t, i] + log_unnormalized_gaussian_density(x_next, transition_means[t, i], transition_sd)


@njit
def log_backward_weights(particles, log_potentials, transition_means, transition_sd, t, j, out):
    """Logarithms of the backward-sampling probabilities B_t(. | j) = B_t(. | x_{t+1}^j), up to an additive constant,
    with log_backward_weights_to.

    Parameters
    ----------
    particles, log_potentials, transition_means : ndarray
        Output of a forward pass.
    transition_sd : float
        Standard deviation s_x of the transition kernels.
    t : int
        Array index t - 1 of the time step t of the note, 0 <= t - 1 <= T - 2.
    j : int
        Index of the particle at time t + 1.
    out : ndarray, shape (N,)
        Output array.
    """
    log_backward_weights_to(log_potentials, transition_means, transition_sd, t, particles[t + 1, j], out)


@njit
def bootstrap_forward_pass(x, y, N, model, ancestor_kernel, conditional_resampling, initial_index, rng):
    """Conditional bootstrap filter with the reference trajectory x: the forward pass of Algorithm 1 of the note or of
    Algorithm S1 of its supplementary material, with the potentials g_t(x_t), the densities of y_t given x_t.

    The index b_1 of the reference at time 1 is drawn with initial_index. At each time t >= 2, the index c of the
    ancestor of the reference state x_t is drawn with ancestor_kernel from b_{t-1}, for the backward-sampling
    probabilities B_{t-1}(. | x_t), and then b_t and the ancestor indices a_{t-1} with conditional_resampling given c.
    The particles other than the reference are drawn from the transitions of the model.

    Parameters
    ----------
    x : ndarray, shape (T, d)
        Reference trajectory.
    y : ndarray, shape (T, d)
        Observations.
    N : int
        Number of particles, N >= 2.
    model : tuple
        (s_1, f, theta_f, s_x, h, theta_h, s_y), see the module docstring.
    ancestor_kernel : function
        Kernel of the ancestor index of the reference, as a jitted function (log_w, i, rng) -> new index:
        identity_kernel for Algorithm 1, independent_draw for the ancestor sampling of Algorithm S1.
    conditional_resampling : function
        Conditional resampling scheme, as a jitted function (log_w, c, rng) -> (index of the reference, ancestor
        indices): conditional_multinomial_resampling or conditional_stratified_resampling.
    initial_index : function
        Index of the reference at time 1, as a jitted function (N, rng) -> index: first_index or uniform_index.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    particles : ndarray, shape (T, N, d)
        particles[t - 1, n] is the particle with index n at time t.
    ancestors : ndarray of int64, shape (T - 1, N)
        ancestors[t - 1, n] is the index at time t of the ancestor of particles[t, n].
    log_potentials : ndarray, shape (T, N)
        log_potentials[t - 1, n] is log g_t(particles[t - 1, n]), up to an additive constant which
        does not depend on t and n.
    transition_means : ndarray, shape (T - 1, N, d)
        transition_means[t - 1, n] is the mean f(particles[t - 1, n], t + 1; theta_f) of the
        transition from particles[t - 1, n] to time t + 1.
    reference_indices : ndarray of int64, shape (T,)
        reference_indices[t - 1] is the index b_t of the reference at time t.
    """
    initial_sd, transition_mean, transition_params, transition_sd, observation_mean, observation_params, observation_sd = model
    T, d = x.shape
    particles = np.empty((T, N, d))
    ancestors = np.empty((T - 1, N), dtype=np.int64)
    log_potentials = np.empty((T, N))
    transition_means = np.empty((T - 1, N, d))
    reference_indices = np.empty(T, dtype=np.int64)
    log_reference_weights = np.empty(N)
    reference_indices[0] = initial_index(N, rng)
    particles[0, reference_indices[0]] = x[0]
    for n in range(N):
        if n != reference_indices[0]:
            for k in range(d):
                particles[0, n, k] = initial_sd * rng.standard_normal()
    observation_means = observation_mean(particles[0], observation_params)
    for n in range(N):
        log_potentials[0, n] = log_unnormalized_gaussian_density(y[0], observation_means[n], observation_sd)
    for t in range(1, T):
        transition_means[t - 1] = transition_mean(particles[t - 1], t + 1, transition_params)
        log_backward_weights_to(log_potentials, transition_means, transition_sd, t - 1, x[t], log_reference_weights)
        c = ancestor_kernel(log_reference_weights, reference_indices[t - 1], rng)
        reference_indices[t], ancestors[t - 1] = conditional_resampling(log_potentials[t - 1], c, rng)
        particles[t, reference_indices[t]] = x[t]
        for n in range(N):
            if n != reference_indices[t]:
                for k in range(d):
                    particles[t, n, k] = transition_means[t - 1, ancestors[t - 1, n], k] + transition_sd * rng.standard_normal()
        observation_means = observation_mean(particles[t], observation_params)
        for n in range(N):
            log_potentials[t, n] = log_unnormalized_gaussian_density(y[t], observation_means[n], observation_sd)
    return particles, ancestors, log_potentials, transition_means, reference_indices


@njit
def conditional_forward_pass(x, y, N, model, rng):
    """Steps (a) and (b) of Algorithm 1: the bootstrap filter with multinomial resampling, with the
    reference trajectory at index 0 at every time step (bootstrap_forward_pass with identity_kernel,
    conditional_multinomial_resampling, and first_index).

    Parameters
    ----------
    x, y, N, model, rng
        As in bootstrap_forward_pass.

    Returns
    -------
    particles, ancestors, log_potentials, transition_means, reference_indices : ndarray
        As in bootstrap_forward_pass, with ancestors[t - 1, 0] = 0 and reference_indices = 0.
    """
    return bootstrap_forward_pass(x, y, N, model, identity_kernel, conditional_multinomial_resampling, first_index,
                                  rng)


@njit
def stratified_ancestor_sampling_forward_pass(x, y, N, model, rng):
    """Steps (a) and (b) of Algorithm S1 of the supplementary material of the note with the bootstrap filter and
    stratified resampling (bootstrap_forward_pass with independent_draw, conditional_stratified_resampling, and
    uniform_index).

    Parameters
    ----------
    x, y, N, model, rng
        As in bootstrap_forward_pass.

    Returns
    -------
    particles, ancestors, log_potentials, transition_means, reference_indices : ndarray
        As in bootstrap_forward_pass.
    """
    return bootstrap_forward_pass(x, y, N, model, independent_draw, conditional_stratified_resampling, uniform_index,
                                  rng)


@njit
def locally_optimal_proposal_variance(transition_sd, observation_sd):
    """Variance v, with 1 / v = 1 / s_x^2 + 1 / s_y^2, of the locally optimal proposal, the conditional distribution
    N(v (f(x_{t-1}, t; theta_f) / s_x^2 + y_t / s_y^2), v I_d) of x_t given x_{t-1} and y_t, for a model with
    h(x; theta_h) = x.

    Parameters
    ----------
    transition_sd : float
        Standard deviation s_x of the transition kernels.
    observation_sd : float
        Standard deviation s_y of the observations.

    Returns
    -------
    float
        v.
    """
    return 1.0 / (1.0 / transition_sd ** 2 + 1.0 / observation_sd ** 2)


@njit
def locally_optimal_proposal_mean(transition_mean, y_t, transition_sd, observation_sd):
    """Mean v (f(x_{t-1}, t; theta_f) / s_x^2 + y_t / s_y^2) of the locally optimal proposal of
    locally_optimal_proposal_variance.

    Parameters
    ----------
    transition_mean : ndarray, shape (d,)
        Transition mean f(x_{t-1}, t; theta_f).
    y_t : ndarray, shape (d,)
        Observation at time t.
    transition_sd : float
        Standard deviation s_x of the transition kernels.
    observation_sd : float
        Standard deviation s_y of the observations.

    Returns
    -------
    ndarray, shape (d,)
        Mean of the proposal.
    """
    return locally_optimal_proposal_variance(transition_sd, observation_sd) * (transition_mean / transition_sd ** 2
                                                                               + y_t / observation_sd ** 2)


@njit
def predictive_sd(transition_sd, observation_sd):
    """Standard deviation sqrt(s_x^2 + s_y^2) of the conditional distribution N(f(x_{t-1}, t; theta_f),
    (s_x^2 + s_y^2) I_d) of y_t given x_{t-1}, for a model with h(x; theta_h) = x.

    Parameters
    ----------
    transition_sd : float
        Standard deviation s_x of the transition kernels.
    observation_sd : float
        Standard deviation s_y of the observations.

    Returns
    -------
    float
        sqrt(s_x^2 + s_y^2).
    """
    return np.sqrt(transition_sd ** 2 + observation_sd ** 2)


@njit
def locally_optimal_forward_pass(x, y, N, model, ancestor_kernel, conditional_resampling, initial_index, rng):
    """The forward pass of bootstrap_forward_pass with the locally optimal proposal instead of the bootstrap filter,
    for a model with h(x; theta_h) = x.

    Each particle at time t >= 2 is drawn from the conditional distribution of x_t given its ancestor
    x_{t-1} and y_t (locally_optimal_proposal_mean and locally_optimal_proposal_variance), and each particle at
    time 1 from the conditional distribution of x_1 given y_1. This is the particle filter of the Feynman--Kac
    model of Section S3 of the supplementary material of the note, with these proposals as Markov kernels and
    with the predictive densities of the observations as potentials: the potential of a particle at time t >= 2
    is the density of N(f(x_{t-1}, t; theta_f), (s_x^2 + s_y^2) I_d) at y_t (predictive_sd), which depends on its
    ancestor x_{t-1} only, and the potential at time 1 is constant. Because the proposal density times the
    potential is the transition density of the model from x_t to x_{t+1} times the density of y_{t+1} given
    x_{t+1}, the backward-sampling probabilities (S3) of this model are proportional to W_t^i times the transition
    density from x_t^i to x_{t+1}^j, as for the bootstrap filter (log_backward_weights).

    Parameters
    ----------
    x, y, N, ancestor_kernel, conditional_resampling, initial_index, rng
        As in bootstrap_forward_pass.
    model : tuple
        (s_1, f, theta_f, s_x, h, theta_h, s_y), see the module docstring, with h(x; theta_h) = x.

    Returns
    -------
    particles, ancestors, transition_means, reference_indices : ndarray
        As in bootstrap_forward_pass.
    log_potentials : ndarray, shape (T, N)
        log_potentials[t - 1, n] is the logarithm of the potential of particles[t - 1, n], up to an
        additive constant which does not depend on n.
    """
    initial_sd, transition_mean, transition_params, transition_sd, _, _, observation_sd = model
    T, d = x.shape
    particles = np.empty((T, N, d))
    ancestors = np.empty((T - 1, N), dtype=np.int64)
    log_potentials = np.empty((T, N))
    transition_means = np.empty((T - 1, N, d))
    reference_indices = np.empty(T, dtype=np.int64)
    log_reference_weights = np.empty(N)
    initial_proposal_variance = 1.0 / (1.0 / initial_sd ** 2 + 1.0 / observation_sd ** 2)
    proposal_sd = np.sqrt(locally_optimal_proposal_variance(transition_sd, observation_sd))
    potential_sd = predictive_sd(transition_sd, observation_sd)
    reference_indices[0] = initial_index(N, rng)
    particles[0, reference_indices[0]] = x[0]
    for n in range(N):
        if n != reference_indices[0]:
            for k in range(d):
                particles[0, n, k] = (initial_proposal_variance * y[0, k] / observation_sd ** 2
                                      + np.sqrt(initial_proposal_variance) * rng.standard_normal())
    log_potentials[0] = 0.0
    for t in range(1, T):
        transition_means[t - 1] = transition_mean(particles[t - 1], t + 1, transition_params)
        log_backward_weights_to(log_potentials, transition_means, transition_sd, t - 1, x[t], log_reference_weights)
        c = ancestor_kernel(log_reference_weights, reference_indices[t - 1], rng)
        reference_indices[t], ancestors[t - 1] = conditional_resampling(log_potentials[t - 1], c, rng)
        particles[t, reference_indices[t]] = x[t]
        for n in range(N):
            if n != reference_indices[t]:
                proposal_mean = locally_optimal_proposal_mean(transition_means[t - 1, ancestors[t - 1, n]], y[t],
                                                              transition_sd, observation_sd)
                for k in range(d):
                    particles[t, n, k] = proposal_mean[k] + proposal_sd * rng.standard_normal()
        for n in range(N):
            log_potentials[t, n] = log_unnormalized_gaussian_density(y[t], transition_means[t - 1, ancestors[t - 1, n]],
                                                                     potential_sd)
    return particles, ancestors, log_potentials, transition_means, reference_indices


@njit
def locally_optimal_ancestor_sampling_forward_pass(x, y, N, model, rng):
    """Steps (a) and (b) of Algorithm S1 of the supplementary material of the note with the locally optimal proposal
    and multinomial resampling, with the reference trajectory at index 0 at every time step (locally_optimal_forward_pass
    with independent_draw, conditional_multinomial_resampling, and first_index).

    Parameters
    ----------
    x, y, N, model, rng
        As in locally_optimal_forward_pass.

    Returns
    -------
    particles, ancestors, log_potentials, transition_means, reference_indices : ndarray
        As in locally_optimal_forward_pass, with reference_indices = 0.
    """
    return locally_optimal_forward_pass(x, y, N, model, independent_draw, conditional_multinomial_resampling,
                                        first_index, rng)


@njit
def locally_optimal_stratified_ancestor_sampling_forward_pass(x, y, N, model, rng):
    """Steps (a) and (b) of Algorithm S1 of the supplementary material of the note with the locally optimal proposal
    and stratified resampling (locally_optimal_forward_pass with independent_draw, conditional_stratified_resampling,
    and uniform_index).

    Parameters
    ----------
    x, y, N, model, rng
        As in locally_optimal_forward_pass.

    Returns
    -------
    particles, ancestors, log_potentials, transition_means, reference_indices : ndarray
        As in locally_optimal_forward_pass.
    """
    return locally_optimal_forward_pass(x, y, N, model, independent_draw, conditional_stratified_resampling,
                                        uniform_index, rng)


@njit
def select(particles, ancestors, log_potentials, transition_means, reference_indices, transition_sd, final_kernel,
           backward_kernel, rng):
    """Steps (c) and (d) of Algorithm 1 for the rules whose kernels K_{t,j,j'}, j != j', are the backward-sampling
    probabilities B_t(. | j'), in their usual implementation (Section 2.1 of the note), after the forward pass of
    Algorithm 1, in which the ancestor of the reference state at time t + 1 is the reference state at time t: l_T is
    drawn with final_kernel from b_T, for the log-weights log W_T^i, and l_t, t < T, with backward_kernel, the kernel
    K_{t,l_{t+1},l_{t+1}}, from the ancestor a_t^{l_{t+1}}, for the log-weights of B_t(. | l_{t+1})
    (log_backward_weights).

    Parameters
    ----------
    particles, ancestors, log_potentials, transition_means, reference_indices : ndarray
        Output of conditional_forward_pass.
    transition_sd : float
        Standard deviation s_x of the transition kernels.
    final_kernel : function
        The kernel K_T, as a jitted function (log_w, i, rng) -> new index.
    backward_kernel : function
        The kernels K_{t,j,j}, as a jitted function (log_w, i, rng) -> new index.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    ndarray of int64, shape (T,)
        Indices l_1, ..., l_T of the output trajectory.
    """
    T, N = log_potentials.shape
    indices = np.empty(T, dtype=np.int64)
    log_output_weights = np.empty(N)
    indices[T - 1] = final_kernel(log_potentials[T - 1], reference_indices[T - 1], rng)
    for t in range(T - 2, -1, -1):
        j = indices[t + 1]
        log_backward_weights(particles, log_potentials, transition_means, transition_sd, t, j, log_output_weights)
        indices[t] = backward_kernel(log_output_weights, ancestors[t, j], rng)
    return indices


@njit
def select_from_reference(particles, log_potentials, transition_means, reference_indices, transition_sd, final_kernel,
                          backward_kernel, rng):
    """Steps (c) and (d) of Algorithm 1 for the rules whose kernels K_{t,j,j'}, j != j', are the backward-sampling
    probabilities B_t(. | j'), with the kernels started from the indices of the reference, as in Algorithm S1.c of the
    supplementary material of the note: l_T is drawn with final_kernel from b_T, for the log-weights log W_T^i, and
    l_t, t < T, from the log-weights of B_t(. | l_{t+1}) (log_backward_weights), with backward_kernel, the kernel
    K_{t,b_{t+1},b_{t+1}}, from b_t if l_{t+1} = b_{t+1}, and with independent_draw otherwise.

    Unlike select, this does not use the ancestor indices: after the ancestor sampling of Algorithm S1, the ancestor
    of the reference state at time t + 1 is the index drawn by ancestor sampling, not the index b_t of the reference
    at time t.

    Parameters
    ----------
    particles, log_potentials, transition_means, reference_indices : ndarray
        Output of a forward pass.
    transition_sd, final_kernel, backward_kernel, rng
        As in select.

    Returns
    -------
    ndarray of int64, shape (T,)
        Indices l_1, ..., l_T of the output trajectory.
    """
    T, N = log_potentials.shape
    indices = np.empty(T, dtype=np.int64)
    log_output_weights = np.empty(N)
    indices[T - 1] = final_kernel(log_potentials[T - 1], reference_indices[T - 1], rng)
    for t in range(T - 2, -1, -1):
        j = indices[t + 1]
        log_backward_weights(particles, log_potentials, transition_means, transition_sd, t, j, log_output_weights)
        if j == reference_indices[t + 1]:
            indices[t] = backward_kernel(log_output_weights, reference_indices[t], rng)
        else:
            indices[t] = independent_draw(log_output_weights, reference_indices[t], rng)
    return indices


@njit
def select_coupled(particles, log_potentials, transition_means, reference_indices, transition_sd, rng):
    """Steps (c) and (d) of Algorithm 1 with the kernels of CM-BS, started from the indices b_t of the reference.

    At time T, l_T is drawn by a forced move from b_T, for the log-weights log W_T^i. At time t < T, if
    l_{t+1} = b_{t+1}, then l_t is drawn by a forced move from b_t, for the log-weights of B_t(. | b_{t+1})
    (log_backward_weights), which is the kernel K_{t,b_{t+1},b_{t+1}}. Otherwise, l_t is drawn with coupling_draw from
    b_t, for the log-weights of p = B_t(. | b_{t+1}) and q = B_t(. | l_{t+1}), which is the kernel
    K_{t,b_{t+1},l_{t+1}} = C_{p,q}; B_t(. | b_{t+1}) is only computed in this case.

    Parameters
    ----------
    particles, log_potentials, transition_means, reference_indices : ndarray
        Output of a forward pass.
    transition_sd : float
        Standard deviation s_x of the transition kernels.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    ndarray of int64, shape (T,)
        Indices l_1, ..., l_T of the output trajectory.
    """
    T, N = log_potentials.shape
    indices = np.empty(T, dtype=np.int64)
    log_output_weights = np.empty(N)
    log_reference_weights = np.empty(N)
    indices[T - 1] = forced_move(log_potentials[T - 1], reference_indices[T - 1], rng)
    for t in range(T - 2, -1, -1):
        j = indices[t + 1]
        log_backward_weights(particles, log_potentials, transition_means, transition_sd, t, j, log_output_weights)
        if j == reference_indices[t + 1]:
            indices[t] = forced_move(log_output_weights, reference_indices[t], rng)
        else:
            log_backward_weights(particles, log_potentials, transition_means, transition_sd, t,
                                 reference_indices[t + 1], log_reference_weights)
            indices[t] = coupling_draw(log_reference_weights, log_output_weights, reference_indices[t], rng)
    return indices


@njit
def simulate_data_sets(n, T, d, model, rng):
    """Draw n independent data sets with simulate.

    Parameters
    ----------
    n : int
        Number of data sets.
    T, d, model, rng
        As in simulate.

    Returns
    -------
    x : ndarray, shape (n, T, d)
        States.
    y : ndarray, shape (n, T, d)
        Observations.
    """
    x = np.empty((n, T, d))
    y = np.empty((n, T, d))
    for k in range(n):
        x_k, y_k = simulate(T, d, model, rng)
        x[k] = x_k
        y[k] = y_k
    return x, y


@njit
def conditional_forward_passes(forward_pass, x, y, N, model, rng):
    """Run a forward pass for each data set.

    Parameters
    ----------
    forward_pass : function
        conditional_forward_pass, stratified_ancestor_sampling_forward_pass,
        locally_optimal_ancestor_sampling_forward_pass, or locally_optimal_stratified_ancestor_sampling_forward_pass.
    x : ndarray, shape (n, T, d)
        Reference trajectories.
    y : ndarray, shape (n, T, d)
        Observations.
    N, model, rng
        As in forward_pass.

    Returns
    -------
    particles, ancestors, log_potentials, transition_means, reference_indices : ndarray
        The outputs of forward_pass, stacked along a first axis of length n.
    """
    n, T, d = x.shape
    particles = np.empty((n, T, N, d))
    ancestors = np.empty((n, T - 1, N), dtype=np.int64)
    log_potentials = np.empty((n, T, N))
    transition_means = np.empty((n, T - 1, N, d))
    reference_indices = np.empty((n, T), dtype=np.int64)
    for k in range(n):
        particles_k, ancestors_k, log_potentials_k, transition_means_k, reference_indices_k = forward_pass(
            x[k], y[k], N, model, rng)
        particles[k] = particles_k
        ancestors[k] = ancestors_k
        log_potentials[k] = log_potentials_k
        transition_means[k] = transition_means_k
        reference_indices[k] = reference_indices_k
    return particles, ancestors, log_potentials, transition_means, reference_indices


@njit
def select_all(particles, ancestors, log_potentials, transition_means, reference_indices, transition_sd, final_kernel,
               backward_kernel, rng):
    """Run select for each data set.

    Parameters
    ----------
    particles, ancestors, log_potentials, transition_means, reference_indices : ndarray
        Output of conditional_forward_passes with conditional_forward_pass.
    transition_sd, final_kernel, backward_kernel, rng
        As in select.

    Returns
    -------
    ndarray of int64, shape (n, T)
        Indices of the output trajectories.
    """
    n, T = particles.shape[0], particles.shape[1]
    indices = np.empty((n, T), dtype=np.int64)
    for k in range(n):
        indices[k] = select(particles[k], ancestors[k], log_potentials[k], transition_means[k], reference_indices[k],
                            transition_sd, final_kernel, backward_kernel, rng)
    return indices


@njit
def select_all_from_reference(particles, log_potentials, transition_means, reference_indices, transition_sd,
                              final_kernel, backward_kernel, rng):
    """Run select_from_reference for each data set.

    Parameters
    ----------
    particles, log_potentials, transition_means, reference_indices : ndarray
        Output of conditional_forward_passes.
    transition_sd, final_kernel, backward_kernel, rng
        As in select_from_reference.

    Returns
    -------
    ndarray of int64, shape (n, T)
        Indices of the output trajectories.
    """
    n, T = particles.shape[0], particles.shape[1]
    indices = np.empty((n, T), dtype=np.int64)
    for k in range(n):
        indices[k] = select_from_reference(particles[k], log_potentials[k], transition_means[k], reference_indices[k],
                                           transition_sd, final_kernel, backward_kernel, rng)
    return indices


@njit
def select_all_coupled(particles, log_potentials, transition_means, reference_indices, transition_sd, rng):
    """Run select_coupled for each data set.

    Parameters
    ----------
    particles, log_potentials, transition_means, reference_indices : ndarray
        Output of conditional_forward_passes.
    transition_sd, rng
        As in select_coupled.

    Returns
    -------
    ndarray of int64, shape (n, T)
        Indices of the output trajectories.
    """
    n, T = particles.shape[0], particles.shape[1]
    indices = np.empty((n, T), dtype=np.int64)
    for k in range(n):
        indices[k] = select_coupled(particles[k], log_potentials[k], transition_means[k], reference_indices[k],
                                    transition_sd, rng)
    return indices


def ancestor_kernel_rule(final_kernel, backward_kernel):
    """Selection of the output indices of all the data sets with select_all and the given kernels.

    Parameters
    ----------
    final_kernel, backward_kernel : function
        As in select.

    Returns
    -------
    function
        (particles, ancestors, log_potentials, transition_means, reference_indices, model, rng) -> output indices, as
        select_all with these kernels and the standard deviation s_x of the transitions of the model.
    """
    def select_rule(particles, ancestors, log_potentials, transition_means, reference_indices, model, rng):
        _, _, _, transition_sd, _, _, _ = model
        return select_all(particles, ancestors, log_potentials, transition_means, reference_indices, transition_sd,
                          final_kernel, backward_kernel, rng)
    return select_rule


def reference_kernel_rule(final_kernel, backward_kernel):
    """Selection of the output indices of all the data sets with select_all_from_reference and the given kernels.

    Parameters
    ----------
    final_kernel, backward_kernel : function
        As in select_from_reference.

    Returns
    -------
    function
        (particles, ancestors, log_potentials, transition_means, reference_indices, model, rng) -> output indices, as
        select_all_from_reference with these kernels and the standard deviation s_x of the transitions of the model;
        the ancestors are not used.
    """
    def select_rule(particles, ancestors, log_potentials, transition_means, reference_indices, model, rng):
        _, _, _, transition_sd, _, _, _ = model
        return select_all_from_reference(particles, log_potentials, transition_means, reference_indices,
                                         transition_sd, final_kernel, backward_kernel, rng)
    return select_rule


def coupled_rule(particles, ancestors, log_potentials, transition_means, reference_indices, model, rng):
    """Selection of the output indices of all the data sets with select_all_coupled (CM-BS).

    Parameters
    ----------
    particles, ancestors, log_potentials, transition_means, reference_indices : ndarray
        Output of conditional_forward_passes; the ancestors are not used.
    model : tuple
        (s_1, f, theta_f, s_x, h, theta_h, s_y), see the module docstring.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    ndarray of int64, shape (n, T)
        Indices of the output trajectories.
    """
    _, _, _, transition_sd, _, _, _ = model
    return select_all_coupled(particles, log_potentials, transition_means, reference_indices, transition_sd, rng)


# the selection rules of the note, as functions (particles, ancestors, log_potentials, transition_means,
# reference_indices, model, rng) -> output indices of all the data sets: after the forward pass of Algorithm 1, BS,
# FM-BS, and M-BS started from the ancestors (ALGORITHM_1_SELECTORS), and, after any forward pass, all the rules
# started from the indices of the reference (ALGORITHM_S1_SELECTORS, the backward pass of Algorithm S1 of the
# supplementary material)
ALGORITHM_1_SELECTORS = {**{rule: ancestor_kernel_rule(*kernels) for rule, kernels in SELECTION_RULES.items()},
                         "CM-BS": coupled_rule}
ALGORITHM_S1_SELECTORS = {**{rule: reference_kernel_rule(*kernels) for rule, kernels in SELECTION_RULES.items()},
                          "CM-BS": coupled_rule}
