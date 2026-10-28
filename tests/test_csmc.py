"""Tests of csmc.py: the samplers, the kernels of the selection rules, the conditional resampling schemes, the forward
passes of Algorithms 1 and S1, and the steps of Algorithm 1.

The Monte Carlo tests use fixed seeds, and compare frequencies with probabilities up to TOLERANCE
standard errors.
"""
import numpy as np
import pytest
from numba import njit

import csmc
from models import identity_mean, linear_gaussian_model, linear_mean

TOLERANCE = 5.0
N_DRAWS = 200_000
P = np.array([0.5, 0.05, 0.3, 0.15])
# additive constant, to check that the samplers only depend on the log-weights up to a constant
LOG_P = np.log(P) + 3.7
KERNELS = (csmc.independent_draw, csmc.forced_move)


@njit
def repeat_sample_index(log_w, n_draws, rng):
    """n_draws independent calls of sample_index."""
    out = np.empty(n_draws, dtype=np.int64)
    for m in range(n_draws):
        out[m] = csmc.sample_index(log_w, rng)
    return out


@njit
def repeat_sample_index_excluding(log_w, i, n_draws, rng):
    """n_draws independent calls of sample_index_excluding."""
    out = np.empty(n_draws, dtype=np.int64)
    for m in range(n_draws):
        out[m] = csmc.sample_index_excluding(log_w, i, rng)
    return out


@njit
def repeat_kernel(kernel, log_w, starts, rng):
    """One step of the kernel from each of the starting indices."""
    out = np.empty(starts.shape[0], dtype=np.int64)
    for m in range(starts.shape[0]):
        out[m] = kernel(log_w, starts[m], rng)
    return out


def frequencies(draws, n):
    """Frequencies of the indices 0, ..., n - 1 among the draws."""
    return np.bincount(draws, minlength=n) / draws.shape[0]


def assert_frequencies(draws, probabilities):
    """Check that the frequencies of the draws match the probabilities up to TOLERANCE standard errors."""
    n_draws = draws.shape[0]
    standard_errors = np.sqrt(probabilities * (1.0 - probabilities) / n_draws)
    np.testing.assert_array_less(np.abs(frequencies(draws, probabilities.shape[0]) - probabilities),
                                 TOLERANCE * standard_errors + 1.0 / n_draws)


def forced_move_matrix(p):
    """Transition matrix F_p of Section 2.2 of the note: F_p(i, j) = p_j / max(1 - p_i, 1 - p_j), j != i, and F_p(i, i)
    the probability of rejecting the proposal, the sum over k != i with p_k < p_i of p_k (p_i - p_k) / {(1 - p_i)
    (1 - p_k)}, a sum of nonnegative terms; F_p(i, .) is the point mass at i if p_i = 1."""
    n = p.shape[0]
    F = np.zeros((n, n))
    for i in range(n):
        if p[i] == 1.0:
            F[i, i] = 1.0
            continue
        for j in range(n):
            if j != i:
                F[i, j] = p[j] / max(1.0 - p[i], 1.0 - p[j])
        F[i, i] = sum(p[k] * (p[i] - p[k]) / ((1.0 - p[i]) * (1.0 - p[k])) for k in range(n) if k != i and p[k] < p[i])
    return F


def coupling_matrix(p, q):
    """Transition matrix C_{p,q}(i, j) = q_j + q_i {F_{p q}(i, j) - delta_i(j)} of Section 2.3 of the note, with
    F_{p q} the forced move for the probabilities proportional to p_k q_k, or the identity if all p_k q_k vanish."""
    u = p * q
    F = forced_move_matrix(u / u.sum()) if u.sum() > 0.0 else np.eye(p.shape[0])
    return q[None, :] + q[:, None] * (F - np.eye(p.shape[0]))


def test_log_sum_exp_excluding():
    rng = np.random.default_rng(1)
    log_w = 50.0 * rng.standard_normal(7)
    for i in range(log_w.shape[0]):
        others = np.delete(log_w, i)
        expected = np.max(others) + np.log(np.sum(np.exp(others - np.max(others))))
        assert csmc.log_sum_exp_excluding(log_w, i) == pytest.approx(expected, rel=1e-14)
    assert csmc.log_sum_exp_excluding(np.array([-np.inf, 0.0, -np.inf]), 1) == -np.inf


FORCED_MOVE_PROBABILITIES = (P, np.array([0.0, 0.6, 0.0, 0.4]), np.array([0.25, 0.25, 0.25, 0.25]),
                             np.array([0.0, 1.0, 0.0, 0.0]))


@pytest.mark.parametrize("p", FORCED_MOVE_PROBABILITIES)
def test_forced_move_matrix_rows_sum_to_one(p):
    np.testing.assert_allclose(forced_move_matrix(p).sum(axis=1), 1.0, rtol=1e-14)


def test_forced_move_point_mass():
    log_w = np.array([-np.inf, 0.0, -np.inf])
    np.testing.assert_array_equal(repeat_kernel(csmc.forced_move, log_w, np.full(100, 1), np.random.default_rng(31)), 1)


@njit
def repeat_coupling_draw(kernel, log_p, log_q, i, n_draws, rng):
    """n_draws independent steps of the kernel (log_p, log_q, i, rng) -> new index from i."""
    out = np.empty(n_draws, dtype=np.int64)
    for m in range(n_draws):
        out[m] = kernel(log_p, log_q, i, rng)
    return out


def test_coupling_draw():
    q = np.array([0.1, 0.2, 0.3, 0.4])
    expected = coupling_matrix(P, q)
    for i in range(P.shape[0]):
        assert_frequencies(repeat_coupling_draw(csmc.coupling_draw, LOG_P, np.log(q) + 0.8, i, N_DRAWS,
                                                np.random.default_rng(32 + i)), expected[i])


def test_sample_index():
    assert_frequencies(repeat_sample_index(LOG_P, N_DRAWS, np.random.default_rng(2)), P)


def test_sample_index_never_draws_zero_weights():
    log_w = np.array([-np.inf, 0.0, -np.inf, 1.0, -np.inf])
    draws = repeat_sample_index(log_w, N_DRAWS, np.random.default_rng(3))
    assert set(np.unique(draws)) == {1, 3}


def test_sample_index_excluding():
    for i in range(P.shape[0]):
        p = P.copy()
        p[i] = 0.0
        assert_frequencies(repeat_sample_index_excluding(LOG_P, i, N_DRAWS, np.random.default_rng(4 + i)), p / p.sum())


def test_multinomial_resampling():
    draws = csmc.multinomial_resampling(LOG_P, N_DRAWS, np.random.default_rng(8))
    assert np.all(np.diff(draws) >= 0)
    assert_frequencies(draws, P)


@pytest.mark.parametrize("kernel, matrix", [
    (csmc.independent_draw, lambda p: np.tile(p, (p.shape[0], 1))),
    (csmc.forced_move, forced_move_matrix),
])
def test_kernel_transition_probabilities(kernel, matrix):
    expected = matrix(P)
    for i in range(P.shape[0]):
        draws = repeat_kernel(kernel, LOG_P, np.full(N_DRAWS, i), np.random.default_rng(10 + i))
        assert_frequencies(draws, expected[i])


@pytest.mark.parametrize("kernel", KERNELS)
def test_kernel_leaves_target_invariant(kernel):
    rng = np.random.default_rng(20)
    starts = repeat_sample_index(LOG_P, N_DRAWS, rng)
    assert_frequencies(repeat_kernel(kernel, LOG_P, starts, rng), P)


def test_independent_draw_does_not_depend_on_current_index():
    draws = [repeat_kernel(csmc.independent_draw, LOG_P, np.full(1000, i), np.random.default_rng(30))
             for i in range(P.shape[0])]
    for other in draws[1:]:
        np.testing.assert_array_equal(draws[0], other)


def stratified_matrix(w):
    """Probabilities S[m, i] = N |[m / N, (m + 1) / N) cap (F(i - 1), F(i)]| that a^m = i under stratified
    resampling, for the probability vector w with cumulative sums F."""
    N = w.shape[0]
    upper = np.cumsum(w)
    lower = upper - w
    strata = np.arange(N)
    return N * np.clip(np.minimum(upper[None, :], (strata[:, None] + 1) / N)
                       - np.maximum(lower[None, :], strata[:, None] / N), 0.0, None)


@njit
def repeat_conditional_resampling(conditional_resampling, log_w, c, n_draws, rng):
    """n_draws independent calls of a conditional resampling scheme: the indices of the reference and the ancestor
    indices."""
    reference_indices = np.empty(n_draws, dtype=np.int64)
    ancestors = np.empty((n_draws, log_w.shape[0]), dtype=np.int64)
    for m in range(n_draws):
        reference_indices[m], ancestors[m] = conditional_resampling(log_w, c, rng)
    return reference_indices, ancestors


def test_conditional_multinomial_resampling():
    reference_indices, ancestors = repeat_conditional_resampling(csmc.conditional_multinomial_resampling, LOG_P, 2,
                                                                 N_DRAWS // 4, np.random.default_rng(40))
    np.testing.assert_array_equal(reference_indices, 0)
    np.testing.assert_array_equal(ancestors[:, 0], 2)
    assert np.all(np.diff(ancestors[:, 1:], axis=1) >= 0)
    assert_frequencies(ancestors[:, 1:].ravel(), P)


@pytest.mark.parametrize("c", range(P.shape[0]))
def test_conditional_stratified_resampling(c):
    # Finke et al. (2026, Algorithm 6): the index b of the reference has probabilities S[b, c] / (N P_c), where
    # N P_c is the sum of the column S[., c] (test_stratified_matrix), a^b = c, and,
    # given b, the other ancestor indices are those of stratified resampling, with the probabilities S[m, .]
    S = stratified_matrix(P)
    reference_indices, ancestors = repeat_conditional_resampling(csmc.conditional_stratified_resampling, LOG_P, c,
                                                                 N_DRAWS, np.random.default_rng(41 + c))
    np.testing.assert_array_equal(ancestors[np.arange(N_DRAWS), reference_indices], c)
    assert np.all(np.diff(ancestors, axis=1) >= 0)
    assert_frequencies(reference_indices, S[:, c] / S[:, c].sum())
    # the strata m which are not almost surely the stratum of the reference
    for m in np.flatnonzero(S[:, c] < S[:, c].sum()):
        others = reference_indices != m
        assert_frequencies(ancestors[others, m], S[m])


def test_stratified_matrix():
    # the expected number of indices m with a^m = i is N w_i (Definition S1 of the supplementary material)
    w = np.array([0.05, 0.6, 0.1, 0.25])
    S = stratified_matrix(w)
    np.testing.assert_allclose(S.sum(axis=1), 1.0, rtol=1e-14)
    np.testing.assert_allclose(S.sum(axis=0), w.shape[0] * w, rtol=1e-14)


def test_conditional_forward_pass():
    rho, sigma, T, d, N = 0.9, 1.0, 6, 2, 5
    model = linear_gaussian_model(rho, sigma)
    rng = np.random.default_rng(50)
    x, y = csmc.simulate(T, d, model, rng)
    particles, ancestors, log_potentials, transition_means, reference_indices = csmc.conditional_forward_pass(
        x, y, N, model, rng)
    assert particles.shape == (T, N, d) and ancestors.shape == (T - 1, N)
    assert log_potentials.shape == (T, N) and transition_means.shape == (T - 1, N, d)
    np.testing.assert_array_equal(reference_indices, 0)
    np.testing.assert_array_equal(particles[:, 0], x)
    np.testing.assert_array_equal(ancestors[:, 0], 0)
    assert np.all(np.diff(ancestors[:, 1:], axis=1) >= 0)
    np.testing.assert_allclose(transition_means, rho * particles[:-1], rtol=1e-14)
    np.testing.assert_allclose(log_potentials, -0.5 * np.sum((y[:, None, :] - particles) ** 2, axis=2) / sigma ** 2,
                               rtol=1e-14)


def test_stratified_ancestor_sampling_forward_pass():
    rho, sigma, T, d, N = 0.9, 1.0, 6, 2, 5
    model = linear_gaussian_model(rho, sigma)
    rng = np.random.default_rng(55)
    x, y = csmc.simulate(T, d, model, rng)
    particles, ancestors, log_potentials, transition_means, reference_indices = \
        csmc.stratified_ancestor_sampling_forward_pass(x, y, N, model, rng)
    np.testing.assert_array_equal(particles[np.arange(T), reference_indices], x)
    assert np.all(np.diff(ancestors, axis=1) >= 0)
    np.testing.assert_allclose(transition_means, rho * particles[:-1], rtol=1e-14)
    np.testing.assert_allclose(log_potentials, -0.5 * np.sum((y[:, None, :] - particles) ** 2, axis=2) / sigma ** 2,
                               rtol=1e-14)


def test_uniform_index():
    rng = np.random.default_rng(56)
    assert_frequencies(np.array([csmc.uniform_index(4, rng) for _ in range(N_DRAWS)]), np.full(4, 0.25))


def backward_probabilities(particles, log_potentials, transition_means, transition_sd, j):
    """Backward-sampling probabilities B_1(. | j) for T = 2."""
    log_b = (log_potentials[0]
             - 0.5 * np.sum((particles[1, j] - transition_means[0]) ** 2, axis=1) / transition_sd ** 2)
    b = np.exp(log_b - np.max(log_b))
    return b / b.sum()


@pytest.mark.parametrize("rule, final_row, transition_row", [
    # rows of the kernels: transition_row(p, q, a, j), with p = B_1(. | 0), q = B_1(. | j), a = a_1^j, and j = l_2
    ("BS", lambda w: w, lambda p, q, a, j: q),
    ("FM-BS", lambda w: forced_move_matrix(w)[0], lambda p, q, a, j: q),
    ("M-BS", lambda w: forced_move_matrix(w)[0], lambda p, q, a, j: forced_move_matrix(q)[a]),
    ("CM-BS", lambda w: forced_move_matrix(w)[0],
     lambda p, q, a, j: forced_move_matrix(q)[0] if j == 0 else coupling_matrix(p, q)[0]),
])
def test_algorithm_1_selectors(rule, final_row, transition_row):
    # T = 2, after the forward pass of Algorithm 1, with the reference at index 0: l_2 ~ K_T(0, .), then l_1 from
    # a_1^{l_2} for BS, FM-BS, and M-BS, and, for CM-BS, l_1 ~ K_{1,0,0}(0, .) if l_2 = 0 (then a_1^{l_2} = 0), and
    # l_1 ~ K_{1,0,l_2}(0, .) otherwise
    rho, sigma, T, d, N, n = 0.5, 1.0, 2, 1, 3, 20_000
    model = linear_gaussian_model(rho, sigma)
    rng = np.random.default_rng(60)
    x, y = csmc.simulate(T, d, model, rng)
    forward = csmc.conditional_forward_pass(x, y, N, model, rng)
    particles, ancestors, log_potentials, transition_means, _ = forward
    w = np.exp(log_potentials[1] - np.max(log_potentials[1]))
    final = final_row(w / w.sum())
    reference = backward_probabilities(particles, log_potentials, transition_means, sigma, 0)
    expected = np.zeros((N, N))
    for j in range(N):
        output = backward_probabilities(particles, log_potentials, transition_means, sigma, j)
        expected[:, j] = final[j] * transition_row(reference, output, ancestors[0, j], j)
    stacked = tuple(np.repeat(array[None], n, axis=0) for array in forward)
    indices = csmc.ALGORITHM_1_SELECTORS[rule](*stacked, model, rng)
    assert_frequencies(indices[:, 0] * N + indices[:, 1], expected.ravel())


# indices of the reference at times 1 and 2 in test_algorithm_s1_selectors
B_1, B_2 = 2, 1


@pytest.mark.parametrize("rule, final_row, transition_row", [
    # rows of the kernels: transition_row(p, q, b, j), with p = B_1(. | b_2), q = B_1(. | j), b = b_1, and j = l_2
    ("BS", lambda w, b: w, lambda p, q, b, j: q),
    ("FM-BS", lambda w, b: forced_move_matrix(w)[b], lambda p, q, b, j: q),
    ("M-BS", lambda w, b: forced_move_matrix(w)[b], lambda p, q, b, j: forced_move_matrix(q)[b] if j == B_2 else q),
    ("CM-BS", lambda w, b: forced_move_matrix(w)[b],
     lambda p, q, b, j: forced_move_matrix(q)[b] if j == B_2 else coupling_matrix(p, q)[b]),
])
def test_algorithm_s1_selectors(rule, final_row, transition_row):
    # T = 2, with the reference at the indices b_1 = B_1 and b_2 = B_2: l_2 ~ K_T(b_2, .), then
    # l_1 ~ K_{1,b_2,l_2}(b_1, .), whatever the ancestor indices
    rho, sigma, T, d, N, n = 0.5, 1.0, 2, 1, 3, 20_000
    model = linear_gaussian_model(rho, sigma)
    rng = np.random.default_rng(65)
    x, y = csmc.simulate(T, d, model, rng)
    particles, ancestors, log_potentials, transition_means, _ = csmc.stratified_ancestor_sampling_forward_pass(
        x, y, N, model, rng)
    reference_indices = np.array([B_1, B_2])
    w = np.exp(log_potentials[1] - np.max(log_potentials[1]))
    final = final_row(w / w.sum(), B_2)
    reference = backward_probabilities(particles, log_potentials, transition_means, sigma, B_2)
    expected = np.zeros((N, N))
    for j in range(N):
        output = backward_probabilities(particles, log_potentials, transition_means, sigma, j)
        expected[:, j] = final[j] * transition_row(reference, output, B_1, j)
    stacked = tuple(np.repeat(array[None], n, axis=0)
                    for array in (particles, ancestors, log_potentials, transition_means, reference_indices))
    indices = csmc.ALGORITHM_S1_SELECTORS[rule](*stacked, model, rng)
    assert_frequencies(indices[:, 0] * N + indices[:, 1], expected.ravel())


@pytest.mark.parametrize("N", [2, 3, 4])
def test_select_coupled_perfect_proposals(N):
    # equal weights and transitions which do not depend on the previous state: under CM-BS, the output index never
    # coincides with the reference index (Section 4.2 of the note)
    T, d = 6, 1
    rng = np.random.default_rng(140 + N)
    particles = rng.standard_normal((T, N, d))
    reference_indices = rng.integers(0, N, T)
    indices = np.array([csmc.select_coupled(particles, np.zeros((T, N)), np.zeros((T - 1, N, d)), reference_indices,
                                            1.0, rng) for _ in range(5_000)])
    assert np.all(indices != reference_indices)


@pytest.mark.parametrize("N", [2, 3])
def test_select_perfect_proposals_m_bs(N):
    # equal weights and transitions which do not depend on the previous state, so that the ancestors a_t^n, n != 0, are
    # independent uniforms given the positions: under M-BS, the output index coincides with the reference index 0 at
    # time t with probability {1 - (-1/N)^(T - t)} / (N + 1) (Section 4.2 of the note)
    T, d, n_draws = 6, 1, N_DRAWS // 4
    rng = np.random.default_rng(150 + N)
    particles = rng.standard_normal((T, N, d))
    reference_indices = np.zeros(T, dtype=np.int64)
    coincidences = np.empty((n_draws, T), dtype=bool)
    for m in range(n_draws):
        ancestors = np.concatenate((np.zeros((T - 1, 1), dtype=np.int64), rng.integers(0, N, (T - 1, N - 1))), axis=1)
        coincidences[m] = csmc.select(particles, ancestors, np.zeros((T, N)), np.zeros((T - 1, N, d)),
                                      reference_indices, 1.0, *csmc.SELECTION_RULES["M-BS"], rng) == 0
    expected = (1.0 - (-1.0 / N) ** (T - np.arange(1, T + 1))) / (N + 1)
    standard_errors = np.sqrt(expected * (1.0 - expected) / n_draws)
    np.testing.assert_array_less(np.abs(coincidences.mean(axis=0) - expected), TOLERANCE * standard_errors + 1.0 / n_draws)


def log_gaussian_density(x, mean, variance):
    """Logarithm of the density of N(mean, variance I_d) at x, with its normalizing constant."""
    return -0.5 * np.sum((x - mean) ** 2) / variance - 0.5 * x.shape[0] * np.log(2.0 * np.pi * variance)


def test_locally_optimal_proposal_times_potential():
    # proposal density times potential = transition density times observation density, for h(x) = x
    rho, s_x, s_y = 0.7, 1.3, 0.6
    v = 1.0 / (1.0 / s_x ** 2 + 1.0 / s_y ** 2)
    rng = np.random.default_rng(70)
    for _ in range(5):
        x_previous, x_new, y = rng.standard_normal(2), rng.standard_normal(2), rng.standard_normal(2)
        proposal = log_gaussian_density(x_new, v * (rho * x_previous / s_x ** 2 + y / s_y ** 2), v)
        potential = log_gaussian_density(y, rho * x_previous, s_x ** 2 + s_y ** 2)
        expected = log_gaussian_density(x_new, rho * x_previous, s_x ** 2) + log_gaussian_density(y, x_new, s_y ** 2)
        assert proposal + potential == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("forward_pass", [csmc.locally_optimal_ancestor_sampling_forward_pass,
                                         csmc.locally_optimal_stratified_ancestor_sampling_forward_pass])
def test_locally_optimal_forward_pass(forward_pass):
    rho, sigma, T, d, N = 0.9, 1.0, 6, 2, 5
    model = linear_gaussian_model(rho, sigma)
    rng = np.random.default_rng(80)
    x, y = csmc.simulate(T, d, model, rng)
    particles, ancestors, log_potentials, transition_means, reference_indices = forward_pass(x, y, N, model, rng)
    assert particles.shape == (T, N, d) and ancestors.shape == (T - 1, N)
    assert log_potentials.shape == (T, N) and transition_means.shape == (T - 1, N, d)
    np.testing.assert_array_equal(particles[np.arange(T), reference_indices], x)
    np.testing.assert_allclose(transition_means, rho * particles[:-1], rtol=1e-14)
    np.testing.assert_array_equal(log_potentials[0], 0.0)
    parents = rho * particles[np.arange(T - 1)[:, None], ancestors]
    np.testing.assert_allclose(log_potentials[1:], -0.25 * np.sum((y[1:, None, :] - parents) ** 2, axis=2) / sigma ** 2,
                               rtol=1e-14)


def test_locally_optimal_proposals():
    # standardized particles of index 1, given their ancestors, are standard Gaussian
    rho, s_1, s_x, s_y, d = 0.9, 2.0, 1.0, 0.5, 1
    model = (s_1, linear_mean, (rho,), s_x, identity_mean, (), s_y)
    x, y = np.array([[0.3], [-1.0]]), np.array([[1.2], [0.4]])
    rng = np.random.default_rng(90)
    v_1 = 1.0 / (1.0 / s_1 ** 2 + 1.0 / s_y ** 2)
    v = 1.0 / (1.0 / s_x ** 2 + 1.0 / s_y ** 2)
    standardized = np.empty((N_DRAWS // 10, 2))
    for m in range(standardized.shape[0]):
        particles, ancestors, _, _, _ = csmc.locally_optimal_ancestor_sampling_forward_pass(x, y, 2, model, rng)
        parent = particles[0, ancestors[0, 1], 0]
        standardized[m, 0] = (particles[0, 1, 0] - v_1 * y[0, 0] / s_y ** 2) / np.sqrt(v_1)
        standardized[m, 1] = (particles[1, 1, 0] - v * (rho * parent / s_x ** 2 + y[1, 0] / s_y ** 2)) / np.sqrt(v)
    n = standardized.shape[0]
    np.testing.assert_array_less(np.abs(standardized.mean(axis=0)), TOLERANCE / np.sqrt(n))
    np.testing.assert_array_less(np.abs(standardized.var(axis=0) - 1.0), TOLERANCE * np.sqrt(2.0 / n))


@pytest.mark.parametrize("forward_pass, kernel_row", [
    # rows from the index b_1 of the reference of the kernel of the ancestor index of the reference, for the
    # probabilities p
    (csmc.conditional_forward_pass, lambda p, b: np.eye(p.shape[0])[b]),
    (csmc.stratified_ancestor_sampling_forward_pass, lambda p, b: p),
    (csmc.locally_optimal_ancestor_sampling_forward_pass, lambda p, b: p),
    (csmc.locally_optimal_stratified_ancestor_sampling_forward_pass, lambda p, b: p),
])
def test_reference_ancestor(forward_pass, kernel_row):
    # T = 2: given the particles at time 1, the ancestor index of the reference at time 2 is drawn from b_1 with the
    # kernel, for the backward-sampling probabilities B_1(. | x_2) of (S3), proportional to W_1^i times the density of
    # N(rho x_1^i, sigma^2) at x_2; the indicators of the draws minus these transition probabilities, which vary with
    # the particles, have mean zero
    rho, sigma, T, d, N, n = 0.8, 1.0, 2, 1, 3, N_DRAWS // 4
    model = linear_gaussian_model(rho, sigma)
    rng = np.random.default_rng(95)
    x, y = csmc.simulate(T, d, model, rng)
    differences = np.empty((n, N))
    for m in range(n):
        particles, ancestors, log_potentials, _, reference_indices = forward_pass(x, y, N, model, rng)
        log_b = log_potentials[0] - 0.5 * (x[1, 0] - rho * particles[0, :, 0]) ** 2 / sigma ** 2
        b = np.exp(log_b - np.max(log_b))
        differences[m] = (np.eye(N)[ancestors[0, reference_indices[1]]]
                          - kernel_row(b / b.sum(), reference_indices[0]))
    np.testing.assert_array_less(np.abs(differences.mean(axis=0)),
                                 TOLERANCE * differences.std(axis=0, ddof=1) / np.sqrt(n) + 1.0 / n)


# forward passes of Algorithm S1, which, with the selection rules started from the indices of the reference, is
# reversible (Proposition S1 of the supplementary material)
ALGORITHM_S1_FORWARD_PASSES = {
    "stratified": csmc.stratified_ancestor_sampling_forward_pass,
    "locally optimal": csmc.locally_optimal_ancestor_sampling_forward_pass,
    "locally optimal, stratified": csmc.locally_optimal_stratified_ancestor_sampling_forward_pass,
}
ALGORITHM_S1_CASES = [(name, rule, N) for name in ALGORITHM_S1_FORWARD_PASSES for rule in csmc.ALGORITHM_S1_SELECTORS
                      for N in (2, 3)]


def algorithm_s1_outputs(name, rule, N, seed):
    """References x ~ pi and outputs of Algorithm S1 with one of ALGORITHM_S1_FORWARD_PASSES and a selection rule, for
    N_DRAWS data sets of the LG model with T = 3 and d = 1."""
    rho, sigma, T, d = 0.9, 1.0, 3, 1
    model = linear_gaussian_model(rho, sigma)
    rng = np.random.default_rng(seed)
    x, y = csmc.simulate_data_sets(N_DRAWS, T, d, model, rng)
    forward = csmc.conditional_forward_passes(ALGORITHM_S1_FORWARD_PASSES[name], x, y, N, model, rng)
    indices = csmc.ALGORITHM_S1_SELECTORS[rule](*forward, model, rng)
    return x[:, :, 0], forward[0][np.arange(N_DRAWS)[:, None], np.arange(T)[None, :], indices, 0]


@pytest.mark.parametrize("name, rule, N", ALGORITHM_S1_CASES)
def test_algorithm_s1_preserves_second_moments(name, rule, N):
    # from x ~ pi, the output has the same expected squared norm at every time step
    x, output = algorithm_s1_outputs(name, rule, N, 100 + N)
    differences = output ** 2 - x ** 2
    np.testing.assert_array_less(np.abs(differences.mean(axis=0)),
                                 TOLERANCE * differences.std(axis=0, ddof=1) / np.sqrt(N_DRAWS))


@pytest.mark.parametrize("name, rule, N", ALGORITHM_S1_CASES)
def test_algorithm_s1_is_reversible(name, rule, N):
    # from x ~ pi, the reference and the output are exchangeable (Proposition S1 of the supplementary material), so that
    # x_s x'_t - x'_s x_t has mean zero for all s < t
    x, output = algorithm_s1_outputs(name, rule, N, 110 + N)
    s, t = np.triu_indices(x.shape[1], k=1)
    differences = x[:, s] * output[:, t] - output[:, s] * x[:, t]
    np.testing.assert_array_less(np.abs(differences.mean(axis=0)),
                                 TOLERANCE * differences.std(axis=0, ddof=1) / np.sqrt(N_DRAWS))
