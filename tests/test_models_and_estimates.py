"""Tests of models.py and estimates.py."""
import numpy as np
import pytest

from estimates import mean_and_standard_error, ratio_and_standard_error
from models import gordon_salmond_smith_model, linear_gaussian_model


def test_linear_gaussian_model():
    rho, sigma = 0.9, 2.0
    initial_sd, transition_mean, transition_params, transition_sd, observation_mean, observation_params, observation_sd = \
        linear_gaussian_model(rho, sigma)
    x = np.array([[1.0, -2.0], [0.5, 3.0]])
    assert initial_sd == pytest.approx(sigma / np.sqrt(1.0 - rho ** 2))
    np.testing.assert_array_equal(transition_mean(x, 3, transition_params), rho * x)
    np.testing.assert_array_equal(observation_mean(x, observation_params), x)
    assert transition_sd == sigma and observation_sd == sigma


def test_gordon_salmond_smith_model():
    initial_variance, transition_params, transition_variance, observation_params, observation_variance = \
        5.0, (0.5, 25.0, 8.0, 1.2), 10.0, (1.0 / 20.0,), 1.0
    initial_sd, transition_mean, theta_f, transition_sd, observation_mean, theta_h, observation_sd = \
        gordon_salmond_smith_model(initial_variance, transition_params, transition_variance, observation_params,
                                   observation_variance)
    x, t = np.array([[2.0], [-0.5]]), 7
    np.testing.assert_allclose(transition_mean(x, t, theta_f), x / 2 + 25 * x / (1 + x ** 2) + 8 * np.cos(1.2 * t),
                               rtol=1e-14)
    np.testing.assert_allclose(observation_mean(x, theta_h), x ** 2 / 20, rtol=1e-14)
    assert (initial_sd, transition_sd, observation_sd) == pytest.approx((np.sqrt(5.0), np.sqrt(10.0), 1.0))


def test_mean_and_standard_error():
    batch_means = np.array([[1.0, 2.0], [3.0, 2.0], [5.0, 8.0]])
    mean, standard_error = mean_and_standard_error(batch_means)
    np.testing.assert_allclose(mean, [3.0, 4.0])
    np.testing.assert_allclose(standard_error, [2.0 / np.sqrt(3.0), np.sqrt(12.0) / np.sqrt(3.0)])


def test_ratio_and_standard_error():
    numerator = np.array([2.0, 4.0, 3.0, 5.0])
    denominator = np.array([1.0, 2.0, 2.0, 3.0])
    ratio, standard_error = ratio_and_standard_error(numerator, denominator)
    assert ratio == pytest.approx(14.0 / 8.0)
    residuals = numerator - ratio * denominator
    assert standard_error == pytest.approx(np.std(residuals, ddof=1) / (2.0 * 2.0))
