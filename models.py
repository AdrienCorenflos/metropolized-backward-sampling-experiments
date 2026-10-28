"""The two state-space models of Section 5 of the note, as tuples (s_1, f, theta_f, s_x, h, theta_h,
s_y) in the sense of csmc.py."""
import numpy as np
from numba import njit


@njit
def linear_mean(x, t, theta):
    """Transition mean f(x, t; theta) = rho x of the linear-Gaussian model.

    Parameters
    ----------
    x : ndarray, shape (..., d)
        Previous states.
    t : int
        Time index of the new state; the model is time-homogeneous.
    theta : tuple
        (rho,).

    Returns
    -------
    ndarray, shape (..., d)
        Transition means.
    """
    (rho,) = theta
    return rho * x


@njit
def identity_mean(x, theta):
    """Observation mean h(x; theta) = x of the linear-Gaussian model.

    Parameters
    ----------
    x : ndarray, shape (..., d)
        States.
    theta : tuple
        Empty tuple.

    Returns
    -------
    ndarray, shape (..., d)
        Observation means.
    """
    return x


@njit
def gordon_salmond_smith_mean(x, t, theta):
    """Transition mean f(x, t; theta) = a x + b x / (1 + x^2) + c cos(omega t), coordinatewise, of
    the nonlinear model of Gordon, Salmond and Smith (1993).

    Parameters
    ----------
    x : ndarray, shape (..., d)
        Previous states.
    t : int
        Time index of the new state.
    theta : tuple
        (a, b, c, omega).

    Returns
    -------
    ndarray, shape (..., d)
        Transition means.
    """
    a, b, c, omega = theta
    return a * x + b * x / (1.0 + x * x) + c * np.cos(omega * t)


@njit
def scaled_square_mean(x, theta):
    """Observation mean h(x; theta) = kappa x^2, coordinatewise, of the nonlinear model of Gordon,
    Salmond and Smith (1993).

    Parameters
    ----------
    x : ndarray, shape (..., d)
        States.
    theta : tuple
        (kappa,).

    Returns
    -------
    ndarray, shape (..., d)
        Observation means.
    """
    (kappa,) = theta
    return kappa * x * x


def linear_gaussian_model(rho, sigma):
    """Linear-Gaussian model x_1 ~ N(0, sigma^2 / (1 - rho^2) I_d), x_t = rho x_{t-1} + sigma eps_t,
    y_t = x_t + sigma eta_t, whose initial distribution is the stationary distribution of the
    states.

    Parameters
    ----------
    rho : float
        Autoregressive coefficient, |rho| < 1.
    sigma : float
        Standard deviation of the state and observation noises.

    Returns
    -------
    tuple
        (s_1, f, theta_f, s_x, h, theta_h, s_y).
    """
    return sigma / np.sqrt(1.0 - rho ** 2), linear_mean, (rho,), sigma, identity_mean, (), sigma


def gordon_salmond_smith_model(initial_variance, transition_params, transition_variance, observation_params,
                               observation_variance):
    """Nonlinear model of Gordon, Salmond and Smith (1993): x_1 ~ N(0, initial_variance),
    x_t = a x_{t-1} + b x_{t-1} / (1 + x_{t-1}^2) + c cos(omega t) + sigma_V eps_t, and
    y_t = kappa x_t^2 + sigma_W eta_t.

    Parameters
    ----------
    initial_variance : float
        Variance of x_1.
    transition_params : tuple
        (a, b, c, omega).
    transition_variance : float
        Variance sigma_V^2 of the state noise.
    observation_params : tuple
        (kappa,).
    observation_variance : float
        Variance sigma_W^2 of the observation noise.

    Returns
    -------
    tuple
        (s_1, f, theta_f, s_x, h, theta_h, s_y).
    """
    return (np.sqrt(initial_variance), gordon_salmond_smith_mean, transition_params, np.sqrt(transition_variance),
            scaled_square_mean, observation_params, np.sqrt(observation_variance))
