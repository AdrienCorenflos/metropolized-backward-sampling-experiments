"""Means over equally sized batches of independent data sets, ratios of means, and their standard
errors."""
import numpy as np


def mean_and_standard_error(batch_means):
    """Mean over all the data sets, from the means over equally sized batches, and its standard error.

    Parameters
    ----------
    batch_means : ndarray, shape (B, ...)
        Means over B batches.

    Returns
    -------
    mean : ndarray, shape (...)
        Mean over all the data sets.
    standard_error : ndarray, shape (...)
        Standard deviation of the batch means, divided by sqrt(B).
    """
    return batch_means.mean(axis=0), batch_means.std(axis=0, ddof=1) / np.sqrt(batch_means.shape[0])


def ratio_and_standard_error(numerator_batch_means, denominator_batch_means):
    """Ratio of the means of two statistics over the same data sets, and its standard error by the
    delta method.

    Parameters
    ----------
    numerator_batch_means : ndarray, shape (B, ...)
        Means of the numerator statistic over B batches.
    denominator_batch_means : ndarray, shape (B, ...)
        Means of the denominator statistic over the same batches.

    Returns
    -------
    ratio : ndarray, shape (...)
        Ratio of the means over all the data sets.
    standard_error : ndarray, shape (...)
        Standard deviation over the batches of numerator - ratio * denominator, divided by sqrt(B)
        times the mean of the denominator.
    """
    denominator_mean = denominator_batch_means.mean(axis=0)
    ratio = numerator_batch_means.mean(axis=0) / denominator_mean
    residuals = numerator_batch_means - ratio * denominator_batch_means
    return ratio, residuals.std(axis=0, ddof=1) / (np.sqrt(numerator_batch_means.shape[0]) * denominator_mean)
