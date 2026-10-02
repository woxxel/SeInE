import numpy as np

"""
This file contains various spike generation mechanisms to
obtain spike counts from an underlying model
"""


def sample_poisson_counts(mu, rng=None):
    """
    Sample spike counts from a Poisson distribution.

    Parameters
    ----------
    mu : float
        Mean firing rate of the neuron.
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    counts : int
        Sampled spike count from the Poisson distribution.
    """
    rng = np.random.default_rng(rng)
    return rng.poisson(mu)


def sample_modulated_poisson_counts(mu, sigma_gain=0.5, rng=None):
    """
    Sample spike counts from a modulated Poisson process with a lognormal gain.

    Parameters
    ----------
    mu : float
        Mean firing rate of the neuron.
    sigma_gain : float, optional
        Standard deviation of the lognormal gain (default is 0.5).
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    counts : int
        Sampled spike count from the modulated Poisson distribution.
    """
    rng = np.random.default_rng(rng)

    # mean-one lognormal gain
    g = rng.lognormal(mean=-0.5 * sigma_gain**2, sigma=sigma_gain, size=np.shape(mu))
    return rng.poisson(g * mu)


def sample_negbin_counts(mu, alpha, rng=None):
    """
    Equivalent to a Poisson model with a trial-to-trial fluctuating rate:

    Parameters
    ----------
    mu : float
        Mean firing rate of the neuron.
    alpha : float
        Overdispersion parameter. Larger values correspond to more variability.
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    counts : int
        Sampled spike count from the negative binomial distribution.
    """
    rng = np.random.default_rng(rng)

    r = 1.0 / alpha
    p = r / (r + mu)

    return rng.negative_binomial(r, p)


def sample_gaussian_rates(mu, sigma, rng=None):
    """
    Sample spike rates from a Gaussian distribution and ensure non-negativity.
    Useful if the data is already trial-averaged firing rates rather than raw counts.

    Parameters
    ----------
    mu : float
        Mean firing rate of the neuron.
    sigma : float
        Standard deviation of the firing rate.
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    rate : float
        Sampled spike rate, non-negative.
    """
    rng = np.random.default_rng(rng)
    y = rng.normal(mu, sigma)
    return np.maximum(y, 0)
