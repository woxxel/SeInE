import numpy as np


def spike_times_to_frames(spike_times, frame_rate_hz, T=None):
    """
    Convert spike times to per-frame spike counts based on a given frame rate.

    Parameters
    ----------
    spike_times : np.ndarray
        Array of spike times in seconds.
    frame_rate_hz : float
        Frame rate in Hz.
    T : float or None, optional
        Optional time horizon in seconds. If provided, counts are returned for
        frames in the interval [0, T). If omitted, counts are returned up to
        the last spike.

    Returns
    -------
    counts : np.ndarray
        Array of spike counts for each frame.
    """
    frames = np.floor(spike_times * frame_rate_hz).astype(int)
    frames = frames[frames >= 0]

    if T is None:
        n_frames = frames.max() + 1 if frames.size else 0
    else:
        n_frames = int(np.ceil(T * frame_rate_hz))

    counts = np.bincount(frames, minlength=n_frames)
    return counts


def sample_poisson_spike_train(rate_hz, duration_s, rng=None):
    """
    Sample a Poisson spike train given a firing rate and duration.

    Parameters
    ----------
    rate_hz : float
        Firing rate of the neuron in Hz.
    duration_s : float
        Duration of the spike train in seconds.
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    spikes : np.ndarray
        Array of spike times in seconds.
    """
    rng = np.random.default_rng(rng)

    n = rng.poisson(rate_hz * duration_s)
    spikes = rng.uniform(0, duration_s, size=n)
    return np.sort(spikes)


def sample_inhom_poisson(rate_fn, duration_s, rate_max=None, rng=None):
    """
    Sample an inhomogeneous Poisson spike train using the thinning method.

    Parameters
    ----------
    rate_fn : callable
        Function that takes time (in seconds) and returns the instantaneous firing rate.
    duration_s : float
        Duration of the spike train in seconds.
    rate_max : float
        Maximum firing rate of the neuron (used for thinning).
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    spikes : np.ndarray
        Array of spike times in seconds.
    """
    rng = np.random.default_rng(rng)
    if rate_max is None:
        # Estimate the maximum rate by sampling the rate function over the duration
        t_test = np.linspace(0, duration_s, 1000)
        rate_max = np.max(rate_fn(t_test))

    n_cand = rng.poisson(rate_max * duration_s)
    t = rng.uniform(0, duration_s, size=n_cand)

    accept = rng.uniform(size=n_cand) < (rate_fn(t) / rate_max)
    return np.sort(t[accept])


def sample_refractory_poisson(rate_hz, duration_s, refractory_s=0.002, rng=None):
    """
    Sample a spike train from a Poisson process with an absolute refractory period.

    Parameters
    ----------
    rate_hz : float
        Mean firing rate of the neuron in Hz.
    duration_s : float
        Duration of the spike train in seconds.
    refractory_s : float, optional
        Absolute refractory period in seconds (default is 0.002s).
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    spikes : np.ndarray
        Array of spike times in seconds.
    """
    rng = np.random.default_rng(rng)

    spikes = []
    t = 0.0

    # adjusted exponential rate so approximate mean rate is rate_hz
    effective_rate = 1.0 / max(1e-12, (1.0 / rate_hz - refractory_s))

    while t < duration_s:
        t += refractory_s + rng.exponential(1.0 / effective_rate)
        if t < duration_s:
            spikes.append(t)

    return np.array(spikes)


def sample_gamma_renewal(rate_hz, duration_s, shape=2.0, rng=None):
    """
    Sample a spike train from a gamma renewal process.
    Shape parameter "k" controls regularity.
        k=1: Poisson/exponential ISIs
        k>1: more regular spiking
        k<1: burstier spiking

    Parameters
    ----------
    rate_hz : float
        Mean firing rate of the neuron in Hz.
    duration_s : float
        Duration of the spike train in seconds.
    shape : float, optional
        Shape parameter of the gamma distribution (default is 2.0).
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    spikes : np.ndarray
        Array of spike times in seconds.
    """
    rng = np.random.default_rng(rng)

    scale = 1.0 / (rate_hz * shape)

    spikes = []
    t = 0.0
    while t < duration_s:
        t += rng.gamma(shape, scale)
        if t < duration_s:
            spikes.append(t)

    return np.array(spikes)


from scipy.special import gamma


def sample_weibull_renewal(rate_hz, duration_s, shape=2.0, rng=None):
    """
    !! Careful: appears to have initialization effect!!

    Sample a spike train from a Weibull renewal process.

    Shape parameter "k" controls regularity.
        k=1: Poisson/exponential ISIs
        k>1: more regular spiking
        k<1: burstier spiking

    Parameters
    ----------
    rate_hz : float
        Mean firing rate of the neuron in Hz.
    duration_s : float
        Duration of the spike train in seconds.
    shape : float, optional
        Shape parameter of the Weibull distribution (default is 2.0).
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    spikes : np.ndarray
        Array of spike times in seconds.
    """
    rng = np.random.default_rng(rng)

    # Weibull mean = scale * Gamma(1 + 1/shape)
    scale = (1.0 / rate_hz) / gamma(1.0 + 1.0 / shape)

    spikes = []
    t = 0.0
    while t < duration_s:
        isi = scale * rng.weibull(shape)
        t += isi
        if t < duration_s:
            spikes.append(t)

    return np.array(spikes)
