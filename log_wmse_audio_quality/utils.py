import numpy as np
from numpy.typing import NDArray


def calculate_rms(samples: NDArray):
    """Given a numpy array of audio samples, return its RMS power level."""
    return np.sqrt(np.mean(np.square(samples), axis=-1))


def convert_decibels_to_amplitude_ratio(decibels):
    return 10 ** (decibels / 20)
