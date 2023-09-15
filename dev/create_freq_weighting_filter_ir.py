import os
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.signal
from audiomentations import (
    Compose,
    HighPassFilter,
    HighShelfFilter,
    PeakingFilter,
    LowPassFilter,
)
from numpy.typing import NDArray

from log_wmse_audio_quality.constants import N_FFT, INTERNAL_SAMPLE_RATE


def get_human_hearing_sensitivity_filter_set() -> (
    Callable[[NDArray[np.float32], int], NDArray[np.float32]]
):
    """Compose a set of filters that together form a frequency response that resembles
    human hearing sensitivity to different frequencies. Inspired by Fenton & Lee (2017).
    """
    return Compose(
        [
            HighShelfFilter(
                min_center_freq=1500.0,
                max_center_freq=1500.0,
                min_gain_db=5.0,
                max_gain_db=5.0,
                min_q=1 / np.sqrt(2),
                max_q=1 / np.sqrt(2),
                p=1.0,
            ),
            HighPassFilter(
                min_cutoff_freq=120.0,
                max_cutoff_freq=120.0,
                min_rolloff=12,
                max_rolloff=12,
                zero_phase=False,
                p=1.0,
            ),
            PeakingFilter(
                min_center_freq=500.0,
                max_center_freq=500.0,
                min_gain_db=2.5,
                max_gain_db=2.5,
                min_q=1.5 / np.sqrt(2),
                max_q=1.5 / np.sqrt(2),
                p=1.0,
            ),
            LowPassFilter(
                min_cutoff_freq=10_000,
                max_cutoff_freq=10_000,
                min_rolloff=12,
                max_rolloff=12,
                zero_phase=True,
                p=1.0,
            ),
        ]
    )


def get_zero_phase_equivalent_filter_impulse_response(
    filter_func: Callable[[NDArray[np.float32], int], NDArray[np.float32]],
    sample_rate: int,
    n_fft: int = 2 * N_FFT,
) -> NDArray:
    """Extract the target response from the given filter_func (which may be not
    zero-phase). The idea is to construct a zero-phase equivalent of the given
    filter_func. Credits: mmxgn (2023)"""
    # Get the impulse response of the filter
    delta = np.zeros(n_fft, dtype=np.float32)
    delta[len(delta) // 2] = 1.0
    impulse_response = filter_func(delta, sample_rate)

    w, h = scipy.signal.freqz(impulse_response, worN=n_fft // 2 + 1)
    linear_target_response = np.abs(h)

    # Compute impulse response
    impulse_response = np.fft.fftshift(np.fft.irfft(linear_target_response, n_fft))

    # Make it symmetric
    center_sample = len(impulse_response) // 2 + 1
    return impulse_response[center_sample - 2000 : center_sample + 2000]


if __name__ == "__main__":
    """Calculate and write the filter impulse response"""
    ir = get_zero_phase_equivalent_filter_impulse_response(
        get_human_hearing_sensitivity_filter_set(), sample_rate=INTERNAL_SAMPLE_RATE
    )
    here = Path(os.path.abspath(os.path.dirname(__file__)))
    base_dir = here.parent
    package_dir = base_dir / "log_wmse_audio_quality"
    with open(package_dir / "filter_ir.pkl", "wb") as f:
        pickle.dump(ir, f)
