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

from log_wmse_audio_quality.constants import N_FFT


def calculate_impulse_response(filter_arr: NDArray, n_fft=N_FFT) -> NDArray:
    # Compute impulse response
    impulse_response = np.fft.fftshift(np.fft.irfft(filter_arr, n_fft))

    # Make it symmetric
    center_sample = len(impulse_response) // 2 + 1
    return impulse_response[center_sample - 2000 : center_sample + 2000]


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


class ZeroPhaseFilter:
    def __init__(self, filter_func: Callable, sample_rate: int, n_fft: int = 2 * 4096):
        """Extract the target response from the given filter_func (which may be not
        zero-phase). The idea is to construct a zero-phase variant of the given
        filter_func."""
        self.sample_rate = sample_rate

        # Get the impulse response of the filter
        delta = np.zeros(n_fft, dtype=np.float32)
        delta[len(delta) // 2] = 1.0
        impulse_response = filter_func(delta, sample_rate)

        w, h = scipy.signal.freqz(impulse_response, worN=n_fft // 2 + 1)
        n_fft = n_fft
        linear_target_response = np.abs(h)
        self.impulse_response = calculate_impulse_response(
            linear_target_response, n_fft
        )

    def __call__(self, audio: NDArray[np.float32]):
        """Apply the zero-phase filter to the given audio. The sample rate of the audio
        should be the same as the sample_rate given when this class instance was
        initialized."""
        if audio.ndim == 2:
            placeholder = np.zeros(shape=audio.shape, dtype=np.float32)
            for chn_idx in range(audio.shape[0]):
                placeholder[chn_idx, :] = scipy.signal.oaconvolve(
                    audio[chn_idx], self.impulse_response, "same"
                ).astype(np.float32)
            return placeholder
        else:
            return scipy.signal.oaconvolve(audio, self.impulse_response, "same").astype(
                np.float32
            )
