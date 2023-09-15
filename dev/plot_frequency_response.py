from typing import Callable, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from dev.create_freq_weighting_filter_ir import (
    get_human_hearing_sensitivity_filter_set,
    get_zero_phase_equivalent_filter_impulse_response,
)
from log_wmse_audio_quality.freq_weighting_filter import (
    HumanHearingSensitivityFilter,
)


def calculate_rms_db(samples: NDArray):
    """
    Given a numpy array of audio samples, return its RMS power level in decibels.
    Avoids NaNs when provided with zeros as input.
    """
    return (
        20 * np.log10(1000000 * np.sqrt(np.mean(np.square(samples), axis=-1)) + 1)
        - 120.0
    )


def get_frequency_response(
    filter_func: Callable, frequencies: Iterable, sample_rate: int
) -> List[np.float64]:
    T = 1 / sample_rate
    t = np.arange(0, 1, T)  # 1 second worth of samples
    loudness_differences = []

    for freq in tqdm(frequencies):
        # Generate sinusoid
        sinusoid = np.sin(2 * np.pi * freq * t).astype(np.float32)
        unfiltered_rms = calculate_rms_db(sinusoid)

        # Pass through filter
        output = filter_func(sinusoid)

        filtered_rms = calculate_rms_db(output)

        loudness_differences.append(filtered_rms - unfiltered_rms)

    return loudness_differences


if __name__ == "__main__":
    # This script plots the frequency response of the human hearing sensitivity filter.
    sample_rate = 44100

    filters = get_human_hearing_sensitivity_filter_set()
    filters_simple_callable = lambda x: filters(x, sample_rate)
    filters2 = HumanHearingSensitivityFilter(
        get_zero_phase_equivalent_filter_impulse_response(
            filters, sample_rate=sample_rate
        )
    )

    # Frequencies to analyze
    frequencies = np.linspace(20, 20000, 400)

    loudness_differences = get_frequency_response(
        filters_simple_callable, frequencies, sample_rate
    )
    loudness_differences2 = get_frequency_response(filters2, frequencies, sample_rate)

    # Plot
    plt.figure(figsize=(9, 4))
    plt.semilogx(frequencies, loudness_differences, label="audiomentations filter")
    # plt.semilogx(frequencies, loudness_differences2, label="zero phase alternative")
    plt.title("Frequency response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("dB")
    xticks = [25, 50, 100, 200, 400, 800, 1500, 3000, 6000, 10000, 20000]
    xlabels = [25, 50, 100, 200, 400, 800, 1500, "3k", "6k", "10k", "20k"]
    plt.xticks(xticks, xlabels)
    plt.grid()
    # plt.legend()
    plt.show()
