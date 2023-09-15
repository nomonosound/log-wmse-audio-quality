import math
from typing import Callable, Iterable, List

import fast_align_audio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from log_wmse_audio_quality.freq_weighting_filter import (
    get_human_hearing_sensitivity_filter_set,
    Convolver,
    get_zero_phase_equivalent_filter_impulse_response,
)


def get_phase_shifts_deg(
    filter_func: Callable, frequencies: Iterable
) -> List[np.float64]:
    T = 1 / 44100
    t = np.arange(0, 1, T)  # 1 second worth of samples

    phase_shifts = []

    for freq in tqdm(frequencies):
        # Generate sinusoid
        sinusoid = np.sin(2 * np.pi * freq * t).astype(np.float32)

        # Pass through filters
        output = filter_func(sinusoid)

        # Normalize the output for compatibility with the MSE-based offset calculation
        output /= np.amax(output)

        # Find lag
        max_offset_samples = 1 + int(math.ceil(0.5 * 44100 / freq))
        lag, _ = fast_align_audio.find_best_alignment_offset(
            reference_signal=output,
            delayed_signal=sinusoid,
            max_offset_samples=max_offset_samples,
        )

        # Convert lag to phase shift in degrees
        phase_shift = lag * T * freq * 360  # in degrees

        phase_shifts.append(phase_shift)

    return phase_shifts


if __name__ == "__main__":
    # This script plots the phase response of the human hearing sensitivity filter.
    # Ideally, the filter should be zero-phase, so it does not introduce any offsets
    # that may negatively affect the precision of the metric.

    sample_rate = 44100
    filters = get_human_hearing_sensitivity_filter_set()
    filters_simple_callable = lambda x: filters(x, sample_rate)
    linear_phase_filter = Convolver(
        get_zero_phase_equivalent_filter_impulse_response(
            filters, sample_rate=sample_rate
        )
    )

    # Frequencies to analyze
    frequencies = np.linspace(20, 18000, 600)

    phase_shifts = get_phase_shifts_deg(filters_simple_callable, frequencies)
    phase_shifts2 = get_phase_shifts_deg(linear_phase_filter, frequencies)

    # Plot
    plt.figure()
    plt.semilogx(frequencies, phase_shifts)
    plt.semilogx(frequencies, phase_shifts2)
    plt.title("Filter phase response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase Shift (degrees)")
    plt.grid()
    plt.show()
