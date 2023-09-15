import matplotlib.pyplot as plt

from log_wmse_audio_quality.freq_weighting_filter import (
    get_human_hearing_sensitivity_filter_set,
    get_zero_phase_equivalent_filter_impulse_response,
    Convolver,
)

if __name__ == "__main__":
    # This script plots the phase response of the human hearing sensitivity filter.
    # Ideally, the filter should be zero-phase, so it does not introduce any offsets
    # that may negatively affect the precision of the metric.

    sample_rate = 44100
    filters = get_human_hearing_sensitivity_filter_set()
    linear_phase_filter = Convolver(
        get_zero_phase_equivalent_filter_impulse_response(
            filters, sample_rate=sample_rate
        )
    )

    # Plot
    plt.figure()
    plt.plot(linear_phase_filter.impulse_response)
    plt.title("Filter impulse response")
    plt.grid()
    plt.show()
