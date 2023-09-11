import numpy as np
import pytest
import soxr
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

from log_wmse_audio_quality import calculate_log_wmse
from log_wmse_audio_quality.freq_weighting_filter import (
    ZeroPhaseFilter,
    get_human_hearing_sensitivity_filter_set,
)
from plot.plot_filter_phase_shift import get_phase_shifts_deg
from plot.plot_frequency_response import get_frequency_response


def generate_sine_wave(frequency: float, length: int, sample_rate: int) -> NDArray:
    t = np.linspace(0, length / sample_rate, length, endpoint=False)
    return np.sin(2 * np.pi * frequency * t).astype(np.float32)


class TestMetrics:
    def test_no_side_effect(self):
        # Test that calculate_log_wmse does not change the numpy arrays passed to it
        np.random.seed(42)
        sample_rate = 44100
        input_sound = np.random.uniform(low=-1.0, high=1.0, size=(sample_rate,)).astype(
            np.float32
        )
        input_sound_copy = np.copy(input_sound)
        est_sound = input_sound * 0.1
        est_sound_copy = np.copy(est_sound)
        target_sound = np.zeros((sample_rate,), dtype=np.float32)
        target_sound_copy = np.copy(target_sound)

        calculate_log_wmse(input_sound, est_sound, target_sound, sample_rate)

        assert_array_equal(input_sound, input_sound_copy)
        assert_array_equal(est_sound, est_sound_copy)
        assert_array_equal(target_sound, target_sound_copy)

    def test_1d_2d_shape(self):
        np.random.seed(42)
        sample_rate = 44100
        input_sound_1d = np.random.uniform(
            low=-1.0, high=1.0, size=(sample_rate,)
        ).astype(np.float32)
        est_sound_1d = input_sound_1d * 0.1
        target_sound_1d = np.zeros((sample_rate,), dtype=np.float32)

        log_mse_1d = calculate_log_wmse(
            input_sound_1d, est_sound_1d, target_sound_1d, sample_rate
        )

        input_sound_2d_mono = input_sound_1d.reshape((1, -1))
        est_sound_2d_mono = est_sound_1d.reshape((1, -1))
        target_sound_2d_mono = target_sound_1d.reshape((1, -1))
        log_mse_2d_mono = calculate_log_wmse(
            input_sound_2d_mono, est_sound_2d_mono, target_sound_2d_mono, sample_rate
        )

        input_sound_2d_stereo = np.vstack((input_sound_1d, input_sound_1d))
        est_sound_2d_stereo = np.vstack((est_sound_1d, est_sound_1d))
        target_sound_2d_stereo = np.vstack((target_sound_1d, target_sound_1d))
        log_mse_2d_stereo = calculate_log_wmse(
            input_sound_2d_stereo,
            est_sound_2d_stereo,
            target_sound_2d_stereo,
            sample_rate,
        )

        assert log_mse_2d_mono == pytest.approx(log_mse_1d)
        assert log_mse_2d_stereo == pytest.approx(log_mse_1d)

    def test_digital_silence_target(self):
        np.random.seed(42)
        sample_rate = 44100
        input_sound = np.random.uniform(low=-1.0, high=1.0, size=(sample_rate,)).astype(
            np.float32
        )
        est_sound = input_sound * 0.1
        target_sound = np.zeros((sample_rate,), dtype=np.float32)

        log_mse = calculate_log_wmse(input_sound, est_sound, target_sound, sample_rate)
        assert log_mse == pytest.approx(18.42, abs=0.01)

    def test_digital_silence_input_and_target(self):
        np.random.seed(42)
        sample_rate = 44100
        input_sound = np.zeros((sample_rate,), dtype=np.float32)
        est_sound = np.zeros_like(input_sound)
        target_sound = np.zeros_like(input_sound)

        log_mse = calculate_log_wmse(input_sound, est_sound, target_sound, sample_rate)
        assert log_mse == pytest.approx(73.68, abs=0.01)

    def test_exact_match(self):
        np.random.seed(42)
        sample_rate = 44100
        target_sound = np.random.uniform(
            low=-0.1, high=0.1, size=(sample_rate,)
        ).astype(np.float32)
        input_sound = np.copy(target_sound)
        est_sound = np.copy(target_sound)

        log_mse = calculate_log_wmse(input_sound, est_sound, target_sound, sample_rate)
        assert log_mse == pytest.approx(73.68, abs=0.01)

    def test_greater_is_better(self):
        np.random.seed(42)
        sample_rate = 44100
        target_sound = np.random.uniform(
            low=-0.1, high=0.1, size=(sample_rate,)
        ).astype(np.float32)
        input_sound = target_sound + np.random.uniform(
            low=-0.5, high=0.5, size=(sample_rate,)
        ).astype(np.float32)
        est_sound1 = target_sound + np.random.uniform(
            low=-0.01, high=0.01, size=(sample_rate,)
        ).astype(np.float32)
        est_sound12 = target_sound + np.random.uniform(
            low=-0.001, high=0.001, size=(sample_rate,)
        ).astype(np.float32)

        log_mse1 = calculate_log_wmse(
            input_sound, est_sound1, target_sound, sample_rate
        )
        log_mse2 = calculate_log_wmse(
            input_sound, est_sound12, target_sound, sample_rate
        )
        assert log_mse2 > log_mse1

    def test_scale_invariance(self):
        np.random.seed(42)
        sample_rate = 44100
        target_sound = np.random.uniform(
            low=-0.1, high=0.1, size=(sample_rate,)
        ).astype(np.float32)
        input_sound = target_sound + np.random.uniform(
            low=-0.5, high=0.5, size=(sample_rate,)
        ).astype(np.float32)
        est_sound = target_sound + np.random.uniform(
            low=-0.01, high=0.01, size=(sample_rate,)
        ).astype(np.float32)

        log_mse1 = calculate_log_wmse(input_sound, est_sound, target_sound, sample_rate)

        target_sound *= 0.01
        est_sound *= 0.01
        input_sound *= 0.01

        log_mse2 = calculate_log_wmse(input_sound, est_sound, target_sound, sample_rate)
        assert log_mse2 == pytest.approx(log_mse1)

    def test_small_error_tolerance(self):
        np.random.seed(42)
        sample_rate = 44100
        target_sound = np.random.uniform(
            low=-0.1, high=0.1, size=(sample_rate,)
        ).astype(np.float32)
        input_sound = target_sound + np.random.uniform(
            low=-1.0, high=1.0, size=(44100,)
        ).astype(np.float32)
        est_sound1 = np.copy(target_sound)

        weak_hiss = np.random.uniform(
            low=-0.0001, high=0.0001, size=(sample_rate,)
        ).astype(np.float32)
        est_sound2 = est_sound1 + weak_hiss

        log_mse1 = calculate_log_wmse(
            input_sound, est_sound1, target_sound, sample_rate
        )
        log_mse2 = calculate_log_wmse(
            input_sound, est_sound2, target_sound, sample_rate
        )
        assert log_mse2 == pytest.approx(log_mse1, abs=0.01)

    def test_frequency_weighting_filter_response(self):
        np.random.seed(42)
        sample_rate = 44100
        filters = ZeroPhaseFilter(
            get_human_hearing_sensitivity_filter_set(), sample_rate
        )

        frequencies = np.linspace(20, 20000, 2000)
        frequency_response = get_frequency_response(
            filter_func=filters, frequencies=frequencies, sample_rate=sample_rate
        )

        freq_50hz_index = np.argmin(np.abs(frequencies - 50))
        freq_1000hz_index = np.argmin(np.abs(frequencies - 1000))
        freq_3000hz_index = np.argmin(np.abs(frequencies - 3000))
        freq_16000hz_index = np.argmin(np.abs(frequencies - 16000))

        assert frequency_response[freq_50hz_index] == pytest.approx(-16.0, abs=1.0)
        assert frequency_response[freq_1000hz_index] == pytest.approx(1.3, abs=1.0)
        assert frequency_response[freq_3000hz_index] == pytest.approx(4.2, abs=1.0)
        assert frequency_response[freq_16000hz_index] == pytest.approx(-11.8, abs=1.0)

    def test_frequency_weighting_filter_zero_phase(self):
        np.random.seed(42)
        sample_rate = 44100
        filters = ZeroPhaseFilter(
            get_human_hearing_sensitivity_filter_set(), sample_rate
        )
        frequencies = np.linspace(20, 18000, 400)
        phase_shifts_deg = get_phase_shifts_deg(filters, frequencies)
        # Check that the phase is no more than 1 degree off
        assert np.amax(phase_shifts_deg) < 1.0
        assert np.amin(phase_shifts_deg) > -1.0

    def test_human_hearing_frequency_sensitivity(self):
        np.random.seed(42)

        sample_rate = 44100

        target_sound = np.random.uniform(low=-0.03, high=0.3, size=(44100,)).astype(
            np.float32
        )
        low_freq_sine = 0.3 * generate_sine_wave(
            50.0, target_sound.shape[-1], sample_rate
        )
        high_freq_sine = 0.3 * generate_sine_wave(
            3000.0, target_sound.shape[-1], sample_rate
        )
        input_sound = np.copy(target_sound) + low_freq_sine + high_freq_sine
        # The 1st est reduced the low freq sine quite well, but not the high freq sine
        est_sound_enhanced_only_low_freq = (
            np.copy(target_sound) + 0.05 * low_freq_sine + high_freq_sine
        )
        # The 2nd est reduced the high freq sine quite well, but not the low freq sine
        est_sound_enhanced_only_high_freq = (
            np.copy(target_sound) + low_freq_sine + 0.05 * high_freq_sine
        )

        # Because human hearing is much more sensitive to 3000 Hz than 50 Hz, the
        # estimate that reduced the 3200 Hz sine should get a significantly higher score
        log_mse1 = calculate_log_wmse(
            input_sound, est_sound_enhanced_only_low_freq, target_sound, sample_rate
        )
        log_mse2 = calculate_log_wmse(
            input_sound, est_sound_enhanced_only_high_freq, target_sound, sample_rate
        )
        # We check that log_mse1 is much lower than log_mse2
        assert log_mse1 == pytest.approx(log_mse2 - 17.2, abs=0.5)

    def test_various_sample_rates(self):
        # Check that the metric returns similar scores for similar-sounding audio at
        # different sample rates
        np.random.seed(42)

        sample_rate = 44100

        target_sound = np.random.uniform(low=-0.03, high=0.3, size=(44100,)).astype(
            np.float32
        )
        low_freq_sine = 0.3 * generate_sine_wave(
            50.0, target_sound.shape[-1], sample_rate
        )
        high_freq_sine = 0.3 * generate_sine_wave(
            3000.0, target_sound.shape[-1], sample_rate
        )
        input_sound = np.copy(target_sound) + low_freq_sine + high_freq_sine
        # The 1st est reduced the low freq sine quite well, but not the high freq sine
        est_sound = np.copy(target_sound) + 0.05 * low_freq_sine + 0.05 * high_freq_sine

        # low-pass filter the sounds by downsampling them
        low_sr = 36000
        input_sound = soxr.resample(
            input_sound,
            sample_rate,
            low_sr,
            quality="HQ",
        )
        est_sound = soxr.resample(
            est_sound,
            sample_rate,
            low_sr,
            quality="HQ",
        )
        target_sound = soxr.resample(
            target_sound,
            sample_rate,
            low_sr,
            quality="HQ",
        )

        scores = []
        for sample_rate in [36000, 44100, 48000]:
            resampled_input_sound = soxr.resample(
                input_sound,
                low_sr,
                sample_rate,
                quality="HQ",
            )
            resampled_est_sound = soxr.resample(
                est_sound,
                low_sr,
                sample_rate,
                quality="HQ",
            )
            resampled_target_sound = soxr.resample(
                target_sound,
                low_sr,
                sample_rate,
                quality="HQ",
            )
            log_mse = calculate_log_wmse(
                resampled_input_sound,
                resampled_est_sound,
                resampled_target_sound,
                sample_rate,
            )
            scores.append(log_mse)

        # Check that all scores are equal
        assert scores[0] == pytest.approx(scores[1], abs=0.01)
        assert scores[0] == pytest.approx(scores[2], abs=0.01)
