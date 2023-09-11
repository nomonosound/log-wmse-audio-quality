from log_wmse_audio_quality.utils import convert_decibels_to_amplitude_ratio

# N_FFT relates to the calculation of the impulse response of the human hearing
# sensitivity filter
N_FFT = 4096

# ERROR_TOLERANCE_THRESHOLD is relative to 0 dB RMS
ERROR_TOLERANCE_THRESHOLD = convert_decibels_to_amplitude_ratio(-68.0)

# This scaler makes the scale of values closer to SDR, where an increase
# in the tenths place is a meaningful improvement. I believe this makes it easier to
# compare numbers at a glance, e.g. when numbers are presented in a table.
SCALER = -4.0

# The main purpose of EPS is to avoid taking the log of a zero value
EPS = 1e-8

# INTERNAL_SAMPLE_RATE is the sample rate at which the metric operates internally,
# mainly due to the human hearing sensitivity filter. When audio with different
# sample rate gets passed, it gets resampled to INTERNAL_SAMPLE_RATE.
INTERNAL_SAMPLE_RATE = 44100
