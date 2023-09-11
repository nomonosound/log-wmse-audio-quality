# logWMSE

This audio quality metric, logWMSE, tries to address a limitation of existing metrics:
The lack of support for digital silence. Existing audio quality metrics, like VISQOL,
CDPAM, SDR, SIR, SAR, ISR, STOI and SI-SDR are not well-behaved when the target is
digital silence.

# Installation

`pip install git+https://github.com/nomonosound/log-wmse-audio-quality`

# Usage example

```python
import numpy as np
from log_wmse_audio_quality import calculate_log_wmse

sample_rate = 44100
input_sound = np.random.uniform(low=-1.0, high=1.0, size=(sample_rate,)).astype(
    np.float32
)
input_sound_copy = np.copy(input_sound)
est_sound = input_sound * 0.1
est_sound_copy = np.copy(est_sound)
target_sound = np.zeros((sample_rate,), dtype=np.float32)
target_sound_copy = np.copy(target_sound)

log_wmse = calculate_log_wmse(input_sound, est_sound, target_sound, sample_rate)
print(log_wmse)  # Expected output: ~18.42
```

# Motivation and more info

Here are some examples of use cases where the target (reference) is pure digital silence:

* **Music source separation:** Imagine separating music into stems like "vocal", "drums",
 "bass", and "other". A song without bass would naturally have a silent target for the "bass" stem.
* **Speech denoising:** If you have a recording with only noise, and no speech, the target
 should be digital silence.
* **Multichannel speaker separation** When evaluating speaker separation in a windowed
 approach, periods when a speaker isn't speaking should be evaluated against silence.

Mean squared error (MSE) is well-defined for digital silence targets, but has several drawbacks:

* The values are commonly ridiculously small, like between 1e-8 and 1e-3, which makes number formatting and sight-reading hard
* It's not tailored specifically for audio
* Lack of scale-invariance
* Doesn't consider the frequency sensitivity of human hearing
* It's not invariant to tiny errors that don't matter (because humans can't hear those errors anyway)
* It's not logarithmic, like human hearing is

**logWMSE** attempts to solve all the problems mentioned above. It's essentially the **log**
of a frequency-**weighted MSE**, with a few bells and whistles.

The frequency weighting is like this:
![frequency_weighting.png](plot/frequency_weighting.png)

This audio quality metric is optimized for high sample rates, like 36000, 44100 and 48000 Hz.

Unlike many audio quality metrics, logWMSE accepts a *triple* of audio inputs:

* unprocessed audio (e.g. a raw, noisy recording)
* processed audio (e.g. a denoised recording)
* target audio (e.g. a clean reference without noise)

Relative audio quality metrics usually only input the two latter. However, logWMSE
additionally needs the unprocessed audio, because it needs to be able to measure how
well input audio was attenuated to the given target when the target is digital silence
(all zeros). And it needs to do this in a "scale-invariant" way. In other words, the
metric score should be the same if you gain the audio triplet by an arbitrary amount.

However, note the following:

* The metric isn't invariant to arbitrary scaling, polarity inversion, or offsets in the estimated audio *relative to the target*.
* Although it incorporates frequency filtering inspired by human auditory sensitivity, it doesn't fully model human auditory perception. For instance, it doesn't consider auditory masking.
