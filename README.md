# logWMSE

This audio quality metric, logWMSE, tries to fix a shortcoming of common metrics: The
support for digital silence. Existing audio quality metrics, like VISQOL, CDPAM, SDR,
SIR, SAR, ISR, STOI and SI-SDR are not well-behaved when the target is digital silence.

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
print(log_wmse)  # 18.42
```

# Motivation and more info

Here are some examples of use cases where the target (reference) is pure digital silence:

* Music source separation with many stems, e.g. "vocal", "drums", "bass" and "other".
 Let's say you have a singer-songwriter song without bass in your test set. For that
 example it makes sense to have digital silence in the target for that stem.
* Speech denoising. If you have a recording with only noise, and no speech, the target
 should be digital silence.
* Multichannel speaker separation wth one target audio channel for each speaker. If you
 apply the metric in a windowed way, some segments are going to have digital silence in
 the target, because the speaker of interest is not speaking at that time.

MSE is well-defined for digital silence targets, but has a bunch of other issues:

* The values are commonly ridiculously small, like between 1e-8 and 1e-3, which makes number formatting and sight-reading hard
* It's not tailored for audio
* It's not scale-invariant
* It doesn't align with frequency sensitivity of human hearing
* It's not invariant to tiny errors that don't matter because humans can't hear those errors anyway
* It's not logarithmic, like human hearing is

So this custom metric attempts to solve all the problems mentioned above.
It's essentially the **log** of a **frequency-weighted MSE**, with a few bells and whistles.

The frequency weighting is like this:
![frequency_weighting.png](plot/frequency_weighting.png)

This audio quality metric is made for high frequencies, like 36000, 44100 and 48000 Hz.

Unlike many audio quality metrics, logWMSE accepts a *triple* of audio inputs:

* unprocessed audio (e.g. a raw, noisy recording)
* processed audio (e.g. a denoised recording)
* target audio (e.g. a clean reference without noise)

Relative audio quality metrics usually only input the two latter. However, logWMSE
additionally needs the unprocessed audio, because it needs to be able to measure how
well input audio was attenuated to the given target when the target is digital silence
(all zeros). And it needs to do this in a "scale-invariant" way. In other words, the
metric score should be the same if you gain the audio triplet by an arbitrary amount.

Note that this metric is not invariant to:

* Arbitrary scaling (gain) in the estimated audio (compared to the target audio)
* Opposite polarity in the estimated audio (compared to the target audio)
* An offset/delay in the estimated audio (compared to the target audio)

And although this metric implements frequency filtering, which is motivated by human
hearing sensitivity to different frequencies, it does not come with a complete
psychoacoustic models for perceptual audio. For example, it has no concept of auditory
masking.
