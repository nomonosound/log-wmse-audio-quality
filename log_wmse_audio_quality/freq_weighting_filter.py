import pickle
from typing import Optional
from importlib.resources import files
import numpy as np
import scipy.signal
from numpy.typing import NDArray


class HumanHearingSensitivityFilter:
    def __init__(self, impulse_response: Optional[NDArray] = None):
        if impulse_response is None:
            ir_path = files("log_wmse_audio_quality").joinpath("filter_ir.pkl")
            with ir_path.open("rb") as f:
                self.impulse_response = pickle.load(f)
        else:
            self.impulse_response = impulse_response

    def __call__(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply the filter to the given audio. The sample rate of the audio
        should be the same as the impulse response."""
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
