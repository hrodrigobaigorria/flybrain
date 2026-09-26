"""Neural runtime settings for the fly-brain engine. No trading, no market fields."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    neural_ms: float = 500  # neural time advanced per observation
    neural_bin_ms: float = 10  # integration bin size (must be <= 10 ms)
    pulse_ms: float = 200  # width of a reinforcement pulse
    pulse_current: float = 20  # reinforcement pulse amplitude (mV-equivalent)
    decoder_threshold_hz: float = 2  # DNp20 right-left difference needed to steer
    learning: bool = True  # apply the candidate KC->MBON plasticity

    def __post_init__(self):
        for x in (
            self.neural_ms,
            self.neural_bin_ms,
            self.pulse_ms,
            self.pulse_current,
            self.decoder_threshold_hz,
        ):
            if not math.isfinite(x) or x <= 0:
                raise ValueError("Settings values must be finite and positive")
        if self.neural_bin_ms > 10 or self.pulse_ms > self.neural_ms:
            raise ValueError("Use <=10 ms bins; a pulse must fit inside one decision window")
