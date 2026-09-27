"""Baseline detector: numpy-only onset gate + hook for Basic Pitch."""
from __future__ import annotations
import time
import numpy as np


class EnergyGate:
    """Placeholder polyphonic onset detector to validate 200ms budget."""
    def __init__(self, sr: int = 22050, thresh: float = 0.02):
        self.sr = sr
        self.thresh = thresh
        self.t = 0.0

    def process(self, chunk: np.ndarray) -> tuple[list[dict], float]:
        t0 = time.perf_counter()
        rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))
        events: list[dict] = []
        if rms > self.thresh:
            events.append({
                "pitch": 60, "velocity": 80,
                "start_seconds": self.t,
                "end_seconds": self.t + len(chunk) / self.sr,
            })
        self.t += len(chunk) / self.sr
        dt_ms = (time.perf_counter() - t0) * 1000
        return events, dt_ms


class BasicPitchWrapper:
    """Drop-in polyphonic model. Needs: pip install basic-pitch."""
    def __init__(self):
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
        self._predict = predict
        self._model = ICASSP_2022_MODEL_PATH

    def transcribe_file(self, wav_path: str, out_mid: str) -> None:
        _, midi_data, _ = self._predict(str(wav_path))
        midi_data.write(str(out_mid))
