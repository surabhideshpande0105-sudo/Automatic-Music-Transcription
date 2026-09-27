"""Audio sources yielding fixed-size mono float32 chunks."""
from __future__ import annotations
import queue
import wave
from pathlib import Path
import numpy as np


def read_wav_mono(path: str | Path, sr_target: int = 22050) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        n_ch, sw, sr, n, _, _ = w.getparams()
        raw = w.readframes(n)
    dtype = np.int16 if sw == 2 else np.int8
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if sw == 2:
        audio /= 32768.0
    else:
        audio /= 128.0
    if n_ch > 1:
        audio = audio.reshape(-1, n_ch).mean(axis=1)
    if sr != sr_target:
        ratio = sr_target / sr
        idx = (np.arange(int(len(audio) * ratio)) / ratio).astype(int)
        idx = np.clip(idx, 0, len(audio) - 1)
        audio = audio[idx]
        sr = sr_target
    return audio, sr


class FileChunker:
    def __init__(self, path: str | Path, chunk_ms: int = 200, sr: int = 22050):
        self.audio, self.sr = read_wav_mono(path, sr)
        self.chunk = int(sr * chunk_ms / 1000)

    def __iter__(self):
        for i in range(0, len(self.audio), self.chunk):
            yield self.audio[i:i + self.chunk]


class MicStream:
    def __init__(self, sr: int = 22050, chunk_ms: int = 200, device=None):
        self.sr = sr
        self.chunk = int(sr * chunk_ms / 1000)
        self.device = device
        self.q: queue.Queue = queue.Queue(maxsize=64)

    def __enter__(self):
        import sounddevice as sd
        def cb(indata, frames, time, status):
            try:
                self.q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass
        self.stream = sd.InputStream(
            samplerate=self.sr, channels=1, dtype="float32",
            blocksize=self.chunk, device=self.device, callback=cb,
        )
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()

    def __iter__(self):
        while True:
            yield self.q.get()
