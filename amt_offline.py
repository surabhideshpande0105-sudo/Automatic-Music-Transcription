"""Offline file -> MIDI using librosa (no TensorFlow). For Basic Pitch later use Python 3.11."""
from __future__ import annotations
import numpy as np

def transcribe_file_librosa(wav_path: str, out_mid: str, sr: int = 22050) -> int:
    import librosa
    from amt_score import write_midi
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    pitches, mags = librosa.piptrack(S=S, sr=sr)
    times = librosa.times_like(S, sr=sr, hop_length=512)
    notes = []
    for i in range(0, S.shape[1], 43):  # ~1s steps
        j = min(i, S.shape[1]-1)
        idx = mags[:, j].argmax()
        f = float(pitches[idx, j])
        if 55 < f < 2000 and mags[idx, j] > np.median(mags[:, j]):
            midi = int(round(69 + 12*np.log2(f/440.0)))
            notes.append({"pitch": max(21, min(108, midi)), "velocity": 80,
                          "start_seconds": float(times[j]),
                          "end_seconds": float(times[j]+0.5)})
    write_midi(out_mid, notes)
    return len(notes)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from amt_score import midi_to_musicxml
    inp = sys.argv[1] if len(sys.argv) > 1 else "my_first_audio.wav"
    out = sys.argv[2] if len(sys.argv) > 2 else f"generated/{Path(inp).stem}.mid"
    n = transcribe_file_librosa(inp, out)
    print(f"wrote {out} notes={n}")
    xml = str(Path(out).with_suffix(".musicxml"))
    print(f"musicxml: {xml} ok={midi_to_musicxml(out, xml)}")
