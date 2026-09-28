"""Offline file -> MIDI using librosa (no TensorFlow). For Basic Pitch later use Python 3.11."""
from __future__ import annotations
import numpy as np

def transcribe_file_auto(wav_path: str, out_mid: str, sr: int = 22050) -> tuple[int, str]:
    """Try Basic Pitch if installed, else improved librosa. Returns (note_count, engine)."""
    try:
        from amt_baseline import BasicPitchWrapper
        BasicPitchWrapper().transcribe_file(wav_path, out_mid)
        import pretty_midi
        n = len([x for i in pretty_midi.PrettyMIDI(out_mid).instruments for x in i.notes])
        return n, "Basic Pitch (polyphonic)"
    except Exception:
        n = transcribe_file_librosa_v2(wav_path, out_mid, sr=sr)
        return n, "Built-in pitch finder (single notes)"


def transcribe_file_librosa_v2(wav_path: str, out_mid: str, sr: int = 22050) -> int:
    """Onset-segmented monophonic transcription: one note per onset segment."""
    import librosa
    from amt_score import write_midi
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    if y.size == 0:
        write_midi(out_mid, [])
        return 0
    hop = 512
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    pitches, mags = librosa.piptrack(S=np.abs(librosa.stft(y, n_fft=2048, hop_length=hop)), sr=sr)
    times = librosa.times_like(S, sr=sr, hop_length=hop)
    try:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True)
    except Exception:
        onsets = []
    bounds = sorted(set([0] + [int(o) for o in onsets] + [S.shape[1]]))
    notes = []
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        seg = mags[:, a:b]
        if seg.size == 0:
            continue
        j = a + int(np.argmax(seg.mean(axis=1)))
        f = float(pitches[min(j, pitches.shape[0] - 1), min(a, pitches.shape[1] - 1)])
        peak = float(seg.max())
        if 55 < f < 2000 and peak > 0.01:
            midi = int(round(69 + 12 * np.log2(f / 440.0)))
            t0 = float(times[a])
            t1 = float(times[min(b, len(times) - 1)])
            if t1 - t0 < 0.12:
                t1 = t0 + 0.3
            notes.append({"pitch": max(21, min(108, midi)), "velocity": 80,
                          "start_seconds": t0, "end_seconds": min(t1, t0 + 2.0)})
    write_midi(out_mid, notes)
    return len(notes)


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
    n, engine = transcribe_file_auto(inp, out)
    print(f"wrote {out} notes={n} engine={engine}")
    xml = str(Path(out).with_suffix(".musicxml"))
    print(f"musicxml: {xml} ok={midi_to_musicxml(out, xml)}")
