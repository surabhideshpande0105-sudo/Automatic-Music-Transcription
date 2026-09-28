"""Offline file -> MIDI. Basic Pitch if present (any single instrument), else built-in finder."""
from __future__ import annotations
import numpy as np

INSTRUMENT_PRESETS: dict[str, dict] = {
    "Any single instrument": {"min_f": 30.0, "max_f": 4200.0, "lo": 21, "hi": 108},
    "Piano": {"min_f": 27.5, "max_f": 4200.0, "lo": 21, "hi": 108},
    "Guitar": {"min_f": 70.0, "max_f": 1400.0, "lo": 28, "hi": 84},
    "Violin": {"min_f": 190.0, "max_f": 3500.0, "lo": 55, "hi": 100},
    "Flute": {"min_f": 250.0, "max_f": 2400.0, "lo": 60, "hi": 96},
    "Bass": {"min_f": 30.0, "max_f": 500.0, "lo": 21, "hi": 60},
}

def transcribe_file_auto(wav_path: str, out_mid: str, sr: int = 22050,
                         instrument: str = "Any single instrument") -> tuple[int, str]:
    """Try Basic Pitch if installed, else improved librosa. Returns (note_count, engine)."""
    p = INSTRUMENT_PRESETS.get(instrument, INSTRUMENT_PRESETS["Any single instrument"])
    try:
        from amt_baseline import BasicPitchWrapper
        BasicPitchWrapper().transcribe_file(wav_path, out_mid,
                                            min_freq=p["min_f"], max_freq=p["max_f"])
        import pretty_midi
        n = len([x for i in pretty_midi.PrettyMIDI(out_mid).instruments for x in i.notes])
        return n, f"Basic Pitch ({instrument})"
    except Exception:
        n = transcribe_file_librosa_v2(wav_path, out_mid, sr=sr, min_f=p["min_f"],
                                       max_f=p["max_f"], lo_midi=p["lo"], hi_midi=p["hi"])
        return n, f"Built-in finder ({instrument}, single notes)"


def transcribe_file_librosa_v2(wav_path: str, out_mid: str, sr: int = 22050,
                               min_f: float = 30.0, max_f: float = 4200.0,
                               lo_midi: int = 21, hi_midi: int = 108,
                               slice_seconds: float = 0.5) -> int:
    """Onset-segmented monophonic transcription covering the full duration.

    Long onset segments are subdivided into fixed slices so sustained/
    legato passages (few onsets) still emit notes across the whole file.
    """
    import librosa
    from amt_score import write_midi
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    if y.size == 0:
        write_midi(out_mid, [])
        return 0
    hop = 512
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    pitches, mags = librosa.piptrack(S=S, sr=sr)
    times = librosa.times_like(S, sr=sr, hop_length=hop)
    frame_seconds = hop / sr
    try:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True)
    except Exception:
        onsets = []
    bounds = sorted(set([0] + [int(o) for o in onsets] + [S.shape[1]]))
    max_slice_frames = max(1, int(slice_seconds / frame_seconds))
    notes = []
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        # Subdivide long legato stretches so output spans the whole file.
        s = a
        while s < b:
            e = min(b, s + max_slice_frames)
            seg = mags[:, s:e]
            if seg.size == 0:
                s = e
                continue
            mean_mag = seg.mean(axis=1)
            j = int(np.argmax(mean_mag))
            f = float(pitches[min(j, pitches.shape[0] - 1), min(s, pitches.shape[1] - 1)])
            peak = float(seg.max())
            floor = float(np.median(seg) * 1.5)
            if min_f < f < max_f and peak > max(0.003, floor):
                midi = int(round(69 + 12 * np.log2(f / 440.0)))
                t0 = float(times[s])
                t1 = float(times[min(e, len(times) - 1)])
                if t1 - t0 < 0.12:
                    t1 = t0 + 0.3
                notes.append({"pitch": max(lo_midi, min(hi_midi, midi)), "velocity": 80,
                              "start_seconds": t0, "end_seconds": t1})
            s = e
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
