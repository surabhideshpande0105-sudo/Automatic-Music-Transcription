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
        return n, f"Built-in finder v3 ({instrument}, lead + chord voices)"


def transcribe_file_librosa_v2(wav_path: str, out_mid: str, sr: int = 22050,
                               min_f: float = 30.0, max_f: float = 4200.0,
                               lo_midi: int = 21, hi_midi: int = 108,
                               slice_seconds: float = 0.5,
                               max_voices: int = 2) -> int:
    """HPSS + pyin lead line + residual second voice, covering the full duration.

    - Harmonic/percussive separation drops drums so pitched lines survive mixes.
    - pyin (with HMM smoothing) octave-jumps far less than raw spectral argmax,
      fixing high-note errors.
    - A residual peak after masking the lead's harmonics adds one chord tone.
    """
    import librosa
    from amt_score import write_midi
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    if y.size == 0:
        write_midi(out_mid, [])
        return 0
    hop = 512
    try:
        yh = librosa.effects.harmonic(y, margin=4.0)
    except Exception:
        yh = y
    S = np.abs(librosa.stft(yh, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    times = librosa.times_like(S, sr=sr, hop_length=hop)
    frame_seconds = hop / sr
    try:
        f0, voiced, _ = librosa.pyin(yh, fmin=max(20.0, min_f), fmax=max_f,
                                      sr=sr, hop_length=hop)
    except Exception:
        f0, voiced = None, None
    try:
        onsets = librosa.onset.onset_detect(y=yh, sr=sr, hop_length=hop, backtrack=True)
    except Exception:
        onsets = []
    bounds = sorted(set([0] + [int(o) for o in onsets] + [S.shape[1]]))
    max_slice_frames = max(1, int(slice_seconds / frame_seconds))
    notes = []

    def add_note(midi_float: float, t0: float, t1: float) -> None:
        midi = int(round(midi_float))
        if lo_midi <= midi <= hi_midi:
            if t1 - t0 < 0.12:
                t1 = t0 + 0.3
            notes.append({"pitch": midi, "velocity": 80,
                          "start_seconds": t0, "end_seconds": t1})

    def lead_midi(a: int, b: int) -> float | None:
        seg_spec = S[:, a:b].mean(axis=1)
        if seg_spec.size == 0:
            return None
        spec_midi: float | None = None
        j = int(np.argmax(seg_spec))
        f = float(freqs[j])
        if min_f < f < max_f and float(seg_spec[j]) > max(0.003, float(np.median(seg_spec)) * 1.5):
            spec_midi = float(librosa.hz_to_midi(f))
        pm: float | None = None
        if f0 is not None and voiced is not None:
            seg = f0[a:b]
            ok = seg[voiced[a:b] & ~np.isnan(seg)]
            if ok.size >= 3:
                pm = float(librosa.hz_to_midi(float(np.median(ok))))
        if pm is None:
            return spec_midi
        # pyin period-doubles on chords: lift sub-octave guesses toward the
        # observed spectral peak so masking hits the right harmonic series.
        if spec_midi is not None:
            while spec_midi - pm >= 11 and pm + 12 <= hi_midi:
                pm += 12
        if pm is not None and (pm < lo_midi - 12 or pm > hi_midi + 12):
            return spec_midi
        return pm

    def chord_midi(a: int, b: int, lead: float | None) -> float | None:
        if max_voices < 2 or lead is None:
            return None
        seg = S[:, a:b].mean(axis=1)
        if seg.size == 0:
            return None
        lead_f = float(librosa.midi_to_hz(lead))
        mask = np.ones_like(seg, dtype=bool)
        for k in range(1, 9):
            mask &= np.abs(freqs - k * lead_f) > (freqs[1] - freqs[0]) * 1.5
        if not mask.any():
            return None
        rest = seg[mask]
        j = int(np.argmax(rest))
        idx = np.flatnonzero(mask)[j]
        f = float(freqs[idx])
        if min_f < f < max_f and float(rest[j]) > float(np.median(seg)) * 3.0:
            cand = float(librosa.hz_to_midi(f))
            if abs(cand - lead) >= 3:
                return cand
        return None

    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        s = a
        while s < b:
            e = min(b, s + max_slice_frames)
            lead = lead_midi(s, e)
            if lead is not None:
                t0 = float(times[s])
                t1 = float(times[min(e, len(times) - 1)])
                add_note(lead, t0, t1)
                second = chord_midi(s, e, lead)
                if second is not None:
                    add_note(second, t0, t1)
            s = e
    notes.sort(key=lambda n: (n["start_seconds"], n["pitch"]))
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
