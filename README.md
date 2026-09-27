# Real-Time Automatic Music Transcription (Audio → Notation)

Polyphonic piano AMT with file + mic input. Target: MIDI events ~200–500ms, engraved score ~1–2s delayed.
Portfolio project in Python.

## How it works

```
.wav (file or mic) → transcription → .mid → MusicXML → viewer
```

* `amt_audio.py` — `FileChunker` (file) + `MicStream` (mic via `sounddevice`)
* `amt_offline.py` — `transcribe_file_librosa()` (no TF baseline; monophonic placeholder)
* `amt_baseline.py` — `EnergyGate` 200ms loop check + `BasicPitchWrapper` (real polyphonic, needs Python 3.11)
* `amt_score.py` — stdlib `write_midi()` + `midi_to_musicxml()` via `music21`
* `app.py` — Streamlit app: Upload / Record → Transcribe → piano-roll + downloads
* `generated/view_notation.html` — no-install score viewer (OSMD, pick a `.musicxml`)
* `train_model.py` / `readable.py` — earlier MAESTRO MIDI-generation baseline (not transcription)

## Quickstart (Windows, Python 3.12)

```powershell
cd project_1
pip install -r requirements_baseline.txt
python test_latency.py
python amt_offline.py my_first_audio.wav generated/test_new.mid
streamlit run app.py
# open http://localhost:8501 in Edge/Chrome
```

Transcribe any new file:

```powershell
python amt_offline.py my_song.wav generated/my_song.mid
# outputs generated/my_song.mid + generated/my_song.musicxml
```

View without MuseScore: double-click `generated/view_notation.html` → Choose File → pick a `.musicxml`. Or open `generated/pipeline_check.png`.

## Real polyphonic quality (Basic Pitch)

`basic-pitch 0.4.0` needs `tensorflow<2.15.1` → Python 3.11 only (you have 3.12/3.13).

```powershell
py -3.11 -m venv .venv-amt
.\.venv-amt\Scripts\Activate
pip install -r requirements_baseline.txt
pip install basic-pitch
python run_bp.py
```

Then swap `transcribe_file_librosa` for `BasicPitchWrapper().transcribe_file` in `app.py`.

## Status / roadmap

* [x] 200ms chunk I/O loop (`test_latency.py`: mean ~0.15ms, `budget_ok=True`)
* [x] File → MIDI → MusicXML + Streamlit file/mic app
* [ ] Streaming inference with 2s window / 200ms hop + overlap
* [ ] Beat/key quantization for stable notation
* [ ] Deploy (Streamlit Cloud / FastAPI)

## Data note

`maestro-v3.0.0/` here is MIDI-only. Full AMT training needs paired audio+MIDI (full MAESTRO, MAPS) or audio synthesized from MIDI. Ignored in git by default.
