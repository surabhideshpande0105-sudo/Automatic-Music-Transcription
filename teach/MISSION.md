# Mission: Ship polyphonic piano AMT app

## Why
Turn any piano recording (file or mic) into playable notation to share as a portfolio piece and use for practice.

## Success looks like
- Upload or record audio in `app.py` and download correct `.mid` + `.musicxml`
- Explain every stage: audio chunks, transcription model, MIDI, notation, viewer
- Demo 200ms MIDI loop vs 1–2s score update tradeoff from memory

## Constraints
- Windows, Python 3.12 now; Python 3.11 venv only for Basic Pitch
- No MuseScore installed; use browser viewer + downloads
- Short sessions; one small win per lesson

## Out of scope
- Training a new neural net from scratch
- Multi-instrument separation and singing voice
- Mobile app packaging
