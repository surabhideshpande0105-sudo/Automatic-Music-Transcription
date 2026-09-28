# AMT Glossary

Compressed language for this project. Only terms the user can already use.

## Terms

**AMT**:
Audio to note-list conversion preserving pitch, onset, and duration.
_Avoid_: transcription magic, audio understanding

**MIDI**:
Performance data of pitch, velocity, and timing without engraving.
_Avoid_: sheet music, audio file

**MusicXML**:
Notation interchange format encoding measures, clefs, and noteheads.
_Avoid_: MIDI file, PDF score

**Chunk**:
Fixed 200ms mono audio block fed to the detector.
_Avoid_: sample, buffer blob

**VLQ**:
Variable-length byte code storing MIDI tick counts compactly.
_Avoid_: timestamp, delta code

**Onset**:
Sudden energy rise marking where a note likely starts.
_Avoid_: beat, peak sound

**HPSS**:
Split of audio into sustained harmonic versus percussive parts.
_Avoid_: noise filter, equalizer

**pyin**:
Probabilistic pitch tracker that smooths octave jumps over time.
_Avoid_: peak picker, tuner app
