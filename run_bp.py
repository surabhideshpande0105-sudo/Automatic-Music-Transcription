from amt_baseline import BasicPitchWrapper
from amt_score import midi_to_musicxml

WAV = "my_first_audio.wav"
MID = "generated/bp.mid"
XML = "generated/bp.musicxml"

BasicPitchWrapper().transcribe_file(WAV, MID)
print(f"wrote {MID}")
ok = midi_to_musicxml(MID, XML)
print(f"musicxml: {'wrote ' + XML if ok else 'skipped (install music21)'}")
