"""AMT app: upload or record -> MIDI + MusicXML + piano-roll. Run: streamlit run app.py"""
from __future__ import annotations
from pathlib import Path
import tempfile
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pretty_midi

from amt_offline import transcribe_file_librosa
from amt_score import midi_to_musicxml

st.set_page_config(page_title="AMT - Audio to Notation", layout="wide")
st.title("Automatic Music Transcription: Audio -> Notation")

tab_file, tab_mic = st.tabs(["Upload file", "Record mic"])

audio_bytes, fname = None, "input.wav"
with tab_file:
    up = st.file_uploader("WAV works best (MP3 needs ffmpeg)", type=["wav", "mp3", "ogg", "flac"])
    if up:
        audio_bytes, fname = up.read(), up.name
with tab_mic:
    rec = st.audio_input("Record, then transcribe")
    if rec:
        audio_bytes, fname = rec.read(), "mic.wav"

if audio_bytes:
    st.audio(audio_bytes)
    if st.button("Transcribe", type="primary"):
        with tempfile.TemporaryDirectory() as d:
            src = str(Path(d) / fname)
            Path(src).write_bytes(audio_bytes)
            mid = str(Path(d) / "out.mid")
            n = transcribe_file_librosa(src, mid)
            m = pretty_midi.PrettyMIDI(mid)
            notes = [x for i in m.instruments for x in i.notes]
            fig, ax = plt.subplots(figsize=(10, 3))
            for x in notes[:500]:
                ax.broken_barh([(x.start, max(0.05, x.end-x.start))], (x.pitch-0.4, 0.8))
            ax.set_xlabel("sec"); ax.set_ylabel("pitch"); ax.set_ylim(21, 108)
            ax.set_title(f"notes={len(notes)}")
            st.pyplot(fig)
            xml = str(Path(d) / "out.musicxml")
            ok = midi_to_musicxml(mid, xml)
            st.write(f"Transcribed {n} notes. MusicXML: {'ok' if ok else 'failed'}")
            st.download_button("Download MIDI", Path(mid).read_bytes(), "transcription.mid")
            if ok:
                st.download_button("Download MusicXML (open in viewer / MuseScore)", Path(xml).read_bytes(), "transcription.musicxml")
                st.info("View: save the MusicXML, then open generated/view_notation.html > Choose File.")
else:
    st.info("Upload a file or record to start.")
