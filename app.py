"""Piano to Sheet Music. Run: streamlit run app.py, open http://localhost:8501"""
from __future__ import annotations
from pathlib import Path
import tempfile
import html
import streamlit as st
import streamlit.components.v1 as components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pretty_midi

from amt_offline import transcribe_file_auto
try:
    from amt_offline import INSTRUMENT_PRESETS
except ImportError:
    INSTRUMENT_PRESETS = {
        "Any single instrument": {"min_f": 30.0, "max_f": 4200.0, "lo": 21, "hi": 108},
        "Piano": {"min_f": 27.5, "max_f": 4200.0, "lo": 21, "hi": 108},
        "Guitar": {"min_f": 70.0, "max_f": 1400.0, "lo": 28, "hi": 84},
        "Violin": {"min_f": 190.0, "max_f": 3500.0, "lo": 55, "hi": 100},
        "Flute": {"min_f": 250.0, "max_f": 2400.0, "lo": 60, "hi": 96},
        "Bass": {"min_f": 30.0, "max_f": 500.0, "lo": 21, "hi": 60},
    }
from amt_score import midi_to_musicxml

st.set_page_config(page_title="Music to Sheet Music", page_icon="🎵", layout="wide")
st.title("🎵 Music to Sheet Music")
st.write("Upload or record an instrument, get sheet music. One instrument at a time works best.")

with st.expander("How does it work? (30 seconds)", expanded=False):
    st.write("1. Pick your music below. 2. Press the big **Transcribe** button. "
             "3. Listen, read the score, download. Short piano clips under 2 minutes work best.")

# ---- Step 1: input ----
st.header("Step 1 — Pick your music")
choice = st.radio("Choose one:", ["Upload a file", "Record with mic", "Try a sample"], horizontal=True)
audio_bytes: bytes | None = None
fname = "input.wav"

if choice == "Upload a file":
    up = st.file_uploader("Drop a WAV/MP3 recording (one instrument is best)", type=["wav", "mp3", "ogg", "flac", "m4a"])
    if up:
        audio_bytes, fname = up.read(), up.name
elif choice == "Record with mic":
    rec = st.audio_input("Press record, play a few notes, then stop")
    if rec:
        audio_bytes, fname = rec.read(), "mic-recording.wav"
else:
    sample = Path("my_first_audio.wav")
    if sample.exists() and st.button("Load sample clip"):
        audio_bytes, fname = sample.read_bytes(), sample.name
    else:
        st.info("Sample: my_first_audio.wav (press the button after selecting this option).")

if not audio_bytes:
    st.info("👆 Choose music above to continue.")
    st.stop()

st.header("Step 2 — Listen and transcribe")
st.audio(audio_bytes)
st.write(f"Selected: **{html.escape(fname)}** ({len(audio_bytes)/1024:.0f} KB)")
instrument = st.selectbox("Which instrument is this?",
                          list(INSTRUMENT_PRESETS.keys()),
                          help="Narrows the pitch range. Full band mixes are not supported yet — record one instrument at a time.")

if not st.button("✨ Transcribe my music", type="primary", use_container_width=True):
    st.stop()

# ---- Transcribe ----
with st.spinner("Listening to every note… (takes a few seconds)"):
    with tempfile.TemporaryDirectory() as d:
        src = str(Path(d) / Path(fname).name)
        Path(src).write_bytes(audio_bytes)
        mid = str(Path(d) / "transcription.mid")
        try:
            n, engine = transcribe_file_auto(src, mid, instrument=instrument)
        except Exception as e:
            st.error(f"Sorry, I could not read that file ({e}). Try a WAV piano clip.")
            st.stop()
        if n == 0:
            st.warning("I heard sound but found no clear notes. Try a louder solo-instrument clip.")
            st.stop()
        m = pretty_midi.PrettyMIDI(mid)
        notes = [x for i in m.instruments for x in i.notes]
        names = [pretty_midi.note_number_to_name(x.pitch) for x in notes]
        try:
            import librosa
            y, sr = librosa.load(src, sr=22050, mono=True)
        except Exception:
            y, sr = None, 22050
        # sonify transcription so users can compare by ear
        try:
            import soundfile as sf
            wav_data = m.synthesize(fs=22050)
            son_path = str(Path(d) / "transcription.wav")
            sf.write(son_path, wav_data, 22050)
            son_bytes = Path(son_path).read_bytes()
        except Exception:
            son_bytes = None
        xml = str(Path(d) / "transcription.musicxml")
        ok = midi_to_musicxml(mid, xml)
        xml_text = Path(xml).read_text(encoding="utf-8") if ok else ""
        mid_bytes = Path(mid).read_bytes()

st.header("Step 3 — Your sheet music")
c1, c2 = st.columns(2)
with c1:
    st.subheader("🔊 Compare by ear")
    st.write("Original:")
    st.audio(audio_bytes)
    if son_bytes:
        st.write("My transcription (computer playing what it heard):")
        st.audio(son_bytes)
with c2:
    st.subheader("🎵 Notes found")
    st.write(f"**{n} notes** ({engine}). First notes: **{', '.join(names[:12])}**")
    if y is not None:
        fig, ax = plt.subplots(figsize=(8, 2.2))
        ax.plot(y[::20])
        ax.set_title("Your recording (loudness)")
        ax.set_xlabel("time →")
        st.pyplot(fig)

st.subheader("📊 Piano-roll")
fig2, ax2 = plt.subplots(figsize=(10, 3))
for x in notes[:500]:
    ax2.broken_barh([(x.start, max(0.05, x.end - x.start))], (x.pitch - 0.4, 0.8))
ax2.set_xlabel("seconds →")
ax2.set_ylabel("pitch (low → high)")
ax2.set_ylim(21, 108)
st.pyplot(fig2)

st.subheader("📄 Sheet music (read here, no extra app needed)")
if xml_text:
    esc = xml_text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    components.html(f"""
      <script src="https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@1.8.5/build/opensheetmusicdisplay.min.js"></script>
      <div id="osmd" style="border:1px solid #ddd;background:#fff;padding:8px"></div>
      <script>const osmd=new opensheetmusicdisplay.OpenSheetMusicDisplay("osmd");
      osmd.load(`{esc}`).then(()=>osmd.render());</script>
    """, height=620, scrolling=True)
else:
    st.warning("Sheet-music view failed, but MIDI below still works.")

st.subheader("⬇️ Keep your music")
st.download_button("Download MIDI (playable notes)", mid_bytes, "my-transcription.mid", use_container_width=True)
if xml_text:
    st.download_button("Download Sheet Music (MusicXML)", xml_text, "my-transcription.musicxml", use_container_width=True)
    st.caption("MusicXML opens in MuseScore, Flat.io, or soundslice.com — or just read it above.")
with st.expander("Advanced details"):
    st.write(f"Engine: {engine}. Basic Pitch handles any single instrument with chords; "
             "dense band mixes need source separation first (roadmap: Demucs stems + per-stem transcription).")
