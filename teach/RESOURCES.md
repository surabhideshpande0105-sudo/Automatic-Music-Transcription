# AMT App Resources

## Knowledge

- [Basic Pitch repo + ICASSP 2022 paper (Bittner et al.)](https://github.com/spotify/basic-pitch)
  Lightweight instrument-agnostic polyphonic CNN; paper at https://arxiv.org/abs/2203.09893. Use for: why Basic Pitch is the real model, ONNX vs TF runtimes, `predict()` API.
- [MAESTRO dataset (Magenta)](https://magenta.tensorflow.org/datasets/maestro)
  Paired piano audio+MIDI corpus. Use for: why `maestro-v3.0.0/` MIDI-only here can't train audio models, what full training needs.
- [librosa docs](https://librosa.org/doc/latest/index.html)
  STFT, piptrack, load/resample reference. Use for: `amt_offline.py` STFT→piptrack placeholder.
- [music21 docs](https://web.mit.edu/music21/)
  `converter.parse` + MusicXML writing. Use for: `midi_to_musicxml()` stage.
- [pretty_midi repo](https://github.com/craffel/pretty-midi)
  MIDI container + piano-roll + synth. Use for: `PrettyMIDI()` display and sonification.
- [Streamlit docs: file_uploader, audio_input, pyplot, download_button](https://docs.streamlit.io/)
  Frontend primitives used in `app.py`. Use for: upload/record/transcribe flow.
- [OpenSheetMusicDisplay](https://opensheetmusicdisplay.github.io/)
  Browser MusicXML renderer used in `generated/view_notation.html`. Use for: no-install score viewing.
- [python-sounddevice docs](https://python-sounddevice.readthedocs.io/)
  Callback input streams. Use for: `MicStream` 200ms blocks.
- [Standard MIDI Files spec (midi.org)](https://midi.org/specifications)
  Defines VLQ delta-time encoding. Use for: why byte order in `_varlen` matters.
- [PYIN pitch estimator (Mauch & Dixon, ICASSP 2014) via librosa](https://librosa.org/doc/latest/generated/librosa.pyin.html)
  Probabilistic thresholding with HMM smoothing. Use for: lead-line tracking in finder v3.

## Wisdom (Communities)

- [basicpitch.io demo](https://basicpitch.io/)
  Try the reference model on your own audio. Use for: calibrating what good polyphonic output looks like.
- [Magenta discussion](https://github.com/magenta/magenta/discussions)
  Practitioner Q&A on piano transcription. Use for: dataset and evaluation questions.

## Gaps

- No stable Windows Python 3.12 TensorFlow path for `basic-pitch[tf]`; ONNX default is the practical route until a 3.11 venv exists.
