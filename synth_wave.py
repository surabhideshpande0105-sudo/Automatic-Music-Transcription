import pretty_midi
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

# Load the MIDI file
midi = pretty_midi.PrettyMIDI(r"C:\Users\Surabhi Deshpande\Downloads\Pirates of the Caribbean - He's a Pirate (3).mid")  # Replace with your filename

# Synthesize to audio waveform
waveform = midi.synthesize(fs=44100)

# Save the synthesized audio as a WAV file
sf.write(r"C:\Users\Surabhi Deshpande\Downloads\pirates.wav", waveform, 44100)

# Plot the waveform
plt.figure(figsize=(15, 4))
librosa.display.waveshow(waveform, sr=44100)

plt.title("Waveform of Pirates of the Caribbean MIDI")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()