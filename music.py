import pretty_midi

midi = pretty_midi.PrettyMIDI(r"C:\Users\Surabhi Deshpande\Downloads\Pirates of the Caribbean - He's a Pirate (3).mid")

print("Tempo:", midi.estimate_tempo())
print("Number of instruments:", len(midi.instruments))

for instrument in midi.instruments:
    print("\nInstrument:", instrument.name)
    print("Program:", instrument.program)
    print("Notes:", len(instrument.notes))

    for note in instrument.notes:
        print(
            "Pitch:", note.pitch,
            "Start:", note.start,
            "End:", note.end,
            "Velocity:", note.velocity
        )