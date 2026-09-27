import pretty_midi

midi = pretty_midi.PrettyMIDI(r"C:\Users\Surabhi Deshpande\Downloads\Pirates of the Caribbean - He's a Pirate (3).mid")

for instrument in midi.instruments:

    print("\nInstrument:", instrument.name)

    for note in instrument.notes:

        note_name = pretty_midi.note_number_to_name(note.pitch)

        print(
            note_name,
            "Start:", round(note.start,2),
            "End:", round(note.end,2),
            "Velocity:", note.velocity
        )