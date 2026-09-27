from music21 import converter
from music21 import note

score = converter.parse(r"C:\Users\Surabhi Deshpande\Downloads\Pirates of the Caribbean - He's a Pirate (3).mid")

count = 0

for element in score.recurse():
    if isinstance(element, note.Note):
        print(element.nameWithOctave)
        count += 1

        if count == 20:
            break