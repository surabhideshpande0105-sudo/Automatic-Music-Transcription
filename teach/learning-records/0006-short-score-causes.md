# Two bugs made scores short, not the music

Chariots rendered as 2 bars for two stacked reasons: reversed VLQ bytes corrupted every MIDI timestamp past 0.13s, and one-note-per-onset emitted ~2 notes for legato passages. Fixed encoder in both copies plus 0.5s slice subdivision. Lesson: verify writer output with an independent reader before blaming the model.
