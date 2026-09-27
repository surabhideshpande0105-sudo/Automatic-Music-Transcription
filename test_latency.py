"""Verify 200ms chunk budget on existing wav using numpy-only baseline."""
from __future__ import annotations
import numpy as np
from amt_audio import FileChunker
from amt_baseline import EnergyGate
from amt_score import write_midi

WAV = "my_first_audio.wav"
CHUNK_MS = 200

chunks = list(FileChunker(WAV, chunk_ms=CHUNK_MS))
det = EnergyGate()
lats, events = [], []
for c in chunks:
    ev, dt = det.process(c)
    lats.append(dt)
    events.extend(ev)

lats = np.array(lats)
print(f"chunks={len(chunks)} chunk_ms={CHUNK_MS}")
print(f"mean={lats.mean():.3f}ms p95={np.percentile(lats,95):.3f}ms max={lats.max():.3f}ms")
print(f"budget_ok={bool((lats < CHUNK_MS).all())} events={len(events)}")
write_midi("generated/baseline_stream.mid", events)
print("wrote generated/baseline_stream.mid")
