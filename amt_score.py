"""Stdlib MIDI writer (no pretty_midi needed) + optional music21 export."""
from __future__ import annotations
import struct
from pathlib import Path


def _varlen(v: int) -> bytes:
    buf = v & 0x7F
    v >>= 7
    out = bytearray()
    while v:
        out.insert(0, (buf | 0x80) & 0xFF)
        buf = v & 0x7F
        v >>= 7
    out.append(buf)
    return bytes(out)


def write_midi(path: str | Path, notes: list[dict], bpm: float = 120.0) -> None:
    tpb = 480
    spb = 60.0 / bpm
    ev = []
    for n in notes:
        s = round(float(n["start_seconds"]) / spb * tpb)
        e = round(float(n.get("end_seconds", n["start_seconds"] + 0.4)) / spb * tpb)
        ev.append((s, 1, int(n["pitch"]), int(n.get("velocity", 80))))
        ev.append((e, 0, int(n["pitch"]), 0))
    ev.sort(key=lambda x: (x[0], x[1]))
    track = bytearray(b"\x00\xff\x51\x03\x07\xa1\x20")
    prev = 0
    for tick, on, p, v in ev:
        track.extend(_varlen(max(0, tick - prev)))
        prev = tick
        track.extend(bytes([0x90 if on else 0x80, p, v]))
    track.extend(b"\x00\xff\x2f\x00")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpb))
        f.write(b"MTrk" + struct.pack(">I", len(track)))
        f.write(track)


def midi_to_musicxml(mid_path: str | Path, out_xml: str | Path) -> bool:
    try:
        from music21 import converter
    except ImportError:
        return False
    score = converter.parse(str(mid_path))
    score.write("musicxml", str(out_xml))
    return True
