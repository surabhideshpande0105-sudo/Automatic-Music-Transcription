#!/usr/bin/env python3
"""
Convert the MAESTRO MIDI-only dataset into readable CSV/JSON files.

Outputs:
  - pieces.csv: one row per MAESTRO performance from metadata
  - notes/<piece_id>.notes.csv: note-level rows for each MIDI file
  - notes_index.csv: one row per converted MIDI file with note counts

This script uses only Python's standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


DEFAULT_DATASET = Path(__file__).resolve().parent / "maestro-v3.0.0"


@dataclass
class MidiNote:
    track: int
    channel: int
    pitch: int
    velocity: int
    start_tick: int
    end_tick: int
    start_seconds: float
    end_seconds: float

    @property
    def duration_ticks(self) -> int:
        return self.end_tick - self.start_tick

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def read_exact(handle: BinaryIO, count: int) -> bytes:
    data = handle.read(count)
    if len(data) != count:
        raise EOFError("Unexpected end of MIDI file")
    return data


def read_varlen(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, pos


def ticks_to_seconds(tick: int, tempo_map: list[tuple[int, int]], ticks_per_beat: int) -> float:
    seconds = 0.0
    last_tick = 0
    current_tempo = 500000

    for tempo_tick, tempo in tempo_map:
        if tempo_tick >= tick:
            break
        seconds += (tempo_tick - last_tick) * current_tempo / 1_000_000 / ticks_per_beat
        last_tick = tempo_tick
        current_tempo = tempo

    seconds += (tick - last_tick) * current_tempo / 1_000_000 / ticks_per_beat
    return seconds


def parse_track_events(track_data: bytes) -> tuple[list[tuple[int, int, int, int, int]], list[tuple[int, int]]]:
    """Return note events and tempo changes.

    Note event tuple:
      absolute_tick, event_type, channel, pitch, velocity
      event_type 1 = note_on, event_type 0 = note_off
    Tempo tuple:
      absolute_tick, microseconds_per_beat
    """
    pos = 0
    tick = 0
    running_status: int | None = None
    note_events: list[tuple[int, int, int, int, int]] = []
    tempos: list[tuple[int, int]] = []

    while pos < len(track_data):
        delta, pos = read_varlen(track_data, pos)
        tick += delta

        status = track_data[pos]
        if status & 0x80:
            pos += 1
            running_status = status
        elif running_status is not None:
            status = running_status
        else:
            raise ValueError("MIDI running status used before status byte")

        if status == 0xFF:
            meta_type = track_data[pos]
            pos += 1
            length, pos = read_varlen(track_data, pos)
            payload = track_data[pos : pos + length]
            pos += length

            if meta_type == 0x51 and length == 3:
                tempos.append((tick, int.from_bytes(payload, "big")))
            elif meta_type == 0x2F:
                break
            running_status = None
            continue

        if status in (0xF0, 0xF7):
            length, pos = read_varlen(track_data, pos)
            pos += length
            running_status = None
            continue

        event_type = status & 0xF0
        channel = status & 0x0F
        data_len = 1 if event_type in (0xC0, 0xD0) else 2
        event_data = track_data[pos : pos + data_len]
        pos += data_len

        if event_type == 0x90:
            pitch, velocity = event_data
            note_events.append((tick, 1 if velocity else 0, channel, pitch, velocity))
        elif event_type == 0x80:
            pitch, velocity = event_data
            note_events.append((tick, 0, channel, pitch, velocity))

    return note_events, tempos


def parse_midi_notes(midi_path: Path) -> list[MidiNote]:
    with midi_path.open("rb") as handle:
        if read_exact(handle, 4) != b"MThd":
            raise ValueError(f"{midi_path} is not a MIDI file")
        header_length = struct.unpack(">I", read_exact(handle, 4))[0]
        header = read_exact(handle, header_length)
        midi_format, track_count, division = struct.unpack(">HHH", header[:6])

        if division & 0x8000:
            raise ValueError("SMPTE time division is not supported")
        ticks_per_beat = division

        raw_tracks: list[bytes] = []
        for _ in range(track_count):
            chunk_type = read_exact(handle, 4)
            chunk_length = struct.unpack(">I", read_exact(handle, 4))[0]
            chunk_data = read_exact(handle, chunk_length)
            if chunk_type == b"MTrk":
                raw_tracks.append(chunk_data)

    all_track_events: list[tuple[int, int, int, int, int, int]] = []
    tempo_map = [(0, 500000)]

    for track_number, track_data in enumerate(raw_tracks):
        note_events, tempos = parse_track_events(track_data)
        tempo_map.extend(tempos)
        for tick, event_type, channel, pitch, velocity in note_events:
            all_track_events.append((tick, track_number, event_type, channel, pitch, velocity))

    tempo_map = sorted(set(tempo_map), key=lambda item: item[0])
    all_track_events.sort(key=lambda item: (item[0], item[2]))

    active: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    notes: list[MidiNote] = []

    for tick, track_number, event_type, channel, pitch, velocity in all_track_events:
        key = (track_number if midi_format != 0 else 0, channel, pitch)
        if event_type == 1:
            active.setdefault(key, []).append((tick, velocity))
            continue

        starts = active.get(key)
        if not starts:
            continue
        start_tick, start_velocity = starts.pop(0)
        if not starts:
            active.pop(key, None)

        notes.append(
            MidiNote(
                track=track_number,
                channel=channel,
                pitch=pitch,
                velocity=start_velocity,
                start_tick=start_tick,
                end_tick=tick,
                start_seconds=ticks_to_seconds(start_tick, tempo_map, ticks_per_beat),
                end_seconds=ticks_to_seconds(tick, tempo_map, ticks_per_beat),
            )
        )

    return sorted(notes, key=lambda note: (note.start_seconds, note.pitch, note.end_seconds))


def safe_piece_id(row_number: int, midi_filename: str) -> str:
    stem = Path(midi_filename).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return f"{row_number:04d}_{stem[:90]}"


def write_notes_csv(path: Path, notes: list[MidiNote]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "track",
                "channel",
                "pitch",
                "velocity",
                "start_seconds",
                "end_seconds",
                "duration_seconds",
                "start_tick",
                "end_tick",
                "duration_ticks",
            ]
        )
        for note in notes:
            writer.writerow(
                [
                    note.track,
                    note.channel,
                    note.pitch,
                    note.velocity,
                    f"{note.start_seconds:.6f}",
                    f"{note.end_seconds:.6f}",
                    f"{note.duration_seconds:.6f}",
                    note.start_tick,
                    note.end_tick,
                    note.duration_ticks,
                ]
            )


def convert_dataset(dataset_dir: Path, output_dir: Path, limit: int | None) -> None:
    metadata_csv = dataset_dir / "maestro-v3.0.0.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Could not find metadata CSV: {metadata_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    notes_dir = output_dir / "notes"
    notes_dir.mkdir(exist_ok=True)

    with metadata_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if limit is not None:
        rows = rows[:limit]

    pieces_path = output_dir / "pieces.csv"
    index_path = output_dir / "notes_index.csv"

    with pieces_path.open("w", newline="", encoding="utf-8") as pieces_handle, index_path.open(
        "w", newline="", encoding="utf-8"
    ) as index_handle:
        piece_fields = ["piece_id"] + list(rows[0].keys()) if rows else ["piece_id"]
        pieces_writer = csv.DictWriter(pieces_handle, fieldnames=piece_fields)
        pieces_writer.writeheader()

        index_writer = csv.writer(index_handle)
        index_writer.writerow(
            [
                "piece_id",
                "notes_csv",
                "note_count",
                "composer",
                "title",
                "split",
                "year",
                "duration",
                "midi_filename",
            ]
        )

        for row_number, row in enumerate(rows, start=1):
            piece_id = safe_piece_id(row_number, row["midi_filename"])
            midi_path = dataset_dir / row["midi_filename"]
            notes_path = notes_dir / f"{piece_id}.notes.csv"

            notes = parse_midi_notes(midi_path)
            write_notes_csv(notes_path, notes)

            piece_row = {"piece_id": piece_id, **row}
            pieces_writer.writerow(piece_row)
            index_writer.writerow(
                [
                    piece_id,
                    str(notes_path.relative_to(output_dir)),
                    len(notes),
                    row["canonical_composer"],
                    row["canonical_title"],
                    row["split"],
                    row["year"],
                    row["duration"],
                    row["midi_filename"],
                ]
            )

    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "pieces_converted": len(rows),
        "files": ["pieces.csv", "notes_index.csv", "notes/*.notes.csv"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MAESTRO MIDI files into readable note CSV files.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to the MAESTRO v3.0.0 folder.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("maestro_readable"),
        help="Folder where readable CSV/JSON files will be written.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Convert only the first N pieces for a quick test.")
    args = parser.parse_args()

    convert_dataset(args.dataset.resolve(), args.output.resolve(), args.limit)
    print(f"Wrote readable files to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
