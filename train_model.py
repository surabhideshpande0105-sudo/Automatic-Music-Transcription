#!/usr/bin/env python3
"""
Train a simple MAESTRO note-generation model from readable note CSV files.

This is a dependency-free baseline:
  - Reads maestro_readable/notes_index.csv and notes/*.notes.csv
  - Learns note-to-note transition probabilities
  - Saves the trained model as JSON
  - Generates a readable notes CSV and a playable MIDI file

It is intentionally simple so you can verify the dataset and training loop before
installing heavier ML libraries like PyTorch or TensorFlow.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import struct
from collections import Counter, defaultdict
from pathlib import Path


Token = tuple[int, int, int, int]


def quantize(value: float, step: float, max_bucket: int) -> int:
    return max(0, min(max_bucket, round(value / step)))


def token_to_key(token: Token) -> str:
    return ",".join(str(part) for part in token)


def key_to_token(key: str) -> Token:
    pitch, velocity_bin, duration_bin, gap_bin = key.split(",")
    return int(pitch), int(velocity_bin), int(duration_bin), int(gap_bin)


def weighted_choice(counts: dict[str, int]) -> str:
    total = sum(counts.values())
    target = random.randint(1, total)
    running = 0
    for key, count in counts.items():
        running += count
        if running >= target:
            return key
    return next(iter(counts))


def read_note_tokens(notes_csv: Path) -> list[Token]:
    notes: list[dict[str, float]] = []
    with notes_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            notes.append(
                {
                    "pitch": int(row["pitch"]),
                    "velocity": int(row["velocity"]),
                    "start": float(row["start_seconds"]),
                    "duration": max(0.03, float(row["duration_seconds"])),
                }
            )

    notes.sort(key=lambda item: (item["start"], item["pitch"]))
    tokens: list[Token] = []
    previous_start = 0.0

    for note in notes:
        gap = max(0.0, note["start"] - previous_start)
        previous_start = note["start"]
        tokens.append(
            (
                int(note["pitch"]),
                quantize(note["velocity"], step=8, max_bucket=15),
                quantize(note["duration"], step=0.05, max_bucket=80),
                quantize(gap, step=0.05, max_bucket=80),
            )
        )

    return tokens


def train(readable_dir: Path, output_model: Path, max_pieces: int | None) -> None:
    index_csv = readable_dir / "notes_index.csv"
    if not index_csv.exists():
        raise FileNotFoundError(f"Could not find {index_csv}")

    start_counts: Counter[str] = Counter()
    transitions: dict[str, Counter[str]] = defaultdict(Counter)
    pieces_seen = 0
    notes_seen = 0

    with index_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if max_pieces is not None and pieces_seen >= max_pieces:
                break

            notes_csv = readable_dir / row["notes_csv"]
            if not notes_csv.exists():
                continue

            tokens = read_note_tokens(notes_csv)
            if len(tokens) < 2:
                continue

            pieces_seen += 1
            notes_seen += len(tokens)
            start_counts[token_to_key(tokens[0])] += 1

            for previous, current in zip(tokens, tokens[1:]):
                transitions[token_to_key(previous)][token_to_key(current)] += 1

            if pieces_seen % 50 == 0:
                print(f"trained on {pieces_seen} pieces / {notes_seen} notes")

    model = {
        "type": "maestro_markov_note_model",
        "version": 1,
        "quantization": {
            "velocity_step": 8,
            "duration_step_seconds": 0.05,
            "gap_step_seconds": 0.05,
        },
        "pieces_seen": pieces_seen,
        "notes_seen": notes_seen,
        "start_counts": dict(start_counts),
        "transitions": {key: dict(value) for key, value in transitions.items()},
    }

    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_model.write_text(json.dumps(model), encoding="utf-8")
    print(f"saved model: {output_model}")
    print(f"pieces: {pieces_seen}")
    print(f"notes: {notes_seen}")


def generate_notes(model_path: Path, note_count: int, seed: int | None) -> list[dict[str, float | int]]:
    if seed is not None:
        random.seed(seed)

    model = json.loads(model_path.read_text(encoding="utf-8"))
    transitions: dict[str, dict[str, int]] = model["transitions"]
    current_key = weighted_choice(model["start_counts"])

    start_seconds = 0.0
    generated: list[dict[str, float | int]] = []

    for _ in range(note_count):
        pitch, velocity_bin, duration_bin, gap_bin = key_to_token(current_key)
        start_seconds += gap_bin * 0.05
        duration = max(0.05, duration_bin * 0.05)
        velocity = max(1, min(127, velocity_bin * 8))

        generated.append(
            {
                "pitch": pitch,
                "velocity": velocity,
                "start_seconds": round(start_seconds, 6),
                "end_seconds": round(start_seconds + duration, 6),
                "duration_seconds": round(duration, 6),
            }
        )

        next_counts = transitions.get(current_key)
        if not next_counts:
            current_key = weighted_choice(model["start_counts"])
        else:
            current_key = weighted_choice(next_counts)

    return generated


def write_generated_csv(path: Path, notes: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pitch", "velocity", "start_seconds", "end_seconds", "duration_seconds"],
        )
        writer.writeheader()
        writer.writerows(notes)


def write_varlen(value: int) -> bytes:
    if value < 0:
        raise ValueError("VLQ value must be non-negative")
    out = bytearray([value & 0x7F])
    value >>= 7
    while value:
        out.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(out)


def write_midi(path: Path, notes: list[dict[str, float | int]]) -> None:
    ticks_per_beat = 480
    seconds_per_beat = 0.5

    events: list[tuple[int, int, int, int]] = []
    for note in notes:
        start_tick = round(float(note["start_seconds"]) / seconds_per_beat * ticks_per_beat)
        end_tick = round(float(note["end_seconds"]) / seconds_per_beat * ticks_per_beat)
        pitch = int(note["pitch"])
        velocity = int(note["velocity"])
        events.append((start_tick, 1, pitch, velocity))
        events.append((end_tick, 0, pitch, 0))

    events.sort(key=lambda item: (item[0], item[1]))

    track = bytearray()
    track.extend(b"\x00\xff\x51\x03\x07\xa1\x20")  # 120 BPM
    previous_tick = 0

    for tick, is_on, pitch, velocity in events:
        delta = max(0, tick - previous_tick)
        previous_tick = tick
        track.extend(write_varlen(delta))
        if is_on:
            track.extend(bytes([0x90, pitch, velocity]))
        else:
            track.extend(bytes([0x80, pitch, 0]))

    track.extend(b"\x00\xff\x2f\x00")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"MThd")
        handle.write(struct.pack(">IHHH", 6, 0, 1, ticks_per_beat))
        handle.write(b"MTrk")
        handle.write(struct.pack(">I", len(track)))
        handle.write(track)


def generate(model_path: Path, output_dir: Path, note_count: int, seed: int | None) -> None:
    notes = generate_notes(model_path, note_count=note_count, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    notes_csv = output_dir / "generated_notes.csv"
    midi_path = output_dir / "generated.mid"
    write_generated_csv(notes_csv, notes)
    write_midi(midi_path, notes)
    print(f"saved notes: {notes_csv}")
    print(f"saved midi: {midi_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and sample a simple model from MAESTRO readable CSV files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model.")
    train_parser.add_argument("--readable-dir", type=Path, default=Path("maestro_readable"))
    train_parser.add_argument("--output-model", type=Path, default=Path("models/maestro_markov.json"))
    train_parser.add_argument("--max-pieces", type=int, default=100)

    generate_parser = subparsers.add_parser("generate", help="Generate a new notes CSV and MIDI file.")
    generate_parser.add_argument("--model", type=Path, default=Path("models/maestro_markov.json"))
    generate_parser.add_argument("--output-dir", type=Path, default=Path("generated"))
    generate_parser.add_argument("--note-count", type=int, default=500)
    generate_parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()

    if args.command == "train":
        train(args.readable_dir.resolve(), args.output_model.resolve(), args.max_pieces)
    elif args.command == "generate":
        generate(args.model.resolve(), args.output_dir.resolve(), args.note_count, args.seed)


if __name__ == "__main__":
    main()
