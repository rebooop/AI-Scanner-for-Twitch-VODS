"""Turn timestamped transcript segments into overlapping ranking windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def format_timestamp(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_windows(transcript: dict[str, Any], window_seconds: float = 90, overlap_seconds: float = 30) -> list[dict[str, Any]]:
    if window_seconds <= 0 or not 0 <= overlap_seconds < window_seconds:
        raise ValueError("window_seconds must be positive and overlap_seconds must be smaller than it")
    duration = float(transcript["duration"])
    step = window_seconds - overlap_seconds
    windows: list[dict[str, Any]] = []
    start = 0.0
    index = 0
    while start < duration:
        end = min(duration, start + window_seconds)
        included = [segment for segment in transcript["segments"] if segment["end"] > start and segment["start"] < end]
        lines = [f"[{format_timestamp(segment['start'])}-{format_timestamp(segment['end'])}] {segment['text']}" for segment in included]
        windows.append({
            "id": f"w_{index:04d}", "start": round(start, 3), "end": round(end, 3),
            "segment_ids": [segment["id"] for segment in included], "text": "\n".join(lines),
            "speech_seconds": round(sum(max(0, min(segment["end"], end) - max(segment["start"], start)) for segment in included), 3),
        })
        if end >= duration:
            break
        start += step
        index += 1
    return windows


def write_windows(transcript_path: Path, output_path: Path, window_seconds: float, overlap_seconds: float) -> dict[str, Any]:
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    data = {
        "transcript": str(transcript_path),
        "window_seconds": window_seconds,
        "overlap_seconds": overlap_seconds,
        "windows": build_windows(transcript, window_seconds, overlap_seconds),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data
