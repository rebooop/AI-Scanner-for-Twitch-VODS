"""Local, resumable VOD transcription using FFmpeg and faster-whisper.

This module deliberately keeps every intermediate artifact on disk. A multi-hour
VOD can take a while, so completed chunks are never re-transcribed unless the
user explicitly deletes their chunk JSON file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PrerequisiteError(RuntimeError):
    """Raised when an external program or Python package is unavailable."""


# Keep Windows DLL-directory handles alive for the life of the Python process.
_CUDA_DLL_DIRECTORIES: list[Any] = []


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float
    duration: float
    core_end: float


def require_program(name: str) -> str:
    path = shutil.which(name)
    # WinGet commonly installs command aliases here. Some non-interactive
    # processes do not inherit that directory in PATH even when PowerShell does.
    if path is None:
        winget_alias = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / f"{name}.exe"
        if winget_alias.is_file():
            path = str(winget_alias)
    if path is None:
        raise PrerequisiteError(
            f"'{name}' was not found on PATH. Install it and restart your terminal."
        )
    return path


def run_command(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{detail}")
    return completed.stdout


def probe_duration(source: Path) -> float:
    """Return the duration of a VOD in seconds using ffprobe."""
    ffprobe = require_program("ffprobe")
    output = run_command([
        ffprobe, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(source),
    ])
    try:
        duration = float(output.strip())
    except ValueError as error:
        raise RuntimeError(f"ffprobe did not return a usable duration for {source}") from error
    if duration <= 0:
        raise RuntimeError(f"The source duration must be positive; got {duration}.")
    return duration


def build_chunks(total_seconds: float, chunk_minutes: float, overlap_seconds: float) -> list[Chunk]:
    """Build source-timeline chunks. The overlap provides boundary context."""
    if total_seconds <= 0:
        raise ValueError("total_seconds must be positive")
    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be positive")
    chunk_seconds = chunk_minutes * 60
    if not 0 <= overlap_seconds < chunk_seconds:
        raise ValueError("overlap_seconds must be at least 0 and smaller than a chunk")

    chunks: list[Chunk] = []
    core_start = 0.0
    index = 0
    while core_start < total_seconds:
        start = max(0.0, core_start - (overlap_seconds if index else 0.0))
        core_end = min(total_seconds, core_start + chunk_seconds)
        end = min(total_seconds, core_end + overlap_seconds)
        chunks.append(Chunk(index=index, start=start, duration=end - start, core_end=core_end))
        core_start = core_end
        index += 1
    return chunks


def job_id_for(source: Path, settings: dict[str, Any]) -> str:
    """Create a stable job ID for one source *and* one transcription configuration."""
    stat = source.stat()
    fingerprint = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(
        f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{fingerprint}".encode()
    ).hexdigest()[:10]
    safe_stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", source.stem).strip(".-") or "vod"
    return f"{safe_stem}-{digest}"


def write_json(path: Path, value: Any) -> None:
    """Write atomically so an interrupted run cannot leave corrupt JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_audio(source: Path, chunk: Chunk, destination: Path) -> None:
    """Create a compact, Whisper-friendly mono 16 kHz WAV for one VOD chunk."""
    ffmpeg = require_program("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{chunk.start:.3f}", "-i", str(source),
        "-t", f"{chunk.duration:.3f}", "-map", "0:a:0?",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination),
    ])


def load_whisper_model(model_name: str, device: str, compute_type: str) -> Any:
    """Load the model once per VOD run, rather than once per audio chunk."""
    if device == "cuda":
        configure_cuda_libraries()
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise PrerequisiteError("faster-whisper is not installed. Run the setup commands in README.md.") from error
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def configure_cuda_libraries() -> None:
    """Make the local CUDA runtime bundle discoverable on Windows.

    This is intentionally process-local: it avoids copying third-party DLLs to
    System32 or permanently modifying the user's global PATH.
    """
    candidates = [
        Path(os.environ["CUDA_LIB_DIR"]) if os.environ.get("CUDA_LIB_DIR") else None,
        Path.home() / "cuda-libs" / "CUDA12_v3",
    ]
    for directory in candidates:
        if directory and (directory / "cublas64_12.dll").is_file() and (directory / "cudnn_ops64_9.dll").is_file():
            directory_text = str(directory)
            if directory_text not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = directory_text + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                _CUDA_DLL_DIRECTORIES.append(os.add_dll_directory(directory_text))
            return


def transcribe_audio(
    audio_path: Path,
    chunk: Chunk,
    model: Any,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
) -> dict[str, Any]:
    """Transcribe one chunk and translate relative timestamps to VOD timestamps."""
    started_at = time.perf_counter()
    segments, info = model.transcribe(
        str(audio_path), language=language, beam_size=5, vad_filter=True,
        word_timestamps=False, condition_on_previous_text=True,
    )

    output_segments = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            output_segments.append({
                "start": round(chunk.start + segment.start, 3),
                "end": round(chunk.start + segment.end, 3),
                "text": text,
            })

    return {
        "chunk": {"index": chunk.index, "start": chunk.start, "duration": chunk.duration, "core_end": chunk.core_end},
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "runtime_seconds": round(time.perf_counter() - started_at, 3),
        "segments": output_segments,
    }


def combine_chunks(chunk_files: list[Path], destination: Path, source: Path, duration: float, settings: dict[str, Any]) -> dict[str, Any]:
    """Combine completed chunks, retaining each segment only in its core range."""
    all_segments: list[dict[str, Any]] = []
    for chunk_file in chunk_files:
        data = read_json(chunk_file)
        is_last_chunk = data["chunk"]["core_end"] >= duration
        for segment in data["segments"]:
            if is_last_chunk or segment["start"] < data["chunk"]["core_end"]:
                all_segments.append(segment)

    all_segments.sort(key=lambda item: (item["start"], item["end"]))
    transcript = {
        "source": str(source),
        "duration": round(duration, 3),
        "settings": settings,
        "segment_count": len(all_segments),
        "segments": [{"id": index, **segment} for index, segment in enumerate(all_segments)],
    }
    write_json(destination, transcript)
    return transcript


def transcribe_vod(
    source: Path,
    work_root: Path,
    transcripts_root: Path,
    model_name: str,
    device: str,
    compute_type: str,
    chunk_minutes: float,
    overlap_seconds: float,
    language: str | None = None,
    keep_audio: bool = False,
    dry_run: bool = False,
) -> Path:
    """Run (or resume) an entire VOD transcription and return its transcript path."""
    if not source.is_file():
        raise FileNotFoundError(f"VOD file not found: {source}")
    duration = probe_duration(source)
    chunks = build_chunks(duration, chunk_minutes, overlap_seconds)
    settings = {
        "model": model_name, "device": device, "compute_type": compute_type,
        "chunk_minutes": chunk_minutes, "overlap_seconds": overlap_seconds,
        "language": language,
    }
    # A model comparison is a distinct run. This prevents a `medium` run from
    # incorrectly resuming `large-v3-turbo` chunk files for the same source.
    job_id = job_id_for(source, settings)
    job_dir = work_root / job_id
    audio_dir = job_dir / "audio"
    chunk_dir = job_dir / "transcript_chunks"
    transcript_path = transcripts_root / f"{job_id}.json"
    write_json(job_dir / "job.json", {"source": str(source), "duration": duration, "settings": settings, "chunks": [chunk.__dict__ for chunk in chunks]})
    print(f"VOD duration: {duration / 3600:.2f} hours | chunks: {len(chunks)} | job: {job_id}")
    if dry_run:
        return transcript_path

    pending_chunks = [chunk for chunk in chunks if not (chunk_dir / f"chunk_{chunk.index:04d}.json").is_file()]
    model: Any | None = None
    if pending_chunks:
        print(f"Loading model once on {device} ({model_name})")
        model = load_whisper_model(model_name, device, compute_type)

    for chunk in chunks:
        chunk_json = chunk_dir / f"chunk_{chunk.index:04d}.json"
        if chunk_json.is_file():
            print(f"[{chunk.index + 1}/{len(chunks)}] already complete; skipping")
            continue
        audio_path = audio_dir / f"chunk_{chunk.index:04d}.wav"
        print(f"[{chunk.index + 1}/{len(chunks)}] extracting {chunk.start / 60:.1f}-{(chunk.start + chunk.duration) / 60:.1f} minutes")
        if not audio_path.is_file():
            extract_audio(source, chunk, audio_path)
        print(f"[{chunk.index + 1}/{len(chunks)}] transcribing on {device} ({model_name})")
        assert model is not None
        result = transcribe_audio(audio_path, chunk, model, model_name, device, compute_type, language)
        write_json(chunk_json, result)
        if not keep_audio:
            audio_path.unlink(missing_ok=True)
        factor = result["runtime_seconds"] / chunk.duration
        print(f"[{chunk.index + 1}/{len(chunks)}] saved {len(result['segments'])} segments in {result['runtime_seconds']:.1f}s (RTF {factor:.3f})")

    chunk_files = [chunk_dir / f"chunk_{chunk.index:04d}.json" for chunk in chunks]
    transcript = combine_chunks(chunk_files, transcript_path, source, duration, settings)
    print(f"Complete: {transcript_path} ({transcript['segment_count']} segments)")
    return transcript_path
