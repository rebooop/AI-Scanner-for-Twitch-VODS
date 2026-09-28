"""Command line entry point for the local-first gaming clip finder."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from transcribe import PrerequisiteError, transcribe_vod
from windows import format_timestamp, write_windows
from rank import RANKING_VERSION, RankingError, rank_windows
from select_results import write_selections
from download import DownloadError, download_video


PROJECT_ROOT = Path(__file__).resolve().parent


def add_transcription_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="large-v3-turbo", help="faster-whisper model name.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--compute-type", default="float16", help="Use float16 on an NVIDIA GPU; int8 on CPU.")
    parser.add_argument("--chunk-minutes", type=float, default=30, help="Core audio per chunk (default: 30).")
    parser.add_argument("--overlap-seconds", type=float, default=3, help="Audio context overlap between chunks (default: 3).")
    parser.add_argument("--language", default=None, help="Optional ISO language code, such as en.")
    parser.add_argument("--keep-audio", action="store_true", help="Retain extracted WAV chunks for inspection.")
    parser.add_argument("--dry-run", action="store_true", help="Show the chunk plan without extracting or transcribing.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local-first gaming VOD clip finder (transcription stage).")
    subcommands = parser.add_subparsers(dest="command", required=True)

    doctor = subcommands.add_parser("doctor", help="Check local transcription prerequisites.")
    doctor.add_argument("--device", default="cuda", choices=("cuda", "cpu"))

    transcribe = subcommands.add_parser("transcribe", help="Extract and transcribe a local VOD; safely resumes chunks.")
    transcribe.add_argument("source", type=Path, help="Path to a local VOD file.")
    add_transcription_options(transcribe)

    download = subcommands.add_parser("download", help="Download one public YouTube or Twitch VOD URL locally.")
    download.add_argument("url", help="Public VOD URL you own or have permission to process.")

    transcribe_url = subcommands.add_parser("transcribe-url", help="Download a public VOD URL, then transcribe it locally.")
    transcribe_url.add_argument("url", help="Public VOD URL you own or have permission to process.")
    add_transcription_options(transcribe_url)

    view = subcommands.add_parser("view-transcript", help="Print transcript segments in a readable timestamped form.")
    view.add_argument("transcript", type=Path)
    view.add_argument("--from-seconds", type=float, default=0)
    view.add_argument("--to-seconds", type=float, default=None)

    windows = subcommands.add_parser("windows", help="Create overlapping transcript windows for LLM ranking.")
    windows.add_argument("transcript", type=Path)
    windows.add_argument("--window-seconds", type=float, default=90)
    windows.add_argument("--overlap-seconds", type=float, default=30)

    rank = subcommands.add_parser("rank", help="Score transcript windows using an OpenAI-compatible LLM API.")
    rank.add_argument("windows", type=Path)
    rank.add_argument("--model", required=True, help="LLM model name supplied by your provider.")
    rank.add_argument("--endpoint", default="https://api.openai.com/v1", help="OpenAI-compatible API base URL.")
    rank.add_argument("--limit", type=int, default=None, help="Score only the first N windows (useful for a paid smoke test).")
    rank.add_argument("--from-minutes", type=float, default=None, help="Only score windows overlapping this VOD timestamp in minutes.")
    rank.add_argument("--to-minutes", type=float, default=None, help="Only score windows before this VOD timestamp in minutes.")

    select = subcommands.add_parser("select", help="Remove overlapping candidates and write the Top K recommendations.")
    select.add_argument("raw_scores", type=Path)
    select.add_argument("--top-k", type=int, default=10)
    return parser


def run_doctor(device: str) -> int:
    print("Local transcription checks")
    for program in ("ffmpeg", "ffprobe"):
        path = shutil.which(program)
        print(f"  {program}: {'OK (' + path + ')' if path else 'MISSING'}")
    try:
        import faster_whisper  # noqa: F401
        print("  faster-whisper: OK")
    except ImportError:
        print("  faster-whisper: MISSING")
    if device == "cuda":
        try:
            import ctranslate2
            print(f"  CTranslate2 CUDA: {'available' if ctranslate2.get_cuda_device_count() else 'not detected'}")
        except ImportError:
            print("  CTranslate2 CUDA: unavailable (faster-whisper not installed)")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "doctor":
        return run_doctor(args.device)
    if args.command == "download":
        try:
            download_video(args.url, PROJECT_ROOT / "input")
        except DownloadError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
        return 0
    if args.command == "view-transcript":
        import json
        data = json.loads(args.transcript.read_text(encoding="utf-8"))
        for segment in data["segments"]:
            if segment["end"] >= args.from_seconds and (args.to_seconds is None or segment["start"] <= args.to_seconds):
                print(f"[{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}] {segment['text']}")
        return 0
    if args.command == "windows":
        output = PROJECT_ROOT / "windows" / f"{args.transcript.stem}.json"
        data = write_windows(args.transcript, output, args.window_seconds, args.overlap_seconds)
        print(f"Wrote {len(data['windows'])} windows to {output}")
        return 0
    if args.command == "rank":
        range_tag = ""
        if args.from_minutes is not None or args.to_minutes is not None:
            start_tag = "start" if args.from_minutes is None else f"from-{args.from_minutes:g}m"
            end_tag = "end" if args.to_minutes is None else f"to-{args.to_minutes:g}m"
            range_tag = f"--{start_tag}--{end_tag}"
        output = PROJECT_ROOT / "rankings" / f"{args.windows.stem}--{args.model.replace('/', '-')}--{RANKING_VERSION}{range_tag}.json"
        try:
            data = rank_windows(
                args.windows, output, args.model, args.endpoint, args.limit,
                None if args.from_minutes is None else args.from_minutes * 60,
                None if args.to_minutes is None else args.to_minutes * 60,
            )
        except RankingError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
        print(f"Wrote {len(data['scores'])} raw scores to {output}")
        return 0
    if args.command == "select":
        output = PROJECT_ROOT / "results" / f"{args.raw_scores.stem}--top-{args.top_k}.json"
        data = write_selections(args.raw_scores, output, args.top_k)
        print(f"Wrote {len(data['recommendations'])} recommendations to {output}")
        return 0
    try:
        source = args.source.resolve() if args.command == "transcribe" else download_video(args.url, PROJECT_ROOT / "input").resolve()
        transcribe_vod(
            source=source, work_root=PROJECT_ROOT / "work", transcripts_root=PROJECT_ROOT / "transcripts",
            model_name=args.model, device=args.device, compute_type=args.compute_type,
            chunk_minutes=args.chunk_minutes, overlap_seconds=args.overlap_seconds,
            language=args.language, keep_audio=args.keep_audio, dry_run=args.dry_run,
        )
    except (PrerequisiteError, DownloadError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
