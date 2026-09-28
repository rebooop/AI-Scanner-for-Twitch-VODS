"""Download public VOD URLs to a local file using yt-dlp."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from transcribe import PrerequisiteError, require_program


class DownloadError(RuntimeError):
    pass


def validate_video_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DownloadError("Provide a complete http:// or https:// video URL.")


def download_video(url: str, destination: Path) -> Path:
    """Download one public video and return its final local path.

    yt-dlp determines which supported site extractor to use; this project does
    not special-case YouTube or Twitch beyond naming the output predictably.
    """
    validate_video_url(url)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        ffmpeg = require_program("ffmpeg")
    except PrerequisiteError as error:
        raise DownloadError("FFmpeg is required to merge YouTube's separate video and audio streams.") from error
    command = [
        sys.executable, "-m", "yt_dlp", "--no-playlist", "--restrict-filenames",
        "--ffmpeg-location", ffmpeg, "--merge-output-format", "mp4", "-P", str(destination),
        "-o", "%(extractor)s-%(id)s.%(ext)s", "--print", "after_move:filepath", url,
    ]
    print("Downloading source VOD locally...")
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise DownloadError(f"yt-dlp could not download this URL:\n{detail}")
    paths = [Path(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
    if not paths:
        raise DownloadError("yt-dlp completed but did not report a downloaded video path.")
    video_path = paths[-1]
    if not video_path.is_file():
        raise DownloadError(f"yt-dlp reported a missing output file: {video_path}")
    print(f"Downloaded: {video_path}")
    return video_path
