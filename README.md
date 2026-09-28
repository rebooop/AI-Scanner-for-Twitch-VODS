# Local-first Gaming Clip Finder

This is the transcription foundation for a clip-finding pipeline. It is intentionally small: it extracts audio locally with FFmpeg, transcribes it locally with `faster-whisper`, and saves segment timestamps. Ranking comes next, after we validate transcription on real VODs.

## Step 1: Install prerequisites (Windows)

This project still needs a real Python installation and FFmpeg on your `PATH`.

1. Install Python 3.11 (64-bit) from [python.org](https://www.python.org/downloads/). During installation, select **Add python.exe to PATH**.
2. Install FFmpeg and ensure both `ffmpeg` and `ffprobe` are on `PATH`. A convenient option is `winget install Gyan.FFmpeg` in an Administrator PowerShell.
3. Open a new PowerShell in this directory and create a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`faster-whisper` uses CTranslate2. For CUDA on Windows, its current requirements can depend on the installed NVIDIA CUDA/cuDNN libraries. If `doctor` finds no CUDA device or model loading fails, follow the current CTranslate2 CUDA installation instructions before proceeding.

## Step 2: Check the machine

```powershell
python main.py doctor
```

Expected: `ffmpeg`, `ffprobe`, `faster-whisper`, and CTranslate2 CUDA are available.

## Step 3: Plan a full VOD without processing it

Place a video outside the repository or in the ignored `input/` directory. Then ask FFmpeg for its duration and print the chunk plan:

```powershell
python main.py transcribe .\input\my-stream.mp4 --dry-run
```

For a 3-hour VOD, the default is six 30-minute chunks. Each chunk after the first begins three seconds early; that overlap protects speech near a boundary.

## Use a YouTube or Twitch VOD link

For public videos you own or have permission to process, install the updated dependencies and run one command:

```powershell
pip install -r requirements.txt
python main.py transcribe-url "https://www.youtube.com/watch?v=..."
```

`yt-dlp` downloads the source into `input/`; all audio extraction and transcription remain local. Public Twitch VOD URLs use the same command. Private, subscriber-only, expired, DRM-protected, or otherwise inaccessible videos can fail because the source platform does not make them available to the downloader.

## Step 4: Run a transcription

Start with a 10-minute sample or the entire VOD:

```powershell
python main.py transcribe .\input\my-stream.mp4 --model large-v3-turbo --device cuda --compute-type float16
```

To compare the smaller model:

```powershell
python main.py transcribe .\input\my-stream.mp4 --model medium --device cuda --compute-type float16
```

Each completed chunk is saved under `work/<job-id>/transcript_chunks/`. Re-running the same source with the same file metadata skips existing chunk JSON files. The final normalized transcript appears in `transcripts/<job-id>.json`.

## What the output means

`transcripts/*.json` uses absolute seconds from the beginning of the original VOD:

```json
{
  "segments": [
    {"id": 0, "start": 41.2, "end": 44.8, "text": "bro there's no way he wins this"}
  ]
}
```

This exact file becomes the input to the next milestone: overlapping transcript windows and clip ranking.

## Development checks

The chunk-planning logic has dependency-free tests:

```powershell
python -m unittest -v
```
