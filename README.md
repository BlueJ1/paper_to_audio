# Paper to Audio

Convert academic papers (PDF) into an audio file using LangChain + Google Gemini and TTS (Murf.ai or Kokoro).

## Architecture

This project consists of three modular scripts that can be run independently or together:

1. **`pdf_to_text.py`** - Converts PDF to audio-friendly text using LLM
2. **`text_to_speech.py`** - Converts text to speech using TTS engines  
3. **`main.py`** - Integrated pipeline that runs both steps

## TTS Engine Options

This tool supports two TTS engines:

1. **Murf.ai** (default) - Cloud-based, high-quality, paid API
2. **Kokoro** - Local, free, open-source model ([hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M))

## Prerequisites

Install ffmpeg (required for audio processing):

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (via chocolatey)
choco install ffmpeg
```

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set environment variables (create a `.env` file or export in shell):

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys:
# GOOGLE_API_KEY - Get from https://aistudio.google.com/app/apikey
# MURF_API_KEY - Get from https://murf.ai/api (only needed if using Murf.ai)
```

## Usage

### Option 1: Integrated Pipeline (Recommended)

Process everything in one command:

```bash
# Using Murf.ai (cloud, paid)
python main.py papers/Titans.pdf --out output.mp3

# Using Kokoro (local, free)
python main.py papers/Titans.pdf --out output.mp3 --tts-engine kokoro

# Keep intermediate text file
python main.py papers/Titans.pdf --keep-text

# Use existing text file (skip PDF processing)
python main.py papers/Titans.pdf --text-file Titans_audio_text.txt
```

### Option 2: Two-Step Process

Run each step independently for more control:

**Step 1: Convert PDF to audio-friendly text**
```bash
python pdf_to_text.py papers/Titans.pdf
# Output: Titans_audio_text.txt
```

**Step 2: Convert text to speech**
```bash
# Using Murf.ai
python text_to_speech.py Titans_audio_text.txt --out Titans.mp3

# Using Kokoro  
python text_to_speech.py Titans_audio_text.txt --out Titans.mp3 --tts-engine kokoro
```

### Why Use the Two-Step Process?

- **Edit the text**: Review and modify the LLM output before generating audio
- **Try different TTS engines**: Generate audio with both Murf.ai and Kokoro from the same text
- **Save costs**: Reuse the LLM-processed text without re-running Gemini
- **Debug**: Isolate issues in PDF processing vs. speech generation

## Command-Line Options

### main.py (Integrated Pipeline)
- `pdf` - Path to PDF file
- `--out OUTPUT.mp3` - Output audio filename (default: output.mp3)
- `--tts-engine {murf,kokoro}` - TTS engine (default: murf)
- `--max-chars N` - Maximum characters per TTS chunk (default: 2800)
- `--keep-text` - Keep intermediate text file
- `--text-file FILE` - Use existing text file (skips PDF processing)

### pdf_to_text.py (PDF to Text)
- `pdf` - Path to PDF file
- `--out FILE.txt` - Output text filename (default: `<pdf_name>_audio_text.txt`)

### text_to_speech.py (Text to Speech)
- `text_file` - Path to text file
- `--out OUTPUT.mp3` - Output audio filename (default: `<text_name>.mp3`)
- `--tts-engine {murf,kokoro}` - TTS engine (default: murf)
- `--max-chars N` - Maximum characters per TTS chunk (default: 2800)

## Environment Variables

- `GOOGLE_API_KEY` - Required for all engines
- `MURF_API_KEY` - Required only for Murf.ai engine
- `MURF_VOICE_ID` - Murf.ai voice (default: marcus)
- `MURF_CHUNK_CHARS` - Characters per chunk (default: 2800)
- `KOKORO_VOICE` - Kokoro voice (default: af_bella)
  - Available voices: af, af_bella, af_sarah, am_adam, am_michael, bf_emma, bf_isabella, bm_george, bm_lewis

## Notes

- The LLM rewrites the PDF into audio-friendly prose, describes figures where they appear, and integrates footnotes into the flow.
- Bibliography and inline references are omitted for a smoother listening experience.
- If Murf returns a size limit error, lower `MURF_CHUNK_CHARS` or pass `--max-chars`.
- Using `gemini-2.5-flash-lite` model which supports API key authentication (some newer models require OAuth2).
- Kokoro models are downloaded automatically to HuggingFace cache (`~/.cache/huggingface/`) on first use.

## Troubleshooting

If you get `401 UNAUTHENTICATED` errors:
- Make sure your `GOOGLE_API_KEY` is valid and set in `.env`
- The current implementation uses `gemini-2.5-flash-lite` which supports API keys
- Some newer Gemini models require OAuth2 instead of API keys

If Kokoro dependencies fail to install:
- Make sure you have Python 3.8+ installed
- Try installing PyTorch separately first: `pip install torch`
- On Apple Silicon Macs, you may need to install the ARM64 version of torch

