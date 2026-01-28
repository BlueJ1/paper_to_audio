# Paper to Audio

Convert academic papers (PDF) into an audio file using LangChain + Google Gemini and Murf.ai.

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
# MURF_API_KEY - Get from https://murf.ai/api
```

## Run

```bash
python main.py papers/Titans.pdf --out output.mp3
```

## Notes

- The LLM rewrites the PDF into audio-friendly prose, describes figures where they appear, and integrates footnotes into the flow.
- Bibliography and inline references are omitted for a smoother listening experience.
- If Murf returns a size limit error, lower `MURF_CHUNK_CHARS` or pass `--max-chars`.
- Using `gemini-1.5-pro` model which supports API key authentication (some newer models require OAuth2).

## Troubleshooting

If you get `401 UNAUTHENTICATED` errors:
- Make sure your `GOOGLE_API_KEY` is valid and set in `.env`
- The current implementation uses `gemini-2.5-flash-lite` which supports API keys
- Some newer Gemini models require OAuth2 instead of API keys
