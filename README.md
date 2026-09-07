# Paper to Audio

A personal local tool for reading academic PDFs aloud. The layout-aware pipeline preserves prose, narrates detected math and tables, and omits references, appendices, running headers, and footnotes. Review the transcript before generating audio in the web UI.

## Install

Use **Python 3.12**. Install FFmpeg (`brew install ffmpeg` on macOS or `sudo apt-get install ffmpeg` on Ubuntu).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-kokoro.txt -c constraints.txt
python check_setup.py --tts-engine kokoro
```

This installs the core app, local Kokoro engine, and English pronunciation model. Kokoro downloads its speech weights/voice on first use; later runs can use cached resources offline. For deterministic PDF text processing or Murf alone, install `requirements.txt` instead. See [SETUP.md](SETUP.md) for optional providers, clean-install checks, and dependency updates.

## Web workflow

```bash
python app.py
```

Open [the local app](http://127.0.0.1:5000). Upload a readable PDF, choose whether to enable LLM narration, and click **Process PDF**. Edit the transcript, choose Kokoro or Murf, then **Generate Audio**. Play or download the resulting MP3. **Regenerate** uses the current edited text and creates a new audio URL.

The UI defaults to deterministic text processing and Kokoro. LLM narration is optional and requires the selected provider's installation and key. Diagnostics are downloadable after text processing and include warnings, stage counts, source table/equation data, and marked inline math. All conflicting controls are disabled during work. Reloading during a running attempt reconnects to it; progress events are replayable and the terminal result is also available through status lookup.

Jobs live in this Python process. Restarting the app loses active job state. Files expire after 24 hours of inactivity, cleaned on subsequent requests. Local limits are two concurrent operations, 32 jobs, eight retained attempts per job, 50 MB uploads, and 2 million transcript characters. Audio subprocesses time out after 30 minutes. The job deletion API is `DELETE /jobs/<job_id>` and rejects active jobs.

## CLI

```bash
# Fully local; no LLM API key
python main.py papers/example.pdf --no-llm --tts-engine kokoro --out output.mp3 --keep-text

# Optional Google LLM for math and table selection (install requirements-google.txt)
python main.py papers/example.pdf --tts-engine kokoro --out output.mp3 --inspect narration.json --cache narration.sqlite

# Optional Cerebras provider (install requirements-cerebras.txt)
python main.py papers/example.pdf --llm-provider cerebras --tts-engine kokoro --out output.mp3

# Speak an edited transcript directly
python text_to_speech.py edited.txt --tts-engine kokoro --out output.mp3

# Local parallel generation; each worker loads its own model
python text_to_speech.py edited.txt --tts-engine kokoro --kokoro-workers 2 --out output.mp3
```

CLI defaults remain **Google LLM enabled** and **Murf TTS**. Use `--no-llm --tts-engine kokoro` for a local run. Google defaults to `gemma-3-27b-it`; Cerebras defaults to `gpt-oss-120b`. Override using `--llm-model`. `--max-chars` and `--kokoro-workers` must be positive. `--keep-text` saves `<pdf_basename>_audio_text.txt` in the current directory. `--text-file edited.txt` bypasses PDF processing in `main.py` (the positional PDF argument is still required but unused).

## Narration behavior

- Deterministic mode uses symbolic math and reads up to 20 table rows with their source column labels. Additional rows are explicitly noted.
- LLM mode requests equation narration, validated single-variable replacements, and up to three representative table row indices. Table narration is rendered from original cells, preserving row/column associations. Invalid responses produce a visible warning and faithful deterministic fallback.
- Surrounding body prose is immutable during inline-math replacement. Generated equation explanations still need review; validation cannot prove their meaning. Numeric and pronunciation polishing applies afterward in both modes.
- Author metadata and inline citations remain included pending a different narration preference. Unruled tables and labels embedded in figures remain extraction limitations.
- Blank/scanned PDFs with no usable text fail with an actionable message. OCR or the explicit vision library API is needed. Vision is never selected automatically and is not exposed in the UI/CLI.
- Rejected blocks remain in diagnostics and are excluded from speech. Empty or failed audio runs never replace a valid output. Murf MP3, WAV, FLAC, and OGG input chunks are decoded explicitly; final output is always MP3.

## Development

```bash
python -m pip install -r requirements-dev.txt -c constraints.txt
python -m pytest -q
python check_setup.py --dev
```

Core tests include generated PDFs, adversarial LLM responses, CLI validation, audio format/atomic output checks, and web lifecycle/replay failures. These run without cloud providers, Kokoro, or optional paper fixtures. The supplied three-paper corpus provides additional coverage; its existing numeric baselines were preserved during repair. See [MODULAR_GUIDE.md](MODULAR_GUIDE.md), [AGENTS.md](AGENTS.md), and [AUDIT_AND_REPAIR_PLAN.md](AUDIT_AND_REPAIR_PLAN.md).
