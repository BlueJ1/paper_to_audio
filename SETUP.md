# Installation & Setup Guide

## Complete Setup Instructions

### 1. Install System Dependencies

#### macOS
```bash
brew install ffmpeg
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

#### Windows
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

### 2. Install Python Dependencies

```bash
# Navigate to project directory
cd /Users/uni/Programming/paper_to_audio

# Create virtual environment (if not already done)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
GOOGLE_API_KEY=your-google-api-key-here
MURF_API_KEY=your-murf-api-key-here
MURF_VOICE_ID=marcus
MURF_FORMAT=mp3
MURF_CHUNK_CHARS=2800
```

**Where to get API keys:**
- Google Gemini: https://aistudio.google.com/app/apikey
- Murf.ai: https://murf.ai/api

## Usage

### Basic Usage

```bash
python main.py papers/Titans.pdf --out output.mp3
```

### Custom Chunk Size

If you encounter size limit errors from Murf:

```bash
python main.py papers/Titans.pdf --out output.mp3 --max-chars 2000
```

## Testing

Run the full test suite (Phase 1 extraction + Phase 2 classification — see `ADVANCED_PIPELINE_PLAN.md`):

```bash
pytest
```

Run a single suite:

```bash
pytest tests/test_extract.py -v      # Phase 1 — layout-aware extraction
pytest tests/test_classify.py -v     # Phase 2 — block classification
```

Integration tests depend on the sample PDFs in `papers/`; they are skipped automatically when a file is missing.

## Troubleshooting

### Issue: "pydub not found" or "ffmpeg not found"

**Solution:** Install ffmpeg first, then reinstall pydub:
```bash
brew install ffmpeg  # macOS
pip install --upgrade pydub
```

### Issue: "401 UNAUTHENTICATED" from Google

**Solution:** 
- Verify your `GOOGLE_API_KEY` is set in `.env`
- The script uses `gemini-1.5-pro` which supports API keys
- Some newer models require OAuth2 instead

### Issue: Audio file shows wrong duration in Finder/QuickTime

**Solution:** This is now fixed! The script uses `pydub` to properly concatenate audio chunks with correct metadata. Make sure you have:
1. Installed ffmpeg
2. Installed pydub (`pip install pydub`)
3. Run the latest version of the script

### Issue: Murf API errors

**Solution:**
- Check your `MURF_API_KEY` is valid
- Verify you have enough credits/characters remaining
- Try reducing `MURF_CHUNK_CHARS` if you get size limit errors

## What the Script Does

1. **[1/5] Loading PDF** - Extracts text from your PDF file
2. **[2/5] Processing with Gemini** - Rewrites the text into audio-friendly prose
   - Describes figures at their original position
   - Integrates footnotes naturally
   - Removes citations and references
3. **[3/5] Splitting into chunks** - Breaks text into manageable pieces for Murf
4. **[4/5] Generating audio** - Sends each chunk to Murf.ai for speech synthesis
5. **[5/5] Saving audio** - Properly concatenates all chunks with correct metadata

The final MP3 file will have accurate duration information and play correctly in all media players!


## Prompt
Transform this academic paper into audio-friendly narration for a Text-to-Speech engine. Your goal is to make the paper sound like an engaging podcast or audiobook while staying as true to the original as possible. You must follow these strict guidelines: First, do not summarize; reproduce the original paper as faithfully as possible. Second, never output LaTeX code, special symbols, or Markdown formatting—strictly avoid using dollar signs, backslashes, underscores, carats, asterisks, or hashes. Third, translate all math into spoken English exactly as a human would read it aloud. For example, render x sub t, capital M sub index t minus 1, N by d, the set of real numbers, and Greek letters like alpha. Fourth, do not read complex formulas character-by-character; instead, describe the equation’s purpose in plain English, such as "the model calculates the weighted sum of attention scores." Finally, rewrite the content into coherent prose by omitting all citations, inline references like bracketed numbers or author names, and URLs to ensure a smooth, uninterrupted flow. Descriptions of tables and graphs should be included where necessary to understand context.
