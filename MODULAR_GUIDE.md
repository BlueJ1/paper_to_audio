# Modular Pipeline Usage Guide

This guide explains the modular architecture and how to use the three scripts independently or together.

## Architecture Overview

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│             │     │                  │     │              │
│   PDF File  │────►│  pdf_to_text.py  │────►│  Text File   │
│             │     │  (Uses Gemini)   │     │              │
└─────────────┘     └──────────────────┘     └──────┬───────┘
                                                     │
                                                     │
                    ┌──────────────────────┐         │
                    │                      │         │
                    │  text_to_speech.py   │◄────────┘
                    │  (Uses TTS Engine)   │
                    │                      │
                    └──────────┬───────────┘
                               │
                               ▼
                         ┌──────────┐
                         │          │
                         │ MP3 File │
                         │          │
                         └──────────┘
```

**main.py** = Orchestrates both steps automatically

## Three Ways to Use

### Method 1: Integrated Pipeline (Fastest)

Use `main.py` to run everything in one command:

```bash
python main.py papers/Titans.pdf --out Titans.mp3 --tts-engine kokoro
```

**Pros:**
- Single command
- Automatic cleanup
- Fastest for one-time conversion

**Cons:**
- Can't edit intermediate text
- Must re-run LLM if TTS fails

### Method 2: Manual Two-Step (Most Flexible)

Run each step separately for maximum control:

```bash
# Step 1: PDF → Text (uses Gemini API)
python pdf_to_text.py papers/Titans.pdf

# Output: Titans_audio_text.txt
# Now you can review/edit this file!

# Step 2: Text → Audio (uses TTS)
python text_to_speech.py Titans_audio_text.txt --tts-engine kokoro
```

**Pros:**
- Can edit LLM output before TTS
- Reuse text file for multiple audio versions
- Better for debugging
- Save money (don't re-run LLM)

**Cons:**
- Two commands required
- Must manage intermediate files

### Method 3: Hybrid (Best of Both)

Use main.py with `--keep-text` to save intermediate files:

```bash
# First run: generate and keep text
python main.py papers/Titans.pdf --keep-text --out Titans_murf.mp3

# This creates:
# - Titans_murf.mp3 (audio with Murf.ai)
# - Titans_audio_text.txt (intermediate text)

# Later: reuse text with different TTS engine
python text_to_speech.py Titans_audio_text.txt --tts-engine kokoro --out Titans_kokoro.mp3
```

**Pros:**
- Fast first run
- Can reuse text later
- Compare TTS engines easily

## Real-World Workflows

### Workflow 1: Quick One-Off Conversion

```bash
python main.py paper.pdf --tts-engine kokoro
# Done! → output.mp3
```

### Workflow 2: Production Quality (Review Before Audio)

```bash
# Step 1: Generate text
python pdf_to_text.py paper.pdf
# → paper_audio_text.txt

# Step 2: Review and edit text file
vim paper_audio_text.txt  # or any editor

# Step 3: Generate audio
python text_to_speech.py paper_audio_text.txt --tts-engine murf
# → paper.mp3
```

### Workflow 3: Compare TTS Engines

```bash
# Step 1: Generate text once
python pdf_to_text.py paper.pdf

# Step 2: Generate with Murf.ai
python text_to_speech.py paper_audio_text.txt --tts-engine murf --out paper_murf.mp3

# Step 3: Generate with Kokoro
python text_to_speech.py paper_audio_text.txt --tts-engine kokoro --out paper_kokoro.mp3

# Now compare audio quality!
```

### Workflow 4: Batch Processing

```bash
# Process multiple papers to text first (uses Gemini API)
for pdf in papers/*.pdf; do
    python pdf_to_text.py "$pdf"
done

# Review all text files, make edits...

# Then generate all audio (uses free Kokoro)
for txt in *_audio_text.txt; do
    python text_to_speech.py "$txt" --tts-engine kokoro
done
```

## File Naming Conventions

### pdf_to_text.py
- Input: `papers/Titans.pdf`
- Output: `Titans_audio_text.txt` (in current directory)
- Custom: `--out my_custom_name.txt`

### text_to_speech.py
- Input: `Titans_audio_text.txt`
- Output: `Titans.mp3` (strips `_audio_text` suffix)
- Custom: `--out my_custom_name.mp3`

### main.py
- Input: `papers/Titans.pdf`
- Output: `output.mp3` (default) or `--out Titans.mp3`
- Intermediate (if `--keep-text`): `Titans_audio_text.txt`

## Environment Variables

Both scripts respect these `.env` variables:

### For pdf_to_text.py
```bash
GOOGLE_API_KEY=your-key-here  # Required
```

### For text_to_speech.py
```bash
# For Murf.ai engine
MURF_API_KEY=your-key-here
MURF_VOICE_ID=marcus
MURF_CHUNK_CHARS=2800

# For Kokoro engine
KOKORO_VOICE=af_bella
```

## Tips & Tricks

### 1. Save Money on LLM Costs

Generate text once, create multiple audio versions:

```bash
python pdf_to_text.py expensive_paper.pdf
# Edit text to fix any issues
python text_to_speech.py expensive_paper_audio_text.txt --tts-engine kokoro
```

### 2. Debug TTS Issues

If audio generation fails, you still have the text:

```bash
python main.py paper.pdf --keep-text
# If TTS fails, you have paper_audio_text.txt
# Fix the issue, then:
python text_to_speech.py paper_audio_text.txt
```

### 3. Custom Text Processing

Skip the PDF entirely and process your own text:

```bash
echo "Your custom narration text here" > custom.txt
python text_to_speech.py custom.txt --tts-engine kokoro
```

### 4. Incremental Updates

Made a small edit to the paper? Update just one section:

```bash
# Edit the text file to add/change content
vim paper_audio_text.txt

# Regenerate audio without re-running LLM
python text_to_speech.py paper_audio_text.txt
```

## Error Handling

### If pdf_to_text.py fails:
- Check GOOGLE_API_KEY is set
- Verify PDF is readable
- Check Gemini API quota

### If text_to_speech.py fails:
- For Murf.ai: Check MURF_API_KEY
- For Kokoro: Install dependencies (`pip install torch kokoro-onnx scipy numpy`)
- Check text file exists and is readable

### If main.py fails mid-process:
- Use `--keep-text` to save intermediate results
- Continue from step 2 with `text_to_speech.py`

## Performance Comparison

| Method | LLM Calls | TTS Calls | Time | Flexibility |
|--------|-----------|-----------|------|-------------|
| Integrated (main.py) | 1 | 1 | Fast | Low |
| Two-step manual | 1 | 1 | Medium | High |
| Hybrid (--keep-text) | 1 | 1+ | Medium | Medium |
| Batch processing | N | N | Slow | Highest |

## Best Practices

1. **Always keep text files** when experimenting with TTS engines
2. **Review LLM output** before generating expensive Murf.ai audio
3. **Use Kokoro for drafts**, Murf.ai for final production
4. **Version control** your text files alongside audio
5. **Batch process** PDFs to text, then review all before audio generation

## Example: Complete Workflow

```bash
# 1. Convert paper to text
python pdf_to_text.py papers/important_paper.pdf
# Output: important_paper_audio_text.txt

# 2. Review and edit (fix any LLM mistakes)
code important_paper_audio_text.txt  # or vim, nano, etc.

# 3. Generate draft with free Kokoro
python text_to_speech.py important_paper_audio_text.txt \
    --tts-engine kokoro \
    --out draft.mp3

# 4. Listen to draft
# 5. If good, generate final with Murf.ai
python text_to_speech.py important_paper_audio_text.txt \
    --tts-engine murf \
    --out final.mp3

# 6. Keep both versions for comparison
ls -lh draft.mp3 final.mp3 important_paper_audio_text.txt
```

## Integration with Other Tools

### Shell Script Wrapper
```bash
#!/bin/bash
# process_papers.sh
for pdf in papers/*.pdf; do
    echo "Processing $pdf..."
    python pdf_to_text.py "$pdf"
done
echo "Review text files, then run:"
echo "for txt in *_audio_text.txt; do python text_to_speech.py \"\$txt\" --tts-engine kokoro; done"
```

### Make-style Dependency
```makefile
%.mp3: %_audio_text.txt
	python text_to_speech.py $< --out $@ --tts-engine kokoro

%_audio_text.txt: papers/%.pdf
	python pdf_to_text.py $<
```

This modular design gives you maximum flexibility while still offering simple one-command operation when needed!
