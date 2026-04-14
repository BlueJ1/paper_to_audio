# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Converts academic papers (PDF) into audio files. Two-stage pipeline:
1. **PDF → Text**: Extracts text with pymupdf, then applies targeted replacements (regex + optional LLM) to make it TTS-friendly while preserving original wording.
2. **Text → Speech**: Uses either Murf.ai (cloud, paid) or Kokoro (local, free) TTS engine.

The legacy PDF→text path (`pdf_to_text.py`) is being superseded by a layout-aware rewrite under `pipeline/`. Both coexist today; see **Pipeline rewrite status** below.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full pipeline (PDF → audio) — Google Gemma (default)
python main.py papers/example.pdf --out output.mp3 --tts-engine kokoro

# Full pipeline using Cerebras (qwen-3-235b-a22b-instruct-2507)
python main.py papers/example.pdf --out output.mp3 --tts-engine kokoro --llm-provider cerebras

# Regex-only cleanup (no LLM, no API key needed)
python main.py papers/example.pdf --out output.mp3 --tts-engine kokoro --no-llm

# Individual steps
python pdf_to_text.py papers/example.pdf                    # PDF → text
python pdf_to_text.py papers/example.pdf --no-llm           # regex-only
python text_to_speech.py example_audio_text.txt --out example.mp3 --tts-engine kokoro

# Tests
pytest                                                      # full suite
pytest tests/test_extract.py -v                             # Phase 1 only
pytest tests/test_classify.py -v                            # Phase 2 only

# Verify setup
python check_setup.py
```

## Architecture

Three scripts, each runnable standalone or via `main.py`:

- **`pdf_to_text.py`** (legacy): Targeted replacement pipeline. Extracts text via pymupdf, then orchestrated by `clean_for_tts(text, llm=None)`:
  1. `_preprocess_raw_text()` — fix raw extraction artifacts BEFORE semantic regex runs: expand ligatures (ﬁ→fi), rejoin hyphenated line-breaks, rejoin URLs/citations split across lines, strip emails/arXiv metadata/References section/footnote markers/standalone section numbers
  2. `_regex_cleanup()` — deterministic passes: remove citations, URLs, LaTeX commands; map unicode math/Greek to English words; handle subscripts/superscripts; calls `_remove_diagram_blocks()` to drop figure/table graphic fragments while keeping captions
  3. `_identify_math_passages()` — heuristic density-based scoring (≥0.10 math density OR ≥2 indicators) to flag paragraphs needing LLM rewriting
  4. `_llm_rewrite_math()` — sends flagged paragraphs individually to the LLM for plain-English math description (non-math text kept verbatim). Gemma 3 doesn't support system messages, so instructions are embedded in a single human message. HTML tags are stripped from output; outputs <30% of original length are rejected as failures
  5. `_final_sanitize()` — last-pass punctuation/whitespace cleanup
  With `llm=None`, only regex cleanup runs (no API calls).

  LLM providers via `build_llm(model, provider)`:
  - `google` (default) → `ChatGoogleGenerativeAI`, default model `gemma-3-27b-it` (fast, no extended thinking). Other options in `GEMMA_MODELS`.
  - `cerebras` → `ChatOpenAI` pointed at `https://api.cerebras.ai/v1`, default `qwen-3-235b-a22b-instruct-2507`.

- **`text_to_speech.py`**: Splits text into chunks (paragraph/sentence boundary-aware via `split_text`), generates audio via `TTSEngine` protocol, concatenates with `pydub`. `generate_audio_chunks()` dispatches sequential vs parallel generation. Kokoro parallel mode uses `ProcessPoolExecutor` with **spawn context** and `_kokoro_worker_init()` to load one `KPipeline` per worker (must be spawn — fork breaks PyTorch). Key classes: `MurfTTSEngine`, `KokoroTTSEngine`.
- **`main.py`**: Orchestrates both steps. `--tts-engine` defaults to `murf`. Supports `--text-file` to skip PDF processing, `--keep-text` to save intermediate text (named `<pdf_basename>_audio_text.txt`), `--no-llm` for regex-only cleanup, `--kokoro-workers` to override parallelism.

## Pipeline rewrite (`pipeline/` package)

The redesign described in `ADVANCED_PIPELINE_PLAN.md` keeps layout and font metadata end-to-end so each block can be routed to the right handler (prose, equation, table, caption, ...). Every stage is a pure function `Document -> Document`.

Module layout:

```
pipeline/
  model.py        # Span, Block, FontStats, Document dataclasses
  extract.py      # Phase 1 — pymupdf dict-mode extraction, column detection, paragraph rejoin
  classify.py     # Phase 2 — block classification, heading levels, parent_section tree
  policies.py     # Phase 3 — SectionPolicy + filter_sections (skip References, footnotes, ...)
  tables.py       # Phase 4 — pdfplumber detection, block merging, skip/verbatim/prose renderers
  serialize.py    # Document -> plain text (paragraph per block)
```

Key types (`pipeline/model.py`):
- `Span` — text, font, size, flags, bbox. Exposes `is_italic`, `is_bold`, `is_monospace`, `is_superscript` from the pymupdf flag bitfield.
- `Block` — `kind` (`heading|body|caption|equation_display|equation_inline|table|figure|footnote|code|page_header|page_footer|toc|noise`), `spans`, `text`, `page`, `bbox`, `column`, `level`, `parent_section`, `meta`. `dominant_size` is a character-weighted mean of span sizes.
- `FontStats` — `body_size` (character-weighted modal span size), `heading_thresholds` (sorted sizes strictly above body).
- `Document` — `blocks`, `fonts`, `page_rects` (raw per-page bbox for margin/center math), plus `language`, `title`, `abstract`, `source_path`.

Phase 1 (`pipeline/extract.py`) — `extract_layout(pdf_path) -> Document`:
- Normalizes each span (NFC + ligature expansion at creation time via `_LIGATURES` map).
- Rebuilds paragraph text from per-line fragments with `_join_line_texts`, healing hyphenated line breaks. Compound prefixes/suffixes in `_HYPHEN_KEEP_PREFIXES`/`_HYPHEN_KEEP_SUFFIXES` keep their hyphen (`self-attention` stays intact; `intro-duction` fuses).
- `_analyze_columns` flags a page as two-column when ≥2 body-width blocks (>25% page width) sit clearly left of the midline AND ≥2 sit clearly right. `_is_full_width` is midline-crossing, not a pure width ratio, so centered titles ~55% of page width still register as full-width.
- `_order_page_blocks` produces reading order: full-width blocks split the page into horizontal bands; within each band left-column blocks precede right-column.

Phase 2 (`pipeline/classify.py`) — `classify_blocks(doc) -> Document`:
- `_mark_headers_and_footers` is a document-wide pass that marks blocks whose digit-normalized text repeats across ≥3 pages at the top/bottom margin (page numbers, running titles).
- Per-block rules run in priority order: caption > code > heading > footnote > toc > equation_display > noise > body. Caption regex is checked first so `Figure 1.` at heading font size doesn't mis-route.
- `_is_heading` has two paths: (A) `dominant_size >= body * 1.05` with a structural signal (numeric prefix, near-all-caps, or ≤14 words); (B) bold-at-body-size with numeric/all-caps prefix — catches venues that bold inline subsection headings. `_is_rotated_block` rejects narrow, tall blocks (e.g. the vertical arXiv sidebar watermark) which would otherwise dominate heading-size rank.
- Equation-number regex is deliberately tight (`\(\d{1,3}(\.\d{1,3})?\)$`) so citation years like `(2019)` don't match.
- `_assign_heading_levels` ranks unique heading sizes descending (rounded to 0.1pt). Drop-cap letters in some section titles inflate weighted size, so heading levels can be noisier than in print — downstream filtering should key off `text`, not `level`.
- `_assign_parent_sections` walks reading order and sets `parent_section = <nearest preceding heading.text>` on every non-heading block.

Phase 3 (`pipeline/policies.py`) — `filter_sections(doc, policy) -> Document`:
- `SectionPolicy` dataclass holds: `skip_kinds` (default `page_header, page_footer, footnote, toc, noise, figure`), per-section booleans (`skip_references`, `skip_acknowledgments`, `skip_author_contributions`, `skip_supplementary`, `skip_appendix`), `keep_code` (default False), and `extra_skip_patterns` for custom regexes.
- `section_patterns()` assembles the effective regex list from the enabled booleans plus extras. Each canned pattern accepts an optional numeric or single-letter prefix so `6 References`, `6. References`, and `A Appendix` all match.
- Filtering walks blocks in order. A heading whose text matches a skip regex opens a skip at that heading's `level`; subsequent blocks are dropped until a later heading at level ≤ the skip level appears. Skip level defaults to 99 when `level is None`, so headings with unknown rank still end the skip as soon as any heading follows.
- Parent-section fallback: non-heading blocks whose `parent_section` matches a skip pattern are dropped even if the level-based tracking missed them (guards against noisy heading levels per the drop-cap caveat above).
- Pure function: returns a new `Document` via `dataclasses.replace`, so callers can A/B different policies against the same Phase 2 output.

Phase 4 (`pipeline/tables.py`) — `handle_tables(doc, policy, llm=None) -> Document`:
- Lazy-imports `pdfplumber` and scans every page with the default (ruled-line) `find_tables()` strategy. Detections smaller than 2×2 cells are discarded — `find_tables` occasionally flags 1×N equation fragments.
- `_merge_tables` matches each detected bbox to `Block`s on the same page whose bbox has ≥50% area inside the table. The largest-overlap block with `kind in ("body", "noise")` becomes the "primary"; sibling fragments inside the bbox are dropped. When no body-like block overlaps (common for false positives over equation regions) the detection is ignored.
- `_find_caption` attaches the nearest `Table N:` caption on the same page by vertical center distance.
- `TableData` carries rows, bbox, page, 1-based `index` (assigned in reading order across the document), and `caption`. `header` is inferred from the first row only when every cell is non-empty and <50% digits.
- `render_table` switches on `TablePolicy.mode`:
  - `skip` (default): `"A table with N columns and M rows appears here (caption); see the paper."` No LLM.
  - `verbatim`: cell-by-cell narration with `header: value` pairs when a header row exists; truncated to `verbatim_max_rows`. No LLM.
  - `prose`: sends CSV + caption to the LLM under a bounded-sentence prompt; rejects outputs that introduce decimal tokens absent from the CSV (`_prose_is_safe`) and falls back to `skip` on exception, empty output, or safety failure.
- Pure: replaces blocks via `dataclasses.replace`, returns a new `Document`. No-op when `source_path` is None or pdfplumber raises.
- Known limitation: unruled tables (Shaw et al. Table 1, some Titans subtables) slip past the conservative ruled-line strategy. Raising recall would require a fallback text-alignment strategy and per-page confidence gating — deferred.

### Pipeline rewrite status

| Phase | Scope                                                   | State    |
|-------|---------------------------------------------------------|----------|
| 1     | `Document` model, dict-mode extraction, columns, rejoin | ✅ shipped |
| 2     | Block classification + heading tree                     | ✅ shipped |
| 3     | Section-filter policy (skip References, footnotes, ...) | ✅ shipped |
| 4     | pdfplumber table extraction + prose mode                | ✅ shipped |
| 5–6   | Display equations + inline-math span detection          | pending  |
| 7     | `num2words`, acronym/abbreviation expansion             | pending  |
| 8     | LLM cache, regression corpus, inspect mode              | pending  |
| 9     | `--vlm` vision-first mode                               | pending  |

The legacy `pdf_to_text.py` remains wired to `main.py` until Phase 5+ land; new work should go in `pipeline/` and the legacy script stays untouched so the user-facing pipeline keeps working.

## Testing

`tests/` holds pytest suites mirroring the pipeline phases.

- `tests/test_extract.py` — Phase 1: unit tests on fixture blocks (ligature expansion, hyphenation, column detection, reading order, font-stat weighting) plus integration tests on both real PDFs.
- `tests/test_classify.py` — Phase 2: unit tests per rule (heading / caption / code / footnote / equation / noise / header-footer / repeating-text normalization) plus integration tests that assert title-is-heading, `References` heading exists, ≥3 captions tagged, ≥40% of blocks remain body.
- `tests/test_policies.py` — Phase 3: unit tests for pattern assembly, per-kind filtering, section skipping (refs / acknowledgments / appendix), nested subsection handling, sibling heading closes skip, parent-section fallback, purity. Integration tests assert `References` and `Acknowledgments` are gone from the real PDFs and that filtering reduces block count without evicting the title.
- `tests/test_tables.py` — Phase 4: unit tests for `TableData` (header detection, CSV round-trip), overlap math, caption matching, and all three render modes including prose-safety guardrails (mocked LLMs). `handle_tables` tests use `monkeypatch` to stub `_extract_raw_tables` — no real pdfplumber calls in unit tests. Integration tests run the full extract→classify→tables pipeline on both PDFs and assert ≥1 (Shaw) / ≥2 (Titans) `kind="table"` blocks with structured payloads.

Integration tests are skipped when `papers/<name>.pdf` is absent. If you move the sample PDFs, update `TWO_COL_PDF` / `ONE_COL_PDF` constants in each test file.

## Required Environment Variables

Set in `.env` file:
- `GOOGLE_API_KEY` — required for PDF→text step with `--llm-provider google` (default)
- `GOOGLE_GENAI_USE_VERTEXAI=false` — required alongside `GOOGLE_API_KEY`; the library defaults to Vertex AI which lacks Gemma models
- `CEREBRAS_API_KEY` — required for PDF→text step with `--llm-provider cerebras`
- `MURF_API_KEY` — only if using Murf.ai engine

Optional (with defaults):
- `MURF_VOICE_ID` (`marcus`), `MURF_FORMAT` (`mp3`), `MURF_CHUNK_CHARS` (`2800`)
- `KOKORO_VOICE` (`af_bella`), `KOKORO_WORKERS` (`1`)

## System Dependencies

- `ffmpeg` — required by pydub for audio processing
- `espeak-ng` — required by Kokoro for phonemization (macOS: `brew install espeak-ng`)
