# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

Always update this file after relevant project changes.

## Confirmed intent and audit status (2026-09-06)

- The owner confirmed that this is a **personal local tool**. Hosted-service infrastructure is outside the current scope.
- The intended output is **faithful prose with math/table narration**, omitting references and appendices. Do not introduce general paraphrasing or summarization of the paper's prose.
- The current audit and prioritized implementation plan are in `AUDIT_AND_REPAIR_PLAN.md`. The audit changed documentation only; its listed code defects remain open.
- Baseline: `.venv/bin/python -m pytest -q` passed **326 tests**, with no skips and five PyMuPDF/SWIG deprecation warnings. A short complete PDF → Kokoro MP3 CLI run and a separate two-worker Kokoro run both succeeded with Hugging Face/Transformers offline mode enabled. Cloud provider calls, a clean installation, browser interactions, and full-paper audio quality were not verified.
- Tests currently concentrate on `pipeline/`; there are no dedicated CLI, Flask/UI, TTS, or legacy-cleanup suites. Passing corpus count checks does not establish narration fidelity.
- Highest-priority confirmed defects: ordinary `A …` titles can trigger appendix filtering and delete the whole paper; grouped quantities such as `1,000 MB` are corrupted; the entry points' LLM controls never activate LLM policies. Before enabling those policies, protect surrounding prose and table facts from model changes.
- Additional repair work covers rejected/empty narration, oversized TTS chunks, audio formats, web job state and recovery, optional dependencies, setup checks, and stale usage documentation. Consult the audit for reproductions and acceptance criteria.
- The audit began with substantial existing modified and untracked source/test files. Preserve that work and include the necessary untracked files deliberately when preparing commits. Do not refresh corpus baselines merely to hide a regression.

## Project Overview

Converts academic papers (PDF) into audio files. Two-stage pipeline:
1. **PDF → Text**: The layout-aware `pipeline/` package extracts text via pymupdf dict-mode, classifies blocks (headings, body, equations, tables, captions, …), filters non-narration sections, rewrites equations/inline-math, and polishes for TTS (numbers, acronyms, abbreviations, units). Optional LLM calls for table/equation/inline-math rewriting.
2. **Text → Speech**: Uses either Murf.ai (cloud, paid) or Kokoro (local, free) TTS engine.

Both `main.py` and `app.py` use `pipeline.run_pipeline()` + `pipeline.serialize()`. The legacy `pdf_to_text.py` still exists as a standalone script but is no longer wired into the main entry points.

Current integration caveat: both entry points always pass `PipelineConfig()` with table, equation, and inline-math modes set to `skip`. They construct an LLM when requested, but the policies never call it. Cache, inspection, per-phase configuration, and vision-first processing exist as library APIs; they are not exposed by the entry points. These are current limitations, not the intended final behavior.

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

- **`main.py`**: Orchestrates both steps. Calls `pipeline.run_pipeline()` for PDF→text (extract → classify → filter → tables → equations → inline-math → polish) then `pipeline.serialize()` to flatten to plain text. LangChain chat models from `pdf_to_text.build_llm()` are adapted to `Callable[[str], str]` via `_wrap_langchain_llm()`. `--tts-engine` defaults to `murf`. Supports `--text-file` to skip PDF processing, `--keep-text` to save intermediate text (named `<pdf_basename>_audio_text.txt`), `--no-llm` for deterministic-only cleanup, `--kokoro-workers` to override parallelism.
- **`app.py`**: Flask web UI. Calls `pipeline.run_pipeline()` directly for text processing (in-process, threaded). Audio generation still shells out to `text_to_speech.py` as a subprocess for live log streaming.
- **`text_to_speech.py`**: Splits text into chunks (paragraph/sentence boundary-aware via `split_text`), generates audio via `TTSEngine` protocol, concatenates with `pydub`. `generate_audio_chunks()` dispatches sequential vs parallel generation. Kokoro parallel mode uses `ProcessPoolExecutor` with **spawn context** and `_kokoro_worker_init()` to load one `KPipeline` per worker (must be spawn — fork breaks PyTorch). Key classes: `MurfTTSEngine`, `KokoroTTSEngine`.
- **`pdf_to_text.py`** (legacy, standalone only): Flat-text extraction + regex/LLM cleanup. Still runnable standalone but no longer wired into `main.py` or `app.py`. Provides `build_llm()` which both entry points import for LLM construction.

  LLM providers via `build_llm(model, provider)`:
  - `google` (default) → `ChatGoogleGenerativeAI`, default model `gemma-3-27b-it` (fast, no extended thinking). Other options in `GEMMA_MODELS`.
  - `cerebras` → `ChatOpenAI` pointed at `https://api.cerebras.ai/v1`, default `qwen-3-235b-a22b-instruct-2507`.

## Pipeline (`pipeline/` package)

Layout-aware PDF preprocessing pipeline. Keeps layout and font metadata end-to-end so each block can be routed to the right handler (prose, equation, table, caption, ...). The intended stage contract is a pure function `Document -> Document`, but `classify_blocks()` currently mutates its input blocks and returns the same document. Copy inputs before retaining independent classification snapshots until that contract is repaired.

Module layout:

```
pipeline/
  model.py        # Span, Block, FontStats, Document dataclasses
  extract.py      # Phase 1 — pymupdf dict-mode extraction, column detection, paragraph rejoin
  classify.py     # Phase 2 — block classification, heading levels, parent_section tree
  policies.py     # Phase 3 — SectionPolicy + filter_sections (skip References, footnotes, ...)
  tables.py       # Phase 4 — pdfplumber detection, block merging, skip/verbatim/prose renderers
  equations.py    # Phase 5 — display equation narration (skip/symbolic/vision)
  inline_math.py  # Phase 6 — inline variable detection + ⟦…⟧ delimited rewrite
  polish.py       # Phase 7 — numbers/abbreviations/acronyms/units/quotes (+ optional SSML)
  qa.py           # Phase 8 — SQLite LLM cache, stats/diff, JSON dump, fail-soft, run_pipeline
  vlm.py          # Phase 9 — optional vision-first mode (render -> VLM narrate -> polish)
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
- `_is_heading` has three paths: (A) `dominant_size >= body * 1.05` with a structural signal (numeric prefix, near-all-caps, or ≤14 words); (B) bold-at-body-size with numeric/all-caps prefix — catches venues that bold inline subsection headings; (C) all-caps standalone line at body size (≥95% ratio) with ≥3 letters and ≤10 words — catches ICLR/NeurIPS templates that set section headings in the body font itself (`ABSTRACT`, `1 INTRODUCTION`, `3 METHOD`). `_is_rotated_block` rejects narrow, tall blocks (e.g. the vertical arXiv sidebar watermark) which would otherwise dominate heading-size rank.
- `arXiv:<id>` margin stamps are routed straight to `noise` by `_ARXIV_HEADER_RE` before the other rules run. Otherwise pymupdf's variable per-page decoding of the rotated stamp lets it surface as oversized "heading" text on some pages and body text on others; either way it's metadata, not narration, so downstream polish doesn't get to mangle `cs.LG` into `cs.L G`.
- Equation-number regex is deliberately tight (`\(\d{1,3}(\.\d{1,3})?\)$`) so citation years like `(2019)` don't match.
- `_assign_heading_levels` ranks unique heading sizes descending (rounded to 0.1pt). Drop-cap letters in some section titles inflate weighted size, so heading levels can be noisier than in print — downstream filtering should key off `text`, not `level`.
- `_assign_parent_sections` walks reading order and sets `parent_section = <nearest preceding heading.text>` on every non-heading block.

Phase 3 (`pipeline/policies.py`) — `filter_sections(doc, policy) -> Document`:
- `SectionPolicy` dataclass holds: `skip_kinds` (default `page_header, page_footer, footnote, toc, noise, figure`), per-section booleans (`skip_references`, `skip_acknowledgments`, `skip_author_contributions`, `skip_supplementary`, `skip_appendix`), `keep_code` (default False), and `extra_skip_patterns` for custom regexes.
- `section_patterns()` assembles the effective regex list from the enabled booleans plus extras. References, acknowledgments, and author-contribution patterns support an optional numeric or single-letter prefix. Appendix and supplementary patterns do not consistently support it: `A Appendix` and `6 Supplementary Material` currently fail to match. `skip_appendix` additionally installs `_DEFAULT_APPENDIX_LETTER`, which catches multi-token lettered appendices but also ordinary titles such as `A Novel Method for Learning`; a title at the highest heading level can therefore cause the entire paper to be dropped. This heuristic needs document context before it is reliable.
- Filtering walks blocks in order. A heading whose text matches a skip regex opens a skip at that heading's `level`; subsequent blocks are dropped until a later heading with a known level ≤ the skip level appears. Skip level defaults to 99 when the opening heading has `level is None`; a later unknown-level heading does not close the skip. The intended unknown-level behavior needs an explicit regression test.
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

Phase 5 (`pipeline/equations.py`) — `handle_equations(doc, policy, llm=None, vision_llm=None) -> Document`:
- Operates only on blocks classified `kind="equation_display"` by Phase 2; all other kinds pass through unchanged.
- `_extract_equation_number` strips a trailing `(3)` / `(3.4)` from the block text; `_spoken_number` converts the number digit-by-digit (`"3.4"` → `"three point four"`).
- `_reconstruct_with_scripts` synthesizes `^token` / `_token` markers from span metadata: bit-0 `is_superscript` is authoritative for superscripts; subscripts are inferred from a span whose `size < 0.88 × median_size` AND whose y-center sits below the median by more than 15% of the line height. With only one baseline span the y-median resolves to the script span itself, so the heuristic needs ≥2 baseline-sized spans to fire (see the `_infers_subscript_from_y_offset` test — it uses 3 baseline spans for exactly this reason).
- `_symbolic_rewrite` applies Greek + unicode-math maps, Unicode super/subscript digit maps (`x²` → `x^2` → `x to the 2`), `^`/`_` marker expansion, and ASCII `=` → `equals`. Maps are duplicated from the legacy `pdf_to_text.py` so `pipeline/` has no cross-import dependency.
- `EquationData` carries `raw_text`, `symbolic`, `number`, `page`, `bbox`. Attached to promoted blocks as `meta["equation"]`.
- `render_equation` dispatches on `EquationPolicy.mode`:
  - `skip` (default): symbolic form prefixed with the equation number when present (`"Equation three point four: x equals y."`). No LLM.
  - `symbolic`: sends the spelled-out form to an LLM under a bounded prompt; falls back to `skip` on exception, empty output, or narration shorter than `min_narration_ratio × symbolic_len`.
  - `vision`: renders the equation bbox via `fitz.Page.get_pixmap(clip=bbox, dpi=200)`, sends PNG bytes to a `VisionLLMCallable(image, prompt) -> text`; on any failure falls through to `symbolic`, then `skip`.
- `_prepend_equation_number` ensures the eq-number prefix is added exactly once — it's suppressed when the LLM narration already contains the spoken form (`"Equation three point four ..."`), so no double-prefix.
- `_sanitize_narration` strips HTML tags, dollar-sign LaTeX delimiters, and markdown code fences from any LLM output before it reaches the transcript.
- Pure function: replaces blocks via `dataclasses.replace`; `Block.kind` stays `equation_display` (only `text` and `meta` change) so downstream consumers can still find them.
- Size guards: skips LLM calls when `symbolic_len < min_chars` (stray fragments) or `> max_chars` (misclassified multi-paragraph content); defaults 3 / 600.

Phase 6 (`pipeline/inline_math.py`) — `rewrite_inline_math(doc, policy, llm=None) -> Document`:
- Operates only on `kind="body"` blocks. Non-body blocks and body blocks without any detected variables pass through unchanged (identity-preserving).
- `_is_variable_span` treats a span as a variable iff its stripped text is exactly one character AND (italic-ASCII-alpha) OR the codepoint falls in Greek (`U+0370–U+03FF`), Math Alphanumeric Symbols (`U+1D400–U+1D7FF`), or Letterlike Symbols (`U+2100–U+214F`). The Math Alphanumeric range catches PDFs that use `𝑡`, `𝑥`, `𝜃`, etc. directly (e.g. Titans) — these glyphs often don't set the italic flag even though they're visually italic math letters.
- `_mark_inline_math` walks the spans in document order, collects variable texts, then wraps each occurrence in the block's paragraph text with `⟦…⟧` delimiters (`U+27E6` / `U+27E7`) using `_find_standalone` to enforce non-alpha boundaries on both sides — so the `x` in `example` is never marked.
- Duplicate variables are marked in order: the `pos` cursor advances past each match so both occurrences of "x" in `for every x there is x squared` get wrapped.
- `rewrite_inline_math` dispatches on `InlineMathPolicy.mode`:
  - `skip` (default): returns the input `Document` untouched (identity).
  - `symbolic`: `_render_symbolic` NFKC-normalizes each delimited span (math-italic `𝑡` → `t`, `𝜃` → `θ`, `ℓ` → `l`) then routes through `equations._symbolic_rewrite` so Greek names are spelled out. ASCII italic variables effectively pass through unchanged — the value is in the `meta["inline_math"]` annotation for downstream inspection.
  - `llm`: `_render_llm` sends the delimited paragraph to an LLM with a prompt that asks to rewrite only the delimited spans. Falls back to `symbolic` on exception, empty output, or output shorter than `min_output_ratio × marked_len` (default 0.5). Any delimiters the LLM leaves behind are stripped defensively.
- Pure function: replaces blocks via `dataclasses.replace`; the input `Document` is never mutated. `meta["inline_math"]` carries the marked form for A/B comparison.

Phase 7 (`pipeline/polish.py`) — `audio_polish(doc, policy) -> Document`:
- Operates on every block kind not in `PolishPolicy.skip_kinds` (defaults skip `code`, `figure`, `page_header`, `page_footer`, `toc`, `noise`, `footnote`, `equation_inline`); `body`, `heading`, `caption`, `equation_display`, `table` all get polished so the table/equation narrations produced by Phases 4–5 also benefit.
- Per-block transforms run in a deliberate order (see `_polish_text` docstring): quotes → pronunciations → units → acronyms → abbreviations → standalone numbers. Reordering breaks things — units must run before acronyms or `100 GB` becomes `100 G B`; pronunciations must run before acronyms or `ReLU` becomes `R E L U`.
- `_apply_pronunciations` is a longest-first case-sensitive whole-token substitution. `DEFAULT_PRONUNCIATIONS` covers `ReLU/PReLU/GeLU/SiLU/LaTeX/TeX/arXiv/PyTorch/NumPy/SciPy/scikit-learn/GitHub/GPU/CPU/TPU`. Anchored with `(?<!\w)…(?!\w)` (not `\b`) so mixed-case keys like `arXiv` and `PReLU` match cleanly — `\b` triggers on case boundaries inside the token.
- `_apply_acronyms` walks `\b([A-Z][A-Z0-9]{1,7})s?\b(?:\s*\(([^)]{2,80})\))?` and threads document-wide state via `_PolishState.seen_acronyms`. First occurrence with a parenthetical whose word initials match the acronym (`_expansion_matches`, splitting on whitespace AND hyphens so `Short-Term` contributes both `S` and `T`) emits `L S T M, Long Short-Term Memory` and records the acronym. Subsequent occurrences emit `L S T M` and drop any parenthetical. Parentheticals that aren't expansions (citations like `BERT (Devlin et al., 2018)`) are preserved verbatim. `_is_mostly_uppercase` (≥90% of the alphabetic payload is upper case) short-circuits the whole pass for a block — otherwise all-caps titles (`AN IMAGE IS WORTH 16X16 WORDS`) and all-caps section headings (`3 METHOD`) would have every word treated as an acronym and be letter-spaced character-by-character. Mixed-case prose is nowhere near 90% upper even when studded with acronyms, so legitimate `LSTM` / `BERT` / `GLUE` still get expanded.
- `_apply_abbreviations` is also longest-first so `Figs.` beats `Fig.` and `et al.` beats `al.`. Defaults cover Latin abbreviations (`e.g.`, `i.e.`, `cf.`, `vs.`, `etc.`, `et al.`, `approx.`) and academic shorthand (`Fig./Figs.`, `Eq./Eqs.`, `Ref./Refs.`, `Sec./Sect.`, `Tab.`, `Ch.`, `App.`).
- `_apply_units` uses one regex `(\d+(?:\.\d+)?)\s*(GHz|MHz|...)(?!\w)` so the number and unit are converted together — `1 GB` → `one gigabyte` (singular at exactly 1 / 1.0), everything else plural. Trailing `(?!\w)` rejects `100 GBs` so we don't double-pluralize. Unit alternation is sorted longest-first so `GHz` beats `Hz`.
- `_apply_numbers` is the final pass: `(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)(?![\w.])` matches integers (with optional thousands separators), decimals, and signed forms while excluding numbers glued to identifiers (`H2O`, `x_3`). `_spell_number` strips commas, handles the leading `-` as `negative`, splits on `.` for decimals (digit-by-digit fractional reading via `_DIGIT_WORDS`), and falls back to digit-by-digit when the integer exceeds `policy.max_number_words` (default 9.999M) since `num2words` produces unreadable output for 12-digit values. Defensive `try/except` around `num2words` import + call so a missing dep degrades to the literal digits rather than crashing.
- `to_ssml(doc, policy)` is an opt-in serializer alternative: wraps blocks in `<speak><p>…</p></speak>` with a `<break time="…ms"/>` before every non-first `heading`. Minimal XML escaping (`&<>`); no `<sub alias>` injection because the polish pass already replaced known mispronounced tokens. Engines that ignore SSML treat the output as plain text plus angle-bracket noise, so callers must opt in deliberately.
- Pure function: returns a new `Document` via `dataclasses.replace`; per-block identity is preserved when no transform fires (`out.blocks[0] is doc.blocks[0]`).

Phase 8 (`pipeline/qa.py`) — caching, stats, inspection, fail-soft:
- `LLMCache(path, model, prompt_version="v1")` wraps a SQLite file (WAL mode, `check_same_thread=False` so worker pools can share it). `wrap_text(llm)` returns a drop-in replacement for any `LLMCallable`; `wrap_vision(vision_llm)` wraps the `(image_bytes, prompt) -> text` shape. Keys are `sha256(model | prompt_version | prompt [ | sha256(image) ])`, so (a) bumping `prompt_version` invalidates exactly that iteration, (b) different models don't share entries, and (c) vision requests key on the actual image bytes. Errors from the wrapped callable propagate and are NOT cached — a single flaky call doesn't poison the cache. Schema stores `(key, model, prompt_version, response, created_at)`. Context-manager support (`with LLMCache(...)`) closes the connection on exit; `close()` is idempotent.
- `StageStats` (frozen dataclass) is the snapshot returned by `document_stats(doc, stage)`: `total_blocks`, `total_chars`, `by_kind: dict[str, int]`, `chars_by_kind`. `compare_stages(before, after)` returns a structured diff with `block_delta`, `char_delta`, and `by_kind_delta` covering every kind seen in either snapshot. Both functions are linear in block count; safe to call after every phase.
- `document_to_dict(doc)` / `document_to_json(doc, pretty=True)` produce a lossy inspection dump. `meta` values pass through `_encode_meta`, which (a) JSON-passes scalars, (b) flattens dataclasses via `asdict` and tags them with `__type__` (so an `EquationData` payload appears as `{"__type__": "EquationData", "raw_text": ..., "symbolic": ...}`), (c) converts tuples/sets to lists, (d) summarizes bytes as `<bytes len=N>`, and (e) falls back to `repr()` for anything else. Intentionally one-way — use `pickle` for round-trip checkpointing.
- `safe_block(fn, block)` is the fail-soft primitive. On exception in `fn(block)` the block is downgraded to `kind="noise"` with `meta["raw"]` = original text, `meta["original_kind"]` = pre-failure kind, `meta["handler_error"]` = `repr(exc)`. Per-block isolation, so one bad block can't crash the pipeline. Currently wired into `handle_equations` (the handler most likely to hit a pathological input via span reconstruction); other handlers can opt in the same way — `from pipeline.qa import safe_block` uses a local import to avoid a cycle with `qa.py`'s phase imports.
- `PipelineConfig` bundles one `*Policy` per phase with sensible defaults. `run_pipeline(path, config=None, llm=None, vision_llm=None, collect_stats=False) -> (Document, list[StageStats])` runs every shipped phase (extract → classify → filter_sections → handle_tables → handle_equations → rewrite_inline_math → audio_polish) in order. With `collect_stats=True` each phase snapshot is labeled by phase name so callers can feed them pairwise to `compare_stages`. Callers who want caching should call `LLMCache.wrap_text(llm)` / `.wrap_vision(vision_llm)` before passing those into `run_pipeline`.

Phase 9 (`pipeline/vlm.py`) — optional vision-first mode:
- `vlm_extract_document(path, vision_llm, policy) -> Document` bypasses Phases 1-6 entirely. It opens the PDF with pymupdf, renders each page with `page.get_pixmap(dpi=policy.dpi)` (default 200), and calls `vision_llm(png_bytes, policy.prompt)` per page. Each successful page becomes one `kind="body"` block with a synthesized single-span placeholder (font=`"vlm"`, size=10.0) covering the page bbox — Phase 7 is the only downstream consumer and needs at least one span for `dominant_size`.
- `VLMPolicy` fields: `dpi` (200), `prompt` (`DEFAULT_VLM_PROMPT`, the plan prompt: narration verbatim / describe equations & tables in 1-2 sentences / skip figures but keep captions / drop headers-footers-page-numbers), `skip_pages` (frozenset of 0-indexed pages to render-but-not-narrate), `max_pages` (cost cap; `None` means all pages), `min_narration_chars` (default 20; below this, the page is treated as failed).
- Per-page fail-soft: VLM exceptions, too-short narrations, and render failures all produce `kind="noise"` blocks with `meta["handler_error"]` = error repr and `meta["vlm_stage"]` ∈ `{"render", "vlm_call", "too_short"}` — one flaky page cannot kill the whole paper. Narration whitespace is collapsed with `" ".join(out.split())` before assembly.
- `run_pipeline_vlm(path, vision_llm, policy=None, polish=None, collect_stats=False) -> (Document, list[StageStats])` is the convenience orchestrator: runs `vlm_extract_document` then `audio_polish` so numbers/acronyms/abbreviations are still normalized on the VLM output. With `collect_stats=True` it emits `"vlm_extract"` and `"audio_polish"` snapshots — feedable pairwise to `compare_stages`.
- Caching: callers wrap `vision_llm` with `LLMCache.wrap_vision` BEFORE passing it in. The cache keys on `sha256(image_bytes)` so re-running on the same PDF with the same prompt is free. Errors are not cached, so flaky pages retry on next invocation.
- Trade-offs vs the symbolic pipeline: one vision call per page (cost) and the VLM may paraphrase (fidelity) — but pathological layouts, scanned pages, and heavy-math papers all work in one shot. Intended as an opt-in fallback or as a reference output to diff Phases 1-6 against.

The symbolic stages are wired into `main.py` and `app.py` via `run_pipeline()`. Phase 8 cache/inspection utilities require explicit caller use. Phase 9 is a separate `run_pipeline_vlm()` library entry point; it is not called by either CLI or Flask. The legacy `pdf_to_text.py` is still runnable standalone and still supplies `build_llm()` to the main entry points.

## Testing

`tests/` holds pytest suites mirroring the pipeline phases.

- `tests/test_extract.py` — Phase 1: unit tests on fixture blocks (ligature expansion, hyphenation, column detection, reading order, font-stat weighting) plus integration tests on both real PDFs.
- `tests/test_classify.py` — Phase 2: unit tests per rule (heading / caption / code / footnote / equation / noise / header-footer / repeating-text normalization) plus integration tests that assert title-is-heading, `References` heading exists, ≥3 captions tagged, ≥40% of blocks remain body.
- `tests/test_policies.py` — Phase 3: unit tests for pattern assembly, per-kind filtering, section skipping (refs / acknowledgments / appendix), nested subsection handling, sibling heading closes skip, parent-section fallback, purity. Integration tests assert `References` and `Acknowledgments` are gone from the real PDFs and that filtering reduces block count without evicting the title.
- `tests/test_tables.py` — Phase 4: unit tests for `TableData` (header detection, CSV round-trip), overlap math, caption matching, and all three render modes including prose-safety guardrails (mocked LLMs). `handle_tables` tests use `monkeypatch` to stub `_extract_raw_tables` — no real pdfplumber calls in unit tests. Integration tests run the full extract→classify→tables pipeline on both PDFs and assert ≥1 (Shaw) / ≥2 (Titans) `kind="table"` blocks with structured payloads.
- `tests/test_equations.py` — Phase 5: unit tests for equation-number extraction, digit-to-words spoken form, symbolic rewrite (Greek / unicode math / `^_` / Unicode super/subscript digits), span-aware sub/superscript reconstruction (needs ≥2 baseline spans for a reliable median — see `test_infers_subscript_from_y_offset`), and all three render modes (skip / symbolic / vision) with their fallback chains (lambda-based mock LLMs). Integration tests assert at least one equation survives classification on each sample PDF and gets `EquationData` attached in `meta`.
- `tests/test_inline_math.py` — Phase 6: unit tests for variable-span detection (italic ASCII vs Greek vs math-italic Unicode vs non-italic singletons), standalone-token boundaries (`x` in `example` is never marked), duplicate-variable ordering (`pos` advances past each match), and the three render modes including LLM fallback chains. Integration tests assert (a) at least one body block on the two-column paper gets `inline_math` meta, (b) rendered text has no residual `⟦/⟧` delimiters, and (c) the one-column paper (Titans, which uses math-italic Unicode for variables) renders at least one Greek letter name — confirming NFKC normalization + Math Alphanumeric Symbols detection works end-to-end.
- `tests/test_polish.py` — Phase 7: unit tests for each transform (quotes, pronunciations including longest-key-wins, acronyms with state threaded across blocks via `_PolishState`, hyphen-splitting in `_expansion_matches`, abbreviations including `Figs.` beating `Fig.`, units with singular/plural at exactly 1 and `(?!\w)` rejecting `100 GBs`, numbers with commas/decimals/negatives/huge-fallback/identifier-rejection), pipeline ordering (`_polish_text` ensures units run before acronyms and pronunciations before acronyms), and the `audio_polish` pure-function contract (skip-kinds untouched, identity preserved when no rule fires, `meta`/`level`/`parent_section` carried forward). `to_ssml` tests cover wrapper, heading-break placement (no leading break before first block), and XML escaping. Integration tests serialize the polished output for both PDFs and assert `e.g.`/`i.e.`/`Fig. `/`et al.` are gone, plus at least one acronym (LSTM/RNN/MLP/BERT/GPT) appears letter-spaced in Titans.
- `tests/test_vlm.py` — Phase 9: unit tests for `VLMPolicy` defaults, per-page render → VLM-stub → block assembly (one `body` block per page, PNG bytes non-empty, custom prompts threaded, `skip_pages` / `max_pages` honored, page_rects populated, `source_path` preserved, synthesized placeholder span, whitespace-collapsed narration), fail-soft (VLM exception → `kind="noise"` with `meta["vlm_stage"]="vlm_call"`; empty/short narration → `vlm_stage="too_short"`; one bad page doesn't kill the run), and `run_pipeline_vlm` integration with Phase 7 polish (`e.g.` expanded, `100 MB` → words, custom `PolishPolicy` threaded through, `collect_stats=True` emits `vlm_extract` + `audio_polish` snapshots). All tests use stub `vision_llm` lambdas — no real VLM calls in the suite. Tests skip automatically when the sample PDFs are absent.
- `tests/test_qa.py` — Phase 8: unit tests for `document_stats` counts/labels, `compare_stages` deltas (new kind appears, char delta, sign), `document_to_dict` including dataclass `__type__` tagging for `EquationData`/`TableData`, bytes summarization, `repr()` fallback, and JSON validity. `LLMCache` tests cover hit/miss, per-prompt separation, `prompt_version` bump invalidates, per-model separation, persistence across instances (reopen same file), error non-caching (a failed call is retried), and vision wrapping (image-bytes hashed into the key). `safe_block` tests assert success-passthrough, exception downgrade to `noise` with preserved meta, and that `handle_equations` survives a monkey-patched `_build_equation_data` failure. The regression-corpus tests at the bottom run `run_pipeline` on all three sample PDFs: `TestTwoColumnCorpus` (Shaw) and `TestOneColumnCorpus` (Titans) pin block-kind distributions + char counts within ±5% (or ±2 absolute) tolerance; `TestViTCorpus` (An Image is Worth 16x16 Words) covers the ICLR-template-specific regressions — Path C all-caps heading detection, title not letter-spaced, lettered-appendix filtering, arXiv stamp dropped. Refresh the pinned corpus numbers deliberately if a legitimate pipeline change shifts expectations.

Integration tests are skipped when `papers/<name>.pdf` is absent. If you move the sample PDFs, update `TWO_COL_PDF` / `ONE_COL_PDF` / `VIT_PDF` constants in each test file.

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
