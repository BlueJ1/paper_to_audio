# Repository guidance

Always update this file after relevant project changes. Preserve existing user work and deliberately include required untracked files when preparing commits. Do not refresh corpus baselines merely to hide a regression.

## Intent and current repair status (2026-09-07)

This is a **personal local tool**, not a hosted service. Preserve the paper's prose while narrating math and table values. Omit references and appendices. Author details and inline citations remain included pending an explicit preference change. Vision is an opt-in library fallback; never silently enable paid processing for scanned/blank PDFs.

The five milestones in `AUDIT_AND_REPAIR_PLAN.md` have been implemented. The original findings are retained there as historical audit evidence; its repair-status section records current checks and limitations. Baseline corpus expectations were not refreshed. New regression suites use generated PDFs and mocks, so essential coverage does not depend on optional sample files. Cloud calls and full-paper listening quality remain separate verification work.

Current validation: 413 tests passed in both the working and clean environments, with no skips and six deprecation warnings. Clean core and selected-Kokoro installs, local CLI/spawn audio, and browser lifecycle/failure checks are recorded in the audit status. `CLAUDE.md` points here to avoid conflicting guidance. `.env.example` contains empty optional keys and no Vertex AI override.

## Commands and dependencies

Supported Python: **3.12**. The original `.venv` is a working environment; avoid changing it unnecessarily.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -c constraints.txt
# Select optional packages as needed:
python -m pip install -r requirements-kokoro.txt -c constraints.txt
python -m pip install -r requirements-google.txt -c constraints.txt
python -m pip install -r requirements-cerebras.txt -c constraints.txt
python -m pip install -r requirements-dev.txt -c constraints.txt

python check_setup.py --dev
python check_setup.py --tts-engine kokoro
python -m pytest -q

python app.py  # http://127.0.0.1:5000
python main.py papers/example.pdf --no-llm --tts-engine kokoro --out output.mp3 --keep-text
python main.py papers/example.pdf --tts-engine kokoro --inspect narration.json --cache narration.sqlite
python text_to_speech.py edited.txt --tts-engine kokoro --kokoro-workers 2 --out output.mp3
```

`requirements.txt` contains the core deterministic pipeline, Flask, and lightweight audio dependencies. Provider and Kokoro dependencies are optional and lazy-loaded. Kokoro's English pronunciation model is pinned to the official `en_core_web_sm` 3.8.0 wheel in `requirements-kokoro.txt`; its absence is checked before Kokoro can invoke an implicit pip download. Speech weights/voices need a first download or existing Hugging Face cache. FFmpeg is required for audio. The tested English path works without `espeak-ng` on PATH; other language/phonemizer configurations may require it.

`constraints.txt` is a macOS Python 3.12 transitive version snapshot, not a universal hash lock. `scripts/snapshot_constraints.py` regenerates it from an environment containing all optional sets. Validate clean installation after updates. Python 3.13 is not supported by the pinned pydub/audioop stack. CI runs Python 3.12 core setup, CLI help, dependency checks, and all tests without ML/provider installations.

The old `.venv`'s `pip check` warning comes from an unsupported installed PyTorch wheel tag (`macosx_110_0_arm64`); actual Kokoro execution succeeds. A fresh pinned installation passes `pip check`. Do not modify the working environment solely to remove that warning.

## Entry points and policy contracts

- `processing.py`: shared `build_config(use_llm)` and `process_pdf(path, use_llm=False, provider="google", model=None, cache_path=None, llm=None) -> (text, report)`. Reports include context-local fallback/omission warnings, per-stage statistics, and document metadata. Empty narration raises an actionable error before TTS.
- `providers.py`: selected-provider imports and the shared LangChain response adapter. Google explicitly uses `vertexai=False`. Calls have bounded timeouts/retries. Defaults: Google `gemma-3-27b-it`; Cerebras `gpt-oss-120b` (the previous Qwen model is deprecated). Accept explicit model overrides; do not assume account availability without verification.
- `main.py`: CLI defaults to Google LLM enabled and Murf. `--no-llm --tts-engine kokoro` is local. `--keep-text` writes `<pdf_basename>_audio_text.txt` in the command directory; no temporary transcript is needed. `--text-file` skips PDF/model work. `--cache` is optional SQLite caching, keyed by provider/model/`structured-v2`; `--inspect` writes a reviewable JSON report.
- `app.py`: local Flask upload → process → edit → generate → play/download → regenerate. Text uses `process_pdf` in a thread. Audio uses the standalone TTS subprocess for logs. No shared consumable event queue.
- `pdf_to_text.py`: standalone legacy flat-text cleanup only, no longer imported by either modern entry point. It re-exports the shared `build_llm`; optional prompt/provider imports are lazy.

Entry-point deterministic policies: table `verbatim` (up to 20 rows), equation `skip` (symbolic narration), inline math `symbolic`. Enabled policies: table `prose` (validated row selection, at most three rows), equation `symbolic` (LLM narration), inline math `llm` (validated replacements). Bare library `PipelineConfig()` still defaults to all three `skip` modes for compatibility. Do not mistake the two policy levels.

## Pipeline

Stages are extract → classify → section filter → tables → equations → inline math → polish. `Document`, `Block`, `Span`, and `FontStats` retain geometry, typography, heading levels, parent sections, and metadata. Classification now copies blocks; it no longer mutates the input snapshot. Later transformation stages return new documents and replace changed blocks.

- `extract.py`: PyMuPDF dict-mode, NFC/ligatures, hyphenated-line healing, column detection, reading order. Full-width blocks divide bands; left-column blocks precede right-column blocks within bands.
- `classify.py`: repeated margin text is headers/footers; arXiv stamps are noise. Caption precedence protects large caption text. Headings combine typography with structural cues, including all-caps body-size section titles. Equation-number patterns deliberately exclude four-digit citation years. Drop caps can make heading ranks noisy.
- `policies.py`: explicit numbered/dotted appendix and supplementary headings are recognized. Bare letter prefixes require preceding numbered/main-section or back-matter context, so an `A …` title/front matter survives. A later heading with unknown level closes an active skip; known deeper headings remain skipped. Parent fallback applies explicit patterns and sections actually identified as skipped, never a context-free letter regex.
- `tables.py`: conservative ruled-table detection, minimum 2×2, at least 50% overlap with body/noise for promotion. Keep this gating; unruled recall remains deferred. LLM `prose` mode accepts only `{ "rows": [zero_based_data_indices] }`, validates indices, and renders original cells/column associations. No free-form table claims enter output. Invalid/missing responses fall back to source-cell narration, with visible warnings. Truncation/selection is explicitly announced. Table source data stays in metadata.
- `equations.py`: Greek/unicode math narration, grouped script runs before NFKC, Unicode-aware explicit scripts, ASCII operators. Prefix suppression recognizes an actual leading `Equation N`, not a cardinal number elsewhere. The existing short/long region guards and vision → symbolic → deterministic fallback chain remain. EquationData retains raw/symbolic forms. Free-form equation explanations need human review; deterministic guards cannot prove equivalence.
- `inline_math.py`: detects single-variable spans (italic Latin, Greek, mathematical alphanumeric/letterlike symbols). Standalone matching protects letters embedded in words. In LLM mode the model receives only variable regions and returns a JSON array; each replacement must preserve the deterministic spoken identity. Replacements are spliced into immutable surrounding prose. The marked original is inspectable in metadata. Broader region detection requires an appropriate validator before relaxing this contract.
- `polish.py`: quotes → pronunciations → units → acronyms → abbreviations → numbers. Preserve this ordering. Quantities consume whole grouped/signed decimals before unit expansion. Terminal punctuation does not block numbers, while decimal/identifier boundaries remain protected. Math scripts are grouped before compatibility normalization. All-caps headings avoid letter-spacing every word. Acronym expansion state spans the document.
- `serialize.py`: shared final eligibility for plain text and SSML excludes noise, figure/layout debris, footnotes, and handler failures even after filtering. `safe_block` and VLM rejected text stays diagnostic, never spoken. `require_narration()` is the application-level empty-text guard.
- `qa.py`: SQLite LLM cache, phase stats/diffs, inspection encoding (mixed sets sort stably), fail-soft blocks, PipelineConfig/run_pipeline. Cache connection closes at the end of each `process_pdf` call. Provider errors are not cached.
- `vlm.py`: separately selected page-render/vision path, with failed/short pages marked noise. Library-only, potentially paraphrasing, never selected automatically by CLI/UI.

Author metadata, inline citations, figure-internal labels, unruled tables, and possible table-caption duplication remain extraction/narration limitations. Improve recall only with positive and negative layout fixtures; do not loosen global heuristics to satisfy counts.

## Audio boundaries

`text_to_speech.synthesize()` validates engine, format, positive chunk limit/worker count, and usable text before model initialization. `split_text()` preserves exact concatenation of whitespace-normalized input and chooses paragraph, sentence, word, then hard boundaries; every chunk obeys the limit. Boundary spaces are retained. Whitespace-only pieces at a one-character limit are ignored for synthesis.

Kokoro parallelism uses `ProcessPoolExecutor` with **spawn**, one pipeline per worker; never switch to fork (PyTorch breaks). Futures are collected in input order. Murf requests carry uppercase MP3/WAV/FLAC/OGG formats. `concatenate_audio()` decodes the explicit source format, exports and validates a temporary sibling MP3, and atomically replaces the destination. Empty/failed runs preserve previous outputs. Temporary publication and web transcript files are removed in `finally`.

## Local web lifecycle

`jobs.JobManager` uses one condition/RLock for job/attempt transitions. Defaults: two active operations, 32 jobs, eight attempts per job, 1,000 retained events per attempt, 24-hour idle retention. Cleanup occurs on requests and expires stale prior-process directories; in-memory state does not survive app restart. Overlap is 409; capacity is 429. Active jobs cannot be deleted.

Attempts own their immutable identity, event history, and terminal result. SSE listeners replay independently; `Last-Event-ID` resumes. Status lookup returns the retained terminal event even if a connection lost completion. Routes use attempt-specific audio/inspection URLs; a failed attempt cannot serve an earlier result. Full worker lifecycle is caught, including file writes and thread-start failures. Audio subprocess timeout is 30 minutes and terminates its process group.

Validate object-shaped JSON, boolean LLM flags, string models/text, supported providers/engines, readable unencrypted PDFs, upload size, and transcript size before starting work. HTTP errors, including 413, return JSON. Frontend request handling catches status/network/parse errors, shares a busy state across conflicting controls, restores controls on terminal paths, polls status alongside SSE, and saves a running attempt in sessionStorage for reload recovery. Render received text using textContent/value, not HTML.

## Testing expectations

Run `.venv/bin/python -m pytest -q` after relevant changes. The existing phase tests and three-paper corpus remain, alongside:

- `tests/test_repairs.py`: generated PDF title/appendix/value fidelity, rejected/empty narration, purity/metadata, shared modes/cache, adversarial structured responses, chunk limits/order, audio formats/atomic failure, ordered futures/Murf failures, lazy imports, CLI validation.
- `tests/test_web.py`: upload/readability/type checks, 413, process/edit/generate/regenerate routes, replay/status, overlap/capacity, disk/subprocess failures, stale-result exclusion, retention/deletion/event bounds.

Essential generated-fixture tests must run even without `papers/`. Do not mistake corpus character counts for proof of faithful narration. Actual short Kokoro/spawn tests, clean installation, and a real-browser flow complement mocked tests. Do not run paid provider checks implicitly.
