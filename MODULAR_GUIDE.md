# Code and library guide

`main.py` and Flask's `app.py` both use `processing.process_pdf()`. `providers.py` lazily constructs only the selected provider and adapts text responses. `processing.build_config()` defines entry-point narration defaults. Library `PipelineConfig()` defaults remain unchanged for backwards compatibility.

```python
from processing import process_pdf

text, report = process_pdf("papers/example.pdf", use_llm=False)
# report contains warnings, per-stage counts, and inspectable document blocks.
```

For an explicitly configured pipeline:

```python
from pipeline import PipelineConfig, run_pipeline, serialize

config = PipelineConfig()
config.table.mode = "verbatim"
config.inline_math.mode = "symbolic"
document, stats = run_pipeline("papers/example.pdf", config=config, collect_stats=True)
text = serialize(document)
```

Stages: `extract` → `classify` → `filter_sections` → `handle_tables` → `handle_equations` → `rewrite_inline_math` → `audio_polish`. Classification and transformation stages return new documents without mutating input blocks. Layout, spans, heading levels, parent sections, and diagnostic metadata remain available in `pipeline.model` dataclasses.

`pipeline.serialize.serialize()` is the final narration eligibility boundary. It excludes noise, layout debris, and blocks with handler failures, even when a failure occurred after section filtering. `require_narration()` turns an empty transcript into an actionable error at the application boundary. Raw rejected content stays in diagnostics.

Inline LLM responses must be arrays matching detected variable identities. Surrounding prose is never returned by the model or replaced. Table LLM responses select valid data-row indices; the renderer uses only original cells. Unsupported JSON, invented prose, and bad row indices trigger a logged and per-call recorded fallback. Equation prose is model-generated and must be reviewed against its source. `pipeline.diagnostics` uses context-local storage to keep concurrent calls' warning lists separate.

`LLMCache` provides SQLite text/vision caching. `process_pdf(cache_path=...)` keys by provider, model, and the `structured-v2` prompt identity, and closes each connection after use. `main.py --inspect` and the UI diagnostics download expose stage counts and source metadata. Web caching is not enabled by default.

`pipeline.run_pipeline_vlm()` remains a separate opt-in library API for callers supplying a vision adapter. Neither entry point chooses it. It may paraphrase, and requires careful comparison with the paper. Scanned documents can instead be OCR-processed before the normal pipeline.

`text_to_speech.synthesize()` validates limits and usable text before loading a model. Chunks concatenate exactly to whitespace-normalized input; paragraph, sentence, word, and hard boundaries are used in that order. Boundary whitespace is retained to make reconstruction exact; whitespace-only chunks (possible at a limit of one character) are ignored for synthesis. Spawn workers return chunks in original order. Audio is decoded using the selected input format, exported to a temporary sibling file, validated, and atomically replaced. The CLI creates no disposable transcript, and web subprocess transcripts are removed in `finally`.

`jobs.JobManager` owns bounded jobs and attempts. A condition lock guards starts and terminal transitions. Each attempt has its own bounded event log and retained result, so SSE readers cannot consume one another's completion. `GET /status/<job>/<attempt>` retrieves the result, `GET /stream/<job>/<attempt>` replays events with `Last-Event-ID`, and audio URLs identify the producing attempt. All worker exceptions become terminal errors. Retention is opportunistic on requests; in-process jobs do not survive server restarts.

`pdf_to_text.py` is a standalone legacy cleanup tool. It is not called by the modern CLI or UI, and does not provide their provider adapter anymore.
