# Project audit and repair plan

Audited 2026-09-06 against the existing working tree, including uncommitted and untracked development. This audit changes documentation only; the defects below remain open.

## Confirmed product intent

- Personal use on the owner's computer; a hosted service is outside the current scope.
- Faithful reading of the paper's prose, with narration of math and tables.
- References and appendices should be omitted.
- Follow-up decisions pending: handling inline citations and author metadata; preferred default detail for math/table narration. These decisions affect narration policy, not whether the concrete defects below should be fixed.

## Verified baseline

| Check | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q` | 326 passed, no skips, 5 PyMuPDF/SWIG deprecation warnings; 20.78 seconds |
| `main.py --help` | Exits successfully in the current environment |
| Three supplied PDFs through the default pipeline | All complete; output sizes: Shaw 17,717 characters, Titans 75,395, ViT 34,659 |
| Spy LLM supplied to the default pipeline on each PDF | Zero LLM calls on all three |
| Full CLI, generated PDF → deterministic text → Kokoro MP3 | Exit 0; decoded duration 7,925 ms; 128,300 bytes |
| Standalone TTS, two Kokoro workers and two chunks | Exit 0; decoded duration 5,225 ms; 84,908 bytes |
| Flask test client | Home route works; validation and lifecycle failures reproduced below |
| `.venv/bin/python -m pip check` | Exit 1: `torch 2.10.0 is not supported on this platform` |

The environment uses Python 3.12.10, PyMuPDF 1.27.2.2, Kokoro 0.9.4, Flask 3.1.3, and pytest 9.0.3. FFmpeg is available. Although `espeak-ng` is absent from PATH, the tested English Kokoro path works with locally available resources. Neither that absence nor the pip warning established a runtime failure; investigate them before changing the working environment.

Audio checks ran with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, using existing model resources. Generated test artifacts were temporary. Audio decoding and duration were checked; listening quality was not assessed. No paid provider requests were made. Google/Cerebras/Murf authentication, current model/voice availability, cold installation, and full-paper audio performance remain unverified. Browser interactions were inspected in source, not exercised in a browser.

The existing suite covers pipeline modules well, but has no dedicated CLI, Flask/UI, TTS, or legacy-cleanup suites. Most integration assertions measure block counts and output sizes; these do not establish narration fidelity. Several new modules/tests and the ViT fixture are still untracked. A commit that includes only tracked changes would omit substantial working functionality.

## Findings

P1 means a core feature or text fidelity is broken. P2 means a reliability or integration defect. P3 is a secondary maintenance issue.

### F1 — P1: section filtering can delete an entire paper

`pipeline/policies.py:44,111–139`: the lettered-appendix regex also matches ordinary titles such as `A Novel Method for Learning`. A synthetic PDF containing that title, `1 Introduction`, and a body paragraph produces **empty narration**: all three extracted/classified blocks disappear in `filter_sections`. The title has the largest font and opens a skip that never closes at the smaller heading sizes.

Conversely, fixture checks show `A. Technical Details`, `A Proofs`, `A Appendix`, and `6 Supplementary Material` are retained. The documented common-prefix support is missing from some patterns. Unknown-level headings also do not always close a skip as documented.

Repair: identify the title/front matter and require document context before treating bare letter prefixes as appendices. Normalize numbered/dotted appendix and supplementary headings. Define and test the unknown-level boundary behavior. Preserve main-body sections while removing genuine appendix descendants.

### F2 — P1: the LLM controls do not enable LLM processing

`main.py:114–129`, `app.py:64–72`, `pipeline/qa.py:363–412`: both entry points construct a model, then use `PipelineConfig()` with table/equation/inline-math modes all set to `skip`. None of these defaults calls the supplied LLM. The CLI nevertheless requires an API key unless `--no-llm` is passed.

The cache and inspection utilities are library features only. `run_pipeline_vlm()` is a separate exported API and is not selectable from either entry point. Existing AGENTS documentation overstated this integration.

Repair: share a single configuration builder between CLI and web UI. Map LLM-enabled mode to explicit policies and deterministic mode to an explicit fallback policy. Build only the selected provider when required. Keep VLM opt-in; decide whether exposing it is needed after the core path is reliable. Add a shared provider adapter instead of duplicate wrappers and legacy-module imports.

### F3 — P1: polishing and equation rewriting can corrupt meaning

Observed outputs from current functions:

| Input | Actual output |
| --- | --- |
| `We used 1,000 MB of memory.` | `We used one,zero megabytes of memory.` |
| `The value is 42.` | unchanged; the terminal period prevents number matching |
| `The value is 3.14.` | unchanged for the same reason |
| `x_α = y` | `x_ alpha equals y` |
| `x²³` | `x to the 2 to the 3` |
| `𝜃 = 𝑥` | `𝜃 equals 𝑥` |

`pipeline/polish.py:376–410` parses the tail of a grouped number as the unit quantity. `pipeline/equations.py:312` expands Greek symbols before resolving script markers, handles superscript digits individually, and lacks the Unicode normalization already used by inline math. Basic ASCII `+`, `-`, and `/` also survive symbolic narration. At `pipeline/equations.py:459`, merely mentioning `two` anywhere suppresses the equation-two prefix, even when it describes two vectors.

Repair: parse complete numeric quantities before substitution, distinguish terminal punctuation from decimal points, group script runs, normalize supported mathematical glyphs, and recognize an actual equation-number prefix. Preserve ambiguity rather than confidently inventing an interpretation.

### F4 — P1 before enabling LLMs: fidelity checks accept changed claims

`pipeline/inline_math.py:246` accepts a mocked response changing `We do not claim causality` to `We do claim causality`; its output-length check cannot enforce the prompt's requirement to preserve surrounding prose. `pipeline/tables.py:354` accepts `The score is 999 percent.` for a table whose only score is `12`, because only decimal tokens are checked.

These modes are currently dormant in the entry points (F2). Enabling them without stronger validation would expose these failures to normal use.

Repair: have the model return replacements for identified math spans/regions, then splice validated replacements into immutable surrounding prose. Validate table values and their row/column associations, including integer and percentage forms; prefer rendering structured selections from the original table. Reject invalid responses, preserve a faithful fallback, and report the fallback. Deterministic validation cannot prove every natural-language explanation correct; retain the text review step and make generated passages inspectable.

### F5 — P2: rejected content reaches narration; empty output reports success

`pipeline/qa.py:323`, `pipeline/serialize.py:12`, `pipeline/vlm.py:159–165`: `safe_block` changes a failed block to `noise` but retains its text. Filtering has already run, and serialization includes every nonempty block. Consequently failed raw equations are still spoken. A VLM response `Bad output` rejected as too short is also serialized verbatim.

`app.py:74–79` reports `done` for an empty document. `text_to_speech.py:255` exports an empty list of chunks successfully (1,196-byte MP3 in the probe). There is no explicit no-extractable-text outcome for scanned/blank PDFs.

Repair: define a final narration eligibility rule, retain rejected text in diagnostics, and expose omission/fallback warnings. Reject empty usable text before initializing TTS. Report scanned/no-text PDFs with an actionable route to an explicitly selected vision path; do not silently enable paid processing.

### F6 — P2: TTS chunk limits and audio format handling are unreliable

`text_to_speech.py:66–104`: a 4,999-character normalized sentence and a 3,001-character unbroken token each remain a single chunk despite a limit of 2,800. Limits of zero and minus one are accepted. Worker counts also lack positive-value validation.

`text_to_speech.py:255–265` assumes every chunk is MP3, while `MURF_FORMAT` is configurable and is sent to the provider. A valid WAV chunk fails with `CouldntDecodeError`.

Repair: split by paragraph, sentence, word, then hard boundary where necessary; guarantee every nonempty chunk fits and normalized text is preserved in order. Validate limits before model initialization. Carry input audio format explicitly or standardize it, and atomically publish only a validated final MP3. Use `finally` for temporary-file cleanup; `main.py` currently cleans up only on success.

### F7 — P2: web jobs can overlap, lose progress, or remain stuck

`app.py:89,145–211`: repeated requests start overlapping workers for the same job and replace the shared event queue. Threads continue writing through the mutable job dictionary; competing SSE listeners consume the same queue. Job status is never changed to a terminal state. A simulated disk-full error in `text_path.write_text`, outside the `try`, escapes the worker with no error event.

Validation probes: a filename ending in `.pdf` accepts non-PDF bytes; `{"text":7}` and `{"llm_model":7}` cause HTTP 500; an invalid TTS engine returns HTTP 200 and starts work; the string `"false"` enables LLM processing. There is no job/file expiry.

Repair: validate request shapes, types, options, and PDF readability. Give each processing attempt its own identity, immutable event channel, and terminal result. Lock transitions and reject overlapping work for a job. Catch the entire worker lifecycle. Add status lookup/reconnect support and bounded local concurrency and retention. An in-process job manager is sufficient for this personal-local scope; Redis, authentication, and a hosted queue are not prerequisites.

### F8 — P2: browser error paths leave controls disabled

`templates/index.html:453–567`: processing/audio-start fetches do not check HTTP status or catch failures. Upload errors assume a JSON body, but Flask's default 413 response is HTML. Upload network errors and SSE disconnects do not restore disabled controls. `Regenerate` remains enabled during generation, making F7 reachable through the UI.

Repair: centralize request/error handling and operation state, restore controls on all terminal paths, disable all conflicting actions during work, and reconnect or retrieve the terminal result. Use attempt-specific audio URLs to avoid stale playback after regeneration. Verify failures in a real browser as well as at route level.

### F9 — P2: installation guidance and optional dependencies disagree with code

`text_to_speech.py:18` eagerly imports Kokoro, so a Murf-only CLI still requires the local ML stack. `pdf_to_text.py` eagerly imports both provider integrations and is imported even for deterministic processing. `requirements.txt` labels Kokoro optional but installs it unconditionally; dependencies have open-ended lower bounds, and pytest is not declared for development setup.

`check_setup.py` always requires Google and Murf keys while omitting several dependencies of the active pipeline. README/SETUP/MODULAR_GUIDE describe the old architecture and conflicting model defaults; README claims Python 3.8 compatibility despite newer syntax in the source. None documents the complete current web workflow.

Repair: lazy-load engine/provider dependencies, declare a tested Python range and separate development/optional dependency sets, and add a reproducible constraints/lock strategy. Make setup checks mode-aware. Update public usage docs from actual behavior, and verify provider model/voice IDs against official sources when implementing provider smoke tests.

### Secondary issues and known limits

- `classify_blocks()` mutates its input despite the advertised pure-stage contract. Either make classification pure (preferred for comparison workflows) or explicitly retain/document mutation; do not assume input snapshots remain valid.
- `_encode_meta({1, "two"})` raises `TypeError` because mixed encoded set elements are sorted directly. Inspection dumps should tolerate supported mixed metadata.
- Default output still reads author affiliations/emails and inline citations; the user's desired policy is pending. The Shaw sample begins with an author email and retains author-year citations.
- Unruled tables and figure-internal labels remain known extraction limitations. Table captions can be narrated both inside a table replacement and as a separate caption block. Improve recall only with positive/negative layout fixtures, rather than relaxing global heuristics to satisfy output counts.
- Existing corpus tests can pass while pinning undesirable narration. Do not refresh their baselines merely to make tests green.

## Implementation sequence and acceptance criteria

Each milestone should be a reviewable change with targeted regression tests. Preserve the current uncommitted work, and deliberately include the existing untracked source/test files when preparing commits.

1. **Prevent text deletion and value corruption — F1, F3, F5.** Add small generated-PDF fixtures and text examples reproducing the confirmed failures. Fix contextual section filtering, quantity/script handling, and final narration eligibility. Acceptance: an `A …` title and its main body survive; genuine appendix variants disappear; `1,000 MB` retains its value; failed/empty content never masquerades as successful narration. Re-run the three-paper corpus and review changed excerpts before updating numeric baselines.
2. **Make LLM mode useful and faithful — F2, F4.** Resolve the remaining narration preferences, implement shared policy/provider configuration, and add constrained replacement validation before activating LLM policies. Acceptance: spy providers show expected calls only in selected modes; deterministic processing needs no provider imports/keys; surrounding prose and numeric facts survive adversarial responses; errors yield visible faithful fallbacks. Add optional cache/stats wiring with provider/model/prompt identity. Leave vision library-only unless explicitly exposing it is part of this milestone.
3. **Make audio boundaries dependable — F6 and TTS portions of F9.** Add chunk-preservation/limit tests, input validation, explicit format handling, ordered-worker tests, and cleanup/output failure tests. Acceptance: all chunks obey the limit with no lost/reordered text; MP3 and any supported alternate input format decode; failed/empty runs do not replace a valid output. Re-run a short actual Kokoro CLI conversion and spawn-worker smoke test. Mock Murf HTTP failures in CI; perform a small live provider check separately when needed.
4. **Repair the local web workflow — F7, F8.** Introduce a small job/attempt abstraction and consistent validation and terminal state. Update frontend state transitions and retry handling. Acceptance: upload → process → edit → generate → play/download → regenerate works; overlapping submissions are rejected; malformed payloads produce 4xx responses; disk/subprocess/network errors reach a terminal state; reconnect cannot steal or lose completion; a failed attempt does not serve stale audio as its result. Test routes without real providers and exercise the complete browser flow with a short local sample.
5. **Make setup and maintenance repeatable — F9 and secondary issues.** Correct setup checks and docs, declare development dependencies and supported runtimes, and add CI for unit/route/CLI tests plus deterministic PDF fixtures. Fix mutation/metadata issues with focused tests. Acceptance: clean deterministic and selected-TTS installs pass their appropriate checks; the documentation's commands and defaults match behavior; essential integration checks do not silently disappear when optional sample PDFs are absent.

Broader extraction recall improvements and an exposed VLM workflow follow these milestones. For personal local use, prioritize faithful output, recoverable failures, and a usable transcript review step over hosted-service infrastructure.

## Reproduction starting points

Run from the project root with the existing virtual environment:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python main.py --help
.venv/bin/python -m pip check
```

The key deterministic failures can be checked without a provider key:

```python
from pipeline import Block, Document, FontStats, filter_sections
from pipeline.polish import PolishPolicy, _PolishState, _polish_text
from text_to_speech import split_text

title = Block("heading", [], "A Novel Method for Learning", 0,
              (0, 0, 100, 20), level=1)
body = Block("body", [], "Main text that should survive.", 0,
             (0, 30, 100, 50))
print(filter_sections(Document([title, body], FontStats(10))).blocks)
# Current: []
print(_polish_text("We used 1,000 MB of memory.", PolishPolicy(), _PolishState()))
# Current: We used one,zero megabytes of memory.
print([len(chunk) for chunk in split_text("word " * 1000, 2800)])
# Current: [4999]
```
