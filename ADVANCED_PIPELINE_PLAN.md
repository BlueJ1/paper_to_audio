# Advanced Preprocessing Pipeline — Design Plan

A plan for rebuilding `pdf_to_text.py` so that academic PDFs produce audio
output that is actually *listenable*: faithful prose, comprehensible math,
narrated tables, and no layout debris.

This is a design document, not a diff. It is written to be implemented
incrementally, phase by phase, with each phase shipping real improvements.

---

## 1. Diagnosis: why the current pipeline fails

The current `clean_for_tts` operates on a flat string produced by
`pymupdf.get_text("text")`. That representation has already discarded
layout, font, and structural information. Every bug we have debugged so far
traces back to this lossy starting point:

| Failure mode                                          | Root cause                                          | Where observed                       |
|-------------------------------------------------------|-----------------------------------------------------|--------------------------------------|
| Tables extracted as column-wise orphan lines          | No cell/column awareness                            | Tables 1–3 in relative-position paper |
| Subscripts lost (`h_t` → `ht`)                        | PDF extraction flattens position info               | Line 38, `ht 1`                      |
| Figure captions merged with graphic debris            | No caption/figure distinction                       | Diagram removal heuristic            |
| Two-column layouts interleave across columns          | No column detection                                 | (papers not yet tested)              |
| References section partially read                     | Skip logic relies on exact string                   | Varies by paper                      |
| Display equations mangled before reaching the LLM     | Equations treated as prose                          | Figure 1 caption, equations 3–4      |
| LLM narrates truncated table as "math"                | Math detector fires on numeric residue              | Line 264 table narration             |
| "Available at …" sentence fragments                   | URL regex has no sentence context                   | Line 190 (fixed)                     |
| `Veličković et al.` citation leaks                    | Citation regex was ASCII-only                       | Line 147 (fixed)                     |
| Math paragraph not flagged; non-math paragraph flagged | Heuristic can't see font / italic / bbox           | LLM false positives and misses       |

The unifying observation is that every failure is trying to recover
information that *still exists in the PDF* but is thrown away before our
pipeline sees it.

---

## 2. Architectural shift

**From:** flat string + layered regex + LLM rewriting of math-heavy paragraphs.
**To:** typed document model + routed per-block processing.

Three principles drive the rebuild:

1. **Extract with layout.** Keep font, bbox, page, and column information
   from extraction all the way to serialization.
2. **Classify, then route.** Every block has a `kind`. Each kind has a
   dedicated handler (prose → prose; table → narrated prose; equation →
   LaTeX-or-symbolic → spoken prose; figure → caption only).
3. **Defer to vision only where symbolic fails.** Deterministic extraction
   stays the default. A vision LLM is a fallback for equations, irregular
   tables, and pathologically-extracted pages — never the first call.

---

## 3. Document model

```python
from dataclasses import dataclass, field
from typing import Literal

BlockKind = Literal[
    "heading", "body", "caption",
    "equation_display", "equation_inline",
    "table", "figure", "footnote", "code",
    "page_header", "page_footer", "toc", "noise",
]

@dataclass
class Span:
    text: str
    font: str
    size: float
    flags: int                                # bold/italic/mono (pymupdf flags)
    bbox: tuple[float, float, float, float]

@dataclass
class Block:
    kind: BlockKind
    spans: list[Span]
    text: str                                 # flattened span text
    page: int
    bbox: tuple[float, float, float, float]
    level: int | None = None                  # heading depth or table index
    parent_section: str | None = None         # e.g. "3. Proposed Architecture"
    meta: dict = field(default_factory=dict)  # type-specific payload

@dataclass
class FontStats:
    body_size: float                          # modal size
    heading_thresholds: list[float]           # sorted cutoffs above body

@dataclass
class Document:
    blocks: list[Block]
    fonts: FontStats
    language: str
    title: str | None = None
    abstract: str | None = None
    source_path: str | None = None
```

Every stage is a pure function `Document → Document`. The final audio text
is produced by one `serialize(doc)` walk that visits blocks in reading
order and dispatches on `kind`.

---

## 4. Pipeline stages

```
load_pdf(path)
  → extract_layout(doc)            # Phase 1 — pymupdf dict-mode, columns, paragraphs
  → classify_blocks(doc)           # Phase 2 — kind tags + heading tree
  → filter_sections(doc, policy)   # Phase 3 — drop References, Appendix, …
  → handle_tables(doc)             # Phase 4 — pdfplumber cells → prose
  → handle_equations(doc)          # Phase 5 — display equations → prose
  → rewrite_inline_math(doc)       # Phase 6 — span-marked math → prose
  → audio_polish(doc)              # Phase 7 — numbers, acronyms, units
  → serialize(doc)                 # Document → final plain text / SSML
```

Each stage is independently testable. Intermediate `Document` objects
serialize to JSON for checkpointing and inspection.

---

## 5. Phase 1 — layout-aware extraction

Replace `page.get_text("text")` with `page.get_text("dict")`. That returns
blocks → lines → spans with bbox and font metadata:

```python
{"blocks": [
  {"lines": [{"spans": [{"text": "...", "font": "CMR10", "size": 9.9,
                         "flags": 20, "bbox": [x0, y0, x1, y1]}]}]},
  ...]}
```

Work:

- **Body-font detection.** Histogram span sizes weighted by text length;
  the modal size is body. All other sizes are candidates for headings,
  footnotes, captions, or super/subscripts.
- **Column detection.** K-means on span x-midpoints. If two well-separated
  clusters form, the page is two-column; assemble reading order
  accordingly. Handle mixed-layout pages (some single-column, some
  two-column, common on abstract pages).
- **Reading order.** Sort blocks within a column top-to-bottom, then
  columns left-to-right. Pages are concatenated in order.
- **Paragraph reconstruction.** Merge consecutive lines when (a) vertical
  gap ≈ line height, (b) font matches, (c) no initial indent mismatch.
  Insert paragraph breaks otherwise. This eliminates the "each sentence on
  its own line" artifact from the current flat extraction.
- **Hyphenation rejoin.** Already works at text level
  (`_rejoin_hyphenated_linebreaks`); move it to span-level so we can use
  font continuity as an additional signal.
- **NFC normalization.** Per span, before rejoin, so decomposed accents
  like "c" + combining caron become "č".

Libraries: `pymupdf` (primary), `pdfplumber` (secondary, for tables),
`pymupdf4llm` (optional, for a quick-win markdown export that already
seeds basic heading and list detection).

---

## 6. Phase 2 — block classification

Each block gets a `kind`. Rules fire first; an LLM classifier is a
fallback only when rules produce low confidence.

Rule set:

- `heading` — font size ≥ body × 1.15 **and** (matches `^\d+(\.\d+)*\s`, is all-caps, or ≤ 10 words).
- `caption` — first line matches `^(Figure|Table|Algorithm)\s+\d+[.:]`.
- `code` — dominant span font is monospace (`Courier`, `Consolas`, `Inconsolata`, `Mono` substrings).
- `footnote` — font size < body × 0.85 **and** block is in the bottom 20% of the page.
- `page_header` / `page_footer` — same text appears at the top / bottom of ≥ 3 pages.
- `equation_display` — block is horizontally centered, has high math-glyph density, often flanked by `(N)` on the right.
- `table` — block's bbox overlaps a bounding box returned by `pdfplumber.page.find_tables()`.
- `figure` — block has no text but has image resources per `pymupdf.get_images()`, or is adjacent to a `caption` block with "Figure".
- `toc` — dense run of `\.\s{2,}\d+$` patterns (dot-leaders).
- `noise` — fewer than ~4 alphabetic words and not matched by any rule above.
- Otherwise → `body`.

Once blocks are classified, walk them in order to build the section tree:
every `heading` opens a new section; every subsequent non-heading block
inherits `parent_section`. Heading `level` is computed from rank-order of
font size observed in the document (not assumed).

Output of this phase: a `Document` where every block has `kind` and
`parent_section` set, and the `fonts` field is populated.

---

## 7. Phase 3 — section filtering and skip policies

Skip logic becomes a one-liner:

```python
doc.blocks = [
    b for b in doc.blocks
    if policy.keep_block(b)
]
```

Default policy:

- **Skip sections matching**: `References`, `Bibliography`, `Acknowledgments?`, `Author Contributions`, `Supplementary.*`, `Appendix [A-Z]\b.*`.
- **Skip block kinds**: `page_header`, `page_footer`, `footnote`, `toc`, `noise`.
- **Keep captions, drop figure contents.**
- **Code**: configurable (default skip; can read in full for CS tutorials).
- **Appendix**: configurable. Default skip; `--include-appendix` overrides.

Policy lives in `pipeline/policies.py` as a dataclass with CLI and YAML
surfaces. Policy changes do **not** require re-extraction — Phase 3 runs
against a cached Phase-2 `Document`.

---

## 8. Phase 4 — tables as prose

Use `pdfplumber.Page.extract_tables()` with a conservative settings profile
(ruled lines OR aligned-text strategy). For each detected table:

1. Build a `Table` payload: list of rows, header row, table index, caption.
2. Attach it to the corresponding `table` block's `meta`.
3. At serialization time, render per `--table-mode`:
   - `prose` (default) — send to an LLM with this prompt skeleton:

     > *You are converting a table into spoken narration for an audiobook.
     > The table title is "{caption}". Columns are {columns}. Rows follow
     > in CSV form. Produce a 2–4 sentence spoken summary that names each
     > column and reads at most three illustrative rows. Never invent data.*

   - `skip` — emit: *"A table with N columns and M rows appears here; see
     the paper."*
   - `verbatim` — read cell by cell with column headers prepended.

Guardrails: LLM output sentence count is bounded by table size; output is
validated not to introduce tokens absent from the CSV input (simple
vocab-containment check).

If `pdfplumber` detects no table in a region but Phase 2 classified the
block as `table`, fall back to `skip` mode with a warning — we do not want
the current silent data loss.

---

## 9. Phase 5 — display equations

Detected by Phase 2 rules. Handler cascades through three tiers:

1. **LaTeX reconstruction (preferred).**
   Pull the underlying text stream with `page.get_text("rawdict")`; look
   for sequences resembling LaTeX (`\frac`, `\sum`, `\alpha`, subscript
   tokens). When found, feed the LaTeX snippet to a dedicated prompt:

   > *Convert this LaTeX expression to one or two sentences of spoken
   > English suitable for audiobook narration. Do not include LaTeX
   > syntax in the output.*

   LaTeX → English produces much better results than prose → prose
   because the structure is unambiguous.

2. **Symbolic fallback.**
   The current unicode-math maps plus Greek substitutions, then an LLM
   rewrite of the whole paragraph. This is the current pipeline,
   retained for cases where no LaTeX structure is recoverable.

3. **Vision fallback (flag-gated).**
   Render the equation bbox to PNG via `page.get_pixmap(clip=bbox)`.
   Send to the configured vision model (Gemini 3 Flash / Claude 4.6) with:

   > *Describe this mathematical expression in one or two spoken English
   > sentences suitable for text-to-speech. Do not include LaTeX or
   > mathematical notation.*

   Off by default; opt-in via `--equations=vlm` or per-equation when the
   previous two tiers produce length-ratio failures.

Equation numbers (`(3.4)`) are preserved in narration: *"Equation three
point four says …"*.

---

## 10. Phase 6 — inline math rewriting

Rather than flagging entire paragraphs as math-heavy, identify **spans**:

- A span with the italic flag set (`span.flags & 2`) and matching
  `^[A-Za-z]$` → variable.
- Adjacent variable-spans plus math-symbol spans → inline math expression.
- Expressions are marked in the paragraph text as `⟦…⟧` delimiters.

Then batch by section and send to the LLM:

> *The following paragraph contains math expressions delimited by ⟦ and ⟧.
> Rewrite only the delimited spans as spoken English. Leave surrounding
> prose exactly as written. Preserve paragraph boundaries. Output the
> full paragraph.*

Benefits over the current paragraph-level heuristic:

- **Higher precision** — we know what is math because we looked at font flags, not patterns.
- **Fewer LLM calls** — one call per section instead of per flagged paragraph; ~5–10× fewer calls on a typical paper.
- **No false positives** — paragraphs without inline math never reach the LLM.
- **Lower cost** — combined with caching (Phase 8), iteration is nearly free.

Fallback: papers without italic-font variable typography (rare but
exists) revert to the current heuristic detector with a confidence
warning logged.

---

## 11. Phase 7 — audio polish

Block-level transforms focused on pronunciation:

- **Number normalization** via `num2words`: `27.3` → `twenty-seven point three`, `10^-9` (already preprocessed to `ten to the negative nine`) passes through.
- **Abbreviation expansion**: `Fig.` → `Figure`, `e.g.` → `for example`, `i.e.` → `that is`, `cf.` → `compare`, `vs.` → `versus`.
- **Acronym handling**: on first occurrence, narrate the parenthetical expansion the author provided (`LSTM (Long Short-Term Memory)` → `L S T M, long short-term memory`). Track seen acronyms per document.
- **Pronunciation overrides**: small lookup table for common ML / math tokens (`ReLU` → `relu`, `GPT` → `G P T`, `LaTeX` → `lay-tek`). User-editable dict.
- **Units**: `100 GB` → `one hundred gigabytes`, `1.5 GHz` → `one point five gigahertz`.
- **Quote normalization**: straight / curly / nested quotes flattened.
- **Optional SSML pass**: `<break strength="strong"/>` between sections,
  `<sub alias="…">` for awkward tokens. Opt-in via `--ssml`; only emitted
  for TTS engines that accept SSML.

---

## 12. Phase 8 — caching, QA, observability

- **Persistent LLM cache.**
  SQLite cache keyed by `sha256(paragraph, prompt_version, model)`. On
  repeat runs (prompt iteration, adding a new section to the paper, etc.),
  completed paragraphs are free. Invalidation is explicit via prompt
  version bump.
- **Per-stage diff metrics.**
  Already started in `clean_for_tts` logging; extend with Block-kind
  distributions: *"3 tables, 2 display equations, 47 body blocks, 4
  headings, 19 footnotes skipped"*.
- **Dry-run inspect mode.**
  `--inspect` dumps the `Document` as JSON and opens an HTML viewer that
  overlays block kinds on the rendered PDF pages. Indispensable for
  debugging new papers.
- **Regression corpus.**
  Pin 5–10 representative PDFs (one- and two-column, math-heavy,
  table-heavy, code-heavy). For each, snapshot expected block-kind
  distribution and output character length ± 5%. Run in CI.
- **Fail-soft per block.**
  Any handler that raises converts its block to `kind="noise"` with the
  original text preserved in `meta["raw"]`. Pipeline never crashes on a
  single weird block.

---

## 13. Phase 9 — optional vision-first mode

A single `--vlm` flag swaps Phases 1–6 for a direct vision LLM pass:

- Render each page to PNG at 200 DPI.
- Prompt Gemini 3 Flash (cheap) or Claude 4.6 with:

  > *Produce clean audiobook narration for this PDF page. Include all
  > prose verbatim. Describe equations and tables in one or two spoken
  > sentences each. Skip figures but include their captions. Skip page
  > numbers, running headers, footers, and footnote reference marks.*

- Concatenate pages; run Phase 7 polish; run Phase 8 QA.

This is lossier for fidelity (the model may paraphrase) and costs one
VLM call per page, but it bypasses every extraction problem in one shot.
It is also useful as a *reference output* to diff the symbolic pipeline
against — when the two diverge sharply on a page, that page is a
candidate for vision fallback on individual blocks.

Cost back-of-envelope: 20-page paper × ~2k image tokens × Gemini Flash
pricing ≈ $0.004 per paper. Cheap enough for routine use; not cheap
enough to waste on trivial cases.

---

## 14. Module layout

```
pipeline/
  __init__.py
  model.py           # dataclasses (Span, Block, Document, FontStats)
  extract.py         # Phase 1
  classify.py        # Phase 2
  sections.py        # Phase 2b — heading tree + parent_section
  policies.py        # Phase 3 — skip rules, config dataclass
  tables.py          # Phase 4
  equations.py       # Phase 5
  inline_math.py     # Phase 6
  polish.py          # Phase 7
  qa.py              # Phase 8 — cache, inspect mode, regressions
  vlm.py             # Phase 9 (optional)
  serialize.py       # Document → string / SSML
  llm.py             # build_llm, retry, cache (reused from pdf_to_text.py)
```

`pdf_to_text.py` becomes a thin orchestrator:

```python
def pdf_to_audio_text(path, policy, llm_config):
    doc = extract_layout(load_pdf(path))
    doc = classify_blocks(doc)
    doc = filter_sections(doc, policy)
    doc = handle_tables(doc, llm_config)
    doc = handle_equations(doc, llm_config)
    doc = rewrite_inline_math(doc, llm_config)
    doc = audio_polish(doc)
    return serialize(doc)
```

---

## 15. Rollout — smallest shippable increments

Nothing here is a big-bang migration. Each increment keeps the CLI and UI
working and produces visible improvement:

| Inc | Phases | Smallest shippable unit                                                    | Expected gain                                 |
|-----|--------|----------------------------------------------------------------------------|-----------------------------------------------|
| 1   | 1      | `Document` dataclass, dict-mode extraction, column detection, paragraph rejoin | Fewer orphan sentences; two-column papers work |
| 2   | 2–3    | Block classification + heading-based skip policy                           | Reliable References skip; footnotes gone      |
| 3   | 4      | pdfplumber table extraction + prose mode                                   | Tables stop disappearing                      |
| 4   | 5–6    | Display-equation handler + inline-math span detection                      | Math finally readable; fewer LLM calls        |
| 5   | 7      | num2words, abbreviation/acronym expansion                                  | More natural narration                        |
| 6   | 8      | LLM cache + regression corpus + inspect mode                               | Iteration speed; no silent regressions        |
| 7   | 9      | `--vlm` mode (opt-in)                                                      | Safety net for pathological PDFs              |

Checkpoint the `Document` as JSON after each phase so later phases can be
re-run without re-paying for extraction or LLM rewriting.

---

## 16. Open questions

- **pdfplumber vs pymupdf for tables.** pdfplumber is slower but detects
  more academic tables correctly. Benchmark on the regression corpus
  before committing.
- **Inline italic-math detection reliability.** Not every paper uses
  italic font for variables (some use roman + manual subscript layout).
  Needs a per-document confidence check; fall back to heuristic detector.
- **VLM pricing variance.** Gemini 3 Flash is cheap today; pricing may
  shift. Keep VLM behind a flag to avoid runaway costs.
- **Section-skip matching.** Regex is transparent but fragile. An
  embedding-based classifier is robust but adds a dependency. Ship regex
  first; revisit after the regression corpus grows.
- **Config surface.** Recommendation: YAML file with CLI overrides. Fits
  the modular architecture and keeps runs reproducible across machines.

---

## 17. Non-goals

- **OCR'd PDFs.** Scanned papers are out of scope for the symbolic
  pipeline. The VLM mode handles them when needed.
- **Full-fidelity math.** We aim for *listenable*, not *reconstructable*.
  If you need LaTeX, read the paper.
- **Multi-language support.** English only. Adding languages means
  configurable skip lists, `num2words` locales, per-language acronym
  tables, and a language detector.
- **Interactive editing.** The Flask UI already has a post-processing
  edit step; that is the final human gate and stays unchanged.

---

## 18. Testing strategy

- **Unit tests per module.** Pure transforms on `Document` are trivial to
  test with fixture documents.
- **Snapshot tests per phase.** Run each phase on the regression corpus
  and diff the output against a committed snapshot. Snapshot refresh is
  a deliberate, reviewed action.
- **End-to-end integration tests.** Full pipeline on 2–3 of the corpus
  papers, asserting character-length and block-kind distributions are
  within tolerance.
- **Property tests.** Random paragraph shuffles should produce proportional
  LLM cost; block classification is idempotent; serialization is the
  inverse of parse for prose blocks.

---

## 19. Risk register

| Risk                                                   | Likelihood | Mitigation                                            |
|--------------------------------------------------------|------------|-------------------------------------------------------|
| Rewrite loses quality we already have                  | Medium     | Regression corpus; keep old pipeline behind `--legacy` during rollout |
| pdfplumber fails on oddly-ruled tables                 | Medium     | Fall back to `skip` mode with a warning; never silent data loss |
| LLM inline-math prompt drifts between providers        | Medium     | Per-provider prompt variants; cache keyed on prompt version |
| Vision mode gets expensive                             | Low        | Flag-gated; cache per-page outputs                    |
| Classification rules over-fit to a handful of papers   | Medium     | Grow the regression corpus; add LLM fallback classifier |
| Section-skip regex matches a legitimate section        | Low        | Log every skip; `--no-skip` escape hatch              |

---

## 20. Summary

Today's pipeline is a series of regex passes over a string that has
already lost most of the information we need. The proposed pipeline
keeps that information — font, layout, structure — and uses it to route
each block to the right handler. The LLM is called less often, on
smaller units, with better context, and with cached results.

The plan is sequenced so Increment 1 alone is shippable and delivers
visible improvement. Every later increment builds on the same
`Document` substrate, so there is never a grand migration — just a
steady widening of what the pipeline understands about the PDF in front
of it.
