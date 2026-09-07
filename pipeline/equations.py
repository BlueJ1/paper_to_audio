"""Phase 5 — display equation narration.

After classification, every equation_display block still carries the raw
glyph stream pymupdf pulled from the PDF — Greek letters, sub/superscripts
that the dict-mode reader flattened, operator symbols from the math block
of Unicode. That is unspeakable. This stage rewrites each equation block
into one of three narration forms:

- `skip` (no LLM needed): symbolic form only, prefixed with the equation
  number when one is present ("Equation three point four: x sub t equals
  f of x sub t minus one."). Always safe, always deterministic.
- `symbolic` (LLM): feed the spelled-out symbolic form to an LLM for a
  bounded spoken English description. Falls back to `skip` when the LLM
  is missing, raises, or returns empty output.
- `vision` (VLM, opt-in): render the equation bbox to a PNG via pymupdf
  and send it to a vision model. Falls back to `symbolic` then `skip` if
  the vision call isn't available or fails.

The stage is a pure function Document -> Document. Blocks whose `kind` is
not `equation_display` pass through untouched. Equation numbers are
extracted before narration and re-prepended as "Equation N" so the LLM
can't forget them.
"""
from __future__ import annotations

import re
import unicodedata
from pipeline.diagnostics import fallback
from dataclasses import dataclass, field, replace
from typing import Callable, Literal

from pipeline.model import BBox, Block, Document, Span


EquationMode = Literal["skip", "symbolic", "vision"]

LLMCallable = Callable[[str], str]
# Vision callable: (png_bytes, prompt) -> narration text.
VisionLLMCallable = Callable[[bytes, str], str]


# ---------------------------------------------------------------------------
# Equation number extraction
# ---------------------------------------------------------------------------

# Matches "(3)", "(3.4)", "(12.7)" at the very end of the text. The same tight
# bound as classify._EQUATION_NUMBER_RE so citation years can't sneak in.
_EQUATION_NUMBER_RE = re.compile(r"\(\s*(\d{1,3}(?:\.\d{1,3})?)\s*\)\s*$")

_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def _extract_equation_number(text: str) -> tuple[str, str | None]:
    m = _EQUATION_NUMBER_RE.search(text)
    if not m:
        return text, None
    return text[: m.start()].rstrip(), m.group(1)


def _spoken_number(number: str) -> str:
    """`3.4` -> `three point four`; `12` -> `twelve` isn't needed — digit-by-digit is fine."""
    parts = number.split(".")
    if not all(p.isdigit() for p in parts):
        return number
    left = " ".join(_DIGIT_WORDS[c] for c in parts[0]) if parts[0] else ""
    if len(parts) == 1:
        return left
    right = " ".join(_DIGIT_WORDS[c] for c in parts[1])
    return f"{left} point {right}"


# ---------------------------------------------------------------------------
# Symbolic rewrite maps — mirrors the legacy pdf_to_text.py maps but kept
# local so the pipeline package has no hidden cross-dependency.
# ---------------------------------------------------------------------------

_GREEK_MAP = {
    "\u0391": "Alpha", "\u0392": "Beta", "\u0393": "Gamma", "\u0394": "Delta",
    "\u0395": "Epsilon", "\u0396": "Zeta", "\u0397": "Eta", "\u0398": "Theta",
    "\u0399": "Iota", "\u039A": "Kappa", "\u039B": "Lambda", "\u039C": "Mu",
    "\u039D": "Nu", "\u039E": "Xi", "\u039F": "Omicron", "\u03A0": "Pi",
    "\u03A1": "Rho", "\u03A3": "Sigma", "\u03A4": "Tau", "\u03A5": "Upsilon",
    "\u03A6": "Phi", "\u03A7": "Chi", "\u03A8": "Psi", "\u03A9": "Omega",
    "\u03B1": "alpha", "\u03B2": "beta", "\u03B3": "gamma", "\u03B4": "delta",
    "\u03B5": "epsilon", "\u03B6": "zeta", "\u03B7": "eta", "\u03B8": "theta",
    "\u03B9": "iota", "\u03BA": "kappa", "\u03BB": "lambda", "\u03BC": "mu",
    "\u03BD": "nu", "\u03BE": "xi", "\u03BF": "omicron", "\u03C0": "pi",
    "\u03C1": "rho", "\u03C2": "sigma", "\u03C3": "sigma", "\u03C4": "tau",
    "\u03C5": "upsilon", "\u03C6": "phi", "\u03C7": "chi", "\u03C8": "psi",
    "\u03C9": "omega",
    "\u03F5": "epsilon", "\u03D5": "phi",
}

_UNICODE_MATH_MAP = {
    "\u2264": " less than or equal to ",
    "\u2265": " greater than or equal to ",
    "\u2260": " not equal to ",
    "\u2248": " approximately ",
    "\u2192": " leads to ",
    "\u2190": " from ",
    "\u21D2": " implies ",
    "\u21D0": " is implied by ",
    "\u2194": " if and only if ",
    "\u2208": " in ",
    "\u2209": " not in ",
    "\u2282": " is a subset of ",
    "\u2283": " is a superset of ",
    "\u222A": " union ",
    "\u2229": " intersection ",
    "\u2200": " for all ",
    "\u2203": " there exists ",
    "\u221E": " infinity ",
    "\u2202": " partial ",
    "\u2211": " sum of ",
    "\u220F": " product of ",
    "\u222B": " integral of ",
    "\u00D7": " times ",
    "\u00B1": " plus or minus ",
    "\u221D": " is proportional to ",
    "\u2207": " nabla ",
    "\u2218": " composed with ",
    "\u2297": " tensor product ",
    "\u2295": " direct sum ",
    "\u22C5": " times ",
    "\u2261": " is identical to ",
    "\u221A": " square root of ",
    "\u2212": " minus ",
}

# Superscript / subscript Unicode digit maps — pymupdf sometimes emits these
# as literal \u207X / \u208X code points even when the glyph is rendered as
# a smaller normal digit.
_SUPERSCRIPT_MAP = {
    "\u2070": "0", "\u00B9": "1", "\u00B2": "2", "\u00B3": "3", "\u2074": "4",
    "\u2075": "5", "\u2076": "6", "\u2077": "7", "\u2078": "8", "\u2079": "9",
    "\u207A": "+", "\u207B": "-",
}
_SUBSCRIPT_MAP = {
    "\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3", "\u2084": "4",
    "\u2085": "5", "\u2086": "6", "\u2087": "7", "\u2088": "8", "\u2089": "9",
    "\u208A": "+", "\u208B": "-",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EquationData:
    """Structured equation payload attached to the promoted block's `meta`."""
    raw_text: str           # as it arrived from Phase 1 (minus trailing eq-number)
    symbolic: str           # after Greek / math / sub-super substitution
    number: str | None      # "3.4", "12", ... or None
    page: int
    bbox: BBox


@dataclass
class EquationPolicy:
    """Knobs for Phase 5.

    `mode` selects the primary narration strategy. `symbolic` and `vision`
    both fall back to `skip` on failure, so the pipeline never crashes on a
    single weird equation block.
    """
    mode: EquationMode = "skip"
    # Below this char count the block is almost certainly a stray fragment;
    # use the skip renderer and don't waste LLM tokens on it.
    min_chars: int = 3
    # Equations longer than this are likely misclassified multi-line content.
    # Fall back to skip rather than asking the LLM to narrate 2 KB of symbols.
    max_chars: int = 600
    # Reject LLM narration this much shorter than the symbolic form as likely
    # truncation; fall back to skip. Ratio of narration length / symbolic length.
    min_narration_ratio: float = 0.3


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def handle_equations(
    doc: Document,
    policy: EquationPolicy | None = None,
    llm: LLMCallable | None = None,
    vision_llm: VisionLLMCallable | None = None,
) -> Document:
    """Rewrite every `equation_display` block's `text` into spoken form.

    Pure function: returns a new Document via `dataclasses.replace`. Non-
    equation blocks pass through unchanged. Each replaced block carries an
    `EquationData` payload in `meta["equation"]` so downstream consumers
    (or the Flask editor) can see the original form.

    Per-block fail-soft: if span reconstruction or symbolic rewrite raises
    on a pathological block, `safe_block` downgrades that one block to
    `kind="noise"` (preserving original text in `meta["raw"]`) rather than
    crashing the whole pipeline. The vast majority of bad inputs are
    already handled inside `render_equation` via its skip fallback; this
    catches the rare case where `_build_equation_data` itself raises.
    """
    from pipeline.qa import safe_block  # local import to avoid cycle

    policy = policy or EquationPolicy()

    def _process(b: Block) -> Block:
        data = _build_equation_data(b)
        text = render_equation(
            data, policy, llm=llm, vision_llm=vision_llm,
            source_path=doc.source_path,
        )
        return replace(b, text=text, meta={**b.meta, "equation": data})

    new_blocks: list[Block] = []
    for b in doc.blocks:
        if b.kind != "equation_display":
            new_blocks.append(b)
            continue
        new_blocks.append(safe_block(_process, b))
    return replace(doc, blocks=new_blocks)


# ---------------------------------------------------------------------------
# Extraction / symbolic rewrite
# ---------------------------------------------------------------------------

def _build_equation_data(b: Block) -> EquationData:
    """Produce the `EquationData` payload for a classified equation block."""
    raw = b.text.strip()
    body, number = _extract_equation_number(raw)
    # Prefer span-aware script reconstruction if the spans carry super/sub
    # information we can use; fall back to the already-flattened `text`.
    script_text = _reconstruct_with_scripts(b, number) or body
    symbolic = _symbolic_rewrite(script_text)
    return EquationData(
        raw_text=body,
        symbolic=symbolic,
        number=number,
        page=b.page,
        bbox=b.bbox,
    )


def _reconstruct_with_scripts(b: Block, number: str | None) -> str:
    """Rebuild the equation text using span flags / y-offset for scripts.

    pymupdf sets bit 0 of `flags` for superscripts (reliable). Subscripts
    are NOT flagged directly — they're just smaller spans that sit below
    the baseline. We detect them by comparing span size to the character-
    weighted median size and by y-center relative to the line's median.

    Emits `^token` for superscripts and `_token` for subscripts. `_symbolic_rewrite`
    converts those into "to the N" / "sub N" phrases.

    Returns "" when there's nothing to gain (e.g. no spans, or the heuristic
    identifies no script spans). Callers fall back to the flat `Block.text`.
    """
    if not b.spans:
        return ""
    # Drop any trailing spans whose combined text is the equation number so it
    # doesn't distort the median. We only look at spans whose text is not
    # entirely whitespace / bracket / digit.
    meaningful = [s for s in b.spans if s.text.strip()]
    if not meaningful:
        return ""

    sizes = sorted(s.size for s in meaningful)
    median_size = sizes[len(sizes) // 2]
    centers = sorted((s.bbox[1] + s.bbox[3]) / 2 for s in meaningful)
    median_center = centers[len(centers) // 2]
    # Line height guard: if the block is only one line tall, don't try to
    # infer subscript position from y-offset — it would be noise.
    line_height = max(s.bbox[3] - s.bbox[1] for s in meaningful)
    y_threshold = max(1.0, line_height * 0.15)

    out: list[str] = []
    found_script = False
    for s in b.spans:
        t = s.text
        if not t:
            continue
        # Skip an eq-number span at the tail.
        if number and _EQUATION_NUMBER_RE.search(t):
            continue

        token = t.strip()
        if not token:
            out.append(t)
            continue

        center = (s.bbox[1] + s.bbox[3]) / 2
        smaller = median_size > 0 and s.size < median_size * 0.88
        above = center < median_center - y_threshold
        below = center > median_center + y_threshold

        if s.is_superscript or (smaller and above):
            out.append(f"^{token} ")
            found_script = True
        elif smaller and below:
            out.append(f"_{token} ")
            found_script = True
        else:
            # Preserve leading/trailing whitespace the span carried.
            out.append(t)
    if not found_script:
        return ""
    return "".join(out)


def _symbolic_rewrite(text: str) -> str:
    """Replace Greek letters, math operators, and script markers with English."""
    # Preserve script runs before NFKC flattens their vertical placement.
    for mapping, marker in ((_SUPERSCRIPT_MAP, "^"), (_SUBSCRIPT_MAP, "_")):
        text = re.sub("[" + re.escape("".join(mapping)) + "]+",
                      lambda m: marker + "{" + "".join(mapping[c] for c in m[0]) + "}", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"_\{([^}]+)\}", r" sub \1 ", text)
    text = re.sub(r"\^\{([^}]+)\}", r" to the \1 ", text)
    text = re.sub(r"_([^\W_]+)", r" sub \1 ", text)
    text = re.sub(r"\^([^\W_]+)", r" to the \1 ", text)
    for sym, name in _GREEK_MAP.items():
        text = text.replace(sym, f" {name} ")
    for sym, repl in _UNICODE_MATH_MAP.items():
        text = text.replace(sym, repl)
    for sym, word in (("=", "equals"), ("+", "plus"), ("-", "minus"), ("/", "over")):
        text = text.replace(sym, f" {word} ")
    # Whitespace polish.
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_SYMBOLIC_PROMPT = """\
You are narrating a mathematical expression for an audiobook. Read it as a
fluent speaker would: prefer words over symbols, expand variables and
subscripts, and keep it brief.

{number_line}\
Expression (symbols already spelled out): {expression}

Produce one or two sentences of spoken English. Do not include LaTeX,
Markdown, code fences, or mathematical notation. Do not restate the
equation number — it will be added automatically."""


_VISION_PROMPT = (
    "Describe this mathematical expression in one or two spoken English "
    "sentences suitable for an audiobook. Do not include LaTeX, Markdown, "
    "or mathematical notation in your output. Do not restate any equation "
    "number — it will be added automatically."
)


def render_equation(
    data: EquationData,
    policy: EquationPolicy,
    llm: LLMCallable | None = None,
    vision_llm: VisionLLMCallable | None = None,
    source_path: str | None = None,
) -> str:
    """Dispatch on `policy.mode`. Falls through to `skip` on any failure."""
    symbolic_len = len(data.symbolic)
    if symbolic_len < policy.min_chars or symbolic_len > policy.max_chars:
        return _render_skip(data)

    narration: str | None = None
    if policy.mode == "vision":
        narration = _render_vision(data, vision_llm, source_path)
        if narration is None:
            # Vision failure -> try symbolic LLM before skipping.
            narration = _render_symbolic(data, policy, llm)
    elif policy.mode == "symbolic":
        narration = _render_symbolic(data, policy, llm)
    # else: skip mode -> narration stays None

    if narration:
        return _prepend_equation_number(narration, data.number)
    # No narration produced. Emit the symbolic form with the eq-number prefix
    # if we have one — even without an LLM the listener gets something that
    # at least names the equation's components.
    if policy.mode != "skip":
        fallback("Equation narration rejected or unavailable; using symbolic source.")
    return _render_skip(data)


def _render_symbolic(
    data: EquationData, policy: EquationPolicy, llm: LLMCallable | None
) -> str | None:
    if llm is None:
        return None
    number_line = (
        f"This is equation {_spoken_number(data.number)}.\n"
        if data.number else ""
    )
    prompt = _SYMBOLIC_PROMPT.format(
        number_line=number_line, expression=data.symbolic
    )
    try:
        out = llm(prompt).strip()
    except Exception:
        return None
    out = _sanitize_narration(out)
    if not out:
        return None
    # Sanity check: absurdly short narration is usually a truncated response.
    if len(out) < max(20, policy.min_narration_ratio * len(data.symbolic)):
        return None
    return out


def _render_vision(
    data: EquationData,
    vision_llm: VisionLLMCallable | None,
    source_path: str | None,
) -> str | None:
    if vision_llm is None or not source_path:
        return None
    try:
        import fitz  # pymupdf, already a required dep
        pdf = fitz.open(source_path)
        try:
            page = pdf[data.page]
            clip = fitz.Rect(*data.bbox)
            pix = page.get_pixmap(clip=clip, dpi=200)
            image_bytes = pix.tobytes("png")
        finally:
            pdf.close()
    except Exception:
        return None
    try:
        out = vision_llm(image_bytes, _VISION_PROMPT).strip()
    except Exception:
        return None
    out = _sanitize_narration(out)
    if not out:
        return None
    return out


def _render_skip(data: EquationData) -> str:
    """Deterministic fallback: the spelled-out symbolic form, eq-number prefixed."""
    body = data.symbolic.strip() or data.raw_text.strip()
    if not body:
        if data.number:
            return f"Equation {_spoken_number(data.number)} appears here; see the paper."
        return "An equation appears here; see the paper."
    body = body.rstrip(".;,") + "."
    if data.number:
        return f"Equation {_spoken_number(data.number)}: {body}"
    return body


def _prepend_equation_number(narration: str, number: str | None) -> str:
    """Prefix `Equation N` if the LLM didn't already mention it."""
    if not number:
        return narration
    spoken = _spoken_number(number)
    if re.match(r"^\s*Equation\s+" + re.escape(spoken) + r"(?!\w|\s+point\b)", narration, re.I):
        return narration
    # Capitalize the narration and prepend.
    narration = narration.lstrip()
    return f"Equation {spoken}: {narration}"


def _sanitize_narration(text: str) -> str:
    """Strip HTML tags, LaTeX delimiters, and markdown code fences from LLM output."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("$", "").replace("\\(", "").replace("\\)", "")
    text = re.sub(r"^```\w*\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    # Collapse internal whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text
