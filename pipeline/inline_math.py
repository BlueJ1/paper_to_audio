"""Phase 6 — inline math span rewriting.

Body paragraphs often sprinkle math into running prose: "the hidden state
x at time t is passed through W ...". Treating the whole paragraph as
math (what the legacy pipeline does) drowns out the non-math text; leaving
it alone produces TTS that mispronounces "x" as the word "ex" and reads
Greek letters as a vowel soup.

This stage takes a surgical approach. Using span-level font flags from
Phase 1, it detects `variable` spans — italic single ASCII letters and any
single Greek letter — and wraps each occurrence in `⟦…⟧` delimiters in
the block text. Then it rewrites according to `InlineMathPolicy.mode`:

- `skip`: leave body blocks untouched (no marking, no rewriting).
- `symbolic`: run `pipeline.equations._symbolic_rewrite` on each delimited
  span. Deterministic and cheap; handles Greek letters well (`α` → `alpha`)
  but effectively a no-op on plain italic ASCII (`x` → `x`). Still useful
  for the meta annotation that downstream consumers can inspect.
- `llm`: send only detected variable regions and validate a JSON array of
  replacements against their spoken identities. Splice these into immutable
  prose; fall back to `symbolic` with a warning on failure.

The pure function contract matches the rest of the pipeline: body blocks
without any detected variables pass through untouched; rewritten blocks
get a `meta["inline_math"]` payload with the marked text so callers can
A/B the rendering without re-running detection.
"""
from __future__ import annotations

import re
import json
from pipeline.diagnostics import fallback
import unicodedata
from dataclasses import dataclass, replace
from typing import Callable, Literal

from pipeline.equations import _symbolic_rewrite
from pipeline.model import Block, Document, Span


InlineMathMode = Literal["skip", "symbolic", "llm"]

LLMCallable = Callable[[str], str]


# Delimiters match the plan's proposed markers; chosen because they never
# appear in extracted PDF text, so we can strip or round-trip them safely.
_MATH_DELIM_OPEN = "\u27E6"   # ⟦
_MATH_DELIM_CLOSE = "\u27E7"  # ⟧

# Unicode ranges a single-character variable span can fall into. We treat
# every codepoint inside these blocks as variable-like regardless of the
# italic flag, because:
#   * Greek math characters are often rendered from fonts that don't set
#     the italic bit, yet visually they're slanted math letters.
#   * Math Alphanumeric Symbols (U+1D400–U+1D7FF) is the canonical block
#     academic PDFs use for italic Latin/Greek variables (e.g. Titans uses
#     𝑡, 𝑥, 𝜃 directly).
#   * Letterlike Symbols include ℓ, ℏ, ℝ, etc. that frequently act as
#     single-symbol variables.
_MATH_CHAR_RANGES: tuple[tuple[int, int], ...] = (
    (0x0370, 0x03FF),   # Greek and Coptic
    (0x1D400, 0x1D7FF), # Mathematical Alphanumeric Symbols
    (0x2100, 0x214F),   # Letterlike Symbols
)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass
class InlineMathPolicy:
    """Knobs for Phase 6.

    `max_chars` gates the LLM call: a paragraph above this length is almost
    certainly a misclassified code block or a concatenated section; don't
    waste tokens on it. `min_runs_for_llm` defends against paragraphs with
    a single incidental italic — symbolic rewrite handles those for free.
    """
    mode: InlineMathMode = "skip"
    min_runs_for_llm: int = 1
    max_chars: int = 6000
    # Retained for caller compatibility; structured validation supersedes it.
    min_output_ratio: float = 0.5


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def rewrite_inline_math(
    doc: Document,
    policy: InlineMathPolicy | None = None,
    llm: LLMCallable | None = None,
) -> Document:
    """Rewrite body blocks with delimited-then-rendered inline math.

    Non-body blocks and blocks without any detected variables pass through
    unchanged. The returned `Document` is a new object (pure function).
    """
    policy = policy or InlineMathPolicy()
    if policy.mode == "skip":
        return doc
    new_blocks: list[Block] = []
    for b in doc.blocks:
        if b.kind != "body":
            new_blocks.append(b)
            continue
        marked = _mark_inline_math(b)
        if marked == b.text:
            new_blocks.append(b)
            continue
        rendered = _render(marked, policy, llm)
        new_blocks.append(
            replace(b, text=rendered, meta={**b.meta, "inline_math": marked})
        )
    return replace(doc, blocks=new_blocks)


# ---------------------------------------------------------------------------
# Variable detection and marking
# ---------------------------------------------------------------------------

def _is_variable_span(span: Span) -> bool:
    """True when the span is a single math variable in running prose.

    Accepts:
      * italic single ASCII letters (the overwhelming convention for
        math variables in English papers),
      * any single character from the Greek, Math Alphanumeric Symbols,
        or Letterlike Symbols blocks regardless of italic flag — many
        math fonts don't set the italic bit on those glyphs even though
        they're visually italic variables.
    """
    text = span.text.strip()
    if len(text) != 1:
        return False
    cp = ord(text)
    for lo, hi in _MATH_CHAR_RANGES:
        if lo <= cp <= hi:
            return True
    return span.is_italic and text.isascii() and text.isalpha()


def _collect_variables(spans: list[Span]) -> list[str]:
    """Stripped text of each variable span, in document order (with duplicates)."""
    return [s.text.strip() for s in spans if _is_variable_span(s)]


def _mark_inline_math(block: Block) -> str:
    """Wrap each variable occurrence in `block.text` with delimiters.

    We advance a `pos` cursor past each match so repeated variables ("for
    each x, pick y such that x = y") get each occurrence marked in order.
    """
    variables = _collect_variables(block.spans)
    if not variables:
        return block.text

    text = block.text
    pos = 0
    for var in variables:
        idx = _find_standalone(text, var, pos)
        if idx < 0:
            continue
        text = (
            text[:idx]
            + _MATH_DELIM_OPEN + var + _MATH_DELIM_CLOSE
            + text[idx + len(var):]
        )
        # Advance past the inserted delimiters + variable.
        pos = idx + len(var) + 2
    return text


def _find_standalone(text: str, var: str, start: int) -> int:
    """Next occurrence of `var` in `text` at or after `start` with non-alpha boundaries.

    Prevents marking the `x` in `example` or the `a` in `paragraph`. We do
    NOT use `\\b` regex boundaries because Unicode word boundaries classify
    Greek letters as letters on both sides — good — but the direct check is
    simpler and works identically for ASCII and Greek here.
    """
    i = max(0, start)
    while i <= len(text) - len(var):
        idx = text.find(var, i)
        if idx < 0:
            return -1
        before_ok = idx == 0 or not text[idx - 1].isalpha()
        end = idx + len(var)
        after_ok = end == len(text) or not text[end].isalpha()
        if before_ok and after_ok:
            return idx
        i = idx + 1
    return -1


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_DELIM_RE = re.compile(
    rf"{_MATH_DELIM_OPEN}([^{_MATH_DELIM_OPEN}{_MATH_DELIM_CLOSE}]+){_MATH_DELIM_CLOSE}"
)


def _render(
    marked: str, policy: InlineMathPolicy, llm: LLMCallable | None
) -> str:
    """Dispatch on `policy.mode`; fall back to `symbolic` on `llm` failure."""
    if policy.mode == "llm":
        out = _render_llm(marked, policy, llm)
        if out is not None:
            return out
    return _render_symbolic(marked)


def _render_symbolic(marked: str) -> str:
    """Replace each `⟦…⟧` with `_symbolic_rewrite(contents)`.

    Strips the delimiters in the process. Contents are NFKC-normalized
    first so math-italic Latin (`𝑡` → `t`), math-italic Greek (`𝜃` → `θ`),
    and letterlike symbols (`ℓ` → `l`) collapse to their base forms that
    the reused symbolic rewriter understands. Without NFKC, `_symbolic_rewrite`
    would pass `𝑡` through unchanged and the TTS would mispronounce it.
    """
    def _sub(m: re.Match) -> str:
        content = unicodedata.normalize("NFKC", m.group(1))
        return _symbolic_rewrite(content)
    return _DELIM_RE.sub(_sub, marked)


_LLM_PROMPT = """Return only a JSON array of spoken replacements, one per math span in
order. Spell Greek letters by name and preserve variable identity. No commentary.
Math spans: {runs}"""


def _render_llm(marked: str, policy: InlineMathPolicy, llm: LLMCallable | None) -> str | None:
    runs = _DELIM_RE.findall(marked)
    if llm is None or len(marked) > policy.max_chars or len(runs) < policy.min_runs_for_llm:
        fallback("Inline math used deterministic narration (provider unavailable or size guard).")
        return None
    try:
        replacements = json.loads(llm(_LLM_PROMPT.format(runs=json.dumps(runs))))
        if not isinstance(replacements, list) or len(replacements) != len(runs):
            raise ValueError("expected one replacement per span")
        # Detected regions currently contain single variables, so their spoken
        # identity is deterministic. A future region detector needs its own validator.
        for raw, spoken in zip(runs, replacements):
            expected = _render_symbolic(_MATH_DELIM_OPEN + raw + _MATH_DELIM_CLOSE)
            if not isinstance(spoken, str) or spoken.strip() != expected.strip():
                raise ValueError("replacement changes variable identity")
        replacements = iter(replacements)
        return _DELIM_RE.sub(lambda _: next(replacements), marked)
    except Exception as exc:
        fallback(f"Inline math response rejected; using symbolic narration: {exc}")
        return None
