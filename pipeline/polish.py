"""Phase 7 — audio polish.

After classification + per-kind handlers run, every kept block carries
narratable prose. What's left are the surface-level pronunciation issues
that wreck a TTS engine's output even when the text is otherwise correct:

- raw numerals (`27.3`) instead of words (`twenty-seven point three`),
- Latin abbreviations (`e.g.`, `i.e.`) read as letters,
- acronyms (`LSTM`) read as nonsense words,
- units (`100 GB`) read as letters,
- known-mispronounced tokens (`ReLU`, `LaTeX`).

`audio_polish(doc, policy)` is a pure `Document -> Document` pass that
applies all of these as deterministic per-block text rewrites, in an order
designed so each rule sees the input it expects (quote-norm before
abbreviations, pronunciation overrides before acronym lettering, units
before standalone number conversion).

Acronym handling threads document-wide state: the first time `LSTM (Long
Short-Term Memory)` is encountered we emit `L S T M, Long Short-Term
Memory` and remember `LSTM`; every subsequent `LSTM` becomes `L S T M`
without the parenthetical. State is internal — callers don't have to plumb
it through.

`to_ssml(doc)` is an optional secondary serializer (Phase 7 SSML opt-in
per the plan). Most TTS engines treat the plain-text output fine; SSML is
useful when the engine respects `<break>` and `<sub>` for richer pacing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Iterable

from pipeline.model import Block, BlockKind, Document


# ---------------------------------------------------------------------------
# Defaults — kept as module constants so callers can read or extend them.
# ---------------------------------------------------------------------------

# Order matters: longer / more specific keys first so `Figs.` doesn't get
# matched as `Fig.` + `s.`. Applied as case-sensitive literal substitutions
# with word-boundary anchoring (see `_apply_abbreviations`).
DEFAULT_ABBREVIATIONS: dict[str, str] = {
    "et al.": "and others",
    "e.g.,": "for example,",
    "i.e.,": "that is,",
    "e.g.": "for example",
    "i.e.": "that is",
    "cf.": "compare",
    "vs.": "versus",
    "etc.": "et cetera",
    "Eqs.": "Equations",
    "Eq.": "Equation",
    "Figs.": "Figures",
    "Fig.": "Figure",
    "Refs.": "references",
    "Ref.": "reference",
    "Sec.": "Section",
    "Sect.": "Section",
    "Tab.": "Table",
    "Ch.": "Chapter",
    "App.": "Appendix",
    "approx.": "approximately",
}

# Tokens that look like acronyms but should be read as words / overrides.
# Applied as case-sensitive whole-token substitutions before acronym
# lettering, so `ReLU` never becomes `R E L U`.
DEFAULT_PRONUNCIATIONS: dict[str, str] = {
    "ReLU": "relu",
    "PReLU": "P relu",
    "GeLU": "gelu",
    "SiLU": "silu",
    "LaTeX": "lay-tek",
    "TeX": "tek",
    "arXiv": "archive",
    "PyTorch": "pie-torch",
    "TensorFlow": "tensor-flow",
    "NumPy": "num-pie",
    "SciPy": "sigh-pie",
    "scikit-learn": "sigh-kit learn",
    "GitHub": "git hub",
    "GPU": "G P U",
    "CPU": "C P U",
    "TPU": "T P U",
}

# Unit suffix → spoken form (singular). Plural appended automatically when
# the preceding number is not exactly 1 / 1.0. Order in the regex follows
# longest-first to avoid `kHz` matching `Hz` first.
DEFAULT_UNITS: dict[str, str] = {
    "TB": "terabyte",
    "GB": "gigabyte",
    "MB": "megabyte",
    "KB": "kilobyte",
    "kB": "kilobyte",
    "GHz": "gigahertz",
    "MHz": "megahertz",
    "kHz": "kilohertz",
    "Hz": "hertz",
    "ms": "millisecond",
    "ns": "nanosecond",
    "μs": "microsecond",
    "kg": "kilogram",
    "mg": "milligram",
    "km": "kilometer",
    "cm": "centimeter",
    "mm": "millimeter",
}

# Block kinds we never polish: their text is either already special-cased
# upstream (figure has no narration) or intentionally machine-readable
# (code is read verbatim by the dedicated handler when enabled).
_DEFAULT_SKIP_KINDS: tuple[BlockKind, ...] = (
    "code", "figure", "page_header", "page_footer",
    "toc", "noise", "footnote", "equation_inline",
)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass
class PolishPolicy:
    """Knobs for Phase 7. Every transform is independently toggleable so a
    caller can A/B (e.g. disable acronym lettering for a paper that already
    spells out every initialism)."""

    enable_quotes: bool = True
    enable_pronunciations: bool = True
    enable_acronyms: bool = True
    enable_abbreviations: bool = True
    enable_units: bool = True
    enable_numbers: bool = True

    # Caller extensions / overrides — merged with defaults at apply time so
    # users can add domain-specific entries without losing the defaults.
    extra_pronunciations: dict[str, str] = field(default_factory=dict)
    extra_abbreviations: dict[str, str] = field(default_factory=dict)
    extra_units: dict[str, str] = field(default_factory=dict)

    # Block kinds skipped entirely. Override to e.g. polish footnotes.
    skip_kinds: tuple[BlockKind, ...] = _DEFAULT_SKIP_KINDS

    # Numbers above this (after stripping commas) are read digit-by-digit
    # — `num2words` for a 12-digit integer is unreadable in audio.
    max_number_words: int = 9_999_999

    # SSML wrapper inserts strong breaks before headings. No-op unless
    # `to_ssml` is invoked.
    ssml_heading_break_ms: int = 600


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def audio_polish(
    doc: Document, policy: PolishPolicy | None = None
) -> Document:
    """Apply the polish transforms block-by-block.

    Pure: returns a new `Document`. Acronym state is threaded internally so
    only the first occurrence in reading order produces the lettered-plus-
    expansion form; subsequent occurrences get the lettered form alone.
    """
    policy = policy or PolishPolicy()
    state = _PolishState()
    new_blocks: list[Block] = []
    for b in doc.blocks:
        if b.kind in policy.skip_kinds or not b.text.strip():
            new_blocks.append(b)
            continue
        text = _polish_text(b.text, policy, state)
        if text == b.text:
            new_blocks.append(b)
        else:
            new_blocks.append(replace(b, text=text))
    return replace(doc, blocks=new_blocks)


# ---------------------------------------------------------------------------
# Internal state and orchestration
# ---------------------------------------------------------------------------

@dataclass
class _PolishState:
    """Document-wide mutable state across blocks (acronyms only, for now)."""
    seen_acronyms: set[str] = field(default_factory=set)


def _polish_text(text: str, policy: PolishPolicy, state: _PolishState) -> str:
    """Run the pipeline on one block's text. Order matters here:

    1. Quotes — clean up curlies before any rule that anchors on punctuation.
    2. Pronunciations — replace `ReLU`/`GPU` first so they're not seen by
       the acronym pass as `R E L U` / `G P U`.
    3. Units — `100 GB` must be handled before acronyms (which would eat
       `GB` as a 2-letter acronym) and before standalone numbers (which
       would convert `100` and leave `GB` orphaned).
    4. Acronyms — letter-space remaining capital runs.
    5. Abbreviations — expand `e.g.`/`Fig.` etc.
    6. Standalone numbers — final pass over whatever digits are left.
    """
    if policy.enable_quotes:
        text = _normalize_quotes(text)
    if policy.enable_pronunciations:
        text = _apply_pronunciations(text, policy)
    if policy.enable_units:
        text = _apply_units(text, policy)
    if policy.enable_acronyms:
        text = _apply_acronyms(text, state)
    if policy.enable_abbreviations:
        text = _apply_abbreviations(text, policy)
    if policy.enable_numbers:
        text = _apply_numbers(text, policy)
    # Whitespace polish: collapse the runs we just inserted.
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Quote normalization — straight quotes only; TTS engines pronounce some
# Unicode quote codepoints inconsistently.
# ---------------------------------------------------------------------------

_QUOTE_MAP = {
    "\u201C": '"', "\u201D": '"', "\u201E": '"', "\u201F": '"',
    "\u2018": "'", "\u2019": "'", "\u201A": "'", "\u201B": "'",
    "\u00AB": '"', "\u00BB": '"',
    "\u2032": "'", "\u2033": '"',
}


def _normalize_quotes(text: str) -> str:
    for src, dst in _QUOTE_MAP.items():
        text = text.replace(src, dst)
    return text


# ---------------------------------------------------------------------------
# Pronunciation overrides — token-level case-sensitive substitutions.
# ---------------------------------------------------------------------------

def _apply_pronunciations(text: str, policy: PolishPolicy) -> str:
    """Replace each known token with its phonetic spelling.

    We use a regex anchored on non-word boundaries (rather than `\\b`) so
    tokens with internal mixed case like `arXiv` and `PReLU` match cleanly
    — Python's `\\b` treats both lower→upper and upper→lower transitions
    as word characters, which would let `RELU` inside `PReLU` match.
    """
    table = {**DEFAULT_PRONUNCIATIONS, **policy.extra_pronunciations}
    if not table:
        return text
    # Longest first so `PReLU` beats `ReLU` when both would match.
    keys = sorted(table.keys(), key=len, reverse=True)
    for key in keys:
        pat = re.compile(rf"(?<!\w){re.escape(key)}(?!\w)")
        text = pat.sub(table[key], text)
    return text


# ---------------------------------------------------------------------------
# Acronym handling — first occurrence emits "L S T M, Long Short-Term
# Memory"; later occurrences emit "L S T M".
# ---------------------------------------------------------------------------

# Acronym pattern: 2–8 uppercase letters / digits, optional hyphen-suffix,
# optionally followed by a parenthetical expansion.
_ACRONYM_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,7})s?\b(?:\s*\(([^)]{2,80})\))?")
# Inside the expansion, count "real" words (letters, hyphens) — drop short
# stop-words like `and`, `of`, `the` when checking initial letters.
_EXPANSION_STOPWORDS = {"a", "an", "and", "of", "or", "the", "for", "to", "in"}


def _apply_acronyms(text: str, state: _PolishState) -> str:
    """Letter-space known acronyms; absorb parenthetical expansion on first sight.

    A parenthetical like `(Long Short-Term Memory)` is treated as the
    expansion only when the leading letters of its non-stopword words match
    the acronym's letters, in order. This guards against
    `BERT (Devlin et al., 2018)` getting treated as "Devlin" being the
    expansion of `BERT`.

    Skips blocks whose text is overwhelmingly uppercase (e.g. a titlecased-
    title like "AN IMAGE IS WORTH 16X16 WORDS" or a section heading like
    "1 INTRODUCTION"): in such blocks every word looks like an acronym to
    the regex, and letter-spacing every word turns the title into gibberish.
    Legitimate acronyms overwhelmingly appear inside mixed-case prose.
    """
    if _is_mostly_uppercase(text):
        return text

    def _sub(m: re.Match) -> str:
        acronym = m.group(1)
        paren = m.group(2)
        spelled = " ".join(acronym)
        if paren and _expansion_matches(acronym, paren):
            if acronym not in state.seen_acronyms:
                state.seen_acronyms.add(acronym)
                return f"{spelled}, {paren}"
            # Seen before: drop the parenthetical, just letter-space the acronym.
            return spelled
        # Either no parenthetical, or the parenthetical isn't an expansion
        # (citation, page reference, etc.) — letter-space the acronym but
        # preserve any captured parenthetical text exactly so e.g.
        # `BERT (Devlin et al., 2018)` keeps its citation.
        if paren is not None:
            return f"{spelled} ({paren})"
        return spelled
    return _ACRONYM_RE.sub(_sub, text)


def _is_mostly_uppercase(text: str) -> bool:
    """True for blocks whose alphabetic payload is ≥90% upper case.

    Catches titles (`AN IMAGE IS WORTH 16X16 WORDS`) and all-caps headings
    (`3 METHOD`, `ABSTRACT`). Running prose contains enough lowercase to
    keep it well under the threshold even when studded with acronyms —
    `BERT (Devlin et al., 2018) outperforms LSTM on GLUE` is ~35% upper.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) >= 0.90


def _expansion_matches(acronym: str, expansion: str) -> bool:
    """True when `expansion`'s word initials match the acronym letters.

    Splits on whitespace AND on hyphens so `Short-Term` contributes both
    `S` and `T` (matching `LSTM`'s `S T`). Stopwords like `of`, `and` are
    dropped before initials are taken since acronyms typically skip them.
    """
    words = re.findall(r"[A-Za-z]+", expansion)
    significant = [w for w in words if w.lower() not in _EXPANSION_STOPWORDS]
    if not significant:
        return False
    initials = "".join(w[0].upper() for w in significant)
    target = acronym.upper()
    # Allow expansions slightly longer than the acronym (subtitled words),
    # but the first len(target) initials must match exactly.
    return initials.startswith(target)


# ---------------------------------------------------------------------------
# Abbreviation expansion — literal substitution with non-word boundaries.
# ---------------------------------------------------------------------------

def _apply_abbreviations(text: str, policy: PolishPolicy) -> str:
    """Expand Latin / journal abbreviations to spoken form.

    Iteration order matters: the merged dict is sorted longest-key-first so
    `et al.` beats `al.` and `Figs.` beats `Fig.`.
    """
    table = {**DEFAULT_ABBREVIATIONS, **policy.extra_abbreviations}
    if not table:
        return text
    for key in sorted(table.keys(), key=len, reverse=True):
        # Trailing dot in keys is part of the literal — anchor only the left
        # side with a non-word boundary so `e.g.` matches mid-sentence.
        pat = re.compile(rf"(?<!\w){re.escape(key)}")
        text = pat.sub(table[key], text)
    return text


# ---------------------------------------------------------------------------
# Units — find `<number> <unit>` together so we can pluralize.
# ---------------------------------------------------------------------------

def _apply_units(text: str, policy: PolishPolicy) -> str:
    table = {**DEFAULT_UNITS, **policy.extra_units}
    if not table:
        return text
    # Build a single alternation, longest first so `GHz` beats `Hz`.
    keys = sorted(table.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(-?\d+(?:\.\d+)?)\s*(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)"
    )

    def _sub(m: re.Match) -> str:
        number_text = m.group(1)
        unit = m.group(2)
        spoken_number = _spell_number(number_text, policy)
        singular = table[unit]
        word = singular if number_text in ("1", "1.0", "-1") else singular + "s"
        return f"{spoken_number} {word}"

    return pattern.sub(_sub, text)


# ---------------------------------------------------------------------------
# Standalone numbers — last pass, after units have already consumed
# `100 GB` etc.
# ---------------------------------------------------------------------------

# Match a number with optional sign and optional thousands separators or
# decimal. Anchored with non-word boundaries so we don't touch the digit
# inside identifiers like `H2O` or `x_3` (the latter has no leading word
# char, but `_` is a word char so the `(?<!\w)` already excludes it).
_NUMBER_RE = re.compile(r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)(?![\w.])")


def _apply_numbers(text: str, policy: PolishPolicy) -> str:
    return _NUMBER_RE.sub(lambda m: _spell_number(m.group(1), policy), text)


def _spell_number(number_text: str, policy: PolishPolicy) -> str:
    """Convert a literal number to spoken English. Falls back to digit-by-digit
    for absurd magnitudes that `num2words` would render incomprehensibly."""
    cleaned = number_text.replace(",", "")
    sign_prefix = ""
    if cleaned.startswith("-"):
        sign_prefix = "negative "
        cleaned = cleaned[1:]
    if not cleaned:
        return number_text
    if "." in cleaned:
        left, right = cleaned.split(".", 1)
        if not left:
            left = "0"
        if not (left.isdigit() and right.isdigit()):
            return number_text
        left_words = _spell_integer(left, policy)
        # Fractional part: digit-by-digit, the only sane reading.
        right_words = " ".join(_DIGIT_WORDS[c] for c in right)
        return f"{sign_prefix}{left_words} point {right_words}"
    if not cleaned.isdigit():
        return number_text
    return f"{sign_prefix}{_spell_integer(cleaned, policy)}"


_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def _spell_integer(digits: str, policy: PolishPolicy) -> str:
    """Spell a non-negative integer; fall back to digit-by-digit when huge."""
    try:
        value = int(digits)
    except ValueError:
        return digits
    if value > policy.max_number_words:
        return " ".join(_DIGIT_WORDS[c] for c in digits)
    try:
        from num2words import num2words
    except ImportError:
        return digits
    try:
        return num2words(value)
    except Exception:
        return digits


# ---------------------------------------------------------------------------
# SSML — opt-in serializer alternative to `pipeline.serialize`.
# ---------------------------------------------------------------------------

def to_ssml(doc: Document, policy: PolishPolicy | None = None) -> str:
    """Render `doc` as SSML with section breaks before headings.

    We escape the bare minimum (`&<>`) and don't inject `<sub alias>` tags
    — the polish pass already replaced known mispronounced tokens, so a
    second alias layer would just complicate downstream debugging. Engines
    that ignore SSML treat the output as plain text plus angle-bracket
    noise, so callers must opt in deliberately.
    """
    policy = policy or PolishPolicy()
    parts: list[str] = ["<speak>"]
    first_block = True
    for b in doc.blocks:
        if not b.text.strip() or b.kind in policy.skip_kinds:
            continue
        if b.kind == "heading" and not first_block:
            parts.append(
                f'<break time="{policy.ssml_heading_break_ms}ms"/>'
            )
        parts.append(f"<p>{_xml_escape(b.text)}</p>")
        first_block = False
    parts.append("</speak>")
    return "".join(parts)


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


__all__ = [
    "DEFAULT_ABBREVIATIONS",
    "DEFAULT_PRONUNCIATIONS",
    "DEFAULT_UNITS",
    "PolishPolicy",
    "audio_polish",
    "to_ssml",
]
