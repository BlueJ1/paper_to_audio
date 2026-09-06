"""Phase 2 — block classification and section tree.

Every Phase 1 block arrives with `kind="body"`. This stage inspects the span
metadata (font size, font name, flags, bbox, page geometry) and assigns a
definitive `kind`. Then it walks the reading order, assigns heading `level`
from font-size rank, and sets `parent_section` on every non-heading block.

Rules are deterministic. An LLM fallback is a Phase 8 concern; this file is
pure heuristics so classification is cheap and reproducible.
"""
from __future__ import annotations

import re
from collections import defaultdict

from pipeline.model import BBox, Block, Document


# ---------------------------------------------------------------------------
# Patterns and constants
# ---------------------------------------------------------------------------

_CAPTION_RE = re.compile(
    r"^\s*(Figure|Fig\.?|Table|Algorithm)\s+\d+(\.\d+)?\s*[.:\s]",
    re.IGNORECASE,
)
# arXiv margin stamp: `arXiv:2501.00663v1 [cs.LG] 31 Dec 2024`. Pymupdf
# sometimes reports this as oversized rotated text, fooling the heading
# classifier; other times it lands at body size in the header band. Either
# way it is metadata, not narration, so we drop it to noise up front.
_ARXIV_HEADER_RE = re.compile(
    r"^\s*arXiv:\s*\d{4}\.\d{4,5}(v\d+)?\b", re.IGNORECASE
)
_SECTION_NUM_RE = re.compile(r"^\d+(\.\d+)*\.?\s+\S")
# 1–3 digit equation number at the right edge: "(3)" or "(3.4)".
# Tight bound avoids matching citation years like "(2019)".
_EQUATION_NUMBER_RE = re.compile(r"\(\s*\d{1,3}(?:\.\d{1,3})?\s*\)\s*$")
_DOT_LEADER_RE = re.compile(r"\.{3,}\s*\d+")
_MONO_FONT_RE = re.compile(
    r"(courier|consolas|inconsolata|menlo|monaco|sourcecode|firacode|mono)",
    re.IGNORECASE,
)

# Unicode ranges that point strongly at math content.
_MATH_RANGES: tuple[tuple[int, int], ...] = (
    (0x0370, 0x03FF),  # Greek
    (0x2070, 0x209F),  # super/subscripts
    (0x2200, 0x22FF),  # mathematical operators
    (0x27C0, 0x27EF),  # misc math A
    (0x2A00, 0x2AFF),  # supplemental math operators
)
_ASCII_MATH = set("=+<>≤≥≠≈−·×÷")

# Thresholds — kept as module constants for easy tuning.
# Size ratio is deliberately modest (papers routinely set section headings only
# 1.05–1.10× body; the Plan's 1.15 was too strict on real samples). We combine
# the size bump with an additional structural signal (numbered / all-caps /
# short line) to avoid false positives on slightly oversized body text.
_HEADING_SIZE_RATIO = 1.05
# Bold at/near body size also counts as a heading when it carries a structural
# signal (e.g. "3.1 Long-term Memory" bolded at body size).
_HEADING_BOLD_SIZE_RATIO = 0.95
# All-caps standalone headings (ICLR/NeurIPS templates set sections in the body
# font at body size, distinguished only by full uppercase). Require at least
# body-size to avoid catching small-caps legends in figures.
_HEADING_ALLCAPS_SIZE_RATIO = 0.95
_HEADING_ALLCAPS_MAX_WORDS = 10
_HEADING_MAX_CHARS = 160
_HEADING_MAX_WORDS = 14
# Block aspect ratios extreme enough to indicate rotated / sidebar text
# (e.g. the arXiv watermark rendered vertically in the left margin). These
# should never be treated as headings even if their per-span size is large.
_ROTATED_MAX_WIDTH = 50.0
_ROTATED_WIDTH_HEIGHT_RATIO = 0.5
_FOOTNOTE_SIZE_RATIO = 0.85
_FOOTNOTE_TOP_FRAC = 0.80
_EQUATION_DENSITY = 0.12
_EQUATION_MAX_CHARS = 400
_HEADER_TOP_FRAC = 0.10
_FOOTER_BOTTOM_FRAC = 0.92
_HEADER_FOOTER_MIN_PAGES = 3


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def classify_blocks(doc: Document) -> Document:
    """Assign `kind`, heading `level`, and `parent_section` in place."""
    _mark_headers_and_footers(doc)
    body_size = doc.fonts.body_size
    for b in doc.blocks:
        if b.kind in ("page_header", "page_footer"):
            continue
        b.kind = _classify_block(b, body_size, doc.page_rects)
    _assign_heading_levels(doc)
    _assign_parent_sections(doc)
    return doc


# ---------------------------------------------------------------------------
# Per-block rules
# ---------------------------------------------------------------------------

def _classify_block(b: Block, body_size: float, page_rects: list[BBox]) -> str:
    text = b.text.strip()
    if not text:
        return "noise"
    if _ARXIV_HEADER_RE.match(text):
        return "noise"
    if _CAPTION_RE.match(text):
        return "caption"
    if _is_code(b):
        return "code"
    if _is_heading(b, body_size):
        return "heading"
    if _is_footnote(b, body_size, page_rects):
        return "footnote"
    if _is_toc(text):
        return "toc"
    if _is_equation_display(text):
        return "equation_display"
    if _is_noise(text):
        return "noise"
    return "body"


def _is_rotated_block(b: Block) -> bool:
    """Reject narrow, tall blocks (vertical / rotated text like arXiv sidebars).

    Rotated glyphs still report their raw font size, which can falsely boost
    an arXiv margin watermark to "largest text in the document".
    """
    w = b.bbox[2] - b.bbox[0]
    h = b.bbox[3] - b.bbox[1]
    if w <= 0 or h <= 0:
        return True
    return w < _ROTATED_MAX_WIDTH and (w / h) < _ROTATED_WIDTH_HEIGHT_RATIO


def _bold_fraction(b: Block) -> float:
    total = sum(len(s.text) for s in b.spans)
    if total == 0:
        return 0.0
    bold = sum(len(s.text) for s in b.spans if s.is_bold)
    return bold / total


def _is_heading(b: Block, body_size: float) -> bool:
    """Short block with a structural heading signal.

    Two qualifying paths:
      A) font size is noticeably above body (≥ 1.05×) AND has a structural
         signal (numeric prefix, near-all-caps, or short line);
      B) font is bold at body size AND the block is numbered / all-caps.
    Path B catches venues that bold inline subsection headings at body size.
    Rotated blocks are excluded unconditionally.
    """
    if body_size <= 0:
        return False
    text = b.text.strip()
    if not text or len(text) > _HEADING_MAX_CHARS:
        return False
    if _is_rotated_block(b):
        return False

    numbered = bool(_SECTION_NUM_RE.match(text))
    letters = [c for c in text if c.isalpha()]
    all_caps = bool(letters) and sum(1 for c in letters if c.isupper()) / len(letters) >= 0.8
    words = text.split()
    short = len(words) <= _HEADING_MAX_WORDS
    size_ratio = b.dominant_size / body_size

    # Path A — size bump.
    if size_ratio >= _HEADING_SIZE_RATIO and (numbered or all_caps or short):
        return True
    # Path B — bold inline heading at body size.
    if (
        _bold_fraction(b) >= 0.5
        and size_ratio >= _HEADING_BOLD_SIZE_RATIO
        and short
        and (numbered or all_caps)
    ):
        return True
    # Path C — all-caps standalone heading at body size (ICLR/NeurIPS style:
    # "ABSTRACT", "1 INTRODUCTION", "2 RELATED WORK" set in the body font).
    # Needs enough letters to rule out stray caps fragments, and a short
    # standalone line — body prose spilling into all-caps would be longer.
    if (
        all_caps
        and size_ratio >= _HEADING_ALLCAPS_SIZE_RATIO
        and len(words) <= _HEADING_ALLCAPS_MAX_WORDS
        and len(letters) >= 3
    ):
        return True
    return False


def _is_code(b: Block) -> bool:
    """Dominant span font is monospace (by flag or by name)."""
    total = sum(len(s.text) for s in b.spans)
    if total == 0:
        return False
    mono = sum(
        len(s.text)
        for s in b.spans
        if s.is_monospace or _MONO_FONT_RE.search(s.font or "")
    )
    return mono / total >= 0.6


def _is_footnote(b: Block, body_size: float, page_rects: list[BBox]) -> bool:
    """Smaller-than-body text anchored to the bottom of the page."""
    if body_size <= 0 or b.dominant_size >= body_size * _FOOTNOTE_SIZE_RATIO:
        return False
    if b.page < 0 or b.page >= len(page_rects):
        return False
    _x0, y0, _x1, y1 = page_rects[b.page]
    page_h = y1 - y0
    if page_h <= 0:
        return False
    rel_top = (b.bbox[1] - y0) / page_h
    return rel_top >= _FOOTNOTE_TOP_FRAC


def _is_toc(text: str) -> bool:
    """Dot-leader-dominant text: `Introduction . . . . . . 1`."""
    return len(_DOT_LEADER_RE.findall(text)) >= 2


def _is_equation_display(text: str) -> bool:
    """Short, math-heavy, no sentence prose.

    Two qualifying modes: (a) ends with an equation number like `(3.4)` AND
    has at least one math glyph, or (b) math-glyph density is high enough
    that it can't plausibly be prose.
    """
    if len(text) > _EQUATION_MAX_CHARS:
        return False
    math_n = _count_math_chars(text)
    if math_n == 0:
        return False
    ends_with_eqnum = bool(_EQUATION_NUMBER_RE.search(text))
    density = math_n / max(len(text), 1)
    if not ends_with_eqnum and density < _EQUATION_DENSITY:
        return False
    # Prose guardrail: genuine equations rarely have many full stops.
    if text.count(". ") > 2:
        return False
    return True


def _count_math_chars(text: str) -> int:
    n = 0
    for ch in text:
        if ch in _ASCII_MATH:
            n += 1
            continue
        cp = ord(ch)
        for lo, hi in _MATH_RANGES:
            if lo <= cp <= hi:
                n += 1
                break
    return n


def _is_noise(text: str) -> bool:
    """Too few alphabetic words to carry meaning on its own."""
    return len(re.findall(r"[A-Za-z]{2,}", text)) < 3


# ---------------------------------------------------------------------------
# Repeating page header / footer detection
# ---------------------------------------------------------------------------

def _mark_headers_and_footers(doc: Document) -> None:
    """Mark blocks whose normalized text repeats across ≥ 3 pages at the
    same top/bottom margin. Sets `kind` directly so per-block classification
    can skip them.
    """
    page_rects = doc.page_rects
    if not page_rects:
        return

    groups: dict[tuple[str, str], list[Block]] = defaultdict(list)
    for b in doc.blocks:
        if b.page < 0 or b.page >= len(page_rects):
            continue
        _x0, y0, _x1, y1 = page_rects[b.page]
        page_h = y1 - y0
        if page_h <= 0:
            continue
        rel_top = (b.bbox[1] - y0) / page_h
        rel_bot = (b.bbox[3] - y0) / page_h
        text = b.text.strip()
        if not text or len(text) > 200:
            continue
        norm = _normalize_repeating(text)
        if not norm:
            continue
        if rel_top < _HEADER_TOP_FRAC:
            groups[("h", norm)].append(b)
        elif rel_bot > _FOOTER_BOTTOM_FRAC:
            groups[("f", norm)].append(b)

    for (side, _norm), blocks in groups.items():
        pages = {b.page for b in blocks}
        if len(pages) < _HEADER_FOOTER_MIN_PAGES:
            continue
        target = "page_header" if side == "h" else "page_footer"
        for b in blocks:
            b.kind = target


def _normalize_repeating(text: str) -> str:
    """Collapse whitespace and digits so `Page 3` == `Page 4`."""
    collapsed = re.sub(r"\s+", " ", text)
    collapsed = re.sub(r"\d+", "#", collapsed)
    return collapsed.strip().lower()


# ---------------------------------------------------------------------------
# Heading levels and section tree
# ---------------------------------------------------------------------------

def _assign_heading_levels(doc: Document) -> None:
    """Rank unique heading sizes (rounded to 0.1pt) descending; largest is 1."""
    sizes = sorted(
        {round(b.dominant_size, 1) for b in doc.blocks if b.kind == "heading"},
        reverse=True,
    )
    size_to_level = {s: i + 1 for i, s in enumerate(sizes)}
    for b in doc.blocks:
        if b.kind == "heading":
            b.level = size_to_level.get(round(b.dominant_size, 1))


def _assign_parent_sections(doc: Document) -> None:
    """Set `parent_section` to the nearest preceding heading's text.

    Nested headings: a new heading replaces `current` only when its level is
    ≤ the previous heading's level. A deeper heading (level > current.level)
    becomes a subsection within the current section — we still track it as
    the immediate parent for following blocks, so subsection-aware filtering
    works in Phase 3.
    """
    current: str | None = None
    for b in doc.blocks:
        if b.kind == "heading":
            current = b.text.strip()
            b.parent_section = None
        else:
            b.parent_section = current
