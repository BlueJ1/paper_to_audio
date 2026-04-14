"""Phase 1 — layout-aware extraction.

Replaces `page.get_text("text")` with `page.get_text("dict")` and reconstructs
a `Document` with reading-order-preserving blocks, paragraph-joined text, and
per-span font/bbox metadata.

The output of this phase is the substrate every later phase operates on.
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass

import fitz  # pymupdf

from pipeline.model import BBox, Block, Document, FontStats, Span


# Ligatures — expand at span creation so no downstream stage sees them.
_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}

# Compound prefixes/suffixes — when a hyphen appears at a line break AND one
# side is in these sets, keep the hyphen (e.g. "self-\nattention" stays
# "self-attention"). Covers the common compound vocabulary in ML papers.
_HYPHEN_KEEP_PREFIXES = frozenset({
    "self", "non", "multi", "pre", "post", "inter", "intra", "sub", "super",
    "cross", "semi", "quasi", "pseudo", "anti", "co", "bi", "tri", "extra",
    "mid", "over", "under", "counter", "trans", "ultra", "meta", "mini",
    "micro", "macro", "mega", "neo",
})

_HYPHEN_KEEP_SUFFIXES = frozenset({
    "aware", "based", "wise", "like", "free", "level", "specific", "style",
    "type", "driven", "oriented", "labeled", "weighted", "invariant",
    "dependent", "independent",
})


def load_pdf(path: str) -> fitz.Document:
    """Open a PDF for extraction. Caller is responsible for `doc.close()`."""
    return fitz.open(path)


def _normalize_span_text(text: str) -> str:
    """NFC-normalize and expand ligatures at span creation time."""
    text = unicodedata.normalize("NFC", text)
    for lig, repl in _LIGATURES.items():
        text = text.replace(lig, repl)
    return text


def _mk_span(raw: dict) -> Span:
    return Span(
        text=_normalize_span_text(raw["text"]),
        font=raw.get("font", ""),
        size=float(raw.get("size", 0.0)),
        flags=int(raw.get("flags", 0)),
        bbox=tuple(raw.get("bbox", (0.0, 0.0, 0.0, 0.0))),  # type: ignore[arg-type]
    )


def _join_line_texts(line_texts: list[str]) -> str:
    """Join lines of a paragraph, healing hyphenated line breaks.

    Default behavior strips the hyphen ("intro-\\nduction" -> "introduction").
    When the hyphen is intentional (compound prefix/suffix), keep it.
    """
    if not line_texts:
        return ""
    out = line_texts[0]
    for ln in line_texts[1:]:
        if not ln:
            continue
        # Hyphenated line-break: prev ends "word-", next starts lowercase alpha.
        if (
            len(out) >= 2
            and out.endswith("-")
            and out[-2].isalpha()
            and ln[0].islower()
        ):
            # Find the word fragment before the hyphen.
            left_word = ""
            for ch in reversed(out[:-1]):
                if ch.isalpha():
                    left_word = ch + left_word
                else:
                    break
            right_head = ""
            for ch in ln:
                if ch.isalpha():
                    right_head += ch
                else:
                    break
            keep_hyphen = (
                left_word.lower() in _HYPHEN_KEEP_PREFIXES
                or right_head.lower() in _HYPHEN_KEEP_SUFFIXES
            )
            if keep_hyphen:
                out = out + ln
            else:
                out = out[:-1] + ln
        else:
            out = out + " " + ln
    return out


def _block_from_pymupdf(raw_block: dict, page_index: int) -> Block | None:
    """Convert a single pymupdf text block into our Block model.

    Returns None for non-text blocks (images) or empty blocks.
    """
    if raw_block.get("type") != 0:
        return None
    lines = raw_block.get("lines") or []
    if not lines:
        return None

    spans: list[Span] = []
    line_texts: list[str] = []
    for line in lines:
        raw_spans = line.get("spans") or []
        if not raw_spans:
            continue
        line_span_objs = [_mk_span(rs) for rs in raw_spans]
        spans.extend(line_span_objs)
        line_text = "".join(s.text for s in line_span_objs).strip()
        if line_text:
            line_texts.append(line_text)

    if not spans or not line_texts:
        return None

    text = _join_line_texts(line_texts).strip()
    if not text:
        return None

    return Block(
        kind="body",  # Phase 2 refines this.
        spans=spans,
        text=text,
        page=page_index,
        bbox=tuple(raw_block.get("bbox", (0.0, 0.0, 0.0, 0.0))),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Column detection and reading order (per page)
# ---------------------------------------------------------------------------

@dataclass
class _ColumnLayout:
    """Result of per-page column analysis."""
    two_column: bool
    page_mid: float   # x-coordinate of the page midline
    page_width: float


def _is_full_width(bbox: BBox, page_mid: float, page_width: float) -> bool:
    """True when a block clearly spans the page midline on both sides.

    More robust than a pure width ratio: academic titles span ~55% of the
    page width and are centered, so the reliable signal is that the bbox
    crosses the midline with significant extent on each side.
    """
    x0, _, x1, _ = bbox
    margin = page_width * 0.08
    return x0 < page_mid - margin and x1 > page_mid + margin


def _analyze_columns(blocks: list[Block], page_width: float) -> _ColumnLayout:
    """Decide whether a page is two-column based on block xmid distribution.

    A block contributes to the column count only if it's "body-width"
    (width > 25% of the page). Narrow fragments — equation subblocks,
    small figure tokens — would otherwise trigger false positives on
    single-column pages where such fragments happen to straddle the
    midline.

    Two-column if >=2 body-width blocks sit clearly left of midline AND
    >=2 sit clearly right. Full-width blocks are excluded.
    """
    mid = page_width / 2.0
    margin = page_width * 0.05
    min_body_width = page_width * 0.25
    left = right = 0
    for b in blocks:
        if _is_full_width(b.bbox, mid, page_width):
            continue
        x0, _, x1, _ = b.bbox
        if (x1 - x0) < min_body_width:
            continue
        xmid = (x0 + x1) / 2.0
        if xmid < mid - margin:
            left += 1
        elif xmid > mid + margin:
            right += 1
    return _ColumnLayout(two_column=left >= 2 and right >= 2, page_mid=mid, page_width=page_width)


def _order_page_blocks(blocks: list[Block], layout: _ColumnLayout) -> list[Block]:
    """Return blocks in reading order for a single page.

    Single-column: sort by y0.
    Two-column: full-width blocks split the page into horizontal bands; within
    each band, left-column blocks come before right-column blocks. This
    correctly handles mixed-layout pages where title/abstract span both
    columns above a two-column body.
    """
    if not blocks:
        return []
    if not layout.two_column:
        return sorted(blocks, key=lambda b: b.bbox[1])

    full_width: list[Block] = []
    col: list[list[Block]] = [[], []]
    for b in blocks:
        x0, _, x1, _ = b.bbox
        xmid = (x0 + x1) / 2.0
        if _is_full_width(b.bbox, layout.page_mid, layout.page_width):
            full_width.append(b)
        elif xmid < layout.page_mid:
            b.column = 0
            col[0].append(b)
        else:
            b.column = 1
            col[1].append(b)

    full_width.sort(key=lambda b: b.bbox[1])
    col[0].sort(key=lambda b: b.bbox[1])
    col[1].sort(key=lambda b: b.bbox[1])

    ordered: list[Block] = []
    prev_y = -float("inf")
    for span_b in full_width:
        band_top = span_b.bbox[1]
        for side in (0, 1):
            ordered.extend(b for b in col[side] if prev_y <= b.bbox[1] < band_top)
        ordered.append(span_b)
        prev_y = span_b.bbox[3]  # bottom of full-width block
    for side in (0, 1):
        ordered.extend(b for b in col[side] if b.bbox[1] >= prev_y)

    return ordered


# ---------------------------------------------------------------------------
# Font statistics (document-wide)
# ---------------------------------------------------------------------------

def _compute_font_stats(blocks: list[Block]) -> FontStats:
    """Modal span size weighted by character count -> body_size.

    Heading thresholds = unique sizes strictly above body_size, sorted
    ascending. Phase 2 maps each into a heading level. We round to 0.1pt
    to collapse trivial float jitter.
    """
    size_weights: Counter[float] = Counter()
    for block in blocks:
        for span in block.spans:
            size_weights[round(span.size, 1)] += len(span.text)
    if not size_weights:
        return FontStats(body_size=0.0, heading_thresholds=[])
    body_size = size_weights.most_common(1)[0][0]
    heading_thresholds = sorted({s for s in size_weights if s > body_size})
    return FontStats(body_size=body_size, heading_thresholds=heading_thresholds)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def extract_layout(pdf_path: str) -> Document:
    """Extract a layout-aware `Document` from a PDF path.

    One-column and two-column pages both produce reading-ordered blocks.
    Every block starts with kind="body"; later phases refine.
    """
    blocks: list[Block] = []
    page_rects: list[BBox] = []
    pdf = load_pdf(pdf_path)
    try:
        for page_index, page in enumerate(pdf):
            page_rect = page.rect
            page_rects.append(
                (float(page_rect.x0), float(page_rect.y0), float(page_rect.x1), float(page_rect.y1))
            )
            page_width = float(page_rect.width)
            raw = page.get_text("dict")
            page_blocks: list[Block] = []
            for raw_block in raw.get("blocks") or []:
                blk = _block_from_pymupdf(raw_block, page_index)
                if blk is not None:
                    page_blocks.append(blk)
            layout = _analyze_columns(page_blocks, page_width)
            ordered = _order_page_blocks(page_blocks, layout)
            blocks.extend(ordered)
    finally:
        pdf.close()

    fonts = _compute_font_stats(blocks)
    return Document(blocks=blocks, fonts=fonts, page_rects=page_rects, source_path=pdf_path)
