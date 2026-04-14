"""Typed document model.

The pipeline operates on a `Document` — a flat list of `Block`s in reading order,
each carrying layout metadata (spans, bbox, page, column). Every pipeline stage
is a pure function Document -> Document. See ADVANCED_PIPELINE_PLAN.md sections
3 and 4.

Phase 1 populates Span, Block (with default kind="body"), FontStats, and
Document. Subsequent phases refine `Block.kind` and `parent_section`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BlockKind = Literal[
    "heading", "body", "caption",
    "equation_display", "equation_inline",
    "table", "figure", "footnote", "code",
    "page_header", "page_footer", "toc", "noise",
]

BBox = tuple[float, float, float, float]


@dataclass
class Span:
    """A contiguous run of text sharing one font/size/flags from pymupdf dict mode.

    `flags` is pymupdf's span flag bitfield: bit 0 superscript, bit 1 italic,
    bit 2 serif, bit 3 monospace, bit 4 bold.
    """
    text: str
    font: str
    size: float
    flags: int
    bbox: BBox

    @property
    def is_italic(self) -> bool:
        return bool(self.flags & 0b10)

    @property
    def is_bold(self) -> bool:
        return bool(self.flags & 0b10000)

    @property
    def is_monospace(self) -> bool:
        return bool(self.flags & 0b1000)

    @property
    def is_superscript(self) -> bool:
        return bool(self.flags & 0b1)


@dataclass
class Block:
    """A reading-order unit: a paragraph, caption, heading, equation, etc.

    `text` is the reconstructed paragraph string (lines joined, hyphens healed);
    it is the canonical string used by downstream stages.

    `column` is 0-indexed within the page (0 for single-column pages).
    """
    kind: BlockKind
    spans: list[Span]
    text: str
    page: int
    bbox: BBox
    column: int = 0
    level: int | None = None
    parent_section: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def dominant_size(self) -> float:
        """Character-length-weighted mean of span sizes. Used for heading tests."""
        if not self.spans:
            return 0.0
        total_chars = sum(len(s.text) for s in self.spans)
        if total_chars == 0:
            return self.spans[0].size
        return sum(s.size * len(s.text) for s in self.spans) / total_chars


@dataclass
class FontStats:
    """Document-wide font statistics used for heading thresholds and noise filtering.

    `body_size` is the character-weighted modal span size. `heading_thresholds`
    is a sorted list of sizes observed strictly above body — Phase 2 maps each
    into a heading level.
    """
    body_size: float
    heading_thresholds: list[float] = field(default_factory=list)


@dataclass
class Document:
    """The full paper as a reading-ordered list of blocks with document-wide metadata.

    `page_rects` stores the raw `(x0, y0, x1, y1)` for each page in order, so
    downstream stages can compute per-page margins (header/footer detection)
    and column centers (equation-centering detection) without reopening the PDF.
    """
    blocks: list[Block]
    fonts: FontStats
    page_rects: list[BBox] = field(default_factory=list)
    language: str = "en"
    title: str | None = None
    abstract: str | None = None
    source_path: str | None = None
