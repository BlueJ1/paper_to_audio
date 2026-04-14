"""Phase 4 — table extraction and narration.

After classification and section filtering, this stage opens the source PDF
with pdfplumber, detects tables with the conservative ruled-line strategy,
matches each detected bbox to one or more `Block`s from the pymupdf
extraction, and replaces the matched block(s) with a single `kind="table"`
block whose `meta["table"]` carries the structured `TableData` payload.

Three render modes are available (`TablePolicy.mode`):

- `skip` (default) — emits "A table with N columns and M rows appears here;
  see the paper." Never calls the LLM.
- `verbatim` — reads cells sequentially with column headers prepended. Never
  calls the LLM.
- `prose` — sends the CSV plus caption to an LLM for a bounded, vocab-checked
  summary. Falls back to `skip` if the LLM call fails or the output contains
  invented numbers.

Detection is intentionally conservative: pdfplumber runs with the default
(ruled-line) strategy and detections smaller than 2×2 cells are discarded.
When a detected region overlaps no body-like block, the detection is treated
as a false positive (common for equation fragments) and ignored.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field, replace
from typing import Callable, Literal

from pipeline.model import BBox, Block, Document


TableMode = Literal["skip", "verbatim", "prose"]

# An LLM callable that takes a prompt string and returns narration.
LLMCallable = Callable[[str], str]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TableData:
    """Structured table payload. Stored in `Block.meta["table"]`."""
    rows: list[list[str]]
    page: int
    bbox: BBox
    index: int = 0
    caption: str | None = None

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    @property
    def header(self) -> list[str] | None:
        """First row if it looks like headers — every cell non-empty, mostly non-numeric."""
        if not self.rows:
            return None
        first = self.rows[0]
        if not first or any(not str(c or "").strip() for c in first):
            return None
        total_chars = sum(len(str(c or "")) for c in first)
        digit_chars = sum(ch.isdigit() for c in first for ch in str(c or ""))
        if total_chars and digit_chars / total_chars > 0.5:
            return None
        return [str(c).strip() for c in first]

    def to_csv(self) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        for row in self.rows:
            writer.writerow(["" if c is None else str(c) for c in row])
        return buf.getvalue()


@dataclass
class TablePolicy:
    """Knobs for Phase 4."""
    mode: TableMode = "skip"
    # Fraction of a block's area that must fall inside the table bbox to count.
    min_overlap_ratio: float = 0.5
    # pdfplumber occasionally returns 1×N or N×1 regions that are really
    # equation fragments — drop anything smaller than 2×2.
    min_rows: int = 2
    min_cols: int = 2
    # Verbatim mode reads up to this many rows, then emits a continuation line.
    verbatim_max_rows: int = 20
    # Prose prompts tell the LLM to read at most this many illustrative rows.
    prose_illustrative_rows: int = 3


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def handle_tables(
    doc: Document,
    policy: TablePolicy | None = None,
    llm: LLMCallable | None = None,
) -> Document:
    """Detect tables on the source PDF and replace matching blocks.

    No-op when `doc.source_path` is None (e.g. fixture documents) or when
    pdfplumber fails to parse the file.
    """
    policy = policy or TablePolicy()
    if not doc.source_path:
        return doc
    try:
        raw = _extract_raw_tables(doc.source_path, policy)
    except Exception:
        return doc
    if not raw:
        return doc
    new_blocks = _merge_tables(doc.blocks, raw, policy, llm)
    return replace(doc, blocks=new_blocks)


# ---------------------------------------------------------------------------
# pdfplumber extraction
# ---------------------------------------------------------------------------

def _extract_raw_tables(path: str, policy: TablePolicy) -> list[TableData]:
    import pdfplumber  # imported lazily so the module loads without the dep

    out: list[TableData] = []
    with pdfplumber.open(path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            try:
                finds = page.find_tables()
            except Exception:
                continue
            for t in finds:
                try:
                    rows = t.extract()
                except Exception:
                    continue
                if not rows:
                    continue
                cleaned = [
                    ["" if c is None else str(c) for c in row]
                    for row in rows
                ]
                n_rows = len(cleaned)
                n_cols = max((len(r) for r in cleaned), default=0)
                if n_rows < policy.min_rows or n_cols < policy.min_cols:
                    continue
                bbox = (
                    float(t.bbox[0]),
                    float(t.bbox[1]),
                    float(t.bbox[2]),
                    float(t.bbox[3]),
                )
                out.append(TableData(rows=cleaned, page=page_index, bbox=bbox))
    return out


# ---------------------------------------------------------------------------
# Block matching
# ---------------------------------------------------------------------------

def _merge_tables(
    blocks: list[Block],
    tables: list[TableData],
    policy: TablePolicy,
    llm: LLMCallable | None,
) -> list[Block]:
    """Return a new block list with detected tables collapsed into kind=table blocks."""
    to_drop: set[int] = set()
    primary_for: dict[int, TableData] = {}

    for table in tables:
        matched: list[int] = []
        best_ratio = 0.0
        best_idx: int | None = None
        for idx, b in enumerate(blocks):
            if b.page != table.page:
                continue
            ratio = _overlap_ratio(b.bbox, table.bbox)
            if ratio < policy.min_overlap_ratio:
                continue
            matched.append(idx)
            # Only body-like blocks can be promoted to a table; equations and
            # captions inside the bbox are typically false-positive fragments.
            if b.kind in ("body", "noise") and ratio > best_ratio:
                best_ratio = ratio
                best_idx = idx

        if best_idx is None:
            # pdfplumber flagged a region but nothing body-like lives there.
            continue
        table.caption = _find_caption(table, blocks)
        primary_for[best_idx] = table
        for i in matched:
            if i != best_idx:
                to_drop.add(i)

    # Assign a 1-based reading-order index to each surviving table.
    for i, t in enumerate(
        sorted(primary_for.values(), key=lambda t: (t.page, t.bbox[1])),
        start=1,
    ):
        t.index = i

    out: list[Block] = []
    for idx, b in enumerate(blocks):
        if idx in to_drop:
            continue
        t = primary_for.get(idx)
        if t is None:
            out.append(b)
            continue
        text = render_table(t, policy, llm)
        out.append(replace(b, kind="table", text=text, meta={**b.meta, "table": t}))
    return out


def _overlap_ratio(block_bbox: BBox, table_bbox: BBox) -> float:
    """Fraction of the block's area contained within the table bbox."""
    bx0, by0, bx1, by1 = block_bbox
    tx0, ty0, tx1, ty1 = table_bbox
    iw = max(0.0, min(bx1, tx1) - max(bx0, tx0))
    ih = max(0.0, min(by1, ty1) - max(by0, ty0))
    intersection = iw * ih
    block_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    if block_area <= 0:
        return 0.0
    return intersection / block_area


_TABLE_CAPTION_RE = re.compile(r"^\s*Table\s+\d", re.IGNORECASE)


def _find_caption(table: TableData, blocks: list[Block]) -> str | None:
    """Nearest 'Table N:' caption on the same page, measured by vertical center distance."""
    y_center = (table.bbox[1] + table.bbox[3]) / 2
    candidates: list[tuple[float, str]] = []
    for b in blocks:
        if b.kind != "caption" or b.page != table.page:
            continue
        if not _TABLE_CAPTION_RE.match(b.text):
            continue
        cy = (b.bbox[1] + b.bbox[3]) / 2
        candidates.append((abs(cy - y_center), b.text.strip()))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_table(
    table: TableData,
    policy: TablePolicy,
    llm: LLMCallable | None = None,
) -> str:
    if policy.mode == "verbatim":
        return _render_verbatim(table, policy.verbatim_max_rows)
    if policy.mode == "prose":
        return _render_prose(table, policy, llm)
    return _render_skip(table)


def _render_skip(table: TableData) -> str:
    cap = f" ({table.caption})" if table.caption else ""
    return (
        f"A table with {table.n_cols} columns and {table.n_rows} rows "
        f"appears here{cap}; see the paper."
    )


def _render_verbatim(table: TableData, max_rows: int) -> str:
    lines: list[str] = []
    if table.caption:
        lines.append(table.caption.rstrip(".") + ".")
    header = table.header
    data_rows = table.rows[1:] if header else table.rows
    for row in data_rows[:max_rows]:
        if header:
            pairs = [
                f"{h.strip()}: {(c or '').strip()}"
                for h, c in zip(header, row)
                if (c or "").strip()
            ]
            if pairs:
                lines.append("; ".join(pairs) + ".")
        else:
            cells = [(c or "").strip() for c in row if (c or "").strip()]
            if cells:
                lines.append(", ".join(cells) + ".")
    remaining = len(data_rows) - max_rows
    if remaining > 0:
        lines.append(f"Table continues with {remaining} more rows.")
    if not lines:
        return _render_skip(table)
    return " ".join(lines)


_PROSE_PROMPT_TEMPLATE = """\
You are converting a table into spoken narration for an audiobook.
{caption_line}
Produce a {min_s}-to-{max_s} sentence spoken summary. Name each column and
read at most {illus} illustrative rows. Do not invent data. Do not include
LaTeX, Markdown, or CSV in the output.

CSV:
{csv}
"""


def _render_prose(
    table: TableData,
    policy: TablePolicy,
    llm: LLMCallable | None,
) -> str:
    if llm is None:
        return _render_skip(table)
    max_s = min(6, max(2, table.n_rows // 2))
    caption_line = (
        f'The table title is "{table.caption}".'
        if table.caption
        else "The table has no caption in the paper."
    )
    prompt = _PROSE_PROMPT_TEMPLATE.format(
        caption_line=caption_line,
        min_s=2,
        max_s=max_s,
        illus=policy.prose_illustrative_rows,
        csv=table.to_csv(),
    )
    try:
        out = llm(prompt).strip()
    except Exception:
        return _render_skip(table)
    if not out or not _prose_is_safe(out, table):
        return _render_skip(table)
    return out


_FLOAT_RE = re.compile(r"-?\d+\.\d+")


def _prose_is_safe(prose: str, table: TableData) -> bool:
    """Reject outputs that introduce numeric values absent from the CSV.

    Only decimal tokens are checked: small integers commonly appear as
    lexical counts ('three illustrative rows') that aren't data fabrication.
    """
    csv_text = table.to_csv()
    for tok in _FLOAT_RE.findall(prose):
        if tok not in csv_text:
            return False
    return True
