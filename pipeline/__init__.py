"""Layout-aware PDF preprocessing pipeline.

Each stage is a pure function Document -> Document (see ADVANCED_PIPELINE_PLAN.md).
Phase 1 implemented: extraction with column detection and paragraph reconstruction.
"""

from pipeline.model import Block, BlockKind, Document, FontStats, Span
from pipeline.extract import extract_layout, load_pdf
from pipeline.classify import classify_blocks
from pipeline.policies import SectionPolicy, filter_sections
from pipeline.serialize import serialize
from pipeline.tables import TableData, TablePolicy, handle_tables, render_table

__all__ = [
    "Block",
    "BlockKind",
    "Document",
    "FontStats",
    "SectionPolicy",
    "Span",
    "TableData",
    "TablePolicy",
    "classify_blocks",
    "extract_layout",
    "filter_sections",
    "handle_tables",
    "load_pdf",
    "render_table",
    "serialize",
]
