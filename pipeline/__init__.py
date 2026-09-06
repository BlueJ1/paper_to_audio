"""Layout-aware PDF preprocessing pipeline.

Each stage is a pure function Document -> Document (see ADVANCED_PIPELINE_PLAN.md).
Phase 1 implemented: extraction with column detection and paragraph reconstruction.
"""

from pipeline.model import Block, BlockKind, Document, FontStats, Span
from pipeline.extract import extract_layout, load_pdf
from pipeline.classify import classify_blocks
from pipeline.policies import SectionPolicy, filter_sections
from pipeline.equations import (
    EquationData,
    EquationPolicy,
    handle_equations,
    render_equation,
)
from pipeline.inline_math import InlineMathPolicy, rewrite_inline_math
from pipeline.polish import PolishPolicy, audio_polish, to_ssml
from pipeline.qa import (
    LLMCache,
    PipelineConfig,
    StageStats,
    compare_stages,
    document_stats,
    document_to_dict,
    document_to_json,
    run_pipeline,
    safe_block,
)
from pipeline.serialize import serialize
from pipeline.tables import TableData, TablePolicy, handle_tables, render_table
from pipeline.vlm import (
    DEFAULT_VLM_PROMPT,
    VLMPolicy,
    run_pipeline_vlm,
    vlm_extract_document,
)

__all__ = [
    "Block",
    "BlockKind",
    "DEFAULT_VLM_PROMPT",
    "Document",
    "EquationData",
    "EquationPolicy",
    "FontStats",
    "InlineMathPolicy",
    "LLMCache",
    "PipelineConfig",
    "PolishPolicy",
    "SectionPolicy",
    "Span",
    "StageStats",
    "TableData",
    "TablePolicy",
    "VLMPolicy",
    "audio_polish",
    "classify_blocks",
    "compare_stages",
    "document_stats",
    "document_to_dict",
    "document_to_json",
    "extract_layout",
    "filter_sections",
    "handle_equations",
    "handle_tables",
    "load_pdf",
    "render_equation",
    "render_table",
    "rewrite_inline_math",
    "run_pipeline",
    "run_pipeline_vlm",
    "safe_block",
    "serialize",
    "to_ssml",
    "vlm_extract_document",
]
