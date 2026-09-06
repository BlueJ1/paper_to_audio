"""Phase 9 — optional vision-first mode.

Bypasses Phases 1-6 (extract/classify/filter/tables/equations/inline-math)
by rendering each PDF page as a PNG and asking a vision LLM to produce
clean audiobook narration for the page directly. The resulting narration
blocks are still fed through Phase 7 (`audio_polish`) for numbers /
acronyms / abbreviations, so the downstream surface is identical.

This mode trades fidelity (the VLM may paraphrase) for robustness: it
bypasses every extraction edge case in one shot. Useful for pathological
PDFs (heavy math, unusual layouts, scanned pages) or as a reference
output to diff the symbolic pipeline against.

Cost model: one vision call per page. Callers who want caching should
wrap `vision_llm` with `LLMCache.wrap_vision` (keyed on the image bytes)
before passing it to `vlm_extract_document` — re-running on the same PDF
with an unchanged prompt is then free.

The page -> narration call is wrapped with page-level fail-soft: a
failing VLM call on one page yields a `kind="noise"` block carrying the
error repr under `meta["handler_error"]`, so a single flaky page can't
kill the whole paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

from pipeline.model import BBox, Block, Document, FontStats, Span
from pipeline.polish import PolishPolicy, audio_polish
from pipeline.qa import StageStats, document_stats


VisionLLMCallable = Callable[[bytes, str], str]


DEFAULT_VLM_PROMPT = (
    "Produce clean audiobook narration for this PDF page. Include all "
    "prose verbatim. Describe equations and tables in one or two spoken "
    "sentences each. Skip figures but include their captions. Skip page "
    "numbers, running headers, footers, and footnote reference marks. "
    "Return only the narration text, with no preamble or formatting."
)


@dataclass
class VLMPolicy:
    """Configuration for `vlm_extract_document`.

    `dpi` controls the page render resolution (200 is the sweet spot for
    Gemini-class VLMs — higher pays in tokens without improving reading).
    `prompt` is sent with each page image; override to specialize narration
    style. `skip_pages` is a set of 0-indexed page numbers to skip entirely
    (useful for known cover pages). `max_pages` caps total pages processed
    for cost control; `None` means "all pages".
    """
    dpi: int = 200
    prompt: str = DEFAULT_VLM_PROMPT
    skip_pages: frozenset[int] = field(default_factory=frozenset)
    max_pages: int | None = None
    min_narration_chars: int = 20


def _render_page_png(pdf, page_index: int, dpi: int) -> tuple[bytes, BBox]:
    """Render `pdf[page_index]` at `dpi` and return (png_bytes, page_bbox)."""
    import fitz  # pymupdf, already required
    page = pdf[page_index]
    pix = page.get_pixmap(dpi=dpi)
    png = pix.tobytes("png")
    rect = page.rect
    bbox: BBox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
    return png, bbox


def _placeholder_span(text: str, bbox: BBox, size: float = 10.0) -> Span:
    """Synthesize a single span covering a VLM-narration block.

    VLM narration doesn't come back with span-level font metadata, but
    Phase 7 (and any caller inspecting `block.spans`) expects at least one
    span. We emit a single span covering the page bbox at the document's
    assumed body size so `dominant_size` and friends behave.
    """
    return Span(text=text, font="vlm", size=size, flags=0, bbox=bbox)


def _narrate_page(
    vision_llm: VisionLLMCallable,
    image: bytes,
    prompt: str,
) -> str:
    """Call the VLM and normalize whitespace on the return."""
    out = vision_llm(image, prompt)
    if not isinstance(out, str):
        return ""
    return " ".join(out.split()).strip()


def vlm_extract_document(
    path: str,
    vision_llm: VisionLLMCallable,
    policy: VLMPolicy | None = None,
) -> Document:
    """Render every page of `path` and narrate it with `vision_llm`.

    Returns a `Document` whose blocks are one `kind="body"` block per
    successful page, in page order. Pages that fail (VLM exception, empty
    output, below `min_narration_chars`) become `kind="noise"` blocks
    preserving the error under `meta["handler_error"]` — by design, one
    bad page does not kill the run.

    The returned Document's `FontStats` is a synthesized default (body
    size 10.0, no heading thresholds) since we don't have per-character
    font metadata from the VLM; Phase 7 doesn't need those.
    """
    import fitz  # pymupdf

    policy = policy or VLMPolicy()
    pdf = fitz.open(path)
    try:
        n_pages = len(pdf)
        if policy.max_pages is not None:
            n_pages = min(n_pages, policy.max_pages)

        blocks: list[Block] = []
        page_rects: list[BBox] = []

        for i in range(n_pages):
            try:
                png, bbox = _render_page_png(pdf, i, policy.dpi)
            except Exception as exc:
                page_rects.append((0.0, 0.0, 0.0, 0.0))
                blocks.append(_noise_block(
                    text="",
                    page=i,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    error=repr(exc),
                    stage="render",
                ))
                continue

            page_rects.append(bbox)

            if i in policy.skip_pages:
                continue

            try:
                narration = _narrate_page(vision_llm, png, policy.prompt)
            except Exception as exc:
                blocks.append(_noise_block(
                    text="", page=i, bbox=bbox,
                    error=repr(exc), stage="vlm_call",
                ))
                continue

            if len(narration) < policy.min_narration_chars:
                blocks.append(_noise_block(
                    text=narration, page=i, bbox=bbox,
                    error=f"narration below {policy.min_narration_chars} chars",
                    stage="too_short",
                ))
                continue

            blocks.append(Block(
                kind="body",
                spans=[_placeholder_span(narration, bbox)],
                text=narration,
                page=i,
                bbox=bbox,
                column=0,
                meta={"source": "vlm", "page": i},
            ))
    finally:
        pdf.close()

    return Document(
        blocks=blocks,
        fonts=FontStats(body_size=10.0, heading_thresholds=[]),
        page_rects=page_rects,
        source_path=path,
    )


def _noise_block(text: str, page: int, bbox: BBox, error: str, stage: str) -> Block:
    """Build a fail-soft placeholder block preserving the failure cause."""
    return Block(
        kind="noise",
        spans=[_placeholder_span(text, bbox)] if text else [],
        text=text,
        page=page,
        bbox=bbox,
        column=0,
        meta={
            "source": "vlm",
            "page": page,
            "handler_error": error,
            "vlm_stage": stage,
        },
    )


def run_pipeline_vlm(
    path: str,
    vision_llm: VisionLLMCallable,
    policy: VLMPolicy | None = None,
    polish: PolishPolicy | None = None,
    collect_stats: bool = False,
) -> tuple[Document, list[StageStats]]:
    """Run the VLM-first pipeline on `path`.

    Equivalent to `run_pipeline` but replaces phases 1-6 with a single
    vision pass. Phase 7 (`audio_polish`) still runs over the narration so
    numbers, acronyms, and abbreviations are normalized consistently.

    Returns `(doc, stats)`; `stats` is `[]` unless `collect_stats=True`,
    in which case it carries a `"vlm_extract"` snapshot and an
    `"audio_polish"` snapshot so the caller can feed them to
    `compare_stages`.
    """
    polish = polish or PolishPolicy()
    stats: list[StageStats] = []

    doc = vlm_extract_document(path, vision_llm, policy)
    if collect_stats:
        stats.append(document_stats(doc, "vlm_extract"))

    doc = audio_polish(doc, polish)
    if collect_stats:
        stats.append(document_stats(doc, "audio_polish"))

    return doc, stats


__all__ = [
    "DEFAULT_VLM_PROMPT",
    "VLMPolicy",
    "VisionLLMCallable",
    "run_pipeline_vlm",
    "vlm_extract_document",
]
