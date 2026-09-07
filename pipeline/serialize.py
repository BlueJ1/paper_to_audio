"""Final narration eligibility, independent of earlier filtering stages."""
from pipeline.model import Block, Document

REJECTED_KINDS = frozenset({"noise", "figure", "page_header", "page_footer", "toc", "footnote"})


def is_narratable(block: Block) -> bool:
    return bool(block.text.strip() and block.kind not in REJECTED_KINDS
                and not block.meta.get("handler_error"))


def serialize(doc: Document) -> str:
    """Keep diagnostics in the document, but never speak rejected content."""
    return "\n\n".join(b.text for b in doc.blocks if is_narratable(b))


def require_narration(text: str) -> str:
    if not text.strip():
        raise ValueError("No usable narration was extracted. For scanned PDFs, use OCR first "
                         "or explicitly select the vision library API; no paid processing was started automatically.")
    return text
