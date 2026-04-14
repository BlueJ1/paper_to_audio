"""Minimal Document -> text serializer.

Phase 1 uses this as a diffing tool against the legacy flat-string extraction.
Later phases replace it with a kind-dispatching walker (prose -> prose,
equation -> narrated prose, etc.).
"""
from __future__ import annotations

from pipeline.model import Document


def serialize(doc: Document) -> str:
    """Flatten a `Document` to plain text, one paragraph per block."""
    return "\n\n".join(b.text for b in doc.blocks if b.text.strip())
