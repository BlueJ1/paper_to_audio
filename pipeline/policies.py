"""Phase 3 — section filtering and skip policies.

After Phase 2 every block has a `kind` and every non-heading block has a
`parent_section`. Phase 3 turns that into a one-pass filter that drops
layout debris (headers, footers, footnotes) and whole sections the listener
doesn't want in audio (References, Acknowledgments, Appendix, ...).

The policy is a dataclass so it can be configured from CLI, YAML, or the
Flask UI without touching this module. `filter_sections(doc, policy)` is a
pure function — it returns a new `Document` with a filtered block list and
leaves the input untouched, so callers can A/B different policies cheaply.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from pipeline.model import Block, Document


# ---------------------------------------------------------------------------
# Default skip patterns
# ---------------------------------------------------------------------------

# Headings sometimes start with a section number ("6 References", "6. References")
# or an appendix letter ("A Appendix", "A. Technical Details"). The regex allows
# an optional numeric or single-letter prefix on every pattern.
_NUM_PREFIX = r"(?:\d+\.?\s+|[A-Z]\.?\s+)?"

_DEFAULT_REFERENCES = rf"^\s*{_NUM_PREFIX}(references?|bibliograph(?:y|ies)|works?\s+cited)\s*$"
_DEFAULT_ACKNOWLEDGMENTS = rf"^\s*{_NUM_PREFIX}acknowledge?ments?\s*$"
_DEFAULT_AUTHOR_CONTRIB = rf"^\s*{_NUM_PREFIX}author\s+contributions?\s*$"
_DEFAULT_SUPPLEMENTARY = r"^\s*supplement(?:ary|al)(?:\s+material)?.*$"
# "Appendix", "Appendix A", "Appendix A: Proofs", "Appendices", or bare
# letter-numbered "A Proofs" when the heading starts with a single capital.
_DEFAULT_APPENDIX = r"^\s*(?:appendix(?:\s+[A-Z0-9]\S*)?(?:[:\s].*)?|appendices)\s*$"
# ICLR/NeurIPS style: appendix sections lettered rather than named. Two
# typographic conventions seen in practice:
#   "A MULTIHEAD SELF-ATTENTION"                 (ViT — all caps)
#   "C Long-term Memory Module (LMM) as a …"    (Titans — title case)
# Main body sections in the same papers are numbered (1, 2, 3, …), so a
# single capital letter prefix is a reliable appendix signal. Require at
# least two more tokens after the letter to rule out spurious short heads.
_DEFAULT_APPENDIX_LETTER = r"^\s*[A-Z]\s+\S+(?:\s+\S+)+\s*$"


# ---------------------------------------------------------------------------
# Policy dataclass
# ---------------------------------------------------------------------------

@dataclass
class SectionPolicy:
    """User-facing knobs for Phase 3.

    `skip_kinds` drops blocks by their Phase 2 `kind` regardless of section.
    `skip_*` booleans toggle canned section-heading patterns. `extra_skip_patterns`
    lets callers add custom regexes (case-insensitive, matched against the full
    heading text) without disabling the defaults.
    """
    # Block kinds unconditionally dropped.
    skip_kinds: tuple[str, ...] = (
        "page_header", "page_footer", "footnote", "toc", "noise", "figure",
    )
    # Canned section skips.
    skip_references: bool = True
    skip_acknowledgments: bool = True
    skip_author_contributions: bool = True
    skip_supplementary: bool = True
    skip_appendix: bool = True
    # Code is dropped by default; CS tutorials can flip this.
    keep_code: bool = False
    # Extra heading-text patterns (case-insensitive).
    extra_skip_patterns: list[str] = field(default_factory=list)

    def section_patterns(self) -> list[str]:
        """Assemble the effective list of section-heading regex patterns."""
        patterns: list[str] = []
        if self.skip_references:
            patterns.append(_DEFAULT_REFERENCES)
        if self.skip_acknowledgments:
            patterns.append(_DEFAULT_ACKNOWLEDGMENTS)
        if self.skip_author_contributions:
            patterns.append(_DEFAULT_AUTHOR_CONTRIB)
        if self.skip_supplementary:
            patterns.append(_DEFAULT_SUPPLEMENTARY)
        if self.skip_appendix:
            patterns.append(_DEFAULT_APPENDIX)
            patterns.append(_DEFAULT_APPENDIX_LETTER)
        patterns.extend(self.extra_skip_patterns)
        return patterns


# ---------------------------------------------------------------------------
# Filter entry point
# ---------------------------------------------------------------------------

def filter_sections(doc: Document, policy: SectionPolicy | None = None) -> Document:
    """Drop blocks per `policy` and return a new `Document`.

    Section skipping uses heading level to determine where a skipped section
    ends: a skip opened by a heading at level L stays active until another
    heading at level ≤ L. As a fallback (heading levels can be noisy — see
    CLAUDE.md on drop-cap inflation), any non-heading whose `parent_section`
    matches a skip pattern is also dropped.
    """
    if policy is None:
        policy = SectionPolicy()

    patterns = [re.compile(p, re.IGNORECASE) for p in policy.section_patterns()]
    kept: list[Block] = []
    active_skip_level: int | None = None

    for b in doc.blocks:
        if b.kind == "heading":
            # Close an active skip if this heading is at or above its level.
            if active_skip_level is not None and b.level is not None:
                if b.level <= active_skip_level:
                    active_skip_level = None
            # Open a new skip if the heading text matches.
            if _matches_any(b.text.strip(), patterns):
                # Level may be None if Phase 2 couldn't rank it; treat as deep.
                active_skip_level = b.level if b.level is not None else 99
                continue
            if active_skip_level is not None:
                continue
            kept.append(b)
            continue

        # Non-heading block.
        if active_skip_level is not None:
            continue
        if b.kind in policy.skip_kinds:
            continue
        if b.kind == "code" and not policy.keep_code:
            continue
        # Parent-section fallback: guards against cases where a skip heading
        # was missed (e.g. classified as body) but its children still carry
        # the right parent_section. Safe because we only check non-headings.
        if b.parent_section and _matches_any(b.parent_section, patterns):
            continue
        kept.append(b)

    return replace(doc, blocks=kept)


def _matches_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(p.search(text) for p in patterns)
