"""Phase 3 tests — section filtering and skip policies.

Unit tests build hand-crafted `Document`s to exercise the filter rules in
isolation. Integration tests run the full pipeline (extract → classify →
filter) against the real PDFs and assert the References/Appendix/footnote
debris is actually gone.
"""
from __future__ import annotations

import os

import pytest

from pipeline.classify import classify_blocks
from pipeline.extract import extract_layout
from pipeline.model import BBox, Block, Document, FontStats, Span
from pipeline.policies import SectionPolicy, filter_sections


PAPERS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "papers")
TWO_COL_PDF = os.path.join(
    PAPERS_DIR, "Self-Attention with Relative Position Representations.pdf"
)
ONE_COL_PDF = os.path.join(PAPERS_DIR, "Titans.pdf")


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _block(
    text: str,
    *,
    kind: str = "body",
    level: int | None = None,
    parent_section: str | None = None,
    bbox: BBox = (50, 100, 500, 120),
    page: int = 0,
    size: float = 10.0,
    font: str = "Times",
    flags: int = 0,
) -> Block:
    span = Span(text=text, font=font, size=size, flags=flags, bbox=bbox)
    b = Block(kind=kind, spans=[span], text=text, page=page, bbox=bbox)
    b.level = level
    b.parent_section = parent_section
    return b


def _doc(blocks: list[Block]) -> Document:
    return Document(
        blocks=blocks,
        fonts=FontStats(body_size=10.0, heading_thresholds=[]),
        page_rects=[(0.0, 0.0, 595.0, 842.0)],
    )


# ---------------------------------------------------------------------------
# Policy pattern assembly
# ---------------------------------------------------------------------------

class TestSectionPatterns:
    def test_default_policy_covers_canonical_sections(self):
        p = SectionPolicy()
        assert len(p.section_patterns()) >= 5

    def test_disabling_skip_removes_pattern(self):
        p = SectionPolicy(skip_appendix=False)
        all_patterns = " ".join(p.section_patterns())
        assert "appendix" not in all_patterns.lower()

    def test_extra_patterns_are_appended(self):
        p = SectionPolicy(extra_skip_patterns=[r"^\s*Related Work\s*$"])
        assert any("Related Work" in pat for pat in p.section_patterns())


# ---------------------------------------------------------------------------
# filter_sections — per-rule unit tests
# ---------------------------------------------------------------------------

class TestFilterByKind:
    def test_drops_page_header(self):
        blocks = [
            _block("Paper Title", kind="heading", level=1),
            _block("running header", kind="page_header"),
            _block("Body paragraph text goes here with plenty of words."),
        ]
        out = filter_sections(_doc(blocks))
        assert all(b.kind != "page_header" for b in out.blocks)
        assert len(out.blocks) == 2

    def test_drops_footnote_and_noise(self):
        blocks = [
            _block("Body paragraph one continues to make a point about things."),
            _block("1 This is a footnote.", kind="footnote"),
            _block("Body paragraph two carries on with a separate thought."),
            _block("42", kind="noise"),
        ]
        out = filter_sections(_doc(blocks))
        assert [b.kind for b in out.blocks] == ["body", "body"]

    def test_keeps_captions_and_equations(self):
        blocks = [
            _block("Figure 1: The architecture diagram.", kind="caption"),
            _block("x = y + z     (3)", kind="equation_display"),
            _block("Ordinary body text for a paragraph with many words."),
        ]
        out = filter_sections(_doc(blocks))
        kinds = [b.kind for b in out.blocks]
        assert kinds == ["caption", "equation_display", "body"]

    def test_code_dropped_by_default_and_kept_when_configured(self):
        blocks = [
            _block("print('hi')", kind="code"),
            _block("Prose paragraph with several normal words in it."),
        ]
        out = filter_sections(_doc(blocks))
        assert all(b.kind != "code" for b in out.blocks)

        out2 = filter_sections(_doc(blocks), SectionPolicy(keep_code=True))
        assert any(b.kind == "code" for b in out2.blocks)


class TestFilterBySection:
    def test_drops_entire_references_section(self):
        blocks = [
            _block("1 Introduction", kind="heading", level=1),
            _block("Intro paragraph with a healthy count of words in it.",
                   parent_section="1 Introduction"),
            _block("References", kind="heading", level=1),
            _block("Author A, 2020. Paper title. Journal.",
                   parent_section="References"),
            _block("Author B, 2021. Another paper title. Journal.",
                   parent_section="References"),
        ]
        out = filter_sections(_doc(blocks))
        texts = [b.text for b in out.blocks]
        assert "References" not in texts
        assert not any("Author A" in t for t in texts)
        assert not any("Author B" in t for t in texts)

    def test_numbered_references_heading_matches(self):
        # "6 References" with a section number prefix.
        blocks = [
            _block("6 References", kind="heading", level=1),
            _block("Author A, 2020.", parent_section="6 References"),
        ]
        out = filter_sections(_doc(blocks))
        assert out.blocks == []

    def test_acknowledgments_and_appendix_dropped(self):
        blocks = [
            _block("Conclusion paragraph rounds out the main body nicely here."),
            _block("Acknowledgments", kind="heading", level=1),
            _block("We thank X and Y for useful feedback.",
                   parent_section="Acknowledgments"),
            _block("Appendix A", kind="heading", level=1),
            _block("Proof details follow here and there.",
                   parent_section="Appendix A"),
        ]
        out = filter_sections(_doc(blocks))
        assert len(out.blocks) == 1
        assert out.blocks[0].text.startswith("Conclusion")

    def test_iclr_letter_prefixed_appendix_dropped(self):
        """ICLR/NeurIPS appendix style: lettered, not named 'Appendix'."""
        blocks = [
            _block("1 Introduction", kind="heading", level=1),
            _block("Main body paragraph with plenty of words in it here."),
            # All-caps form (ViT)
            _block("A MULTIHEAD SELF-ATTENTION", kind="heading", level=1),
            _block("Appendix A body content.",
                   parent_section="A MULTIHEAD SELF-ATTENTION"),
            # Title-case form (Titans)
            _block("C Long-term Memory Module (LMM) as a Sequence Model",
                   kind="heading", level=1),
            _block("Appendix C body content.",
                   parent_section="C Long-term Memory Module (LMM) as a Sequence Model"),
        ]
        out = filter_sections(_doc(blocks))
        assert len(out.blocks) == 2
        assert out.blocks[1].text.startswith("Main body")

    def test_numbered_section_not_treated_as_appendix(self):
        """A numeric-prefixed heading must NOT match the appendix-letter pattern."""
        blocks = [
            _block("2 RELATED WORK", kind="heading", level=1),
            _block("Body of section 2.", parent_section="2 RELATED WORK"),
        ]
        out = filter_sections(_doc(blocks))
        # Heading and its body both kept — numbered, not lettered.
        assert len(out.blocks) == 2

    def test_keep_appendix_disables_appendix_skip(self):
        blocks = [
            _block("Appendix A", kind="heading", level=1),
            _block("Proof details follow here.", parent_section="Appendix A"),
        ]
        out = filter_sections(_doc(blocks), SectionPolicy(skip_appendix=False))
        # The heading and its body should both survive.
        assert len(out.blocks) == 2

    def test_nested_subsection_in_skipped_section_stays_skipped(self):
        blocks = [
            _block("References", kind="heading", level=1),
            _block("A.1 Sub-refs", kind="heading", level=2),
            _block("Some nested reference entry.", parent_section="A.1 Sub-refs"),
            _block("2 Next Real Section", kind="heading", level=1),
            _block("Body of next section resumes with several words.",
                   parent_section="2 Next Real Section"),
        ]
        out = filter_sections(_doc(blocks))
        # References, its subsection, and its subsection's body must all be gone.
        assert len(out.blocks) == 2
        assert out.blocks[0].text == "2 Next Real Section"

    def test_sibling_heading_closes_skip(self):
        blocks = [
            _block("References", kind="heading", level=1),
            _block("ref entry", parent_section="References"),
            _block("Discussion", kind="heading", level=1),
            _block("Discussion paragraph with several words of real content.",
                   parent_section="Discussion"),
        ]
        out = filter_sections(_doc(blocks))
        assert [b.text for b in out.blocks] == [
            "Discussion",
            "Discussion paragraph with several words of real content.",
        ]

    def test_parent_section_fallback_when_heading_misclassified(self):
        # Phase 2 could miss a heading; if a body block's parent_section still
        # matches the skip pattern, we drop it anyway.
        blocks = [
            _block("Reference entry that slipped through.",
                   parent_section="References"),
            _block("Normal body paragraph with enough words to matter.",
                   parent_section="1 Introduction"),
        ]
        out = filter_sections(_doc(blocks))
        assert len(out.blocks) == 1
        assert out.blocks[0].text.startswith("Normal body")

    def test_case_insensitive_matching(self):
        blocks = [
            _block("BIBLIOGRAPHY", kind="heading", level=1),
            _block("[1] Foo, 2020.", parent_section="BIBLIOGRAPHY"),
        ]
        out = filter_sections(_doc(blocks))
        assert out.blocks == []

    def test_extra_pattern_is_respected(self):
        blocks = [
            _block("Related Work", kind="heading", level=1),
            _block("Prior papers have explored this space in depth.",
                   parent_section="Related Work"),
            _block("Keep This", kind="heading", level=1),
            _block("Kept body paragraph with enough words for a match.",
                   parent_section="Keep This"),
        ]
        policy = SectionPolicy(extra_skip_patterns=[r"^\s*Related Work\s*$"])
        out = filter_sections(_doc(blocks), policy)
        texts = [b.text for b in out.blocks]
        assert "Related Work" not in texts
        assert "Keep This" in texts


class TestPurity:
    def test_input_doc_is_not_mutated(self):
        blocks = [
            _block("Body paragraph with several plain words."),
            _block("Noise", kind="noise"),
        ]
        doc = _doc(blocks)
        original_ids = [id(b) for b in doc.blocks]
        filter_sections(doc)
        assert [id(b) for b in doc.blocks] == original_ids


# ---------------------------------------------------------------------------
# Integration — real PDFs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(TWO_COL_PDF), reason="sample PDF missing")
class TestTwoColumnPaperFilter:
    @pytest.fixture(scope="class")
    def filtered(self):
        doc = classify_blocks(extract_layout(TWO_COL_PDF))
        return filter_sections(doc)

    @pytest.fixture(scope="class")
    def unfiltered(self):
        return classify_blocks(extract_layout(TWO_COL_PDF))

    def test_references_heading_is_gone(self, filtered):
        for b in filtered.blocks:
            if b.kind == "heading":
                assert "reference" not in b.text.lower()

    def test_reference_body_blocks_are_gone(self, filtered):
        for b in filtered.blocks:
            if b.parent_section:
                assert "reference" not in b.parent_section.lower()

    def test_no_page_headers_or_footnotes_remain(self, filtered):
        dropped_kinds = {"page_header", "page_footer", "footnote", "noise", "toc"}
        for b in filtered.blocks:
            assert b.kind not in dropped_kinds

    def test_body_prose_still_dominates(self, filtered):
        body = sum(1 for b in filtered.blocks if b.kind == "body")
        assert body / max(len(filtered.blocks), 1) >= 0.5

    def test_filter_reduces_block_count(self, filtered, unfiltered):
        # References + footnotes + page headers should drop a meaningful chunk.
        assert len(filtered.blocks) < len(unfiltered.blocks)


@pytest.mark.skipif(not os.path.exists(ONE_COL_PDF), reason="sample PDF missing")
class TestOneColumnPaperFilter:
    @pytest.fixture(scope="class")
    def filtered(self):
        doc = classify_blocks(extract_layout(ONE_COL_PDF))
        return filter_sections(doc)

    def test_references_heading_is_gone(self, filtered):
        for b in filtered.blocks:
            if b.kind == "heading":
                assert "reference" not in b.text.lower()

    def test_acknowledgments_gone_if_present(self, filtered):
        for b in filtered.blocks:
            if b.parent_section:
                assert "acknowledg" not in b.parent_section.lower()
            if b.kind == "heading":
                assert "acknowledg" not in b.text.lower()

    def test_title_heading_survives(self, filtered):
        # Title is level 1 at position 0; filter must not drop it.
        assert filtered.blocks[0].kind == "heading"
