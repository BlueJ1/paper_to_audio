"""Phase 2 tests — block classification and section tree.

Unit tests use hand-built fixture Blocks to exercise individual rules in
isolation. Integration tests run the full pipeline (extract -> classify)
against the real PDFs in `papers/` and assert structural invariants that
any reasonable classifier should satisfy.
"""
from __future__ import annotations

import os

import pytest

from pipeline.classify import (
    _count_math_chars,
    _is_code,
    _is_equation_display,
    _is_footnote,
    _is_heading,
    _is_noise,
    _normalize_repeating,
    classify_blocks,
)
from pipeline.extract import extract_layout
from pipeline.model import BBox, Block, Document, FontStats, Span


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
    bbox: BBox = (50, 100, 500, 120),
    page: int = 0,
    size: float = 10.0,
    font: str = "Times",
    flags: int = 0,
) -> Block:
    span = Span(text=text, font=font, size=size, flags=flags, bbox=bbox)
    return Block(kind="body", spans=[span], text=text, page=page, bbox=bbox)


def _doc(blocks: list[Block], *, body_size: float = 10.0,
         page_rects: list[BBox] | None = None) -> Document:
    sizes = sorted({round(b.dominant_size, 1) for b in blocks if b.dominant_size > body_size})
    return Document(
        blocks=blocks,
        fonts=FontStats(body_size=body_size, heading_thresholds=sizes),
        page_rects=page_rects or [(0.0, 0.0, 595.0, 842.0)],
    )


# ---------------------------------------------------------------------------
# Individual rule unit tests
# ---------------------------------------------------------------------------

class TestIsHeading:
    def test_large_short_text_is_heading(self):
        b = _block("Introduction", size=14.0)
        assert _is_heading(b, body_size=10.0) is True

    def test_section_numbered_heading(self):
        b = _block("3.1 Relative Position Representations", size=12.0)
        assert _is_heading(b, body_size=10.0) is True

    def test_all_caps_heading(self):
        b = _block("METHODS AND MATERIALS", size=12.0)
        assert _is_heading(b, body_size=10.0) is True

    def test_body_size_paragraph_is_not_heading(self):
        b = _block("The quick brown fox jumps over the lazy dog repeatedly every morning.")
        assert _is_heading(b, body_size=10.0) is False

    def test_long_large_text_is_not_heading(self):
        # Big font but too many words to be a heading (e.g. abstract first line).
        long = " ".join(["word"] * 40)
        b = _block(long, size=13.0)
        assert _is_heading(b, body_size=10.0) is False


class TestIsCode:
    def test_monospace_flag_marks_code(self):
        # Flag bit 3 (0b1000 = 8) = monospace in pymupdf.
        b = _block("for i in range(10):", flags=0b1000)
        assert _is_code(b) is True

    def test_courier_font_name_marks_code(self):
        b = _block("print('hi')", font="Courier-Bold")
        assert _is_code(b) is True

    def test_plain_body_text_is_not_code(self):
        b = _block("This is ordinary prose with no monospace spans.")
        assert _is_code(b) is False


class TestIsFootnote:
    def test_small_font_at_bottom_is_footnote(self):
        b = _block("1 This is a footnote.", size=8.0, bbox=(50, 780, 400, 800))
        page_rects: list[BBox] = [(0.0, 0.0, 595.0, 842.0)]
        assert _is_footnote(b, body_size=10.0, page_rects=page_rects) is True

    def test_small_font_not_at_bottom_is_not_footnote(self):
        b = _block("small text mid-page", size=8.0, bbox=(50, 300, 400, 320))
        page_rects: list[BBox] = [(0.0, 0.0, 595.0, 842.0)]
        assert _is_footnote(b, body_size=10.0, page_rects=page_rects) is False

    def test_body_sized_text_at_bottom_is_not_footnote(self):
        b = _block("This line happens to be near the bottom.",
                   size=10.0, bbox=(50, 780, 400, 800))
        page_rects: list[BBox] = [(0.0, 0.0, 595.0, 842.0)]
        assert _is_footnote(b, body_size=10.0, page_rects=page_rects) is False


class TestIsEquationDisplay:
    def test_equation_number_with_math(self):
        # `x = α + β (3.4)` — equation number at end, Greek letters.
        assert _is_equation_display("x = α + β            (3.4)") is True

    def test_high_math_density(self):
        assert _is_equation_display("∑ᵢ xᵢ = αβ + γ ≤ 1") is True

    def test_prose_is_not_equation(self):
        text = (
            "We evaluate our approach on four machine translation datasets "
            "and report results in the next section."
        )
        assert _is_equation_display(text) is False

    def test_citation_year_not_mistaken_for_eq_number(self):
        # (2019) has 4 digits, so it does not match the equation-number regex.
        assert _is_equation_display("Prior work by Vaswani et al. (2019)") is False


class TestIsNoise:
    def test_short_text_is_noise(self):
        assert _is_noise("42") is True
        assert _is_noise("Fig.") is True

    def test_sentence_is_not_noise(self):
        assert _is_noise("This is a plain sentence with several words.") is False


class TestNormalizeRepeating:
    def test_collapses_digits(self):
        assert _normalize_repeating("Page 3") == _normalize_repeating("Page 42")

    def test_case_and_whitespace_insensitive(self):
        assert _normalize_repeating("  ICLR 2025  ") == _normalize_repeating("iclr 2025")


class TestCountMathChars:
    def test_counts_greek_and_operators(self):
        assert _count_math_chars("αβγ =+") == 5

    def test_plain_text_has_none(self):
        assert _count_math_chars("hello world") == 0


# ---------------------------------------------------------------------------
# classify_blocks end-to-end on fixtures
# ---------------------------------------------------------------------------

class TestClassifyBlocks:
    def test_caption_takes_precedence_over_heading(self):
        # Big font + "Figure 1:" — caption regex wins.
        b = _block("Figure 1: The architecture.", size=12.0)
        classify_blocks(_doc([b]))
        assert b.kind == "caption"

    def test_builds_section_tree(self):
        h1 = _block("1 Introduction", size=12.0)
        p1 = _block("We study X and Y, hoping to discover Z and more.")
        p2 = _block("Another body paragraph continues the introduction here.")
        h2 = _block("2 Method", size=12.0)
        p3 = _block("We propose a novel approach with several parts.")
        doc = _doc([h1, p1, p2, h2, p3])
        classify_blocks(doc)
        assert h1.kind == "heading"
        assert h2.kind == "heading"
        assert p1.kind == "body"
        assert p1.parent_section == "1 Introduction"
        assert p2.parent_section == "1 Introduction"
        assert p3.parent_section == "2 Method"

    def test_heading_levels_from_font_rank(self):
        title = _block("Paper Title", size=18.0)
        sec = _block("1 Section", size=13.0)
        sub = _block("1.1 Subsection", size=11.8)
        body = _block("Plain body paragraph with many words in it.")
        doc = _doc([title, sec, sub, body])
        classify_blocks(doc)
        assert title.level == 1
        assert sec.level == 2
        assert sub.level == 3
        assert body.level is None

    def test_repeating_header_marked_across_pages(self):
        # Same running header on 4 pages near the top of each 842pt page.
        blocks = []
        page_rects: list[BBox] = []
        for p in range(4):
            page_rects.append((0.0, 0.0, 595.0, 842.0))
            hdr = _block(
                "Paper Short Title", bbox=(50, 20, 300, 40), page=p, size=9.0,
            )
            body = _block(
                "Ordinary body paragraph with enough words to avoid noise.",
                bbox=(50, 200 + p, 500, 250 + p),
                page=p,
            )
            blocks.extend([hdr, body])
        doc = _doc(blocks, page_rects=page_rects)
        classify_blocks(doc)
        headers = [b for b in blocks if b.kind == "page_header"]
        assert len(headers) == 4

    def test_non_repeating_top_block_is_not_marked_header(self):
        # A single top-of-page block should not be flagged just for position.
        top = _block("Some one-off note", bbox=(50, 20, 300, 40), size=9.0)
        body = _block("Body paragraph with a reasonable number of words in it.")
        doc = _doc([top, body])
        classify_blocks(doc)
        assert top.kind != "page_header"


# ---------------------------------------------------------------------------
# Integration — real PDFs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(TWO_COL_PDF), reason="sample PDF missing")
class TestTwoColumnPaperClassification:
    @pytest.fixture(scope="class")
    def doc(self):
        return classify_blocks(extract_layout(TWO_COL_PDF))

    def test_all_blocks_have_kind(self, doc):
        for b in doc.blocks:
            assert b.kind is not None

    def test_title_is_heading(self, doc):
        first = doc.blocks[0]
        assert first.kind == "heading"
        assert first.level == 1

    def test_detects_at_least_one_section_heading(self, doc):
        headings = [b for b in doc.blocks if b.kind == "heading"]
        # Title + Abstract + intro + several sections.
        assert len(headings) >= 5

    def test_parent_section_populated_for_body_blocks(self, doc):
        body_blocks = [b for b in doc.blocks if b.kind == "body"]
        with_parent = [b for b in body_blocks if b.parent_section]
        # Not every body block has a preceding heading (abstract precedes the
        # first numbered section in some papers), but most should.
        assert len(with_parent) / max(len(body_blocks), 1) >= 0.7

    def test_detects_some_captions(self, doc):
        captions = [b for b in doc.blocks if b.kind == "caption"]
        # Paper has Figures 1–2 and Tables 1–4; some should survive as caption blocks.
        assert len(captions) >= 3
        for c in captions:
            assert c.text.lower().startswith(("figure", "fig.", "table", "algorithm"))

    def test_references_heading_exists(self, doc):
        headings = [b.text.strip().lower() for b in doc.blocks if b.kind == "heading"]
        assert any("reference" in h for h in headings), (
            "Expected a 'References' heading; Phase 3 will key its skip logic off this."
        )


@pytest.mark.skipif(not os.path.exists(ONE_COL_PDF), reason="sample PDF missing")
class TestOneColumnPaperClassification:
    @pytest.fixture(scope="class")
    def doc(self):
        return classify_blocks(extract_layout(ONE_COL_PDF))

    def test_title_is_heading(self, doc):
        first = doc.blocks[0]
        assert first.kind == "heading"
        assert first.level == 1

    def test_finds_references_section(self, doc):
        headings = [b.text.strip().lower() for b in doc.blocks if b.kind == "heading"]
        assert any("reference" in h for h in headings)

    def test_detects_equation_or_caption_blocks(self, doc):
        # Titans is math-heavy; at least some equation_display or caption
        # blocks should be tagged.
        tagged = [b for b in doc.blocks if b.kind in ("equation_display", "caption")]
        assert len(tagged) >= 3

    def test_body_blocks_are_majority(self, doc):
        body_count = sum(1 for b in doc.blocks if b.kind == "body")
        # Prose is the dominant content in an academic paper.
        assert body_count / len(doc.blocks) >= 0.4
