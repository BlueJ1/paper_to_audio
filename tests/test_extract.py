"""Phase 1 tests — layout-aware extraction.

Split into two tiers:
- Unit tests run on hand-built fixtures (pymupdf-free) to exercise the pure
  functions: NFC normalization, ligature expansion, line joining with
  hyphenation, column detection, reading order.
- Integration tests extract from real PDFs in `papers/` and assert
  reading-order and structural invariants.
"""
from __future__ import annotations

import os
import unicodedata

import pytest

from pipeline.extract import (
    _analyze_columns,
    _is_full_width,
    _join_line_texts,
    _normalize_span_text,
    _order_page_blocks,
    _compute_font_stats,
    extract_layout,
)
from pipeline.model import Block, Span


PAPERS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "papers")
TWO_COL_PDF = os.path.join(
    PAPERS_DIR, "Self-Attention with Relative Position Representations.pdf"
)
ONE_COL_PDF = os.path.join(PAPERS_DIR, "Titans.pdf")


# ---------------------------------------------------------------------------
# Helpers for building fixture Blocks
# ---------------------------------------------------------------------------

def _block(text: str, bbox, page: int = 0) -> Block:
    return Block(
        kind="body",
        spans=[Span(text=text, font="X", size=10.0, flags=0, bbox=bbox)],
        text=text,
        page=page,
        bbox=bbox,
    )


# ---------------------------------------------------------------------------
# Span normalization
# ---------------------------------------------------------------------------

class TestNormalizeSpanText:
    def test_expands_fi_ligature(self):
        assert _normalize_span_text("eﬃcient") == "efficient"

    def test_expands_fl_ligature(self):
        assert _normalize_span_text("ﬂow") == "flow"

    def test_nfc_precomposes_decomposed_accents(self):
        # decomposed "č" = "c" + U+030C (combining caron)
        decomposed = "c\u030c"
        assert len(decomposed) == 2
        out = _normalize_span_text(decomposed)
        assert out == "č"
        assert out == unicodedata.normalize("NFC", decomposed)

    def test_passthrough_plain_ascii(self):
        assert _normalize_span_text("hello world") == "hello world"


# ---------------------------------------------------------------------------
# Line joining / hyphenation
# ---------------------------------------------------------------------------

class TestJoinLineTexts:
    def test_single_line(self):
        assert _join_line_texts(["hello"]) == "hello"

    def test_joins_with_space(self):
        assert _join_line_texts(["hello", "world"]) == "hello world"

    def test_strips_hyphen_on_linebreak(self):
        # "intro-\nduction" -> "introduction"
        assert _join_line_texts(["intro-", "duction"]) == "introduction"

    def test_keeps_hyphen_for_compound_prefix(self):
        # "self-\nattention" -> "self-attention"
        assert _join_line_texts(["self-", "attention"]) == "self-attention"

    def test_keeps_hyphen_for_compound_suffix(self):
        # "context-\naware" -> "context-aware"
        assert _join_line_texts(["context-", "aware"]) == "context-aware"

    def test_leaves_hyphen_before_uppercase_next(self):
        # "State-\nSomething" — next line starts uppercase -> not a linebreak
        # hyphenation; keep the hyphen and add a space join.
        assert _join_line_texts(["State-", "Something"]) == "State- Something"

    def test_handles_trailing_punctuation(self):
        assert (
            _join_line_texts(["This is a test.", "Next sentence."])
            == "This is a test. Next sentence."
        )


# ---------------------------------------------------------------------------
# Full-width and column analysis
# ---------------------------------------------------------------------------

class TestIsFullWidth:
    def test_centered_title_is_full_width(self):
        page_width = 595.0
        page_mid = page_width / 2
        # Title spans 55% of page width, centered.
        assert _is_full_width((136, 70, 461, 85), page_mid, page_width) is True

    def test_left_column_block_is_not_full_width(self):
        page_width = 595.0
        page_mid = page_width / 2
        assert _is_full_width((72, 270, 290, 403), page_mid, page_width) is False

    def test_right_column_block_is_not_full_width(self):
        page_width = 595.0
        page_mid = page_width / 2
        assert _is_full_width((307, 270, 525, 403), page_mid, page_width) is False


class TestAnalyzeColumns:
    def test_one_column_when_all_blocks_on_one_side(self):
        blocks = [
            _block("a", (50, 100, 500, 120)),  # full-width
            _block("b", (50, 130, 300, 150)),  # left
            _block("c", (50, 160, 300, 180)),  # left
        ]
        layout = _analyze_columns(blocks, page_width=595.0)
        assert layout.two_column is False

    def test_two_column_when_blocks_split_across_midline(self):
        blocks = [
            _block("L1", (50, 100, 280, 120)),
            _block("L2", (50, 130, 280, 150)),
            _block("R1", (310, 100, 540, 120)),
            _block("R2", (310, 130, 540, 150)),
        ]
        layout = _analyze_columns(blocks, page_width=595.0)
        assert layout.two_column is True


class TestOrderPageBlocks:
    def test_two_column_reading_order_with_full_width_title(self):
        """Title spans both columns; body is two-column. Expect:
        title -> left col top-to-bottom -> right col top-to-bottom.
        """
        title = _block("TITLE", (100, 50, 500, 70))
        l1 = _block("L1", (50, 100, 280, 150))
        l2 = _block("L2", (50, 160, 280, 200))
        r1 = _block("R1", (310, 100, 540, 150))
        r2 = _block("R2", (310, 160, 540, 200))
        # Intentionally scrambled input order.
        blocks = [r2, l2, title, r1, l1]
        layout = _analyze_columns(blocks, page_width=595.0)
        assert layout.two_column is True
        ordered = _order_page_blocks(blocks, layout)
        assert [b.text for b in ordered] == ["TITLE", "L1", "L2", "R1", "R2"]

    def test_single_column_sorted_by_y(self):
        b1 = _block("first", (50, 100, 500, 120))
        b2 = _block("second", (50, 200, 500, 220))
        b3 = _block("third", (50, 300, 500, 320))
        layout = _analyze_columns([b1, b2, b3], page_width=595.0)
        assert layout.two_column is False
        ordered = _order_page_blocks([b3, b1, b2], layout)
        assert [b.text for b in ordered] == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# Font stats
# ---------------------------------------------------------------------------

class TestFontStats:
    def test_body_size_is_character_weighted_mode(self):
        # A few big-font short blocks shouldn't outvote a long body-text block.
        big = Block(
            kind="body",
            spans=[Span(text="H", font="X", size=20.0, flags=0, bbox=(0, 0, 1, 1))],
            text="H",
            page=0,
            bbox=(0, 0, 1, 1),
        )
        body = Block(
            kind="body",
            spans=[Span(text="a" * 500, font="X", size=10.0, flags=0, bbox=(0, 0, 1, 1))],
            text="a" * 500,
            page=0,
            bbox=(0, 0, 1, 1),
        )
        stats = _compute_font_stats([big, body, big])
        assert stats.body_size == 10.0
        assert stats.heading_thresholds == [20.0]


# ---------------------------------------------------------------------------
# Integration — real PDFs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(TWO_COL_PDF), reason="sample PDF missing")
class TestTwoColumnPaper:
    @pytest.fixture(scope="class")
    def doc(self):
        return extract_layout(TWO_COL_PDF)

    def test_produces_blocks(self, doc):
        assert len(doc.blocks) > 50

    def test_body_size_is_plausible(self, doc):
        # Body font in a typical 10pt academic paper falls in this range.
        assert 8.5 <= doc.fonts.body_size <= 11.5

    def test_title_is_first_block(self, doc):
        first = doc.blocks[0]
        assert "Self-Attention" in first.text
        assert "Relative Position" in first.text
        # Title sits at the top of page 0 and is styled larger than body.
        assert first.page == 0
        assert first.dominant_size > doc.fonts.body_size

    def test_two_column_reading_preserves_column_order(self, doc):
        """Within a body page, col 0 blocks should be emitted before col 1
        blocks (given reasonable y ranges). A simple invariant: the first
        col-1 block on page 2 comes after the last col-0 block on page 2.
        """
        page2 = [b for b in doc.blocks if b.page == 2]
        cols = [b.column for b in page2]
        # Once we see a column-1 block, no column-0 block should follow.
        seen_one = False
        for c in cols:
            if c == 1:
                seen_one = True
            elif seen_one and c == 0:
                pytest.fail("col-0 block emitted after col-1 block on same page")

    def test_body_paragraphs_are_joined(self, doc):
        """Find a long body paragraph and check that per-line fragments are
        rejoined into a single paragraph (i.e. no mid-paragraph newlines)."""
        long_bodies = [
            b for b in doc.blocks
            if abs(b.dominant_size - doc.fonts.body_size) < 0.5 and len(b.text) > 200
        ]
        assert long_bodies, "expected at least one long body paragraph"
        for b in long_bodies[:5]:
            assert "\n" not in b.text, f"paragraph has newline: {b.text[:80]!r}"

    def test_hyphenation_is_rejoined(self, doc):
        """No paragraph should end a word with '- ' or contain '- <lowercase>'
        from a broken line-break hyphen."""
        for b in doc.blocks:
            # Allow "self-attention" style; reject "intro- duction" style.
            if "- " in b.text:
                # Must be followed by an uppercase letter or another hyphen-safe token.
                for i, ch in enumerate(b.text):
                    if ch == "-" and i + 1 < len(b.text) and b.text[i + 1] == " ":
                        nxt = b.text[i + 2:i + 3]
                        if nxt and nxt.islower():
                            pytest.fail(
                                f"hyphen not rejoined in block: {b.text[max(0,i-20):i+20]!r}"
                            )

    def test_ligatures_are_expanded(self, doc):
        """No paragraph text should contain raw ligature codepoints."""
        raw_ligatures = "\ufb00\ufb01\ufb02\ufb03\ufb04"
        for b in doc.blocks:
            for ch in raw_ligatures:
                assert ch not in b.text, f"raw ligature in block: {b.text[:80]!r}"


@pytest.mark.skipif(not os.path.exists(ONE_COL_PDF), reason="sample PDF missing")
class TestOneColumnPaper:
    @pytest.fixture(scope="class")
    def doc(self):
        return extract_layout(ONE_COL_PDF)

    def test_produces_blocks(self, doc):
        assert len(doc.blocks) > 50

    def test_title_is_first_block(self, doc):
        first = doc.blocks[0]
        assert "Titans" in first.text
        assert first.page == 0

    def test_predominantly_single_column(self, doc):
        """Titans is a single-column paper. Some pages contain genuine
        side-by-side figure captions (e.g. the MAC/MAG/MAL architectures
        figure on p8) that legitimately classify as two-column, so we
        tolerate a small fraction of col-1 blocks rather than asserting
        zero.
        """
        col1_blocks = [b for b in doc.blocks if b.column != 0]
        assert len(col1_blocks) / len(doc.blocks) < 0.05
