"""Phase 6 tests — inline math span rewriting.

Unit tests exercise variable-span detection (italic ASCII vs Greek vs
non-italic), standalone-token boundaries (never mark the `x` in `example`),
duplicate-variable ordering, and the three render modes. Integration tests
run the full pipeline on the sample PDFs with symbolic mode and assert
body-block text is changed only when variables were detected.
"""
from __future__ import annotations

import os

import pytest

from pipeline.classify import classify_blocks
from pipeline.extract import extract_layout
from pipeline.inline_math import (
    InlineMathPolicy,
    _collect_variables,
    _find_standalone,
    _is_variable_span,
    _mark_inline_math,
    _MATH_DELIM_CLOSE,
    _MATH_DELIM_OPEN,
    _render_symbolic,
    rewrite_inline_math,
)
from pipeline.model import BBox, Block, Document, FontStats, Span


PAPERS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "papers")
TWO_COL_PDF = os.path.join(
    PAPERS_DIR, "Self-Attention with Relative Position Representations.pdf"
)
ONE_COL_PDF = os.path.join(PAPERS_DIR, "Titans.pdf")


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

ITALIC = 0b10   # pymupdf italic bit
BOLD = 0b10000


def _span(
    text: str,
    *,
    flags: int = 0,
    size: float = 10.0,
    bbox: BBox = (0.0, 100.0, 10.0, 110.0),
    font: str = "CMR10",
) -> Span:
    return Span(text=text, font=font, size=size, flags=flags, bbox=bbox)


def _body(text: str, spans: list[Span], *, page: int = 0) -> Block:
    return Block(
        kind="body", spans=spans, text=text, page=page,
        bbox=(0.0, 100.0, 500.0, 200.0),
    )


def _doc(blocks: list[Block], source_path: str | None = None) -> Document:
    return Document(
        blocks=blocks,
        fonts=FontStats(body_size=10.0, heading_thresholds=[]),
        page_rects=[(0.0, 0.0, 595.0, 842.0)],
        source_path=source_path,
    )


def _opened(s: str) -> str:
    return _MATH_DELIM_OPEN + s + _MATH_DELIM_CLOSE


# ---------------------------------------------------------------------------
# Variable detection
# ---------------------------------------------------------------------------

class TestVariableSpan:
    def test_italic_single_letter_is_variable(self):
        assert _is_variable_span(_span("x", flags=ITALIC)) is True

    def test_non_italic_single_letter_is_not_variable(self):
        # Plain "a" in prose shouldn't count — lots of false positives otherwise.
        assert _is_variable_span(_span("a", flags=0)) is False

    def test_italic_multi_letter_is_not_variable(self):
        # Emphasized word in italic (e.g. "Transformer") is NOT a variable.
        assert _is_variable_span(_span("Transformer", flags=ITALIC)) is False

    def test_greek_letter_is_variable_regardless_of_italic(self):
        assert _is_variable_span(_span("\u03B1", flags=0)) is True   # α
        assert _is_variable_span(_span("\u0394", flags=0)) is True   # Δ

    def test_digit_is_not_variable(self):
        assert _is_variable_span(_span("5", flags=ITALIC)) is False

    def test_empty_span_is_not_variable(self):
        assert _is_variable_span(_span("   ")) is False


# ---------------------------------------------------------------------------
# Standalone token matching
# ---------------------------------------------------------------------------

class TestFindStandalone:
    def test_matches_var_surrounded_by_whitespace(self):
        assert _find_standalone("given x equals y", "x", 0) == 6

    def test_does_not_match_inside_another_word(self):
        # The `x` in `example` should NOT match; the standalone `x` at the
        # end should.
        text = "example x equals y"
        assert _find_standalone(text, "x", 0) == 8

    def test_matches_at_start_of_text(self):
        assert _find_standalone("x is a variable", "x", 0) == 0

    def test_matches_at_end_of_text(self):
        assert _find_standalone("a variable is x", "x", 0) == 14

    def test_returns_minus_one_when_not_present(self):
        assert _find_standalone("nothing here", "x", 0) == -1

    def test_skips_before_start(self):
        text = "x and y and x"
        # start at position 5: should find the second `x` at index 12.
        assert _find_standalone(text, "x", 5) == 12


# ---------------------------------------------------------------------------
# Marking
# ---------------------------------------------------------------------------

class TestMarkInlineMath:
    def test_marks_single_italic_variable(self):
        spans = [
            _span("Given "),
            _span("x", flags=ITALIC),
            _span(" we compute."),
        ]
        b = _body("Given x we compute.", spans)
        out = _mark_inline_math(b)
        assert out == f"Given {_opened('x')} we compute."

    def test_marks_greek_letter(self):
        spans = [
            _span("The rate "),
            _span("\u03B1", flags=0),
            _span(" is small."),
        ]
        b = _body("The rate \u03B1 is small.", spans)
        out = _mark_inline_math(b)
        assert _opened("\u03B1") in out

    def test_marks_multiple_distinct_variables(self):
        spans = [
            _span("Both "),
            _span("x", flags=ITALIC),
            _span(" and "),
            _span("y", flags=ITALIC),
            _span(" are real."),
        ]
        b = _body("Both x and y are real.", spans)
        out = _mark_inline_math(b)
        assert _opened("x") in out
        assert _opened("y") in out

    def test_marks_repeated_variable_sequentially(self):
        spans = [
            _span("For every "),
            _span("x", flags=ITALIC),
            _span(" there is "),
            _span("x", flags=ITALIC),
            _span(" squared."),
        ]
        b = _body("For every x there is x squared.", spans)
        out = _mark_inline_math(b)
        # Both occurrences wrapped, in order.
        assert out.count(_opened("x")) == 2

    def test_no_spans_no_change(self):
        spans = [_span("Just prose with no italic letters.")]
        b = _body("Just prose with no italic letters.", spans)
        assert _mark_inline_math(b) == "Just prose with no italic letters."

    def test_does_not_match_inside_prose_words(self):
        # Italic "x" is a variable, but the `x` in `example` shouldn't match.
        spans = [
            _span("The "),
            _span("example", flags=ITALIC),   # multi-letter, not a var
            _span(" uses "),
            _span("x", flags=ITALIC),
            _span("."),
        ]
        # Note: multi-letter italic is NOT a variable, so only the trailing x matters.
        b = _body("The example uses x.", spans)
        out = _mark_inline_math(b)
        # Only the trailing standalone x should be marked.
        assert out == f"The example uses {_opened('x')}."


# ---------------------------------------------------------------------------
# Rendering — symbolic
# ---------------------------------------------------------------------------

class TestRenderSymbolic:
    def test_greek_is_spelled_out(self):
        marked = f"The {_opened(chr(0x03B1))} rate."
        out = _render_symbolic(marked)
        assert "alpha" in out
        assert _MATH_DELIM_OPEN not in out

    def test_ascii_var_passes_through_without_delimiters(self):
        marked = f"Let {_opened('x')} be a value."
        out = _render_symbolic(marked)
        assert out == "Let x be a value."

    def test_multiple_vars_in_one_paragraph(self):
        marked = f"Both {_opened('x')} and {_opened(chr(0x03B2))} appear."
        out = _render_symbolic(marked)
        assert "beta" in out
        assert _MATH_DELIM_OPEN not in out
        assert "x" in out


# ---------------------------------------------------------------------------
# Rendering — LLM mode
# ---------------------------------------------------------------------------

class TestRenderLLM:
    def _body_with_var(self, text_with_var: str = "Given x we compute y."):
        spans = [
            _span("Given "),
            _span("x", flags=ITALIC),
            _span(" we compute "),
            _span("y", flags=ITALIC),
            _span("."),
        ]
        return _body(text_with_var, spans)

    def test_llm_output_used_when_well_formed(self):
        b = self._body_with_var()
        doc = _doc([b])
        out_doc = rewrite_inline_math(
            doc,
            InlineMathPolicy(mode="llm", min_runs_for_llm=1, min_output_ratio=0.0),
            llm=lambda _: '["x", "y"]',
        )
        assert out_doc.blocks[0].text == "Given x we compute y."
        # No delimiters leak through.
        assert _MATH_DELIM_OPEN not in out_doc.blocks[0].text

    def test_llm_failure_falls_back_to_symbolic(self):
        b = self._body_with_var()
        doc = _doc([b])
        def broken(_):
            raise RuntimeError("API down")
        out_doc = rewrite_inline_math(
            doc,
            InlineMathPolicy(mode="llm"),
            llm=broken,
        )
        # Symbolic fallback: delimiters stripped, text essentially unchanged
        # for ASCII italics.
        assert out_doc.blocks[0].text == "Given x we compute y."

    def test_llm_output_stripped_of_residual_delimiters(self):
        b = self._body_with_var()
        doc = _doc([b])
        llm_out = f"Given {_MATH_DELIM_OPEN}x{_MATH_DELIM_CLOSE} compute y, the result is clear enough for readers."
        out_doc = rewrite_inline_math(
            doc,
            InlineMathPolicy(mode="llm", min_output_ratio=0.0),
            llm=lambda _: llm_out,
        )
        assert _MATH_DELIM_OPEN not in out_doc.blocks[0].text
        assert _MATH_DELIM_CLOSE not in out_doc.blocks[0].text

    def test_llm_short_output_triggers_fallback(self):
        b = self._body_with_var()
        doc = _doc([b])
        out_doc = rewrite_inline_math(
            doc,
            InlineMathPolicy(mode="llm", min_output_ratio=10.0),  # impossibly high
            llm=lambda _: "short",
        )
        # Fell back to symbolic → original text shape preserved.
        assert out_doc.blocks[0].text == "Given x we compute y."


# ---------------------------------------------------------------------------
# rewrite_inline_math — Document-level
# ---------------------------------------------------------------------------

class TestRewriteInlineMath:
    def test_skip_mode_returns_same_document(self):
        b = _body(
            "Given x we compute y.",
            [
                _span("Given "), _span("x", flags=ITALIC),
                _span(" we compute "), _span("y", flags=ITALIC), _span("."),
            ],
        )
        doc = _doc([b])
        out = rewrite_inline_math(doc, InlineMathPolicy(mode="skip"))
        assert out is doc

    def test_non_body_blocks_untouched(self):
        h = Block(
            kind="heading", spans=[_span("x", flags=ITALIC)],
            text="x", page=0, bbox=(0, 0, 10, 20),
        )
        doc = _doc([h])
        out = rewrite_inline_math(doc, InlineMathPolicy(mode="symbolic"))
        assert out.blocks[0] is h  # identity preserved for non-body

    def test_body_without_vars_untouched(self):
        b = _body("Plain prose only.", [_span("Plain prose only.")])
        doc = _doc([b])
        out = rewrite_inline_math(doc, InlineMathPolicy(mode="symbolic"))
        assert out.blocks[0] is b

    def test_meta_records_marked_form(self):
        b = _body(
            "Given x.",
            [_span("Given "), _span("x", flags=ITALIC), _span(".")],
        )
        doc = _doc([b])
        out = rewrite_inline_math(doc, InlineMathPolicy(mode="symbolic"))
        assert _opened("x") in out.blocks[0].meta["inline_math"]

    def test_purity(self):
        b = _body(
            "Given x.",
            [_span("Given "), _span("x", flags=ITALIC), _span(".")],
        )
        doc = _doc([b])
        original_text = b.text
        rewrite_inline_math(doc, InlineMathPolicy(mode="symbolic"))
        assert b.text == original_text
        assert "inline_math" not in b.meta


# ---------------------------------------------------------------------------
# Collect variables
# ---------------------------------------------------------------------------

class TestCollectVariables:
    def test_returns_in_document_order(self):
        spans = [
            _span("Given "),
            _span("x", flags=ITALIC),
            _span(" and "),
            _span("y", flags=ITALIC),
            _span("."),
        ]
        assert _collect_variables(spans) == ["x", "y"]

    def test_empty_when_no_italics(self):
        spans = [_span("Plain text.")]
        assert _collect_variables(spans) == []


# ---------------------------------------------------------------------------
# Integration — real PDFs, symbolic mode (no LLM required)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(TWO_COL_PDF), reason="sample PDF missing")
class TestTwoColumnPaperInlineMath:
    @pytest.fixture(scope="class")
    def doc(self):
        d = classify_blocks(extract_layout(TWO_COL_PDF))
        return rewrite_inline_math(d, InlineMathPolicy(mode="symbolic"))

    def test_at_least_one_body_block_has_inline_math_meta(self, doc):
        marked = [b for b in doc.blocks if "inline_math" in b.meta]
        assert len(marked) >= 1

    def test_marked_text_contains_delimiters_in_meta(self, doc):
        for b in doc.blocks:
            if "inline_math" in b.meta:
                assert _MATH_DELIM_OPEN in b.meta["inline_math"]

    def test_rendered_text_has_no_residual_delimiters(self, doc):
        for b in doc.blocks:
            assert _MATH_DELIM_OPEN not in b.text
            assert _MATH_DELIM_CLOSE not in b.text


@pytest.mark.skipif(not os.path.exists(ONE_COL_PDF), reason="sample PDF missing")
class TestOneColumnPaperInlineMath:
    @pytest.fixture(scope="class")
    def doc(self):
        d = classify_blocks(extract_layout(ONE_COL_PDF))
        return rewrite_inline_math(d, InlineMathPolicy(mode="symbolic"))

    def test_greek_letters_spelled_out_somewhere(self, doc):
        # Titans has numerous Greek letters (theta, alpha, ...). At least one
        # body block should have its ⟦α⟧ markup turned into "alpha" etc.
        combined = "\n".join(b.text for b in doc.blocks if b.kind == "body")
        greek_names = ("alpha", "beta", "gamma", "theta", "lambda", "sigma")
        assert any(name in combined for name in greek_names)
