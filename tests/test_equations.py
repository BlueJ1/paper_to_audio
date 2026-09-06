"""Phase 5 tests — display equation narration.

Unit tests exercise equation-number extraction, digit-to-words, the symbolic
rewrite, span-aware sub/superscript reconstruction, and all three render
modes (skip / symbolic / vision) including their failure fallbacks.

Integration tests run extract -> classify -> handle_equations on the sample
PDFs and assert at least one equation block is narrated (without calling a
real LLM — skip mode only).
"""
from __future__ import annotations

import os

import pytest

from pipeline.classify import classify_blocks
from pipeline.extract import extract_layout
from pipeline.equations import (
    EquationData,
    EquationPolicy,
    _extract_equation_number,
    _prepend_equation_number,
    _reconstruct_with_scripts,
    _sanitize_narration,
    _spoken_number,
    _symbolic_rewrite,
    handle_equations,
    render_equation,
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

def _span(
    text: str,
    *,
    size: float = 10.0,
    flags: int = 0,
    bbox: BBox = (0.0, 100.0, 20.0, 110.0),
    font: str = "CMR10",
) -> Span:
    return Span(text=text, font=font, size=size, flags=flags, bbox=bbox)


def _eq_block(
    text: str,
    *,
    spans: list[Span] | None = None,
    bbox: BBox = (100.0, 200.0, 400.0, 220.0),
    page: int = 0,
) -> Block:
    if spans is None:
        spans = [_span(text)]
    return Block(kind="equation_display", spans=spans, text=text, page=page, bbox=bbox)


def _doc(blocks: list[Block], source_path: str | None = None) -> Document:
    return Document(
        blocks=blocks,
        fonts=FontStats(body_size=10.0, heading_thresholds=[]),
        page_rects=[(0.0, 0.0, 595.0, 842.0)],
        source_path=source_path,
    )


# ---------------------------------------------------------------------------
# Number parsing
# ---------------------------------------------------------------------------

class TestEquationNumber:
    def test_trailing_simple_number(self):
        body, n = _extract_equation_number("x = f(y) (3)")
        assert n == "3"
        assert body == "x = f(y)"

    def test_trailing_dotted_number(self):
        body, n = _extract_equation_number("h_t = g(h_{t-1}) (3.4)")
        assert n == "3.4"
        assert body == "h_t = g(h_{t-1})"

    def test_no_number(self):
        body, n = _extract_equation_number("x + y = z")
        assert n is None
        assert body == "x + y = z"

    def test_citation_year_not_matched_because_mid_text(self):
        # Eq-number regex is anchored at end, so a mid-text year should not match.
        body, n = _extract_equation_number("see Vaswani (2017) and x = y")
        assert n is None

    def test_spoken_number_integer(self):
        assert _spoken_number("3") == "three"

    def test_spoken_number_decimal(self):
        assert _spoken_number("3.4") == "three point four"

    def test_spoken_number_two_digits(self):
        assert _spoken_number("12") == "one two"


# ---------------------------------------------------------------------------
# Symbolic rewrite
# ---------------------------------------------------------------------------

class TestSymbolicRewrite:
    def test_greek_letters_spelled_out(self):
        out = _symbolic_rewrite("\u03B1 + \u03B2 = \u03B3")
        assert "alpha" in out
        assert "beta" in out
        assert "gamma" in out

    def test_unicode_operators_mapped(self):
        out = _symbolic_rewrite("x \u2264 y \u2265 z")
        assert "less than or equal to" in out
        assert "greater than or equal to" in out

    def test_ascii_equals_spoken(self):
        out = _symbolic_rewrite("a = b")
        assert "equals" in out

    def test_superscript_notation_becomes_to_the(self):
        out = _symbolic_rewrite("x^2 + y^n")
        assert "to the 2" in out
        assert "to the n" in out

    def test_subscript_notation_becomes_sub(self):
        out = _symbolic_rewrite("h_t + x_i")
        assert "sub t" in out
        assert "sub i" in out

    def test_unicode_superscript_digits(self):
        out = _symbolic_rewrite("x\u00B2")  # x²
        assert "to the 2" in out

    def test_unicode_subscript_digits(self):
        out = _symbolic_rewrite("x\u2081")  # x₁
        assert "sub 1" in out

    def test_whitespace_is_polished(self):
        out = _symbolic_rewrite("  x   +   y  ")
        assert out == "x + y"


# ---------------------------------------------------------------------------
# Span-aware script reconstruction
# ---------------------------------------------------------------------------

class TestReconstructScripts:
    def test_uses_superscript_flag(self):
        # span with bit 0 (superscript) set, at smaller size.
        base = _span("x", size=10.0, bbox=(100, 100, 110, 110))
        sup = _span("2", size=7.0, flags=0b1, bbox=(110, 95, 116, 105))
        block = _eq_block("x2", spans=[base, sup])
        out = _reconstruct_with_scripts(block, number=None)
        assert "^2" in out

    def test_infers_subscript_from_y_offset(self):
        # Smaller span sitting below the majority-baseline -> subscript.
        # Provide several baseline spans so the median resolves to the baseline.
        baseline_bbox = (0.0, 100.0, 10.0, 110.0)  # y-center 105
        baseline = [
            _span("x", size=10.0, bbox=(100, 100, 110, 110)),
            _span(" = ", size=10.0, bbox=(120, 100, 140, 110)),
            _span("f(y)", size=10.0, bbox=(150, 100, 190, 110)),
        ]
        sub = _span("t", size=7.0, bbox=(110, 108, 116, 116))  # smaller + lower
        block = _eq_block(
            "x_t = f(y)", spans=[baseline[0], sub, *baseline[1:]],
            bbox=(100, 100, 190, 116),
        )
        out = _reconstruct_with_scripts(block, number=None)
        assert "_t" in out

    def test_returns_empty_when_no_scripts_detected(self):
        spans = [_span("x", size=10.0), _span("=", size=10.0), _span("y", size=10.0)]
        block = _eq_block("x = y", spans=spans)
        assert _reconstruct_with_scripts(block, number=None) == ""

    def test_skips_equation_number_span(self):
        spans = [
            _span("x", size=10.0, bbox=(100, 100, 110, 110)),
            _span(" = y ", size=10.0, bbox=(110, 100, 160, 110)),
            _span("(3)", size=10.0, bbox=(160, 100, 180, 110)),
        ]
        block = _eq_block("x = y (3)", spans=spans)
        out = _reconstruct_with_scripts(block, number="3")
        assert "(3)" not in out


# ---------------------------------------------------------------------------
# Skip renderer
# ---------------------------------------------------------------------------

class TestRenderSkip:
    def test_prefixes_equation_number(self):
        data = EquationData(
            raw_text="x = y", symbolic="x equals y", number="3.4",
            page=0, bbox=(0, 0, 100, 20),
        )
        out = render_equation(data, EquationPolicy(mode="skip"))
        assert out.startswith("Equation three point four:")
        assert "x equals y" in out

    def test_works_without_equation_number(self):
        data = EquationData(
            raw_text="x = y", symbolic="x equals y", number=None,
            page=0, bbox=(0, 0, 100, 20),
        )
        out = render_equation(data, EquationPolicy(mode="skip"))
        assert "Equation" not in out
        assert "x equals y" in out

    def test_empty_symbolic_produces_placeholder(self):
        data = EquationData(
            raw_text="", symbolic="", number="1",
            page=0, bbox=(0, 0, 100, 20),
        )
        out = render_equation(data, EquationPolicy(mode="skip"))
        assert "appears here" in out


# ---------------------------------------------------------------------------
# Symbolic-mode renderer (LLM path)
# ---------------------------------------------------------------------------

class TestRenderSymbolic:
    def test_uses_llm_output_when_available(self):
        data = EquationData(
            raw_text="h_t = f(h_{t-1}, x_t)",
            symbolic="h sub t equals f of h sub t minus one and x sub t",
            number="3.4", page=0, bbox=(0, 0, 100, 20),
        )
        llm = lambda prompt: "The hidden state at time t is a function of the previous state and the input."
        out = render_equation(data, EquationPolicy(mode="symbolic"), llm=llm)
        # Eq-number prefix is always added when the LLM doesn't include it.
        assert "Equation three point four" in out
        assert "hidden state" in out

    def test_falls_back_to_skip_without_llm(self):
        data = EquationData(
            raw_text="x = y", symbolic="x equals y", number=None,
            page=0, bbox=(0, 0, 100, 20),
        )
        out = render_equation(data, EquationPolicy(mode="symbolic"), llm=None)
        assert "x equals y" in out  # skip-form symbolic readout

    def test_falls_back_when_llm_raises(self):
        data = EquationData(
            raw_text="x = y", symbolic="x equals y", number="1",
            page=0, bbox=(0, 0, 100, 20),
        )
        def broken(_):
            raise RuntimeError("API down")
        out = render_equation(data, EquationPolicy(mode="symbolic"), llm=broken)
        assert out.startswith("Equation one:")

    def test_falls_back_when_llm_returns_empty(self):
        data = EquationData(
            raw_text="x = y", symbolic="x equals y", number=None,
            page=0, bbox=(0, 0, 100, 20),
        )
        out = render_equation(
            data, EquationPolicy(mode="symbolic"),
            llm=lambda _: "   ",
        )
        assert "x equals y" in out

    def test_rejects_suspiciously_short_narration(self):
        data = EquationData(
            raw_text="h_t = f(h_{t-1}, x_t) somewhat long symbolic form here with many tokens",
            symbolic="h sub t equals f of h sub t minus one and x sub t somewhat long symbolic form",
            number=None, page=0, bbox=(0, 0, 100, 20),
        )
        # 2-char narration is vastly shorter than symbolic; must be rejected.
        out = render_equation(
            data, EquationPolicy(mode="symbolic"),
            llm=lambda _: "ok",
        )
        assert "sub t" in out  # fell back to symbolic skip-form

    def test_llm_narration_already_naming_equation_is_not_double_prefixed(self):
        data = EquationData(
            raw_text="x = y", symbolic="x equals y", number="3",
            page=0, bbox=(0, 0, 100, 20),
        )
        llm = lambda _: (
            "Equation three says that x and y are equal. This is a full sentence."
        )
        out = render_equation(data, EquationPolicy(mode="symbolic"), llm=llm)
        # Should not produce "Equation three: Equation three..."
        assert out.count("Equation three") == 1

    def test_strips_html_and_latex_delimiters_from_llm_output(self):
        data = EquationData(
            raw_text="x = y", symbolic="x equals y well enough here", number=None,
            page=0, bbox=(0, 0, 100, 20),
        )
        llm = lambda _: "<p>$x$ equals $y$. They are the same value.</p>"
        out = render_equation(data, EquationPolicy(mode="symbolic"), llm=llm)
        assert "<p>" not in out
        assert "$" not in out


# ---------------------------------------------------------------------------
# Vision renderer
# ---------------------------------------------------------------------------

class TestRenderVision:
    def test_skips_without_vision_llm(self):
        # No vision llm and no symbolic llm -> skip form.
        data = EquationData(
            raw_text="x = y", symbolic="x equals y", number="1",
            page=0, bbox=(0, 0, 100, 20),
        )
        out = render_equation(
            data, EquationPolicy(mode="vision"),
            vision_llm=None, source_path="/fake.pdf",
        )
        assert out.startswith("Equation one:")

    def test_falls_back_to_symbolic_llm_on_vision_failure(self):
        data = EquationData(
            raw_text="x = y", symbolic="x equals y nicely here",
            number="1", page=0, bbox=(0, 0, 100, 20),
        )
        # Vision fails because source_path is bogus and fitz won't open it,
        # so _render_vision returns None and we fall through to symbolic.
        out = render_equation(
            data, EquationPolicy(mode="vision"),
            llm=lambda _: "A description of the equation that is long enough.",
            vision_llm=lambda img, prompt: "ignored",
            source_path="/nonexistent/path.pdf",
        )
        assert "description of the equation" in out.lower()


# ---------------------------------------------------------------------------
# Narration helpers
# ---------------------------------------------------------------------------

class TestSanitize:
    def test_strips_html_tags(self):
        assert _sanitize_narration("<p>hello</p>") == "hello"

    def test_strips_code_fences(self):
        assert _sanitize_narration("```\nhello\n```") == "hello"

    def test_collapses_whitespace(self):
        assert _sanitize_narration("hello    world") == "hello world"


class TestPrependNumber:
    def test_no_number_is_noop(self):
        assert _prepend_equation_number("hello", None) == "hello"

    def test_number_prepended_when_missing(self):
        out = _prepend_equation_number("x is y", "3.4")
        assert out.startswith("Equation three point four:")

    def test_number_not_duplicated_when_already_mentioned(self):
        out = _prepend_equation_number(
            "Equation three point four says x is y.", "3.4"
        )
        assert out.count("three point four") == 1


# ---------------------------------------------------------------------------
# handle_equations — Document-level
# ---------------------------------------------------------------------------

class TestHandleEquations:
    def test_non_equation_blocks_pass_through_untouched(self):
        body = Block(
            kind="body", spans=[_span("hello")], text="hello",
            page=0, bbox=(0, 0, 100, 20),
        )
        doc = _doc([body])
        out = handle_equations(doc)
        assert out.blocks[0].text == "hello"
        assert out.blocks[0].kind == "body"

    def test_equation_block_gets_meta_payload(self):
        eq = _eq_block("x = y (3)")
        doc = _doc([eq])
        out = handle_equations(doc, EquationPolicy(mode="skip"))
        equation = out.blocks[0].meta.get("equation")
        assert isinstance(equation, EquationData)
        assert equation.number == "3"
        assert out.blocks[0].kind == "equation_display"  # kind is unchanged
        assert "Equation three" in out.blocks[0].text

    def test_purity(self):
        """handle_equations must not mutate the input document."""
        eq = _eq_block("x = y (3)")
        doc = _doc([eq])
        original_text = eq.text
        handle_equations(doc, EquationPolicy(mode="skip"))
        assert eq.text == original_text
        assert "equation" not in eq.meta

    def test_multiple_equations_narrated_independently(self):
        e1 = _eq_block("x = y (1)", bbox=(100, 100, 400, 120))
        e2 = _eq_block("a = b (2)", bbox=(100, 200, 400, 220))
        doc = _doc([e1, e2])
        out = handle_equations(doc, EquationPolicy(mode="skip"))
        assert "Equation one" in out.blocks[0].text
        assert "Equation two" in out.blocks[1].text

    def test_symbolic_mode_uses_llm_for_each_equation(self):
        calls: list[str] = []
        def fake_llm(prompt: str) -> str:
            calls.append(prompt)
            return "A short narration describing what the expression says."
        e1 = _eq_block("x = y + z well enough here (1)")
        e2 = _eq_block("a = b + c sufficient length too (2)")
        doc = _doc([e1, e2])
        handle_equations(doc, EquationPolicy(mode="symbolic"), llm=fake_llm)
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# Integration — real PDFs, skip mode only (no LLM required)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(TWO_COL_PDF), reason="sample PDF missing")
class TestTwoColumnPaperEquations:
    @pytest.fixture(scope="class")
    def doc(self):
        d = classify_blocks(extract_layout(TWO_COL_PDF))
        return handle_equations(d, EquationPolicy(mode="skip"))

    def test_at_least_one_equation_detected(self, doc):
        equations = [b for b in doc.blocks if b.kind == "equation_display"]
        assert len(equations) >= 1

    def test_narration_replaces_raw_equation_text(self, doc):
        for b in doc.blocks:
            if b.kind == "equation_display":
                data = b.meta.get("equation")
                assert isinstance(data, EquationData)
                # Text should either be a readable skip-form or start with "Equation".
                assert b.text and not b.text.endswith(")")


@pytest.mark.skipif(not os.path.exists(ONE_COL_PDF), reason="sample PDF missing")
class TestOneColumnPaperEquations:
    @pytest.fixture(scope="class")
    def doc(self):
        d = classify_blocks(extract_layout(ONE_COL_PDF))
        return handle_equations(d, EquationPolicy(mode="skip"))

    def test_equations_carry_structured_payload(self, doc):
        eq_blocks = [b for b in doc.blocks if b.kind == "equation_display"]
        # Titans has many equations.
        assert len(eq_blocks) >= 3
        for b in eq_blocks:
            assert isinstance(b.meta.get("equation"), EquationData)
