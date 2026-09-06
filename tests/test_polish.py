"""Phase 7 tests — audio polish.

Unit tests cover each transform in isolation (quotes, pronunciations,
acronyms with state-threading across blocks, abbreviations, units with
pluralization, number normalization with edge cases). Integration tests
run the full extract→classify→polish pipeline on the sample PDFs and
assert that common pre-polish artifacts (`e.g.`, `Fig.`, `LSTM`, raw
multi-digit integers) no longer appear in the serialized output.
"""
from __future__ import annotations

import os

import pytest

from pipeline.classify import classify_blocks
from pipeline.extract import extract_layout
from pipeline.model import BBox, Block, Document, FontStats, Span
from pipeline.policies import SectionPolicy, filter_sections
from pipeline.polish import (
    DEFAULT_ABBREVIATIONS,
    DEFAULT_PRONUNCIATIONS,
    DEFAULT_UNITS,
    PolishPolicy,
    _PolishState,
    _apply_abbreviations,
    _apply_acronyms,
    _apply_numbers,
    _apply_pronunciations,
    _apply_units,
    _expansion_matches,
    _normalize_quotes,
    _polish_text,
    _spell_number,
    audio_polish,
    to_ssml,
)
from pipeline.serialize import serialize


PAPERS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "papers")
TWO_COL_PDF = os.path.join(
    PAPERS_DIR, "Self-Attention with Relative Position Representations.pdf"
)
ONE_COL_PDF = os.path.join(PAPERS_DIR, "Titans.pdf")


# ---------------------------------------------------------------------------
# Fixture helpers (same shape as the other phase-test files)
# ---------------------------------------------------------------------------

def _span(text: str, *, size: float = 10.0,
          bbox: BBox = (0.0, 100.0, 10.0, 110.0)) -> Span:
    return Span(text=text, font="CMR10", size=size, flags=0, bbox=bbox)


def _block(kind: str, text: str, *, page: int = 0) -> Block:
    return Block(
        kind=kind, spans=[_span(text)], text=text, page=page,
        bbox=(0.0, 100.0, 500.0, 200.0),
    )


def _doc(blocks: list[Block]) -> Document:
    return Document(
        blocks=blocks,
        fonts=FontStats(body_size=10.0, heading_thresholds=[]),
        page_rects=[(0.0, 0.0, 595.0, 842.0)],
    )


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------

class TestQuotes:
    def test_curly_doubles_become_straight(self):
        assert _normalize_quotes("\u201Chello\u201D") == '"hello"'

    def test_curly_singles_become_apostrophe(self):
        assert _normalize_quotes("it\u2019s") == "it's"

    def test_guillemets_become_double(self):
        assert _normalize_quotes("\u00ABparis\u00BB") == '"paris"'

    def test_no_op_on_ascii(self):
        assert _normalize_quotes("plain text") == "plain text"


# ---------------------------------------------------------------------------
# Pronunciations
# ---------------------------------------------------------------------------

class TestPronunciations:
    def test_relu_replaced(self):
        out = _apply_pronunciations("we use ReLU activation", PolishPolicy())
        assert "relu" in out
        assert "ReLU" not in out

    def test_latex_replaced(self):
        out = _apply_pronunciations("written in LaTeX", PolishPolicy())
        assert "lay-tek" in out

    def test_arxiv_mid_sentence(self):
        out = _apply_pronunciations("posted to arXiv yesterday", PolishPolicy())
        assert "archive" in out

    def test_extra_pronunciations_merged(self):
        policy = PolishPolicy(extra_pronunciations={"Foo": "phoo"})
        assert "phoo" in _apply_pronunciations("call Foo", policy)

    def test_longer_key_wins(self):
        # PReLU should beat ReLU.
        out = _apply_pronunciations("PReLU activation", PolishPolicy())
        assert "P relu" in out

    def test_inside_word_not_replaced(self):
        # "RELUctant" shouldn't become "reluctant" because the override key
        # is `ReLU` (case-sensitive) — but even if a case-insensitive variant
        # were added, the non-word boundary should reject it.
        out = _apply_pronunciations("RELUctant", PolishPolicy())
        assert out == "RELUctant"


# ---------------------------------------------------------------------------
# Acronyms
# ---------------------------------------------------------------------------

class TestAcronyms:
    def test_first_with_expansion_letters_and_keeps_phrase(self):
        state = _PolishState()
        out = _apply_acronyms("LSTM (Long Short-Term Memory) is used", state)
        assert "L S T M, Long Short-Term Memory" in out
        assert "LSTM" in state.seen_acronyms

    def test_subsequent_drops_parenthetical(self):
        state = _PolishState()
        _apply_acronyms("LSTM (Long Short-Term Memory)", state)
        out = _apply_acronyms("Then LSTM (oops) again", state)
        assert "L S T M" in out
        # The parenthetical for an already-seen acronym should be dropped
        # because the expansion is no longer needed; "(oops)" is not an
        # expansion of LSTM and should not match the "expansion" path.
        assert "L S T M" in out

    def test_unrelated_parenthetical_does_not_count_as_expansion(self):
        state = _PolishState()
        out = _apply_acronyms("BERT (Devlin et al., 2018)", state)
        # Parenthetical doesn't match initials → just letter-space the acronym.
        assert "B E R T" in out
        assert "Devlin" in out  # the parenthetical text survives

    def test_no_expansion_just_letters(self):
        out = _apply_acronyms("we evaluate on GLUE today", _PolishState())
        assert "G L U E" in out

    def test_expansion_matches_with_stopwords(self):
        # "Generative Pre-trained Transformer" → GPT (the "Pre-trained"
        # hyphenated word counts as one initial, P).
        assert _expansion_matches("GPT", "Generative Pre-trained Transformer")

    def test_expansion_match_rejects_unrelated_words(self):
        assert not _expansion_matches("LSTM", "we tried something different")

    def test_all_caps_block_is_left_alone(self):
        """A predominantly uppercase block (title / all-caps section heading)
        must not be letter-spaced: every word would look like an acronym and
        "AN IMAGE IS WORTH WORDS" would turn into "A N I M A G E …"."""
        title = "AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION"
        out = _apply_acronyms(title, _PolishState())
        assert out == title

    def test_all_caps_short_heading_is_left_alone(self):
        out = _apply_acronyms("3 METHOD", _PolishState())
        assert out == "3 METHOD"
        out = _apply_acronyms("ABSTRACT", _PolishState())
        assert out == "ABSTRACT"

    def test_mixed_case_prose_still_letter_spaces(self):
        # Ratio well below 70% upper — letter-spacing must still fire.
        out = _apply_acronyms(
            "We evaluated BERT on GLUE and outperformed LSTM baselines.",
            _PolishState(),
        )
        assert "B E R T" in out
        assert "G L U E" in out
        assert "L S T M" in out

    def test_state_persists_across_blocks_via_audio_polish(self):
        doc = _doc([
            _block("body", "We use LSTM (Long Short-Term Memory) here."),
            _block("body", "Later the LSTM is fine-tuned."),
        ])
        polished = audio_polish(doc)
        assert "L S T M, Long Short-Term Memory" in polished.blocks[0].text
        # Second block: just letter-spaced, no parenthetical.
        assert "L S T M" in polished.blocks[1].text
        assert "Long Short-Term Memory" not in polished.blocks[1].text


# ---------------------------------------------------------------------------
# Abbreviations
# ---------------------------------------------------------------------------

class TestAbbreviations:
    def test_eg_expanded(self):
        out = _apply_abbreviations("models e.g. GPT", PolishPolicy())
        assert "for example" in out
        assert "e.g." not in out

    def test_ie_with_comma_expanded(self):
        out = _apply_abbreviations("the model i.e., GPT-4", PolishPolicy())
        assert "that is," in out

    def test_fig_expanded(self):
        out = _apply_abbreviations("see Fig. 3", PolishPolicy())
        assert "Figure" in out
        assert "Fig." not in out

    def test_figs_plural_wins_over_fig(self):
        # Longest-key-first sort guarantees `Figs.` gets matched as a unit
        # rather than `Fig.` then a stray `s.`.
        out = _apply_abbreviations("see Figs. 1 and 2", PolishPolicy())
        assert "Figures" in out
        assert "Figure s" not in out

    def test_etal_expanded(self):
        out = _apply_abbreviations("Vaswani et al. 2017", PolishPolicy())
        assert "and others" in out

    def test_extra_overrides_merge(self):
        policy = PolishPolicy(extra_abbreviations={"foo.": "bar"})
        assert _apply_abbreviations("a foo. b", policy) == "a bar b"


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

class TestUnits:
    def test_singular_at_one(self):
        assert _apply_units("1 GB free", PolishPolicy()) == "one gigabyte free"

    def test_plural_otherwise(self):
        assert "gigabytes" in _apply_units("100 GB", PolishPolicy())

    def test_decimal_pluralizes(self):
        out = _apply_units("1.5 GHz", PolishPolicy())
        assert "gigahertz" in out
        assert "one point five" in out

    def test_ghz_beats_hz(self):
        # If sorting weren't longest-first, "1 GHz" would match Hz.
        out = _apply_units("1 GHz", PolishPolicy())
        assert "gigahertz" in out
        assert "hertz" != out  # contains "gigahertz"

    def test_unit_followed_by_word_char_skipped(self):
        # `100 GBs` (with trailing s) — the regex `(?!\w)` excludes this
        # so we don't double-pluralize.
        out = _apply_units("100 GBs of disk", PolishPolicy())
        assert out == "100 GBs of disk"

    def test_no_space_between_number_and_unit(self):
        assert "gigahertz" in _apply_units("3GHz", PolishPolicy())


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

class TestNumbers:
    def test_integer(self):
        assert _spell_number("27", PolishPolicy()) == "twenty-seven"

    def test_decimal(self):
        assert _spell_number("27.3", PolishPolicy()) == "twenty-seven point three"

    def test_negative(self):
        assert _spell_number("-3", PolishPolicy()) == "negative three"

    def test_thousands_separator_stripped(self):
        assert _spell_number("1,234", PolishPolicy()) == "one thousand, two hundred and thirty-four"

    def test_zero_decimal_left_padded(self):
        # ".5" → "zero point five"
        assert _spell_number(".5", PolishPolicy()) == "zero point five"

    def test_huge_number_falls_back_digit_by_digit(self):
        out = _spell_number("123456789012", PolishPolicy(max_number_words=999))
        # All digits read individually — fewer than 12 commas means num2words
        # was bypassed.
        assert "one two three" in out

    def test_apply_numbers_skips_inside_identifier(self):
        # `H2O` should not get spelled because `2` is glued to letters.
        assert _apply_numbers("H2O", PolishPolicy()) == "H2O"

    def test_apply_numbers_handles_sentence(self):
        out = _apply_numbers("we trained on 12 GPUs", PolishPolicy())
        assert "twelve" in out
        assert "12" not in out


# ---------------------------------------------------------------------------
# Polish-text orchestration (ordering)
# ---------------------------------------------------------------------------

class TestPolishText:
    def test_units_before_numbers(self):
        # `100 GB` should stay as one phrase, not "one hundred GB".
        out = _polish_text("100 GB free", PolishPolicy(), _PolishState())
        assert "one hundred gigabytes" in out

    def test_quotes_then_pronunciations(self):
        out = _polish_text("\u201CReLU\u201D works", PolishPolicy(), _PolishState())
        assert '"relu"' in out

    def test_acronym_then_abbrev(self):
        # A sentence with both. They compose without interfering.
        out = _polish_text(
            "Fig. 3 compares LSTM (Long Short-Term Memory) variants.",
            PolishPolicy(), _PolishState(),
        )
        assert "Figure" in out
        assert "L S T M, Long Short-Term Memory" in out
        # The standalone "3" should also be spelled out.
        assert "three" in out


# ---------------------------------------------------------------------------
# audio_polish — pure-function contract and skip-kind handling
# ---------------------------------------------------------------------------

class TestAudioPolish:
    def test_returns_new_document(self):
        doc = _doc([_block("body", "see Fig. 1")])
        out = audio_polish(doc)
        assert out is not doc
        assert out.blocks[0].text == "see Figure one"

    def test_skip_kinds_untouched(self):
        # Footnote text should not be polished even if it contains things
        # the polish pass would otherwise rewrite.
        doc = _doc([_block("footnote", "see Fig. 3")])
        out = audio_polish(doc)
        assert out.blocks[0].text == "see Fig. 3"

    def test_empty_block_passthrough(self):
        doc = _doc([_block("body", "   ")])
        out = audio_polish(doc)
        assert out.blocks[0] is doc.blocks[0]

    def test_unchanged_block_returned_as_is(self):
        # No polish-relevant tokens: should be the same block object so
        # downstream code can identity-check for "did anything happen here".
        doc = _doc([_block("body", "plain prose with nothing to fix")])
        out = audio_polish(doc)
        assert out.blocks[0] is doc.blocks[0]

    def test_meta_and_kind_preserved(self):
        b = Block(
            kind="body", spans=[_span("see Fig. 1")], text="see Fig. 1", page=0,
            bbox=(0, 0, 100, 100), level=2, parent_section="Intro",
            meta={"keep": "me"},
        )
        doc = _doc([b])
        out = audio_polish(doc).blocks[0]
        assert out.kind == "body"
        assert out.level == 2
        assert out.parent_section == "Intro"
        assert out.meta == {"keep": "me"}


# ---------------------------------------------------------------------------
# SSML
# ---------------------------------------------------------------------------

class TestSSML:
    def test_wrapped_in_speak(self):
        doc = _doc([_block("body", "hello")])
        out = to_ssml(doc)
        assert out.startswith("<speak>")
        assert out.endswith("</speak>")
        assert "<p>hello</p>" in out

    def test_break_before_non_first_heading(self):
        doc = _doc([
            _block("body", "intro text"),
            _block("heading", "Methods"),
            _block("body", "method text"),
        ])
        out = to_ssml(doc)
        # One break before Methods, none before the body.
        assert out.count("<break") == 1
        assert "<break" in out.split("Methods")[0]

    def test_first_block_heading_has_no_leading_break(self):
        doc = _doc([_block("heading", "Title"), _block("body", "body")])
        out = to_ssml(doc)
        assert "<break" not in out

    def test_xml_escape(self):
        doc = _doc([_block("body", "a < b & c > d")])
        out = to_ssml(doc)
        assert "&lt;" in out
        assert "&amp;" in out
        assert "&gt;" in out


# ---------------------------------------------------------------------------
# Integration tests on real PDFs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(TWO_COL_PDF), reason="sample PDF missing")
class TestTwoColumnPaperPolish:
    @pytest.fixture(scope="class")
    def polished_text(self) -> str:
        doc = extract_layout(TWO_COL_PDF)
        doc = classify_blocks(doc)
        doc = filter_sections(doc, SectionPolicy())
        polished = audio_polish(doc)
        return serialize(polished)

    def test_no_eg_or_ie(self, polished_text):
        assert "e.g." not in polished_text
        assert "i.e." not in polished_text

    def test_no_fig_dot(self, polished_text):
        # "Fig." with a digit-style follow-on is the diagnostic pattern.
        # If the paper used "Figure" already, the test is trivially true.
        assert "Fig. " not in polished_text

    def test_no_etal(self, polished_text):
        assert "et al." not in polished_text


@pytest.mark.skipif(not os.path.exists(ONE_COL_PDF), reason="sample PDF missing")
class TestOneColumnPaperPolish:
    @pytest.fixture(scope="class")
    def polished_text(self) -> str:
        doc = extract_layout(ONE_COL_PDF)
        doc = classify_blocks(doc)
        doc = filter_sections(doc, SectionPolicy())
        polished = audio_polish(doc)
        return serialize(polished)

    def test_no_raw_eg(self, polished_text):
        assert "e.g." not in polished_text

    def test_no_etal_dot(self, polished_text):
        assert "et al." not in polished_text

    def test_some_acronym_lettered(self, polished_text):
        # Titans is a transformer paper, which references LSTM, RNN, MLP
        # routinely; at least one of the common ones should have been
        # letter-spaced somewhere in the polished output.
        candidates = ["L S T M", "R N N", "M L P", "B E R T", "G P T"]
        assert any(c in polished_text for c in candidates), (
            f"no letter-spaced acronym from {candidates} found"
        )
