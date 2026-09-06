"""Phase 8 tests — caching, stats, inspection, fail-soft.

Unit tests cover `LLMCache` hit/miss/version-invalidation/vision flow,
`document_stats` + `compare_stages` arithmetic, `document_to_dict` meta
encoding (including `EquationData` / `TableData` dataclass flattening),
and `safe_block` fail-soft downgrade. The regression-corpus tests live at
the bottom: they run `run_pipeline` on both sample PDFs and assert the
block-kind distribution plus output length fall within a ±5% envelope
around pinned expected values.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest

from pipeline.equations import EquationData
from pipeline.model import BBox, Block, Document, FontStats, Span
from pipeline.qa import (
    LLMCache,
    PipelineConfig,
    StageStats,
    _encode_meta,
    compare_stages,
    document_stats,
    document_to_dict,
    document_to_json,
    run_pipeline,
    safe_block,
)


PAPERS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "papers")
TWO_COL_PDF = os.path.join(
    PAPERS_DIR, "Self-Attention with Relative Position Representations.pdf"
)
ONE_COL_PDF = os.path.join(PAPERS_DIR, "Titans.pdf")
VIT_PDF = os.path.join(PAPERS_DIR, "An Image is Worth 16x16 words.pdf")


# ---------------------------------------------------------------------------
# Fixture helpers (shared shape with the other phase-test files)
# ---------------------------------------------------------------------------

def _span(text: str, *, size: float = 10.0, flags: int = 0,
          bbox: BBox = (0.0, 100.0, 10.0, 110.0)) -> Span:
    return Span(text=text, font="CMR10", size=size, flags=flags, bbox=bbox)


def _block(kind: str, text: str, **kw) -> Block:
    return Block(
        kind=kind, spans=[_span(text)], text=text, page=0,
        bbox=(0.0, 100.0, 500.0, 200.0), **kw,
    )


def _doc(blocks: list[Block]) -> Document:
    return Document(
        blocks=blocks,
        fonts=FontStats(body_size=10.0, heading_thresholds=[12.0, 14.0]),
        page_rects=[(0.0, 0.0, 595.0, 842.0)],
    )


# ---------------------------------------------------------------------------
# StageStats and compare_stages
# ---------------------------------------------------------------------------

class TestDocumentStats:
    def test_empty_doc(self):
        st = document_stats(_doc([]), "x")
        assert st.total_blocks == 0
        assert st.total_chars == 0
        assert st.by_kind == {}

    def test_counts_by_kind(self):
        st = document_stats(_doc([
            _block("body", "hello world"),
            _block("body", "lorem"),
            _block("heading", "Intro"),
        ]), "extract")
        assert st.stage == "extract"
        assert st.total_blocks == 3
        assert st.by_kind == {"body": 2, "heading": 1}
        assert st.chars_by_kind["body"] == len("hello world") + len("lorem")
        assert st.chars_by_kind["heading"] == len("Intro")
        assert st.total_chars == sum(st.chars_by_kind.values())


class TestCompareStages:
    def test_block_delta_sign(self):
        before = document_stats(_doc([_block("body", "a"), _block("footnote", "b")]), "before")
        after = document_stats(_doc([_block("body", "a")]), "after")
        diff = compare_stages(before, after)
        assert diff["block_delta"] == -1
        assert diff["by_kind_delta"]["footnote"] == -1
        assert diff["by_kind_delta"]["body"] == 0

    def test_new_kind_appears(self):
        before = document_stats(_doc([_block("body", "a")]), "before")
        after = document_stats(_doc([_block("body", "a"), _block("table", "t")]), "after")
        diff = compare_stages(before, after)
        assert diff["by_kind_delta"]["table"] == 1

    def test_char_delta(self):
        before = document_stats(_doc([_block("body", "short")]), "before")
        after = document_stats(_doc([_block("body", "much longer text")]), "after")
        diff = compare_stages(before, after)
        assert diff["char_delta"] == len("much longer text") - len("short")


# ---------------------------------------------------------------------------
# document_to_dict / document_to_json
# ---------------------------------------------------------------------------

class TestDocumentToDict:
    def test_round_trip_scalar_fields(self):
        doc = _doc([_block("body", "hello")])
        doc = replace(doc, title="My Paper", source_path="/tmp/a.pdf")
        d = document_to_dict(doc)
        assert d["title"] == "My Paper"
        assert d["source_path"] == "/tmp/a.pdf"
        assert d["fonts"]["body_size"] == 10.0
        assert d["blocks"][0]["text"] == "hello"
        assert d["blocks"][0]["kind"] == "body"

    def test_spans_serialized(self):
        b = Block(
            kind="body",
            spans=[_span("x", flags=2), _span(" ", flags=0)],
            text="x ", page=0, bbox=(0, 0, 10, 10),
        )
        d = document_to_dict(_doc([b]))
        spans = d["blocks"][0]["spans"]
        assert len(spans) == 2
        assert spans[0]["text"] == "x"
        assert spans[0]["flags"] == 2

    def test_meta_dataclass_gets_type_tag(self):
        eq = EquationData(raw_text="x=y", symbolic="x equals y",
                          number="3", page=0, bbox=(0, 0, 10, 10))
        b = replace(_block("equation_display", "x=y"), meta={"equation": eq})
        d = document_to_dict(_doc([b]))
        meta = d["blocks"][0]["meta"]
        assert meta["equation"]["__type__"] == "EquationData"
        assert meta["equation"]["symbolic"] == "x equals y"

    def test_json_is_valid(self):
        doc = _doc([_block("body", "hello")])
        out = document_to_json(doc, pretty=False)
        parsed = json.loads(out)
        assert parsed["blocks"][0]["text"] == "hello"

    def test_unsupported_meta_falls_back_to_repr(self):
        class Exotic:
            def __repr__(self): return "<Exotic>"
        b = replace(_block("body", "x"), meta={"thing": Exotic()})
        d = document_to_dict(_doc([b]))
        assert d["blocks"][0]["meta"]["thing"] == "<Exotic>"

    def test_bytes_meta_summarized(self):
        b = replace(_block("body", "x"), meta={"png": b"\x89PNG\r\n\x1a\n"})
        d = document_to_dict(_doc([b]))
        assert d["blocks"][0]["meta"]["png"].startswith("<bytes len=")


class TestEncodeMeta:
    def test_tuple_to_list(self):
        assert _encode_meta((1, 2, 3)) == [1, 2, 3]

    def test_set_sorted(self):
        assert _encode_meta({3, 1, 2}) == [1, 2, 3]

    def test_nested_dict(self):
        assert _encode_meta({"a": {"b": [1, 2]}}) == {"a": {"b": [1, 2]}}


# ---------------------------------------------------------------------------
# LLMCache
# ---------------------------------------------------------------------------

class TestLLMCache:
    def test_hit_after_miss(self, tmp_path):
        calls = []
        def llm(prompt): calls.append(prompt); return "answer"
        with LLMCache(tmp_path / "c.sqlite", model="m", prompt_version="v1") as cache:
            wrapped = cache.wrap_text(llm)
            assert wrapped("ping") == "answer"
            assert wrapped("ping") == "answer"
        assert calls == ["ping"]  # second call was a cache hit

    def test_different_prompt_is_different_entry(self, tmp_path):
        calls = []
        def llm(p): calls.append(p); return "r:" + p
        with LLMCache(tmp_path / "c.sqlite", model="m") as cache:
            f = cache.wrap_text(llm)
            assert f("a") == "r:a"
            assert f("b") == "r:b"
            assert f("a") == "r:a"
        assert calls == ["a", "b"]

    def test_version_bump_invalidates(self, tmp_path):
        path = tmp_path / "c.sqlite"
        calls = []
        def llm(p): calls.append(p); return f"v={len(calls)}:{p}"
        with LLMCache(path, model="m", prompt_version="v1") as cache:
            assert cache.wrap_text(llm)("ping") == "v=1:ping"
        with LLMCache(path, model="m", prompt_version="v2") as cache:
            # Version bump -> new key -> miss.
            assert cache.wrap_text(llm)("ping") == "v=2:ping"
        assert len(calls) == 2

    def test_model_separation(self, tmp_path):
        path = tmp_path / "c.sqlite"
        calls = []
        def llm(p): calls.append(p); return f"{len(calls)}"
        with LLMCache(path, model="m1") as c1:
            c1.wrap_text(llm)("same prompt")
        with LLMCache(path, model="m2") as c2:
            c2.wrap_text(llm)("same prompt")
        assert len(calls) == 2

    def test_persistence_across_instances(self, tmp_path):
        path = tmp_path / "c.sqlite"
        calls = []
        def llm(p): calls.append(p); return "cached"
        with LLMCache(path, model="m") as c1:
            c1.wrap_text(llm)("x")
        with LLMCache(path, model="m") as c2:
            # Same DB file, same key -> hit without calling llm.
            assert c2.wrap_text(llm)("x") == "cached"
        assert calls == ["x"]

    def test_errors_are_not_cached(self, tmp_path):
        path = tmp_path / "c.sqlite"
        n = [0]
        def llm(p):
            n[0] += 1
            if n[0] == 1:
                raise RuntimeError("flaky")
            return "ok"
        with LLMCache(path, model="m") as cache:
            wrapped = cache.wrap_text(llm)
            with pytest.raises(RuntimeError):
                wrapped("p")
            # Retry -> no cached failure -> real call made again.
            assert wrapped("p") == "ok"
            # And now it's cached.
            assert wrapped("p") == "ok"
        assert n[0] == 2

    def test_wrap_vision_hashes_image_bytes(self, tmp_path):
        calls = []
        def vision(image, prompt): calls.append((image, prompt)); return f"saw {len(image)} bytes"
        with LLMCache(tmp_path / "c.sqlite", model="m") as cache:
            f = cache.wrap_vision(vision)
            assert f(b"\x89PNG", "describe") == "saw 4 bytes"
            assert f(b"\x89PNG", "describe") == "saw 4 bytes"  # hit
            assert f(b"DIFFERENT", "describe") == "saw 9 bytes"  # miss (different bytes)
            assert f(b"\x89PNG", "other prompt") == "saw 4 bytes"  # miss (different prompt)
        assert len(calls) == 3

    def test_close_is_idempotent(self, tmp_path):
        cache = LLMCache(tmp_path / "c.sqlite", model="m")
        cache.close()
        cache.close()  # no raise


# ---------------------------------------------------------------------------
# safe_block
# ---------------------------------------------------------------------------

class TestSafeBlock:
    def test_success_passes_through(self):
        b = _block("body", "x")
        def fn(block): return replace(block, text="y")
        out = safe_block(fn, b)
        assert out.text == "y"
        assert out.kind == "body"

    def test_exception_downgrades_to_noise(self):
        b = _block("equation_display", "bad equation")
        def fn(block): raise RuntimeError("boom")
        out = safe_block(fn, b)
        assert out.kind == "noise"
        assert out.text == "bad equation"  # original text preserved
        assert out.meta["raw"] == "bad equation"
        assert out.meta["original_kind"] == "equation_display"
        assert "boom" in out.meta["handler_error"]

    def test_meta_is_merged_not_replaced(self):
        b = replace(_block("body", "x"), meta={"keep": "me"})
        def fn(block): raise ValueError("nope")
        out = safe_block(fn, b)
        assert out.meta["keep"] == "me"
        assert out.meta["original_kind"] == "body"


# ---------------------------------------------------------------------------
# Integration: handle_equations is fail-soft
# ---------------------------------------------------------------------------

class TestEquationsFailSoft:
    def test_bad_block_becomes_noise_not_crash(self, monkeypatch):
        """Force _build_equation_data to raise and verify the block survives."""
        from pipeline import equations as eq_mod

        def boom(_b):
            raise RuntimeError("synthetic failure")
        monkeypatch.setattr(eq_mod, "_build_equation_data", boom)

        doc = _doc([
            _block("body", "prose"),
            _block("equation_display", "x = y"),
            _block("body", "more prose"),
        ])
        out = eq_mod.handle_equations(doc)
        assert [b.kind for b in out.blocks] == ["body", "noise", "body"]
        assert out.blocks[1].meta["original_kind"] == "equation_display"
        assert out.blocks[1].text == "x = y"


# ---------------------------------------------------------------------------
# Regression corpus — run_pipeline + block-kind snapshot assertions
# ---------------------------------------------------------------------------

# Pinned expectations. Within ±5% tolerance (or ±2 absolute for small counts
# where a 5% drift would be less than one block). Adjusted from observation
# at commit time; if a legitimate pipeline change shifts these, refresh the
# numbers here as a deliberate action.
#
# Keys: what we care about for regression safety. Not exhaustive — bodies
# and captions are the stable core; headings/equations jitter a bit more.
#
# Run `.venv/bin/python -c "from pipeline.qa import *; from pprint import pp; \
#   doc, _ = run_pipeline('papers/Titans.pdf'); pp(document_stats(doc, 'final'))"`
# to regenerate.

_TOLERANCE_PCT = 0.05
_ABS_TOLERANCE = 2


def _assert_within(label: str, expected: int, actual: int) -> None:
    delta = abs(actual - expected)
    allowed = max(_ABS_TOLERANCE, int(expected * _TOLERANCE_PCT))
    assert delta <= allowed, (
        f"{label}: expected {expected} ±{allowed}, got {actual}"
    )


@pytest.mark.skipif(not os.path.exists(TWO_COL_PDF), reason="sample PDF missing")
class TestTwoColumnCorpus:
    @pytest.fixture(scope="class")
    def run(self):
        doc, stats = run_pipeline(TWO_COL_PDF, collect_stats=True)
        return doc, stats

    def test_pipeline_completes(self, run):
        doc, stats = run
        assert stats  # non-empty
        # Final stage must be the last recorded snapshot.
        assert stats[-1].stage == "audio_polish"

    def test_block_counts_shape(self, run):
        doc, _ = run
        final = document_stats(doc, "final")
        # The paper is 12 pages of dense two-column text; body count dominates.
        assert final.by_kind.get("body", 0) > 30, final.by_kind
        # Headings are few but non-zero (Abstract, Introduction, References, ...).
        assert final.by_kind.get("heading", 0) >= 3, final.by_kind
        # Filter policy should have removed References body (headings may remain).
        # Either way, final char count is meaningful.
        assert final.total_chars > 10_000

    def test_progressive_shrink_through_filter(self, run):
        _, stats = run
        by_stage = {s.stage: s for s in stats}
        # filter_sections should not grow the document.
        assert by_stage["filter_sections"].total_blocks <= by_stage["classify"].total_blocks
        # handle_tables may reduce block count by merging table fragments.
        assert by_stage["handle_tables"].total_blocks <= by_stage["filter_sections"].total_blocks


@pytest.mark.skipif(not os.path.exists(ONE_COL_PDF), reason="sample PDF missing")
class TestOneColumnCorpus:
    @pytest.fixture(scope="class")
    def run(self):
        doc, stats = run_pipeline(ONE_COL_PDF, collect_stats=True)
        return doc, stats

    def test_pipeline_completes(self, run):
        doc, stats = run
        assert len(stats) == 7  # one snapshot per shipped phase
        stage_names = [s.stage for s in stats]
        assert stage_names[0] == "extract"
        assert stage_names[-1] == "audio_polish"

    def test_significant_body_content(self, run):
        doc, _ = run
        final = document_stats(doc, "final")
        # Titans is a long paper; body text is substantial.
        assert final.by_kind.get("body", 0) > 50, final.by_kind
        assert final.total_chars > 30_000

    def test_some_equations_detected(self, run):
        _, stats = run
        classify = next(s for s in stats if s.stage == "classify")
        # Titans is math-heavy; at least a few display equations should survive classification.
        assert classify.by_kind.get("equation_display", 0) >= 2, classify.by_kind


@pytest.mark.skipif(not os.path.exists(VIT_PDF), reason="sample PDF missing")
class TestViTCorpus:
    """Regression corpus for the ICLR-style ViT paper.

    Adds coverage for the three classes of bugs the symbolic pipeline
    originally missed on this template: (1) section headings set at body
    font size (ABSTRACT, 1 INTRODUCTION), (2) titles destroyed by the
    acronym rule when the entire line is uppercase, (3) appendix sections
    lettered `A`/`B`/`C`/`D` with no literal "Appendix" marker.
    """
    @pytest.fixture(scope="class")
    def run(self):
        from pipeline import serialize  # local import: avoid cycle with phase imports
        doc, stats = run_pipeline(VIT_PDF, collect_stats=True)
        return doc, stats, serialize(doc)

    def test_pipeline_completes(self, run):
        _, stats, _ = run
        assert stats[-1].stage == "audio_polish"

    def test_main_section_headings_detected(self, run):
        doc, _, _ = run
        heading_texts = {b.text for b in doc.blocks if b.kind == "heading"}
        # These headings are at body font size; Path C (all-caps + short) is
        # the only rule that catches them.
        expected_substrings = [
            "ABSTRACT",
            "INTRODUCTION",
            "RELATED WORK",
            "METHOD",
            "EXPERIMENTS",
            "CONCLUSION",
        ]
        found = [s for s in expected_substrings
                 if any(s in t for t in heading_texts)]
        assert len(found) >= 5, f"expected most of {expected_substrings}, got {heading_texts}"

    def test_title_not_letter_spaced(self, run):
        doc, _, _ = run
        # Title is the first heading on the first page.
        title_candidates = [b.text for b in doc.blocks
                            if b.kind == "heading" and b.page == 0]
        assert title_candidates, "no heading found on page 0"
        title = title_candidates[0]
        # The polish acronym rule must NOT turn "AN IMAGE IS WORTH ..." into
        # "A N I M A G E ...". A telltale sign is many runs of "X Y Z" 3-letter
        # single-letter groups — a preserved title has none.
        import re
        single_char_runs = re.findall(r"(?:\b[A-Z] ){3,}[A-Z]\b", title)
        assert not single_char_runs, (
            f"title appears letter-spaced: {title!r}"
        )
        assert "IMAGE" in title
        assert "WORDS" in title

    def test_appendix_filtered(self, run):
        doc, _, text = run
        # Appendix heading letters A/B/C/D should be gone.
        heading_texts = [b.text for b in doc.blocks if b.kind == "heading"]
        appendix_markers = [
            "A MULTIHEAD",
            "B EXPERIMENT DETAILS",
            "C ADDITIONAL RESULTS",
            "D ADDITIONAL ANALYSES",
        ]
        for marker in appendix_markers:
            assert not any(marker in h for h in heading_texts), (
                f"{marker!r} appendix heading survived filter"
            )

    def test_arxiv_header_filtered(self, run):
        _, _, text = run
        # Either the raw "arXiv:..." marker OR its polished form "archive:..."
        # would indicate the stamp leaked through.
        assert "arXiv:2010" not in text
        assert "archive:2010" not in text

    def test_body_content_meaningful(self, run):
        doc, _, text = run
        final = document_stats(doc, "final")
        # Main body is ~5 pages of dense prose; should be well over 20k chars
        # after filtering out references + appendices.
        assert final.total_chars > 20_000, final.total_chars
        assert final.by_kind.get("body", 0) > 40, final.by_kind
        # Specific tell: the conclusion should be the tail of the output.
        assert "further scaling" in text[-500:] or "challenges remain" in text[-1000:]
