"""Phase 9 tests — vision-first pipeline.

Unit tests use a stub `vision_llm` lambda (no real VLM calls) to verify
page rendering, block assembly, fail-soft behaviour, and integration
with Phase 7 polish. Integration tests run `run_pipeline_vlm` on the
sample PDFs with a stub narrator to assert the full wiring end-to-end
without burning vision tokens.
"""
from __future__ import annotations

import os

import pytest

from pipeline.polish import PolishPolicy
from pipeline.qa import compare_stages, document_stats
from pipeline.vlm import (
    DEFAULT_VLM_PROMPT,
    VLMPolicy,
    run_pipeline_vlm,
    vlm_extract_document,
)


PAPERS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "papers")
TWO_COL_PDF = os.path.join(
    PAPERS_DIR, "Self-Attention with Relative Position Representations.pdf"
)
ONE_COL_PDF = os.path.join(PAPERS_DIR, "Titans.pdf")


def _stub_narrator(text_per_page: str = "This is page narration with enough text.") -> callable:
    """Return a stub vision_llm: (image_bytes, prompt) -> narration."""
    calls = []

    def narrate(image: bytes, prompt: str) -> str:
        calls.append((len(image), prompt))
        return f"{text_per_page} Page {len(calls)}."

    narrate.calls = calls  # type: ignore[attr-defined]
    return narrate


# ---------------------------------------------------------------------------
# VLMPolicy defaults
# ---------------------------------------------------------------------------

class TestVLMPolicy:
    def test_defaults(self):
        p = VLMPolicy()
        assert p.dpi == 200
        assert p.prompt == DEFAULT_VLM_PROMPT
        assert p.skip_pages == frozenset()
        assert p.max_pages is None
        assert p.min_narration_chars >= 10


# ---------------------------------------------------------------------------
# Per-page behaviour (stubbed VLM, real PDF render)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(TWO_COL_PDF), reason="sample PDF missing")
class TestVlmExtractDocument:
    def test_one_body_block_per_page(self):
        narrate = _stub_narrator()
        doc = vlm_extract_document(
            TWO_COL_PDF, narrate, VLMPolicy(max_pages=2, dpi=72)
        )
        assert len(doc.blocks) == 2
        assert all(b.kind == "body" for b in doc.blocks)
        assert [b.page for b in doc.blocks] == [0, 1]
        assert len(narrate.calls) == 2  # type: ignore[attr-defined]

    def test_prompt_and_bytes_passed(self):
        narrate = _stub_narrator()
        vlm_extract_document(
            TWO_COL_PDF, narrate, VLMPolicy(max_pages=1, dpi=72)
        )
        nbytes, prompt = narrate.calls[0]  # type: ignore[attr-defined]
        assert nbytes > 0  # non-empty PNG
        assert prompt == DEFAULT_VLM_PROMPT

    def test_custom_prompt(self):
        narrate = _stub_narrator()
        custom = "Narrate for a five-year-old."
        vlm_extract_document(
            TWO_COL_PDF, narrate, VLMPolicy(max_pages=1, prompt=custom, dpi=72)
        )
        assert narrate.calls[0][1] == custom  # type: ignore[attr-defined]

    def test_skip_pages_omits_block(self):
        narrate = _stub_narrator()
        doc = vlm_extract_document(
            TWO_COL_PDF,
            narrate,
            VLMPolicy(max_pages=3, skip_pages=frozenset({1}), dpi=72),
        )
        pages = [b.page for b in doc.blocks]
        assert 1 not in pages  # skipped
        assert len(narrate.calls) == 2  # type: ignore[attr-defined]

    def test_max_pages_caps(self):
        narrate = _stub_narrator()
        doc = vlm_extract_document(
            TWO_COL_PDF, narrate, VLMPolicy(max_pages=1, dpi=72)
        )
        assert len(doc.blocks) == 1

    def test_page_rects_populated(self):
        narrate = _stub_narrator()
        doc = vlm_extract_document(
            TWO_COL_PDF, narrate, VLMPolicy(max_pages=2, dpi=72)
        )
        assert len(doc.page_rects) == 2
        for r in doc.page_rects:
            assert r[2] > r[0] and r[3] > r[1]  # non-degenerate

    def test_source_path_preserved(self):
        narrate = _stub_narrator()
        doc = vlm_extract_document(
            TWO_COL_PDF, narrate, VLMPolicy(max_pages=1, dpi=72)
        )
        assert doc.source_path == TWO_COL_PDF

    def test_synthetic_span_per_block(self):
        narrate = _stub_narrator()
        doc = vlm_extract_document(
            TWO_COL_PDF, narrate, VLMPolicy(max_pages=1, dpi=72)
        )
        b = doc.blocks[0]
        assert len(b.spans) == 1
        assert b.spans[0].text == b.text

    def test_narration_whitespace_collapsed(self):
        def narrate(image: bytes, prompt: str) -> str:
            return "Hello\n\n  world.  Multiple   spaces here too end."
        doc = vlm_extract_document(
            TWO_COL_PDF, narrate, VLMPolicy(max_pages=1, dpi=72)
        )
        assert doc.blocks[0].text == "Hello world. Multiple spaces here too end."


# ---------------------------------------------------------------------------
# Fail-soft
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(TWO_COL_PDF), reason="sample PDF missing")
class TestFailSoft:
    def test_vlm_exception_becomes_noise_block(self):
        def boom(image: bytes, prompt: str) -> str:
            raise RuntimeError("synthetic VLM failure")
        doc = vlm_extract_document(
            TWO_COL_PDF, boom, VLMPolicy(max_pages=2, dpi=72)
        )
        assert [b.kind for b in doc.blocks] == ["noise", "noise"]
        assert "synthetic VLM failure" in doc.blocks[0].meta["handler_error"]
        assert doc.blocks[0].meta["vlm_stage"] == "vlm_call"

    def test_empty_narration_becomes_noise(self):
        def narrate(image: bytes, prompt: str) -> str:
            return ""  # empty => below min_narration_chars
        doc = vlm_extract_document(
            TWO_COL_PDF, narrate, VLMPolicy(max_pages=1, dpi=72)
        )
        assert doc.blocks[0].kind == "noise"
        assert doc.blocks[0].meta["vlm_stage"] == "too_short"

    def test_short_narration_becomes_noise(self):
        def narrate(image: bytes, prompt: str) -> str:
            return "Too short."
        doc = vlm_extract_document(
            TWO_COL_PDF,
            narrate,
            VLMPolicy(max_pages=1, dpi=72, min_narration_chars=50),
        )
        assert doc.blocks[0].kind == "noise"
        # Original short text preserved for debugging.
        assert doc.blocks[0].text == "Too short."

    def test_one_bad_page_doesnt_kill_run(self):
        calls = [0]
        def narrate(image: bytes, prompt: str) -> str:
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError("flaky page")
            return "Good narration with enough words here to pass the length check."
        doc = vlm_extract_document(
            TWO_COL_PDF, narrate, VLMPolicy(max_pages=3, dpi=72)
        )
        kinds = [b.kind for b in doc.blocks]
        assert kinds.count("body") == 2
        assert kinds.count("noise") == 1


# ---------------------------------------------------------------------------
# Integration with Phase 7 polish
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(TWO_COL_PDF), reason="sample PDF missing")
class TestRunPipelineVlm:
    def test_polish_runs_on_vlm_output(self):
        def narrate(image: bytes, prompt: str) -> str:
            # "e.g." should be expanded by Phase 7 polish.
            return "The attention mechanism, e.g. self-attention, scales as 100 MB."
        doc, stats = run_pipeline_vlm(
            TWO_COL_PDF, narrate, VLMPolicy(max_pages=1, dpi=72),
        )
        text = doc.blocks[0].text
        assert "e.g." not in text  # polish expanded it
        assert "megabyte" in text.lower()
        assert stats == []  # collect_stats defaults to False

    def test_collect_stats_emits_two_snapshots(self):
        narrate = _stub_narrator()
        doc, stats = run_pipeline_vlm(
            TWO_COL_PDF,
            narrate,
            VLMPolicy(max_pages=2, dpi=72),
            collect_stats=True,
        )
        assert [s.stage for s in stats] == ["vlm_extract", "audio_polish"]
        # Polish doesn't add or remove blocks.
        assert stats[0].total_blocks == stats[1].total_blocks
        # compare_stages works on the pair.
        diff = compare_stages(stats[0], stats[1])
        assert diff["block_delta"] == 0

    def test_custom_polish_policy_threaded_through(self):
        def narrate(image: bytes, prompt: str) -> str:
            return "This has one hundred units of data, namely 100 GB of storage space."
        # Disable number polish; expect the "100" digit to survive.
        doc, _ = run_pipeline_vlm(
            TWO_COL_PDF, narrate, VLMPolicy(max_pages=1, dpi=72),
            polish=PolishPolicy(enable_numbers=False, enable_units=False),
        )
        assert "100" in doc.blocks[0].text


# ---------------------------------------------------------------------------
# Integration with the one-column paper (different page count / layout)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(ONE_COL_PDF), reason="sample PDF missing")
class TestOneColumnVlm:
    def test_one_column_paper_narrates(self):
        narrate = _stub_narrator()
        doc = vlm_extract_document(
            ONE_COL_PDF, narrate, VLMPolicy(max_pages=2, dpi=72)
        )
        assert len(doc.blocks) == 2
        assert all(b.kind == "body" for b in doc.blocks)
        final = document_stats(doc, "vlm_extract")
        assert final.by_kind.get("body", 0) == 2
