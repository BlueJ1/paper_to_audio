"""Phase 4 tests — table extraction and narration.

Unit tests cover the TableData payload, overlap math, caption matching, and
the three render modes (skip / verbatim / prose). Integration tests run the
full pipeline on the real sample PDFs and assert that pdfplumber-detected
tables are actually collapsed into kind="table" blocks with structured
payloads attached.
"""
from __future__ import annotations

import os

import pytest

from pipeline.classify import classify_blocks
from pipeline.extract import extract_layout
from pipeline.model import BBox, Block, Document, FontStats, Span
from pipeline.tables import (
    TableData,
    TablePolicy,
    _find_caption,
    _overlap_ratio,
    _prose_is_safe,
    handle_tables,
    render_table,
)


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
    bbox: BBox = (50, 100, 500, 120),
    page: int = 0,
    size: float = 10.0,
) -> Block:
    span = Span(text=text, font="Times", size=size, flags=0, bbox=bbox)
    return Block(kind=kind, spans=[span], text=text, page=page, bbox=bbox)


def _doc(blocks: list[Block], source_path: str | None = None) -> Document:
    return Document(
        blocks=blocks,
        fonts=FontStats(body_size=10.0, heading_thresholds=[]),
        page_rects=[(0.0, 0.0, 595.0, 842.0)],
        source_path=source_path,
    )


# ---------------------------------------------------------------------------
# TableData
# ---------------------------------------------------------------------------

class TestTableData:
    def test_dimensions(self):
        t = TableData(rows=[["a", "b", "c"], ["1", "2", "3"]], page=0,
                      bbox=(0, 0, 100, 100))
        assert t.n_rows == 2
        assert t.n_cols == 3

    def test_header_detected_when_first_row_is_textual(self):
        t = TableData(rows=[["Model", "BLEU"], ["base", "26.5"]], page=0,
                      bbox=(0, 0, 100, 100))
        assert t.header == ["Model", "BLEU"]

    def test_header_rejected_when_first_row_is_mostly_numeric(self):
        t = TableData(rows=[["26.5", "38.2"], ["27.9", "41.2"]], page=0,
                      bbox=(0, 0, 100, 100))
        assert t.header is None

    def test_header_rejected_when_first_row_has_empty_cells(self):
        t = TableData(rows=[["Model", "", "BLEU"], ["a", "b", "c"]], page=0,
                      bbox=(0, 0, 100, 100))
        assert t.header is None

    def test_to_csv_is_parseable(self):
        t = TableData(rows=[["a", "b"], ["1", "2"]], page=0, bbox=(0, 0, 1, 1))
        csv = t.to_csv()
        assert "a,b" in csv.replace("\r", "")
        assert "1,2" in csv.replace("\r", "")


# ---------------------------------------------------------------------------
# Overlap ratio
# ---------------------------------------------------------------------------

class TestOverlapRatio:
    def test_fully_contained_block_gives_one(self):
        block = (10, 10, 20, 20)
        table = (0, 0, 100, 100)
        assert _overlap_ratio(block, table) == 1.0

    def test_disjoint_bboxes_give_zero(self):
        block = (0, 0, 10, 10)
        table = (50, 50, 100, 100)
        assert _overlap_ratio(block, table) == 0.0

    def test_half_overlap(self):
        block = (0, 0, 10, 10)  # area 100
        table = (5, 0, 15, 10)  # overlap is (5,0)-(10,10) = 50
        assert _overlap_ratio(block, table) == pytest.approx(0.5)

    def test_zero_area_block(self):
        assert _overlap_ratio((0, 0, 0, 0), (0, 0, 100, 100)) == 0.0


# ---------------------------------------------------------------------------
# Caption matching
# ---------------------------------------------------------------------------

class TestFindCaption:
    def test_picks_nearest_table_caption_on_same_page(self):
        table = TableData(rows=[["a", "b"], ["c", "d"]], page=0,
                          bbox=(100, 400, 300, 500))
        near = _block(
            "Table 1: Near caption.", kind="caption",
            bbox=(100, 380, 300, 395),
        )
        far = _block(
            "Table 2: Far caption.", kind="caption",
            bbox=(100, 50, 300, 65),
        )
        unrelated = _block(
            "Figure 3: Figure caption.", kind="caption",
            bbox=(100, 410, 300, 425),
        )
        assert _find_caption(table, [near, far, unrelated]) == "Table 1: Near caption."

    def test_returns_none_when_no_caption_matches(self):
        table = TableData(rows=[["a", "b"], ["c", "d"]], page=0,
                          bbox=(100, 400, 300, 500))
        captions = [_block("Figure 1: Not a table caption.", kind="caption")]
        assert _find_caption(table, captions) is None

    def test_ignores_captions_on_other_pages(self):
        table = TableData(rows=[["a", "b"], ["c", "d"]], page=0,
                          bbox=(100, 400, 300, 500))
        other_page = _block(
            "Table 1: Wrong page.", kind="caption",
            bbox=(100, 395, 300, 410), page=2,
        )
        assert _find_caption(table, [other_page]) is None


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

class TestRenderSkip:
    def test_announces_rows_and_columns(self):
        t = TableData(rows=[["a", "b"], ["1", "2"], ["3", "4"]], page=0,
                      bbox=(0, 0, 1, 1))
        out = render_table(t, TablePolicy(mode="skip"))
        assert "2 columns" in out
        assert "3 rows" in out
        assert "see the paper" in out

    def test_includes_caption_if_present(self):
        t = TableData(
            rows=[["a", "b"], ["1", "2"]], page=0, bbox=(0, 0, 1, 1),
            caption="Table 1: BLEU scores across models.",
        )
        out = render_table(t, TablePolicy(mode="skip"))
        assert "Table 1" in out


class TestRenderVerbatim:
    def test_pairs_headers_with_cells(self):
        t = TableData(
            rows=[["Model", "BLEU"], ["Base", "26.5"], ["Big", "27.9"]],
            page=0, bbox=(0, 0, 1, 1),
        )
        out = render_table(t, TablePolicy(mode="verbatim"))
        assert "Model: Base" in out
        assert "BLEU: 26.5" in out
        assert "Model: Big" in out

    def test_falls_back_to_cells_when_no_header(self):
        t = TableData(
            rows=[["26.5", "38.2"], ["27.9", "41.2"]],
            page=0, bbox=(0, 0, 1, 1),
        )
        out = render_table(t, TablePolicy(mode="verbatim"))
        assert "26.5" in out and "41.2" in out

    def test_truncates_long_tables(self):
        rows = [["A", "B"]] + [["r", str(i)] for i in range(30)]
        t = TableData(rows=rows, page=0, bbox=(0, 0, 1, 1))
        out = render_table(t, TablePolicy(mode="verbatim", verbatim_max_rows=5))
        assert "Table continues with 25 more rows" in out


class TestRenderProse:
    def test_uses_llm_output_when_safe(self):
        t = TableData(
            rows=[["Model", "BLEU"], ["Base", "26.5"], ["Big", "27.9"]],
            page=0, bbox=(0, 0, 1, 1), caption="Table 1: BLEU",
        )
        llm = lambda prompt: "The base model scores 26.5 BLEU; the big model scores 27.9."
        out = render_table(t, TablePolicy(mode="prose"), llm=llm)
        assert "26.5" in out and "27.9" in out

    def test_falls_back_to_source_cells_when_llm_missing(self):
        t = TableData(
            rows=[["Model", "BLEU"], ["Base", "26.5"]],
            page=0, bbox=(0, 0, 1, 1),
        )
        out = render_table(t, TablePolicy(mode="prose"), llm=None)
        assert out == "Model: Base; BLEU: 26.5."

    def test_rejects_invented_numeric_values(self):
        t = TableData(
            rows=[["Model", "BLEU"], ["Base", "26.5"]],
            page=0, bbox=(0, 0, 1, 1),
        )
        # LLM hallucinates a value (99.9) that is not in the CSV.
        llm = lambda prompt: "The base model scored 99.9 BLEU."
        out = render_table(t, TablePolicy(mode="prose"), llm=llm)
        assert "99.9" not in out
        assert out == "Model: Base; BLEU: 26.5."

    def test_rejects_unstructured_prose(self):
        t = TableData(
            rows=[["Model", "BLEU"], ["Base", "26.5"]],
            page=0, bbox=(0, 0, 1, 1),
        )
        llm = lambda prompt: "Three rows compare two approaches on BLEU."
        out = render_table(t, TablePolicy(mode="prose"), llm=llm)
        assert out == "Model: Base; BLEU: 26.5."

    def test_falls_back_when_llm_raises(self):
        t = TableData(rows=[["a", "b"], ["1", "2"]], page=0, bbox=(0, 0, 1, 1))
        def broken(_):
            raise RuntimeError("API down")
        out = render_table(t, TablePolicy(mode="prose"), llm=broken)
        assert out == "a: 1; b: 2."


class TestProseSafety:
    def test_decimal_in_csv_is_safe(self):
        t = TableData(rows=[["a", "b"], ["1.5", "2.5"]], page=0, bbox=(0, 0, 1, 1))
        assert _prose_is_safe("Scores range from 1.5 to 2.5.", t) is True

    def test_decimal_absent_from_csv_is_unsafe(self):
        t = TableData(rows=[["a", "b"], ["1.5", "2.5"]], page=0, bbox=(0, 0, 1, 1))
        assert _prose_is_safe("Average is 9.9.", t) is False


# ---------------------------------------------------------------------------
# handle_tables — monkeypatched integration
# ---------------------------------------------------------------------------

class TestHandleTables:
    def test_noop_without_source_path(self):
        blocks = [_block("Body prose here.")]
        doc = _doc(blocks)
        out = handle_tables(doc)
        assert out is doc  # exact same object, no work done

    def test_collapses_matched_body_block_into_table(self, monkeypatch):
        table_bbox = (100.0, 200.0, 400.0, 300.0)
        table = TableData(
            rows=[["Model", "BLEU"], ["Base", "26.5"]],
            page=0, bbox=table_bbox,
        )
        body = _block(
            "Model BLEU Base 26.5",
            bbox=(110.0, 210.0, 390.0, 290.0),  # fully inside table bbox
        )
        other = _block(
            "Unrelated paragraph elsewhere on the page.",
            bbox=(50.0, 500.0, 500.0, 550.0),
        )
        doc = _doc([body, other], source_path="/fake/path.pdf")
        monkeypatch.setattr(
            "pipeline.tables._extract_raw_tables",
            lambda path, policy: [table],
        )
        out = handle_tables(doc, TablePolicy(mode="skip"))
        kinds = [b.kind for b in out.blocks]
        assert "table" in kinds
        assert kinds.count("table") == 1
        # Non-matching body block is preserved.
        assert any(b.text.startswith("Unrelated") for b in out.blocks)
        # Table payload is attached to the promoted block.
        tblock = next(b for b in out.blocks if b.kind == "table")
        assert isinstance(tblock.meta["table"], TableData)
        assert tblock.meta["table"].n_rows == 2

    def test_false_positive_detection_ignored_when_no_body_block_matches(
        self, monkeypatch
    ):
        # pdfplumber flags a region, but only an equation fragment overlaps —
        # treat as false positive and leave blocks untouched.
        fake_table = TableData(
            rows=[["a", "b"], ["1", "2"]],
            page=0, bbox=(100.0, 100.0, 200.0, 150.0),
        )
        eq = _block(
            "α = β", kind="equation_display",
            bbox=(110.0, 110.0, 190.0, 140.0),
        )
        doc = _doc([eq], source_path="/fake/path.pdf")
        monkeypatch.setattr(
            "pipeline.tables._extract_raw_tables",
            lambda path, policy: [fake_table],
        )
        out = handle_tables(doc, TablePolicy(mode="skip"))
        # Nothing was promoted; equation block remains untouched.
        assert [b.kind for b in out.blocks] == ["equation_display"]

    def test_drops_secondary_fragments_inside_table_bbox(self, monkeypatch):
        table = TableData(
            rows=[["a", "b"], ["1", "2"]], page=0,
            bbox=(100.0, 100.0, 400.0, 300.0),
        )
        # Primary body block and two small noise fragments, all inside bbox.
        primary = _block(
            "Large primary body with multiple cells strung together here.",
            bbox=(110.0, 110.0, 390.0, 290.0),
        )
        frag_a = _block(
            "a", kind="noise", bbox=(120.0, 120.0, 130.0, 130.0),
        )
        frag_b = _block(
            "b", kind="noise", bbox=(200.0, 120.0, 210.0, 130.0),
        )
        outside = _block("Keep me.", bbox=(50.0, 500.0, 500.0, 520.0))
        doc = _doc(
            [primary, frag_a, frag_b, outside], source_path="/fake/path.pdf"
        )
        monkeypatch.setattr(
            "pipeline.tables._extract_raw_tables",
            lambda path, policy: [table],
        )
        out = handle_tables(doc, TablePolicy(mode="skip"))
        # One table + one preserved outside block = 2 blocks total.
        assert len(out.blocks) == 2
        assert out.blocks[0].kind == "table"
        assert out.blocks[1].text == "Keep me."

    def test_indices_assigned_in_reading_order(self, monkeypatch):
        t1 = TableData(
            rows=[["a", "b"], ["1", "2"]], page=0,
            bbox=(100.0, 100.0, 400.0, 150.0),
        )
        t2 = TableData(
            rows=[["c", "d"], ["3", "4"]], page=0,
            bbox=(100.0, 400.0, 400.0, 450.0),
        )
        b1 = _block("row1 body", bbox=(110.0, 110.0, 390.0, 140.0))
        b2 = _block("row2 body", bbox=(110.0, 410.0, 390.0, 440.0))
        doc = _doc([b2, b1], source_path="/fake/path.pdf")  # deliberately reverse
        monkeypatch.setattr(
            "pipeline.tables._extract_raw_tables",
            lambda path, policy: [t2, t1],
        )
        out = handle_tables(doc, TablePolicy(mode="skip"))
        tables = [b.meta["table"] for b in out.blocks if b.kind == "table"]
        # Index 1 should be the higher-on-page (smaller y0) table.
        by_index = sorted(tables, key=lambda t: t.index)
        assert by_index[0].bbox[1] < by_index[1].bbox[1]


# ---------------------------------------------------------------------------
# Integration — real PDFs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(TWO_COL_PDF), reason="sample PDF missing")
class TestTwoColumnPaperTables:
    @pytest.fixture(scope="class")
    def doc(self):
        d = classify_blocks(extract_layout(TWO_COL_PDF))
        return handle_tables(d)

    def test_at_least_one_table_block_detected(self, doc):
        tables = [b for b in doc.blocks if b.kind == "table"]
        assert len(tables) >= 1

    def test_table_blocks_carry_structured_payload(self, doc):
        for b in doc.blocks:
            if b.kind == "table":
                t = b.meta.get("table")
                assert isinstance(t, TableData)
                assert t.n_rows >= 2 and t.n_cols >= 2

    def test_skip_mode_text_replaces_garbled_cells(self, doc):
        for b in doc.blocks:
            if b.kind == "table":
                assert "see the paper" in b.text


@pytest.mark.skipif(not os.path.exists(ONE_COL_PDF), reason="sample PDF missing")
class TestOneColumnPaperTables:
    @pytest.fixture(scope="class")
    def doc(self):
        d = classify_blocks(extract_layout(ONE_COL_PDF))
        return handle_tables(d)

    def test_finds_multiple_tables(self, doc):
        tables = [b for b in doc.blocks if b.kind == "table"]
        # Titans has several evaluation tables.
        assert len(tables) >= 2
