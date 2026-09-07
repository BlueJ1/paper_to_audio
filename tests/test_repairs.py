"""Audit regressions independent of optional paper fixtures and cloud providers."""
import copy
import io
import json
import subprocess
import sys
from pathlib import Path

import fitz
import pytest
from pydub import AudioSegment

from pipeline import Block, Document, FontStats, run_pipeline
from pipeline.classify import classify_blocks
from pipeline.equations import _symbolic_rewrite, _prepend_equation_number
from pipeline.inline_math import _render_llm, InlineMathPolicy
from pipeline.polish import _polish_text, PolishPolicy, _PolishState
from pipeline.policies import filter_sections
from pipeline.qa import safe_block, _encode_meta
from pipeline.serialize import serialize, require_narration
from pipeline.tables import TableData, TablePolicy, render_table
from processing import build_config, process_pdf
from text_to_speech import split_text, concatenate_audio, Settings, synthesize, _collect_ordered, MurfTTSEngine


def block(text, kind="body", level=None):
    return Block(kind, [], text, 0, (0, 0, 100, 20), level=level)


def document(*blocks):
    return Document(list(blocks), FontStats(10))


@pytest.fixture
def tiny_pdf(tmp_path):
    path = tmp_path / "paper.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 60), "A Novel Method for Learning", fontsize=18)
        page.insert_text((50, 100), "1 Introduction", fontsize=13)
        page.insert_text((50, 140), "We used 1,000 MB of memory. The value is 42.", fontsize=10)
        page.insert_text((50, 200), "A. Technical Details", fontsize=13)
        page.insert_text((50, 240), "Appendix secret should be omitted.", fontsize=10)
        pdf.save(path)
    return path


def test_generated_pdf_fidelity(tiny_pdf):
    doc, _ = run_pipeline(str(tiny_pdf))
    text = serialize(doc)
    assert "A Novel Method for Learning" in text
    assert "one thousand megabytes" in text
    assert "forty-two." in text
    assert "Appendix secret" not in text


@pytest.mark.parametrize("heading", ["A. Technical Details", "A Proofs", "A Appendix", "6 Supplementary Material", "Appendix A"])
def test_appendix_variants_and_descendants(heading):
    doc = document(block("A Novel Method", "heading", 1), block("1 Introduction", "heading", 2),
                   block("Main prose."), block(heading, "heading", 2), block("Private appendix."),
                   block("A.1 Details", "heading", 3), block("Nested appendix."))
    out = serialize(filter_sections(doc))
    assert "Main prose" in out and "A Novel Method" in out
    assert "appendix" not in out.lower()


def test_unknown_heading_closes_unknown_skip():
    doc = document(block("References", "heading"), block("Citation"),
                   block("Discussion", "heading"), block("Main prose"))
    assert serialize(filter_sections(doc)) == "Discussion\n\nMain prose"


def test_ambiguous_front_matter_kept():
    doc = document(block("An author"), block("A Novel Method", "heading", 1), block("Main text"))
    assert filter_sections(doc).blocks == doc.blocks


@pytest.mark.parametrize(("source", "expected"), [
    ("We used 1,000 MB of memory.", "We used one thousand megabytes of memory."),
    ("The value is 42.", "The value is forty-two."),
    ("The value is 3.14.", "The value is three point one four."),
    ("-1,000.5 MB", "negative one thousand point five megabytes"),
])
def test_quantities(source, expected):
    assert _polish_text(source, PolishPolicy(), _PolishState()) == expected


@pytest.mark.parametrize(("source", "expected"), [
    ("x_α = y", "x sub alpha equals y"),
    ("x²³", "x to the 23"),
    ("𝜃 = 𝑥", "theta equals x"),
    ("x + y - z / t", "x plus y minus z over t"),
])
def test_equation_meaning(source, expected):
    assert _symbolic_rewrite(source) == expected


def test_equation_prefix_is_not_suppressed_by_cardinality():
    assert _prepend_equation_number("Two vectors are equal.", "2").startswith("Equation two:")
    assert _prepend_equation_number("Equation two: x equals y.", "2") == "Equation two: x equals y."


def test_rejected_content_is_only_diagnostic():
    def broken(_):
        raise ValueError("bad extraction")
    failed = safe_block(broken, block("secret raw equation", "equation_display"))
    assert failed.meta["raw"] == "secret raw equation"
    assert serialize(document(failed, block("Bad output", "noise"), block("Kept prose"))) == "Kept prose"
    with pytest.raises(ValueError, match="No usable narration"):
        require_narration(serialize(document(failed)))


def test_blank_pdf_actionable(tmp_path):
    path = tmp_path / "blank.pdf"
    with fitz.open() as pdf:
        pdf.new_page(); pdf.save(path)
    with pytest.raises(ValueError, match="OCR"):
        process_pdf(path)


def test_classification_pure():
    doc = document(block("arXiv:2501.00663v1 [cs.LG]"))
    before = copy.deepcopy(doc)
    out = classify_blocks(doc)
    assert doc == before and out.blocks[0].kind == "noise"
    assert out is not doc and out.blocks[0] is not doc.blocks[0]


def test_mixed_metadata():
    assert json.loads(json.dumps(_encode_meta({1, "two"}))) == [1, "two"]


def test_shared_modes():
    enabled, disabled = build_config(True), build_config(False)
    assert (enabled.table.mode, enabled.equation.mode, enabled.inline_math.mode) == ("prose", "symbolic", "llm")
    assert (disabled.table.mode, disabled.equation.mode, disabled.inline_math.mode) == ("verbatim", "skip", "symbolic")


def test_selected_spy_calls_and_cache(tmp_path, monkeypatch):
    import processing
    from pipeline.model import Span
    # Exercise all real policy handlers with a deterministic layout, no sample PDF.
    equation = block("x = y + z", "equation_display")
    prose = block("We do not claim causality for x or y.")
    prose.spans = [Span("x", "Times", 10, 2, (0, 0, 1, 1)), Span("y", "Times", 10, 2, (0, 0, 1, 1))]
    doc = document(equation, prose)
    import pipeline.qa as qa
    monkeypatch.setattr(qa, "extract_layout", lambda _: copy.deepcopy(doc))
    monkeypatch.setattr(qa, "classify_blocks", lambda d: d)
    calls = []
    def spy(prompt):
        calls.append(prompt)
        return '["x", "y"]' if "Math spans:" in prompt else "x equals the sum of y and z."
    cache = tmp_path / "cache.sqlite"
    text, report = processing.process_pdf("unused", True, llm=spy, cache_path=cache)
    assert len(calls) == 2 and "We do not claim causality" in text
    processing.process_pdf("unused", True, llm=spy, cache_path=cache)
    assert len(calls) == 2
    processing.process_pdf("unused", False, llm=lambda _: pytest.fail("deterministic called LLM"))
    assert report["stats"] and report["document"]


@pytest.mark.parametrize("response", [
    "We do claim causality for x.", '["x; causality is established"]', '["y"]', '{"text": "x"}',
])
def test_inline_adversarial(response):
    marked = "We do not claim causality for ⟦x⟧."
    assert _render_llm(marked, InlineMathPolicy(min_runs_for_llm=1), lambda _: response) is None


def test_inline_splice():
    marked = "We do not claim causality for ⟦α⟧."
    assert _render_llm(marked, InlineMathPolicy(min_runs_for_llm=1), lambda _: '["alpha"]') == "We do not claim causality for alpha."


@pytest.mark.parametrize("response", ['The score is 999 percent.', '{"rows":[true]}', '{"rows":[9]}',
                                      '{"rows":[0,0]}', '{"rows":[0],"score":99}', '{"rows":[]}'])
def test_table_adversarial(response):
    table = TableData([["Model", "Score"], ["A", "12%"], ["B", "98%"]], (0,0,1,1), 0)
    text = render_table(table, TablePolicy(mode="prose"), lambda _: response)
    assert "Model: A; Score: 12%" in text and "Model: B; Score: 98%" in text
    assert "999" not in text


def test_table_selection_preserves_associations():
    table = TableData([["Model", "Score"], ["A", "12%"], ["B", "98%"]], (0,0,1,1), 0)
    text = render_table(table, TablePolicy(mode="prose"), lambda _: '{"rows":[1]}')
    assert "Model: B; Score: 98%" in text and "12%" not in text


@pytest.mark.parametrize("limit", [1, 2, 10, 2800])
@pytest.mark.parametrize("source", ["word " * 1000, "x" * 3001, "one. two?\n\nthird paragraph!\nlast line", " \n\t "])
def test_chunk_bounds_and_exact_order(source, limit):
    chunks = split_text(source, limit)
    assert all(0 < len(c) <= limit for c in chunks)
    assert "".join(chunks) == " ".join(source.split())


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_limits(limit):
    with pytest.raises(ValueError):
        split_text("a", limit)


@pytest.mark.parametrize("format", ["mp3", "wav", "flac", "ogg"])
def test_audio_formats_and_atomic_success(tmp_path, format):
    buf = io.BytesIO()
    AudioSegment.silent(100).set_frame_rate(48000).export(buf, format=format, **({"codec": "libopus"} if format == "ogg" else {}))
    destination = tmp_path / "result.mp3"
    concatenate_audio([buf.getvalue(), buf.getvalue()], destination, format)
    assert len(AudioSegment.from_mp3(destination)) == 200
    assert list(tmp_path.iterdir()) == [destination]


def test_audio_failure_preserves_output_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "result.mp3"
    path.write_bytes(b"previous output")
    with pytest.raises(ValueError):
        concatenate_audio([], path)
    with pytest.raises(Exception):
        concatenate_audio([b"invalid audio"], path)
    buf = io.BytesIO(); AudioSegment.silent(100).export(buf, format="wav")
    def broken_export(self, destination, **kwargs):
        Path(destination).write_bytes(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(AudioSegment, "export", broken_export)
    with pytest.raises(OSError):
        concatenate_audio([buf.getvalue()], path, "wav")
    assert path.read_bytes() == b"previous output" and list(tmp_path.iterdir()) == [path]


def settings(**overrides):
    return Settings(**{**dict(tts_engine="kokoro", murf_api_key="", murf_voice_id="test", murf_format="mp3",
                             murf_chunk_chars=2800, kokoro_voice="af_bella", kokoro_workers=1), **overrides})


def test_validation_precedes_model_initialization(monkeypatch, tmp_path):
    import text_to_speech as tts
    monkeypatch.setattr(tts, "KokoroTTSEngine", lambda _: pytest.fail("model initialized"))
    for text, opts, workers in [("", {}, 1), ("text", {}, 0), ("text", {"murf_chunk_chars": 0}, 1)]:
        with pytest.raises(ValueError):
            synthesize(text, tmp_path / "out.mp3", settings(**opts), workers)


def test_worker_results_ordered():
    from concurrent.futures import Future
    first, second = Future(), Future()
    second.set_result(b"second"); first.set_result(b"first")
    assert _collect_ordered({second: 1, first: 0}, 2) == [b"first", b"second"]


def test_murf_http_failures(monkeypatch):
    import requests
    class Failure:
        def raise_for_status(self):
            raise requests.HTTPError("503")
    monkeypatch.setattr(requests, "post", lambda *a, **kw: Failure())
    with pytest.raises(requests.HTTPError):
        MurfTTSEngine(settings(tts_engine="murf")).generate_speech("Hello")


def test_optional_dependencies_not_imported():
    code = '''import sys
import main, app, pdf_to_text, text_to_speech
assert not any(n == 'kokoro' or n == 'torch' or n.startswith('langchain') for n in sys.modules)
'''
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("script", ["main.py", "text_to_speech.py"])
def test_cli_invalid_limit_before_providers(script, tmp_path):
    result = subprocess.run([sys.executable, script, "missing", "--max-chars", "0"], capture_output=True, text=True)
    assert result.returncode == 2 and "positive" in result.stderr


def test_missing_pronunciation_model_fails_before_kokoro(monkeypatch):
    import importlib.metadata as metadata
    from text_to_speech import _load_kokoro_pipeline
    def missing(_):
        raise metadata.PackageNotFoundError("en-core-web-sm")
    monkeypatch.setattr(metadata, "version", missing)
    with pytest.raises(RuntimeError, match="requirements-kokoro"):
        _load_kokoro_pipeline()


def test_provider_selection_and_response_adapter(monkeypatch):
    from types import SimpleNamespace
    from providers import build_llm, wrap_langchain_llm, DEFAULT_CEREBRAS_MODEL
    calls = []
    def model(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(invoke=lambda _: SimpleNamespace(content=[{"type": "text", "text": "hello"}]))
    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=model))
    monkeypatch.setitem(sys.modules, "langchain_google_genai", SimpleNamespace(ChatGoogleGenerativeAI=model))
    monkeypatch.setenv("CEREBRAS_API_KEY", "test")
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    chat = build_llm(provider="cerebras")
    assert calls[-1]["model"] == DEFAULT_CEREBRAS_MODEL
    assert wrap_langchain_llm(chat)("prompt") == "hello"
    build_llm(provider="google")
    assert calls[-1]["vertexai"] is False
    assert calls[-1]["timeout"] == 120
    with pytest.raises(ValueError):
        build_llm(provider="unknown")


def test_murf_format_sent_uppercase(monkeypatch):
    import requests
    from types import SimpleNamespace
    payloads = []
    def post(*args, **kwargs):
        payloads.append(kwargs["json"])
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"audioFile": "https://example.test/audio"})
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", lambda *a, **kw: SimpleNamespace(raise_for_status=lambda: None, content=b"wav"))
    assert MurfTTSEngine(settings(tts_engine="murf", murf_format="wav")).generate_speech("hello") == b"wav"
    assert payloads[0]["format"] == "WAV"


def test_setup_only_checks_selected_modes(monkeypatch):
    import check_setup
    modules, keys = [], []
    monkeypatch.setattr(check_setup, "check_import", lambda module: modules.append(module) or True)
    monkeypatch.setattr(check_setup, "check_env_var", lambda key: keys.append(key) or True)
    assert check_setup.main([]) == 0
    assert not keys and "kokoro" not in modules and "langchain_google_genai" not in modules
    assert check_setup.main(["--llm-provider", "cerebras"]) == 0
    assert keys == ["CEREBRAS_API_KEY"] and "langchain_openai" in modules


def test_ssml_uses_final_eligibility_even_with_custom_policy():
    from pipeline.polish import to_ssml
    rejected = block("rejected body")
    rejected.meta["handler_error"] = "failure"
    doc = document(rejected, block("noise", "noise"), block("Kept text"))
    assert to_ssml(doc, PolishPolicy(skip_kinds=frozenset())) == "<speak><p>Kept text</p></speak>"
