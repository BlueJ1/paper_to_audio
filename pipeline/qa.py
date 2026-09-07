"""Phase 8 — caching, stats, inspection, fail-soft.

Earlier phases produce all the value; this phase makes iteration on that
value cheap and diagnosable:

- `LLMCache` wraps any `LLMCallable` (or vision `VisionLLMCallable`) with
  a SQLite-backed cache keyed by `sha256(model | prompt_version | prompt)`.
  Re-runs after a prompt edit are free for every paragraph the new prompt
  version was already computed on; bumping `prompt_version` invalidates
  exactly what needs to be recomputed.
- `document_stats(doc, stage)` and `compare_stages(before, after)` produce
  structured per-stage metrics — block-kind counts and character totals,
  plus the delta between any two snapshots. Useful both for logging and
  for regression tests that pin block-kind distributions.
- `document_to_dict(doc)` / `document_to_json(doc, pretty=True)` emit a
  lossy JSON representation of a `Document` for inspection dumps and
  debugging. `meta` payloads carrying dataclass values (EquationData,
  TableData) are flattened to plain dicts with a `__type__` marker so a
  human reader still sees what kind of payload was attached.
- `safe_block(fn, block)` is the fail-soft primitive: run `fn(block)` and,
  on any exception, return a `kind="noise"` block that preserves the
  original text under `meta["raw"]` and the exception repr under
  `meta["handler_error"]`. Handlers opt in per call site.
- `run_pipeline(path, ...)` is a thin convenience that runs every shipped
  phase in order with a single policy bundle — handy for regression tests
  and for a future `--inspect` CLI.

All functions are pure with respect to their inputs except `LLMCache`,
which is stateful by design (SQLite writes). `LLMCache.close()` is safe
to call multiple times.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

from pipeline.classify import classify_blocks
from pipeline.equations import EquationPolicy, handle_equations
from pipeline.extract import extract_layout
from pipeline.inline_math import InlineMathPolicy, rewrite_inline_math
from pipeline.model import Block, Document
from pipeline.policies import SectionPolicy, filter_sections
from pipeline.polish import PolishPolicy, audio_polish
from pipeline.tables import TablePolicy, handle_tables


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageStats:
    """Snapshot of a Document's shape at one pipeline stage.

    `stage` is a free-form label (e.g. `"extract"`, `"classify"`, `"polish"`)
    chosen by the caller — `document_stats` does not inspect it. `by_kind`
    maps every `BlockKind` observed in the doc to its block count;
    `chars_by_kind` does the same for `sum(len(b.text))`.
    """
    stage: str
    total_blocks: int
    total_chars: int
    by_kind: dict[str, int]
    chars_by_kind: dict[str, int]


def document_stats(doc: Document, stage: str = "unspecified") -> StageStats:
    """Compute block-kind and character-count distributions for `doc`.

    Cheap (one linear pass); safe to call after every phase. Empty blocks
    (whitespace-only `text`) contribute 0 characters but still count as
    blocks — upstream phases don't emit truly empty blocks, but the count
    is preserved as-is so callers see exactly what the Document holds.
    """
    by_kind: dict[str, int] = {}
    chars_by_kind: dict[str, int] = {}
    for b in doc.blocks:
        by_kind[b.kind] = by_kind.get(b.kind, 0) + 1
        chars_by_kind[b.kind] = chars_by_kind.get(b.kind, 0) + len(b.text)
    return StageStats(
        stage=stage,
        total_blocks=len(doc.blocks),
        total_chars=sum(chars_by_kind.values()),
        by_kind=by_kind,
        chars_by_kind=chars_by_kind,
    )


def compare_stages(before: StageStats, after: StageStats) -> dict[str, Any]:
    """Structured delta between two `StageStats` snapshots.

    Keys present in either snapshot appear in `by_kind_delta` (missing sides
    are treated as 0). Useful for assertions like "filter_sections reduced
    footnotes to 0" or "handle_tables merged 47 blocks into 3 tables".
    """
    kinds = set(before.by_kind) | set(after.by_kind)
    return {
        "before_stage": before.stage,
        "after_stage": after.stage,
        "block_delta": after.total_blocks - before.total_blocks,
        "char_delta": after.total_chars - before.total_chars,
        "by_kind_delta": {
            k: after.by_kind.get(k, 0) - before.by_kind.get(k, 0)
            for k in sorted(kinds)
        },
    }


# ---------------------------------------------------------------------------
# JSON dump (lossy — for inspection only, not round-trip)
# ---------------------------------------------------------------------------

def document_to_dict(doc: Document) -> dict[str, Any]:
    """Plain-dict representation of a Document for inspection dumps.

    Lossy: span/block bboxes become lists, dataclass `meta` values are
    flattened via `_encode_meta` with a `__type__` tag, and any value that
    resists JSON encoding falls back to `repr()`. Not intended for
    round-trip — use `pickle` for that.
    """
    return {
        "language": doc.language,
        "title": doc.title,
        "abstract": doc.abstract,
        "source_path": doc.source_path,
        "page_rects": [list(r) for r in doc.page_rects],
        "fonts": {
            "body_size": doc.fonts.body_size,
            "heading_thresholds": list(doc.fonts.heading_thresholds),
        },
        "blocks": [_block_to_dict(b) for b in doc.blocks],
    }


def document_to_json(doc: Document, pretty: bool = True) -> str:
    """Serialize `document_to_dict(doc)` to JSON."""
    d = document_to_dict(doc)
    return json.dumps(d, indent=2 if pretty else None, ensure_ascii=False)


def _block_to_dict(b: Block) -> dict[str, Any]:
    return {
        "kind": b.kind,
        "text": b.text,
        "page": b.page,
        "bbox": list(b.bbox),
        "column": b.column,
        "level": b.level,
        "parent_section": b.parent_section,
        "dominant_size": b.dominant_size,
        "spans": [_span_to_dict(s) for s in b.spans],
        "meta": _encode_meta(b.meta),
    }


def _span_to_dict(s) -> dict[str, Any]:
    return {
        "text": s.text,
        "font": s.font,
        "size": s.size,
        "flags": s.flags,
        "bbox": list(s.bbox),
    }


def _encode_meta(value: Any) -> Any:
    """Best-effort JSON-safe encoding.

    Dataclasses are flattened via `asdict` and tagged with `__type__` so a
    reader sees that the payload was e.g. an `EquationData`. Tuples become
    lists; sets become sorted lists. Anything else we don't recognize
    falls back to its `repr()`.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        d = asdict(value)
        d["__type__"] = type(value).__name__
        return _encode_meta(d)
    if isinstance(value, dict):
        return {str(k): _encode_meta(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_meta(v) for v in value]
    if isinstance(value, set):
        return sorted((_encode_meta(v) for v in value), key=lambda v: (type(v).__name__, repr(v)))
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    return repr(value)


# ---------------------------------------------------------------------------
# LLM cache
# ---------------------------------------------------------------------------

class LLMCache:
    """SQLite-backed cache wrapping one or more LLM callables.

    Keys are `sha256(model | prompt_version | prompt [ | vision_image_hash ])`
    so bumping `prompt_version` invalidates exactly the entries tied to
    that prompt iteration. Caller owns the file (default:
    `~/.cache/paper_to_audio/llm_cache.sqlite`).

    The wrapped callable has the same signature as the original; a cache
    hit short-circuits the network call. On a miss the original is invoked
    and its return is stored before being returned to the caller. Errors
    from the original callable propagate — we don't cache failures.
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS llm_cache (
            key TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """

    def __init__(
        self,
        path: str | Path,
        model: str,
        prompt_version: str = "v1",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.prompt_version = prompt_version
        # `check_same_thread=False` so a handler using a worker pool can
        # share the cache. SQLite itself is serialized by its own mutex.
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(self._SCHEMA)

    # ------------- wrapping -------------

    def wrap_text(
        self, llm: Callable[[str], str]
    ) -> Callable[[str], str]:
        """Return a cached version of an `LLMCallable` (prompt -> text)."""

        def cached(prompt: str) -> str:
            key = self._key(prompt)
            hit = self._get(key)
            if hit is not None:
                return hit
            result = llm(prompt)
            self._put(key, result)
            return result
        return cached

    def wrap_vision(
        self, vision_llm: Callable[[bytes, str], str]
    ) -> Callable[[bytes, str], str]:
        """Return a cached version of a `VisionLLMCallable` ((image, prompt) -> text)."""

        def cached(image: bytes, prompt: str) -> str:
            image_hash = hashlib.sha256(image).hexdigest()
            key = self._key(prompt, extra=image_hash)
            hit = self._get(key)
            if hit is not None:
                return hit
            result = vision_llm(image, prompt)
            self._put(key, result)
            return result
        return cached

    # ------------- housekeeping -------------

    def close(self) -> None:
        """Close the SQLite connection. Safe to call multiple times."""
        try:
            self._conn.close()
        except sqlite3.ProgrammingError:
            pass

    def __enter__(self) -> "LLMCache":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------- internals -------------

    def _key(self, prompt: str, extra: str = "") -> str:
        h = hashlib.sha256()
        h.update(self.model.encode("utf-8"))
        h.update(b"|")
        h.update(self.prompt_version.encode("utf-8"))
        h.update(b"|")
        h.update(prompt.encode("utf-8"))
        if extra:
            h.update(b"|")
            h.update(extra.encode("utf-8"))
        return h.hexdigest()

    def _get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT response FROM llm_cache WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def _put(self, key: str, response: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_cache "
            "(key, model, prompt_version, response, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (key, self.model, self.prompt_version, response, time.time()),
        )


# ---------------------------------------------------------------------------
# Fail-soft
# ---------------------------------------------------------------------------

def safe_block(
    fn: Callable[[Block], Block],
    block: Block,
) -> Block:
    """Run `fn(block)` and downgrade exceptions to a preserved noise block.

    On failure the returned block keeps the original text but its `kind`
    flips to `"noise"` and `meta` gains:
      * `raw` — the original `text`, so downstream inspectors can see what
        was supposed to be there,
      * `original_kind` — what the block had been before the failure,
      * `handler_error` — `repr(exc)` for triage.

    Callers use this inside a `for b in doc.blocks` loop where `fn` would
    otherwise crash the whole pipeline on one weird block. No-op path is
    `fn(b) -> b`, so there's zero overhead on success.
    """
    try:
        return fn(block)
    except Exception as exc:
        return replace(
            block,
            kind="noise",
            meta={
                **block.meta,
                "raw": block.text,
                "original_kind": block.kind,
                "handler_error": repr(exc),
            },
        )


# ---------------------------------------------------------------------------
# Convenience pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Bundle of per-phase policies for `run_pipeline`.

    All fields default to their phase's default policy so a bare
    `run_pipeline(path)` exercises the "safe and deterministic" path (no
    LLM calls, no vision calls). Override individual fields to enable
    LLM-backed modes per phase.
    """
    section: SectionPolicy = field(default_factory=SectionPolicy)
    table: TablePolicy = field(default_factory=TablePolicy)
    equation: EquationPolicy = field(default_factory=EquationPolicy)
    inline_math: InlineMathPolicy = field(default_factory=InlineMathPolicy)
    polish: PolishPolicy = field(default_factory=PolishPolicy)


def run_pipeline(
    path: str,
    config: PipelineConfig | None = None,
    llm: Callable[[str], str] | None = None,
    vision_llm: Callable[[bytes, str], str] | None = None,
    collect_stats: bool = False,
) -> tuple[Document, list[StageStats]]:
    """Run every shipped phase on `path` with one policy bundle.

    Returns `(doc, stats)`. When `collect_stats=False`, `stats` is `[]` —
    skipping the snapshots avoids an extra O(blocks) pass per phase. With
    `collect_stats=True` each phase snapshot is labeled by phase name so a
    caller can feed them pairwise to `compare_stages` for per-stage deltas.

    The handlers are called with `llm` / `vision_llm` as-is; callers who
    want caching should wrap those with `LLMCache.wrap_text` /
    `wrap_vision` before passing them in.
    """
    config = config or PipelineConfig()
    stats: list[StageStats] = []

    def _snap(doc: Document, stage: str) -> Document:
        if collect_stats:
            stats.append(document_stats(doc, stage))
        return doc

    doc = extract_layout(path)
    doc = _snap(doc, "extract")
    doc = _snap(classify_blocks(doc), "classify")
    doc = _snap(filter_sections(doc, config.section), "filter_sections")
    doc = _snap(handle_tables(doc, config.table, llm=llm), "handle_tables")
    doc = _snap(
        handle_equations(doc, config.equation, llm=llm, vision_llm=vision_llm),
        "handle_equations",
    )
    doc = _snap(rewrite_inline_math(doc, config.inline_math, llm=llm), "rewrite_inline_math")
    doc = _snap(audio_polish(doc, config.polish), "audio_polish")
    return doc, stats


__all__ = [
    "LLMCache",
    "PipelineConfig",
    "StageStats",
    "compare_stages",
    "document_stats",
    "document_to_dict",
    "document_to_json",
    "run_pipeline",
    "safe_block",
]
