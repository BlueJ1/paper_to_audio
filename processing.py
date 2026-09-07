"""Shared CLI/web policy, optional cache, and inspectable conversion results."""
from contextlib import nullcontext
from dataclasses import asdict
from pipeline import PipelineConfig, run_pipeline, LLMCache, document_to_dict
from pipeline.serialize import serialize, require_narration
from pipeline.diagnostics import messages
from providers import build_llm, wrap_langchain_llm, DEFAULT_LLM_MODEL, DEFAULT_CEREBRAS_MODEL


def build_config(use_llm=False):
    config = PipelineConfig()
    config.table.mode = "prose" if use_llm else "verbatim"
    config.equation.mode = "symbolic" if use_llm else "skip"
    config.inline_math.mode = "llm" if use_llm else "symbolic"
    return config


def process_pdf(path, use_llm=False, provider="google", model=None, cache_path=None, llm=None):
    model = model or (DEFAULT_CEREBRAS_MODEL if provider == "cerebras" else DEFAULT_LLM_MODEL)
    if use_llm and llm is None:
        llm = wrap_langchain_llm(build_llm(model=model, provider=provider))
    if not use_llm:
        llm = None
    notices = []
    token = messages.set(notices)
    try:
        context = LLMCache(cache_path, model=f"{provider}:{model}", prompt_version="structured-v2") if cache_path and use_llm else nullcontext()
        with context as cache:
            if cache:
                llm = cache.wrap_text(llm)
            doc, stats = run_pipeline(str(path), config=build_config(use_llm), llm=llm, collect_stats=True)
        for b in doc.blocks:
            if b.meta.get("handler_error"):
                notices.append(f"Omitted rejected block on page {b.page + 1}: {b.meta['handler_error']}")
        text = require_narration(serialize(doc))
        if use_llm:
            notices.append("Review generated equation narration against the paper; prose and table cells are preserved by construction.")
        report = {"provider": provider if use_llm else None, "model": model if use_llm else None,
                  "warnings": notices, "stats": [asdict(s) for s in stats], "document": document_to_dict(doc)}
        return text, report
    finally:
        messages.reset(token)
