"""
Paper to Audio - Main Pipeline
Integrates PDF-to-Text and Text-to-Speech processing.
This is the main entry point that orchestrates both steps.
"""
import argparse

from dotenv import load_dotenv

import json
from pathlib import Path
from dataclasses import replace
from processing import process_pdf
from providers import DEFAULT_LLM_MODEL, DEFAULT_CEREBRAS_MODEL
from text_to_speech import load_settings, synthesize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a paper PDF into audio using LLM + TTS (integrated pipeline)"
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--out", default="output.mp3", help="Output audio filename")
    parser.add_argument(
        "--tts-engine",
        choices=["murf", "kokoro"],
        default="murf",
        help="TTS engine to use: 'murf' (cloud API, paid) or 'kokoro' (local, free)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Maximum characters per TTS request (defaults to env MURF_CHUNK_CHARS)",
    )
    parser.add_argument(
        "--kokoro-workers",
        type=int,
        default=None,
        help="Parallel workers for Kokoro (defaults to env KOKORO_WORKERS or 1)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM processing, use deterministic narration (no LLM API cost)",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["google", "cerebras"],
        default="google",
        help="LLM provider for math rewriting: 'google' (Gemma/Gemini) or 'cerebras' (default: google)",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help=(
            f"Model for math rewriting. "
            f"Google default: {DEFAULT_LLM_MODEL}. "
            f"Cerebras default: {DEFAULT_CEREBRAS_MODEL}."
        ),
    )
    parser.add_argument(
        "--keep-text",
        action="store_true",
        help="Save the narration transcript beside the current command directory",
    )
    parser.add_argument(
        "--text-file",
        default=None,
        help="Use existing text file instead of processing PDF (skips PDF processing)",
    )
    parser.add_argument("--cache", help="Optional SQLite LLM cache path")
    parser.add_argument("--inspect", help="Save source blocks, stage counts, and warnings as JSON")
    args = parser.parse_args()

    for value in (args.max_chars, args.kokoro_workers):
        if value is not None and value <= 0:
            parser.error("Chunk limit and worker count must be positive")
    load_dotenv()
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    else:
        text, report = process_pdf(args.pdf, not args.no_llm, args.llm_provider,
                                   args.llm_model, args.cache)
        if args.inspect:
            Path(args.inspect).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        for message in report["warnings"]:
            print(f"Warning: {message}")
        if args.keep_text:
            path = Path(Path(args.pdf).stem + "_audio_text.txt")
            path.write_text(text, encoding="utf-8")
            print(f"Saved transcript to {path}")
    if not text.strip():
        parser.error("No usable text to speak")
    settings = load_settings(args.tts_engine)
    if args.max_chars is not None:
        settings = replace(settings, murf_chunk_chars=args.max_chars)
    synthesize(text, args.out, settings, args.kokoro_workers)
    print(f"Successfully saved audio to {args.out}")


if __name__ == "__main__":
    main()
