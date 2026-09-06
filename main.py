"""
Paper to Audio - Main Pipeline
Integrates PDF-to-Text and Text-to-Speech processing.
This is the main entry point that orchestrates both steps.
"""
import argparse
import os
import tempfile

from dotenv import load_dotenv

from pipeline import run_pipeline, serialize, PipelineConfig
from pdf_to_text import build_llm, DEFAULT_LLM_MODEL, DEFAULT_CEREBRAS_MODEL
from text_to_speech import load_settings, split_text, MurfTTSEngine, KokoroTTSEngine, concatenate_audio, Settings, generate_audio_chunks


def _wrap_langchain_llm(chat_model) -> callable:
    """Adapt a LangChain BaseChatModel to the pipeline's Callable[[str], str]."""
    def call(prompt: str) -> str:
        result = chat_model.invoke(prompt)
        if hasattr(result, "content"):
            return str(result.content)
        return str(result)
    return call


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
        help="Skip LLM processing, use regex-only cleanup (faster, no API cost)",
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
        help="Keep the intermediate text file instead of deleting it",
    )
    parser.add_argument(
        "--text-file",
        default=None,
        help="Use existing text file instead of processing PDF (skips PDF processing)",
    )
    args = parser.parse_args()

    # Resolve model default based on provider
    if args.llm_model is None:
        args.llm_model = (
            DEFAULT_CEREBRAS_MODEL if args.llm_provider == "cerebras" else DEFAULT_LLM_MODEL
        )

    # Load environment variables
    load_dotenv()
    if not args.no_llm and not args.text_file:
        if args.llm_provider == "cerebras":
            if not os.getenv("CEREBRAS_API_KEY", "").strip():
                raise RuntimeError("Missing CEREBRAS_API_KEY in environment.")
        else:
            if not os.getenv("GOOGLE_API_KEY", "").strip():
                raise RuntimeError("Missing GOOGLE_API_KEY in environment.")

    # Determine intermediate text file path
    if args.text_file:
        text_file = args.text_file
        print(f"[1/5] Using existing text file: {text_file}")
        with open(text_file, "r", encoding="utf-8") as f:
            rewritten = f.read()
        print(f"      Loaded {len(rewritten)} characters")
    else:
        # Create temporary or persistent text file
        if args.keep_text:
            pdf_basename = os.path.splitext(os.path.basename(args.pdf))[0]
            text_file = f"{pdf_basename}_audio_text.txt"
        else:
            temp_fd, text_file = tempfile.mkstemp(suffix=".txt", prefix="audio_text_")
            os.close(temp_fd)

        # Step 1-2: Run the layout-aware pipeline
        print(f"[1/5] Loading PDF: {args.pdf}")
        llm = None
        if not args.no_llm:
            print(
                f"[2/5] Processing with pipeline (provider={args.llm_provider}, "
                f"model={args.llm_model})..."
            )
            llm = _wrap_langchain_llm(
                build_llm(model=args.llm_model, provider=args.llm_provider)
            )
        else:
            print("[2/5] Processing with pipeline (no LLM, deterministic only)...")

        doc, _ = run_pipeline(args.pdf, config=PipelineConfig(), llm=llm)
        rewritten = serialize(doc)
        print(f"      Output: {len(rewritten)} characters of audio-friendly text")

        # Save intermediate text file
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(rewritten)
        if args.keep_text:
            print(f"      Saved intermediate text to {text_file}")

    # Load TTS settings
    settings = load_settings(tts_engine=args.tts_engine)
    if args.max_chars:
        settings = Settings(
            tts_engine=settings.tts_engine,
            murf_api_key=settings.murf_api_key,
            murf_voice_id=settings.murf_voice_id,
            murf_format=settings.murf_format,
            murf_chunk_chars=args.max_chars,
            kokoro_voice=settings.kokoro_voice,
            kokoro_workers=settings.kokoro_workers,
        )

    kokoro_workers = (
        args.kokoro_workers if args.kokoro_workers is not None else settings.kokoro_workers
    )

    # Step 3: Split text into chunks
    print(f"[3/5] Splitting into chunks (max {settings.murf_chunk_chars} chars each)...")
    chunks = split_text(rewritten, settings.murf_chunk_chars)
    print(f"      Created {len(chunks)} chunks")

    # Step 4: Generate audio
    if args.tts_engine == "kokoro":
        print(f"[4/5] Generating audio with Kokoro (voice: {settings.kokoro_voice}, local)...")
        tts_engine = None if kokoro_workers > 1 else KokoroTTSEngine(settings)
    else:
        print(f"[4/5] Generating audio with Murf.ai (voice: {settings.murf_voice_id})...")
        tts_engine = MurfTTSEngine(settings)

    audio_chunks = generate_audio_chunks(chunks, settings, tts_engine, kokoro_workers)

    # Step 5: Save audio
    print(f"[5/5] Saving audio to {args.out}...")
    concatenate_audio(audio_chunks, args.out)

    # Cleanup temporary file if not keeping
    if not args.keep_text and not args.text_file:
        try:
            os.unlink(text_file)
        except:
            pass

    print(f"Successfully saved audio to {args.out}")


if __name__ == "__main__":
    main()
