"""
Paper to Audio - Main Pipeline
Integrates PDF-to-Text and Text-to-Speech processing.
This is the main entry point that orchestrates both steps.
"""
import argparse
import os
import tempfile

from pdf_to_text import load_pdf_text, build_llm, rewrite_for_audio
from text_to_speech import load_settings, split_text, MurfTTSEngine, KokoroTTSEngine, concatenate_audio, Settings
from dotenv import load_dotenv


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
        "--keep-text",
        action="store_true",
        help="Keep the intermediate text file instead of deleting it",
    )
    parser.add_argument(
        "--text-file",
        default=None,
        help="Use existing text file instead of processing PDF (skips step 1-2)",
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not google_api_key and not args.text_file:
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
            # Create temporary file
            temp_fd, text_file = tempfile.mkstemp(suffix=".txt", prefix="audio_text_")
            os.close(temp_fd)  # Close the file descriptor

        # Step 1: Load PDF
        print(f"[1/5] Loading PDF: {args.pdf}")
        paper_text = load_pdf_text(args.pdf)
        print(f"      Loaded {len(paper_text)} characters")

        # Step 2: Process with LLM
        print("[2/5] Processing with Gemini (this may take a while)...")
        llm = build_llm()
        rewritten = rewrite_for_audio(llm, paper_text)
        print(f"      Generated {len(rewritten)} characters of audio-friendly text")

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
        )

    # Step 3: Split text into chunks
    print(f"[3/5] Splitting into chunks (max {settings.murf_chunk_chars} chars each)...")
    chunks = split_text(rewritten, settings.murf_chunk_chars)
    print(f"      Created {len(chunks)} chunks")

    # Step 4: Generate audio
    if args.tts_engine == "kokoro":
        print(f"[4/5] Generating audio with Kokoro (voice: {settings.kokoro_voice}, local)...")
        tts_engine = KokoroTTSEngine(settings)
    else:
        print(f"[4/5] Generating audio with Murf.ai (voice: {settings.murf_voice_id})...")
        tts_engine = MurfTTSEngine(settings)

    audio_chunks = []
    for i, chunk in enumerate(chunks, 1):
        print(f"      Processing chunk {i}/{len(chunks)}...")
        audio_chunks.append(tts_engine.generate_speech(chunk))

    # Step 5: Save audio
    print(f"[5/5] Saving audio to {args.out}...")
    concatenate_audio(audio_chunks, args.out)

    # Cleanup temporary file if not keeping
    if not args.keep_text and not args.text_file:
        try:
            os.unlink(text_file)
        except:
            pass

    print(f"✓ Successfully saved audio to {args.out}")


if __name__ == "__main__":
    main()
