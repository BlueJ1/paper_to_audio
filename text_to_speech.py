"""
Text to Speech Generator
Converts audio-friendly text into speech using TTS engines (Murf.ai or Kokoro).
"""
import argparse
import io
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Protocol, Dict, Optional
import concurrent.futures
import multiprocessing

import requests
from dotenv import load_dotenv
from pydub import AudioSegment

from kokoro import KPipeline
import numpy as np
import soundfile as sf


class TTSEngine(Protocol):
    """Protocol for TTS engines."""
    def generate_speech(self, text: str) -> bytes:
        """Generate speech audio from text and return as bytes."""
        ...


@dataclass
class Settings:
    tts_engine: str  # "murf" or "kokoro"
    murf_api_key: str
    murf_voice_id: str
    murf_format: str
    murf_chunk_chars: int
    kokoro_voice: str
    kokoro_workers: int


def load_settings(tts_engine: str = "murf") -> Settings:
    """Load settings from environment variables."""
    load_dotenv()

    murf_api_key = os.getenv("MURF_API_KEY", "").strip()
    if tts_engine == "murf" and not murf_api_key:
        raise RuntimeError("Missing MURF_API_KEY in environment. Required when using Murf TTS engine.")

    murf_voice_id = os.getenv("MURF_VOICE_ID", "marcus").strip() or "marcus"
    murf_format = os.getenv("MURF_FORMAT", "mp3").strip() or "mp3"
    murf_chunk_chars = int(os.getenv("MURF_CHUNK_CHARS", "2800").strip())
    kokoro_voice = os.getenv("KOKORO_VOICE", "af_bella").strip() or "af_bella"
    kokoro_workers = int(os.getenv("KOKORO_WORKERS", "1").strip() or "1")

    return Settings(
        tts_engine=tts_engine,
        murf_api_key=murf_api_key,
        murf_voice_id=murf_voice_id,
        murf_format=murf_format,
        murf_chunk_chars=murf_chunk_chars,
        kokoro_voice=kokoro_voice,
        kokoro_workers=kokoro_workers,
    )


def split_text(text: str, max_chars: int) -> List[str]:
    """Split text into chunks suitable for TTS processing."""
    # Split on paragraph boundaries, then fall back to sentence boundaries.
    paragraphs = [
        " ".join(p.split()) for p in text.split("\n\n") if p.strip()
    ]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        if current:
            chunks.append(" ".join(current).strip())
            current = []
            current_len = 0

    sentence_split = re.compile(r"(?<=[.!?])\s+")

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            # Paragraph too large; split into sentences.
            sentences = sentence_split.split(paragraph)
            for sentence in sentences:
                if current_len + len(sentence) + 1 > max_chars:
                    flush()
                current.append(sentence)
                current_len += len(sentence) + 1
            continue

        if current_len + len(paragraph) + 2 > max_chars:
            flush()
        current.append(paragraph)
        current_len += len(paragraph) + 2

    flush()
    return [chunk for chunk in chunks if chunk]


class MurfTTSEngine:
    """Murf.ai cloud-based TTS engine."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def generate_speech(self, text: str) -> bytes:
        url = "https://api.murf.ai/v1/speech/generate"
        headers = {
            "api-key": self.settings.murf_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "voiceId": self.settings.murf_voice_id,
            "text": text,
            "format": self.settings.murf_format,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        # Murf API returns the audio URL in the "audioFile" field
        audio_url = data.get("audioFile") or data.get("audioUrl") or data.get("audio_url")
        if not audio_url:
            raise RuntimeError(f"Unexpected Murf response: {data}")
        audio_response = requests.get(audio_url, timeout=120)
        audio_response.raise_for_status()
        return audio_response.content


class KokoroTTSEngine:
    """Local Kokoro TTS engine."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Kokoro model from HuggingFace."""
        try:
            # Kokoro uses a phonemizer; ensure it is available on the system.
            self.pipeline = KPipeline(lang_code="a")
        except Exception as e:
            if "espeak" in str(e).lower():
                raise RuntimeError(
                    "Kokoro requires espeak-ng for phonemization. "
                    "Install with: brew install espeak-ng (macOS) or sudo apt-get install espeak-ng (Debian/Ubuntu)."
                ) from e
            raise

    def generate_speech(self, text: str) -> bytes:
        """Generate speech using Kokoro model."""
        return _kokoro_generate_bytes(self.pipeline, self.settings.kokoro_voice, text)


_KOKORO_PIPELINE: Optional[KPipeline] = None
_KOKORO_VOICE: Optional[str] = None


def _kokoro_generate_bytes(pipeline: KPipeline, voice: str, text: str) -> bytes:
    """Generate speech using Kokoro pipeline and return MP3 bytes."""
    generator = pipeline(text, voice=voice)

    audio_chunks = []
    for _, _, audio in generator:
        audio_chunks.append(audio)

    if not audio_chunks:
        raise RuntimeError("No audio generated from Kokoro pipeline")

    samples = np.concatenate(audio_chunks)
    sample_rate = 24000

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, samples, sample_rate, format="WAV")
    wav_buffer.seek(0)

    audio_segment = AudioSegment.from_wav(wav_buffer)
    mp3_buffer = io.BytesIO()
    audio_segment.export(mp3_buffer, format="mp3", bitrate="128k")

    return mp3_buffer.getvalue()


def _kokoro_worker_init(voice: str) -> None:
    global _KOKORO_PIPELINE, _KOKORO_VOICE
    _KOKORO_PIPELINE = KPipeline(lang_code="a")
    _KOKORO_VOICE = voice


def _kokoro_generate_chunk(text: str) -> bytes:
    if _KOKORO_PIPELINE is None or _KOKORO_VOICE is None:
        raise RuntimeError("Kokoro worker not initialized")
    return _kokoro_generate_bytes(_KOKORO_PIPELINE, _KOKORO_VOICE, text)


def _collect_ordered(
    futures: Dict[concurrent.futures.Future, int],
    total: int,
    progress_label: Optional[str] = None,
) -> List[bytes]:
    results: List[bytes] = [b""] * total
    for future in concurrent.futures.as_completed(futures):
        idx = futures[future]
        results[idx] = future.result()
        if progress_label:
            print(f"      Completed {progress_label} {idx + 1}/{total}...")
    return results


def _generate_kokoro_parallel(chunks: List[str], voice: str, workers: int) -> List[bytes]:
    if not chunks:
        return []
    worker_count = min(workers, len(chunks))
    mp_context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp_context,
        initializer=_kokoro_worker_init,
        initargs=(voice,),
    ) as executor:
        futures = {
            executor.submit(_kokoro_generate_chunk, chunk): i
            for i, chunk in enumerate(chunks)
        }
        return _collect_ordered(futures, len(chunks), progress_label="chunk")


def generate_audio_chunks(
    chunks: List[str],
    settings: Settings,
    tts_engine: Optional[TTSEngine],
    kokoro_workers: int,
) -> List[bytes]:
    """Generate audio for all chunks, optionally in parallel for Kokoro."""
    if settings.tts_engine == "kokoro" and kokoro_workers > 1:
        print(f"      Using {kokoro_workers} parallel workers for Kokoro...")
        return _generate_kokoro_parallel(chunks, settings.kokoro_voice, kokoro_workers)

    if tts_engine is None:
        raise RuntimeError("TTS engine is not initialized")

    audio_chunks: List[bytes] = []
    for i, chunk in enumerate(chunks, 1):
        print(f"      Processing chunk {i}/{len(chunks)}...")
        audio_chunks.append(tts_engine.generate_speech(chunk))
    return audio_chunks


def concatenate_audio(chunks: Iterable[bytes], output_path: str) -> None:
    """Properly concatenate audio chunks using pydub to maintain correct duration metadata."""
    combined = AudioSegment.empty()

    for chunk_bytes in chunks:
        # Load each chunk as an AudioSegment
        audio_segment = AudioSegment.from_file(io.BytesIO(chunk_bytes), format="mp3")
        combined += audio_segment

    # Export with proper metadata
    combined.export(output_path, format="mp3", bitrate="128k")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert audio-friendly text into speech using TTS"
    )
    parser.add_argument("text_file", help="Path to the text file")
    parser.add_argument(
        "--out",
        default=None,
        help="Output audio filename (default: <text_basename>.mp3)",
    )
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
    args = parser.parse_args()

    # Determine output filename
    if args.out is None:
        text_basename = os.path.splitext(os.path.basename(args.text_file))[0]
        # Remove "_audio_text" suffix if present
        if text_basename.endswith("_audio_text"):
            text_basename = text_basename[:-11]
        args.out = f"{text_basename}.mp3"

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

    print(f"[1/4] Loading text from: {args.text_file}")
    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"      Loaded {len(text)} characters")

    print(f"[2/4] Splitting into chunks (max {settings.murf_chunk_chars} chars each)...")
    chunks = split_text(text, settings.murf_chunk_chars)
    print(f"      Created {len(chunks)} chunks")

    # Initialize the appropriate TTS engine
    if args.tts_engine == "kokoro":
        print(f"[3/4] Generating audio with Kokoro (voice: {settings.kokoro_voice}, local)...")
        tts_engine = None if kokoro_workers > 1 else KokoroTTSEngine(settings)
    else:
        print(f"[3/4] Generating audio with Murf.ai (voice: {settings.murf_voice_id})...")
        tts_engine = MurfTTSEngine(settings)

    audio_chunks = generate_audio_chunks(chunks, settings, tts_engine, kokoro_workers)

    print(f"[4/4] Saving audio to {args.out}...")
    concatenate_audio(audio_chunks, args.out)

    print(f"✓ Successfully saved audio to {args.out}")


if __name__ == "__main__":
    main()
