"""
Text to Speech Generator
Converts audio-friendly text into speech using TTS engines (Murf.ai or Kokoro).
"""
from __future__ import annotations

import argparse
import tempfile
from bisect import bisect_right
from pathlib import Path
from dataclasses import replace
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

    settings = Settings(
        tts_engine=tts_engine,
        murf_api_key=murf_api_key,
        murf_voice_id=murf_voice_id,
        murf_format=murf_format,
        murf_chunk_chars=murf_chunk_chars,
        kokoro_voice=kokoro_voice,
        kokoro_workers=kokoro_workers,
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings, workers=None):
    if settings.tts_engine not in {"kokoro", "murf"}:
        raise ValueError("Unknown TTS engine")
    if settings.murf_format.lower() not in {"mp3", "wav", "flac", "ogg"}:
        raise ValueError("Unsupported MURF_FORMAT; use mp3, wav, flac, or ogg")
    for label, value in (("Chunk limit", settings.murf_chunk_chars),
                         ("Worker count", settings.kokoro_workers if workers is None else workers)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} must be a positive integer")



def split_text(text: str, max_chars: int) -> List[str]:
    """Split text into chunks suitable for TTS processing."""
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("Chunk limit must be a positive integer")
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    normalized = " ".join(paragraphs)
    paragraph_ends = []
    offset = 0
    for paragraph in paragraphs[:-1]:
        offset += len(paragraph) + 1
        paragraph_ends.append(offset)
    sentence_ends = [m.end() for m in re.finditer(r"[.!?] +", normalized)]
    chunks = []
    start = 0
    while start < len(normalized):
        stop = min(start + max_chars, len(normalized))
        if stop < len(normalized):
            boundary = None
            for boundaries in (paragraph_ends, sentence_ends):
                index = bisect_right(boundaries, stop) - 1
                if index >= 0 and boundaries[index] > start:
                    boundary = boundaries[index]
                    break
            if boundary is not None:
                stop = boundary
            else:
                space = normalized.rfind(" ", start, stop)
                if space > start:
                    stop = space + 1
        chunks.append(normalized[start:stop])
        start = stop
    return chunks


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
            "format": self.settings.murf_format.upper(),
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


def _load_kokoro_pipeline():
    # Misaki otherwise invokes pip implicitly, even with HF offline mode set.
    from importlib.metadata import PackageNotFoundError, version
    try:
        version("en-core-web-sm")
    except PackageNotFoundError as exc:
        raise RuntimeError("Kokoro needs its English pronunciation model. Run: "
                           "python -m pip install -r requirements-kokoro.txt -c constraints.txt") from exc
    from kokoro import KPipeline
    return KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")


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
            self.pipeline = _load_kokoro_pipeline()
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
    import numpy as np
    import soundfile as sf
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
    _KOKORO_PIPELINE = _load_kokoro_pipeline()
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
    validate_settings(settings, kokoro_workers)
    if not chunks or any(not c.strip() or len(c) > settings.murf_chunk_chars for c in chunks):
        raise ValueError("Audio requires nonempty text chunks within the configured limit")
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


def concatenate_audio(chunks: Iterable[bytes], output_path: str, input_format="mp3") -> None:
    """Decode all chunks, then atomically publish a validated MP3."""
    combined = AudioSegment.empty()
    for chunk in chunks:
        segment = AudioSegment.from_file(io.BytesIO(chunk), format=input_format.lower())
        if len(segment) == 0:
            raise ValueError("Empty audio chunk")
        combined += segment
    if not len(combined):
        raise ValueError("No audio generated")
    destination = Path(output_path).resolve()
    fd, temporary = tempfile.mkstemp(prefix=".audio-", suffix=".mp3", dir=destination.parent)
    os.close(fd)
    try:
        combined.export(temporary, format="mp3", bitrate="128k").close()
        if not len(AudioSegment.from_mp3(temporary)):
            raise ValueError("Final audio is empty")
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def synthesize(text, output, settings, workers=None):
    validate_settings(settings, workers)
    chunks = [chunk for chunk in split_text(text, settings.murf_chunk_chars) if chunk.strip()]
    if not chunks:
        raise ValueError("No usable text to speak")
    workers = settings.kokoro_workers if workers is None else workers
    if settings.tts_engine == "kokoro":
        engine = None if workers > 1 else KokoroTTSEngine(settings)
    else:
        engine = MurfTTSEngine(settings)
    audio = generate_audio_chunks(chunks, settings, engine, workers)
    concatenate_audio(audio, output, "mp3" if settings.tts_engine == "kokoro" else settings.murf_format)


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

    for value in (args.max_chars, args.kokoro_workers):
        if value is not None and value <= 0:
            parser.error("Chunk limit and worker count must be positive")
    text = Path(args.text_file).read_text(encoding="utf-8")
    if not text.strip():
        parser.error("No usable text to speak")
    settings = load_settings(tts_engine=args.tts_engine)
    if args.max_chars is not None:
        settings = replace(settings, murf_chunk_chars=args.max_chars)
    synthesize(text, args.out, settings, args.kokoro_workers)
    print(f"Successfully saved audio to {args.out}")


if __name__ == "__main__":
    main()
