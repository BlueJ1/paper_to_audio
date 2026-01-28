"""
Text to Speech Generator
Converts audio-friendly text into speech using TTS engines (Murf.ai or Kokoro).
"""
import argparse
import io
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Protocol

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

    return Settings(
        tts_engine=tts_engine,
        murf_api_key=murf_api_key,
        murf_voice_id=murf_voice_id,
        murf_format=murf_format,
        murf_chunk_chars=murf_chunk_chars,
        kokoro_voice=kokoro_voice,
    )


def split_text(text: str, max_chars: int) -> List[str]:
    """Split text into chunks suitable for TTS processing."""
    # Split on paragraph boundaries, then fall back to sentence boundaries.
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        if current:
            chunks.append("\n".join(current).strip())
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
        # Generate audio samples - pipeline returns a generator
        # Each iteration yields (graphemes, phonemes, audio)
        generator = self.pipeline(text, voice=self.settings.kokoro_voice)

        # Collect all audio chunks
        audio_chunks = []
        for _, _, audio in generator:
            audio_chunks.append(audio)

        # Concatenate all audio chunks
        if not audio_chunks:
            raise RuntimeError("No audio generated from Kokoro pipeline")

        samples = np.concatenate(audio_chunks)
        sample_rate = 24000  # Kokoro uses 24kHz sample rate

        # Write to temporary WAV buffer using soundfile
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, samples, sample_rate, format="WAV")
        wav_buffer.seek(0)

        # Convert WAV to MP3 using pydub
        audio_segment = AudioSegment.from_wav(wav_buffer)
        mp3_buffer = io.BytesIO()
        audio_segment.export(mp3_buffer, format="mp3", bitrate="128k")

        return mp3_buffer.getvalue()


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
        tts_engine = KokoroTTSEngine(settings)
    else:
        print(f"[3/4] Generating audio with Murf.ai (voice: {settings.murf_voice_id})...")
        tts_engine = MurfTTSEngine(settings)

    audio_chunks = []
    for i, chunk in enumerate(chunks, 1):
        print(f"      Processing chunk {i}/{len(chunks)}...")
        audio_chunks.append(tts_engine.generate_speech(chunk))

    print(f"[4/4] Saving audio to {args.out}...")
    concatenate_audio(audio_chunks, args.out)

    print(f"✓ Successfully saved audio to {args.out}")


if __name__ == "__main__":
    main()
