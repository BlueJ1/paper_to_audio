import argparse
import os
import re
from dataclasses import dataclass
from typing import Iterable, List
import io

import requests
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from pydub import AudioSegment



@dataclass
class Settings:
    google_api_key: str
    murf_api_key: str
    murf_voice_id: str
    murf_format: str
    murf_chunk_chars: int


def load_settings() -> Settings:
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    murf_api_key = os.getenv("MURF_API_KEY", "").strip()
    if not google_api_key or not murf_api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY or MURF_API_KEY in environment.")
    murf_voice_id = os.getenv("MURF_VOICE_ID", "marcus").strip() or "marcus"
    murf_format = os.getenv("MURF_FORMAT", "mp3").strip() or "mp3"
    murf_chunk_chars = int(os.getenv("MURF_CHUNK_CHARS", "2800").strip())
    return Settings(
        google_api_key=google_api_key,
        murf_api_key=murf_api_key,
        murf_voice_id=murf_voice_id,
        murf_format=murf_format,
        murf_chunk_chars=murf_chunk_chars,
    )


def load_pdf_text(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return "\n\n".join(page.page_content for page in pages)


def build_llm(settings: Settings) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
    )


def build_prompt() -> ChatPromptTemplate:
    system = (
        "You are transforming academic papers into audio-friendly narration. "
        "Rewrite the content into coherent prose with smooth flow. "
        "Describe all figures accurately and place their descriptions at their original position in the text. "
        "Omit citations, inline references, and bibliography sections that break listening flow. "
        "Integrate footnotes into the main text naturally near where they appear. "
        "Preserve section structure and key equations by describing them in words."
    )
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system),
            ("human", "Paper content:\n\n{paper_text}"),
        ]
    )


def rewrite_for_audio(llm: ChatGoogleGenerativeAI, paper_text: str) -> str:
    prompt = build_prompt()
    response = llm.invoke(prompt.format_messages(paper_text=paper_text))
    return response.content.strip()


def split_text(text: str, max_chars: int) -> List[str]:
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


def generate_speech_chunk(
    settings: Settings,
    text: str,
) -> bytes:
    url = "https://api.murf.ai/v1/speech/generate"
    headers = {
        "api-key": settings.murf_api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "voiceId": settings.murf_voice_id,
        "text": text,
        "format": settings.murf_format,
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
    parser = argparse.ArgumentParser(description="Convert a paper PDF into audio using LLM + Murf.ai")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--out", default="output.mp3", help="Output audio filename")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Maximum characters per Murf request (defaults to env MURF_CHUNK_CHARS)",
    )
    args = parser.parse_args()

    settings = load_settings()
    if args.max_chars:
        settings = Settings(
            google_api_key=settings.google_api_key,
            murf_api_key=settings.murf_api_key,
            murf_voice_id=settings.murf_voice_id,
            murf_format=settings.murf_format,
            murf_chunk_chars=args.max_chars,
        )

    print(f"[1/5] Loading PDF: {args.pdf}")
    paper_text = load_pdf_text(args.pdf)
    print(f"      Loaded {len(paper_text)} characters")

    print("[2/5] Processing with Gemini (this may take a while)...")
    llm = build_llm(settings)
    rewritten = rewrite_for_audio(llm, paper_text)
    print(f"      Generated {len(rewritten)} characters of audio-friendly text")

    print(f"[3/5] Splitting into chunks (max {settings.murf_chunk_chars} chars each)...")
    chunks = split_text(rewritten, settings.murf_chunk_chars)
    print(f"      Created {len(chunks)} chunks")

    print(f"[4/5] Generating audio with Murf.ai (voice: {settings.murf_voice_id})...")
    audio_chunks = []
    for i, chunk in enumerate(chunks, 1):
        print(f"      Processing chunk {i}/{len(chunks)}...")
        audio_chunks.append(generate_speech_chunk(settings, chunk))

    print(f"[5/5] Saving audio to {args.out}...")
    concatenate_audio(audio_chunks, args.out)

    print(f"✓ Successfully saved audio to {args.out}")


if __name__ == "__main__":
    main()
