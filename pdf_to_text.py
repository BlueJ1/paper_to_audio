"""
PDF to Text Processor
Converts academic papers (PDF) into audio-friendly text using LLM.
"""
import argparse
import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader


def load_pdf_text(pdf_path: str) -> str:
    """Load text content from a PDF file."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return "\n\n".join(page.page_content for page in pages)


def build_llm() -> ChatGoogleGenerativeAI:
    """Build the Google Gemini LLM instance."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
    )


def build_prompt() -> ChatPromptTemplate:
    """Build the prompt template for converting papers to audio-friendly text."""
    system = (
        r"You are transforming academic papers into audio-friendly narration for a Text-to-Speech engine. "
        r"Your goal is to make the paper sound like an engaging podcast or audiobook. " "\n\n"
        r"STRICT GUIDELINES:" "\n"
        r"1. **NO LATEX or MARKDOWN**: Never output LaTeX code, special symbols, or Markdown formatting (no '$', '\', '_', '^', '*', '#'). " "\n"
        r"2. **Math to Spoken English**: You must translate all math into how a human would read it aloud." "\n"
        r"   - Instead of '$x_t$', write 'x sub t'." "\n"
        r"   - Instead of '$\mathcal{M}_{t-1}$', write 'capital M sub t minus 1'." "\n"
        r"   - Instead of '$N \times d$', write 'N by d'." "\n"
        r"   - Instead of '$\mathbb{R}$', write 'the set of real numbers'." "\n"
        r"   - For Greek letters like '$\alpha$', write 'alpha'." "\n"
        r"3. **Complex Equations**: Do not read complex formulas character-by-character. Instead, describe the equation's purpose in plain English (e.g., 'The model calculates the weighted sum of attention scores...')." "\n"
        r"4. **Flow**: Rewrite the content into coherent prose. Omit citations, inline references (like '[1]' or '(Smith et al.)'), and URLs." "\n"
        r"5. **Figures**: Describe figures and tables naturally in the text where they appear." "\n"
    )
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system),
            ("human", "Paper content:\n\n{paper_text}"),
        ]
    )


def rewrite_for_audio(llm: ChatGoogleGenerativeAI, paper_text: str) -> str:
    """Rewrite paper text into audio-friendly format using LLM."""
    prompt = build_prompt()
    response = llm.invoke(prompt.format_messages(paper_text=paper_text))
    return response.content.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a paper PDF into audio-friendly text using LLM"
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--out",
        default=None,
        help="Output text filename (default: <pdf_basename>_audio_text.txt)",
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not google_api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in environment.")

    # Determine output filename
    if args.out is None:
        pdf_basename = os.path.splitext(os.path.basename(args.pdf))[0]
        args.out = f"{pdf_basename}_audio_text.txt"

    print(f"[1/3] Loading PDF: {args.pdf}")
    paper_text = load_pdf_text(args.pdf)
    print(f"      Loaded {len(paper_text)} characters")

    print("[2/3] Processing with Gemini (this may take a while)...")
    llm = build_llm()
    rewritten = rewrite_for_audio(llm, paper_text)
    print(f"      Generated {len(rewritten)} characters of audio-friendly text")

    print(f"[3/3] Saving text to {args.out}...")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(rewritten)

    print(f"✓ Successfully saved audio-friendly text to {args.out}")


if __name__ == "__main__":
    main()
