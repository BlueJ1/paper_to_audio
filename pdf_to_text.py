"""
PDF to Text Processor
Converts academic papers (PDF) into audio-friendly text using targeted replacements.
Preserves original text verbatim; only replaces elements that are bad for TTS
(math formulas, citations, LaTeX artifacts, special characters).
"""
from __future__ import annotations

import argparse
import os
import re
import time
import unicodedata
from typing import Dict, List, Optional, Tuple

import fitz  # pymupdf
from dotenv import load_dotenv


def load_pdf_text(pdf_path: str) -> str:
    """Load text content from a PDF file using pymupdf."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n\n".join(pages)


GEMMA_MODELS = [
    "gemma-3-27b-it",        # fast, no extended thinking (recommended default)
    "gemma-3-12b-it",        # smaller, faster
    "gemma-4-31b-it",        # slow (extended thinking), highest quality
    "gemma-4-26b-a4b-it",    # slow (extended thinking)
]
GEMINI_MODELS = [
    "gemini-3.1-flash-lite-preview",  # fast, cheap, preview
]
DEFAULT_LLM_MODEL = GEMMA_MODELS[0]

CEREBRAS_MODELS = [
    "gpt-oss-120b",  # large MoE, fast on Cerebras hardware (recommended)
]
DEFAULT_CEREBRAS_MODEL = CEREBRAS_MODELS[0]
_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"


from providers import build_llm


# ---------------------------------------------------------------------------
# Ligature and pre-processing maps
# ---------------------------------------------------------------------------

_LIGATURES = {
    "\ufb01": "fi",   # ﬁ
    "\ufb02": "fl",   # ﬂ
    "\ufb03": "ffi",  # ﬃ
    "\ufb04": "ffl",  # ﬄ
    "\ufb00": "ff",   # ﬀ
    "\ufb05": "st",   # ﬅ
    "\ufb06": "st",   # ﬆ
}

# Prefixes that signal an intentional compound (keep hyphen when joining a
# line-break split): "self-\nattention" -> "self-attention".
_HYPHEN_KEEP_PREFIXES = frozenset({
    "self", "non", "multi", "pre", "post", "inter", "intra", "sub", "super",
    "cross", "semi", "quasi", "pseudo", "anti", "co", "bi", "tri", "extra",
    "mid", "over", "under", "counter", "trans", "ultra", "meta", "mini",
    "micro", "macro", "mega", "neo",
})

# Suffixes that signal a compound (keep hyphen): "context-\naware" ->
# "context-aware".
_HYPHEN_KEEP_SUFFIXES = frozenset({
    "aware", "based", "wise", "like", "free", "level", "specific", "style",
    "type", "driven", "oriented", "labeled", "weighted", "invariant",
    "dependent", "independent",
})


def _rejoin_hyphenated_linebreaks(text: str) -> str:
    """Rejoin words broken by a line-break hyphen.

    Default: strip the hyphen ("intro-\\nduction" -> "introduction").
    Preserve the hyphen for intentional compound words, detected via:
      1. Known compound prefixes (self-, non-, multi-, ...)
      2. Known compound suffixes (-aware, -based, -labeled, ...)
      3. Both halves appear as standalone words elsewhere in the document
         (e.g. if "graph" and "labeled" both occur alone, "graph-\\nlabeled"
         is probably the compound "graph-labeled", not a hyphenation artifact).
    """
    # Pre-scan the document for hyphenated word pairs that appear WITHOUT a
    # line break between them. If "self-attention" occurs anywhere in that
    # form, "self-\nattention" is almost certainly the same compound.
    inline_pairs = {
        (m.group(1).lower(), m.group(2).lower())
        for m in re.finditer(r"([A-Za-z]+)-([A-Za-z]+)", text)
    }

    def _join(m: re.Match) -> str:
        left, right = m.group(1), m.group(2)
        left_lc = left.lower()
        if left_lc in _HYPHEN_KEEP_PREFIXES:
            return f"{left}-{right}"
        right_head = re.match(r"^[a-zA-Z]+", right)
        right_head_lc = right_head.group(0).lower() if right_head else ""
        if right_head_lc in _HYPHEN_KEEP_SUFFIXES:
            return f"{left}-{right}"
        # Same hyphenated pair exists mid-line elsewhere -> real compound.
        if (left_lc, right_head_lc) in inline_pairs:
            return f"{left}-{right}"
        return f"{left}{right}"

    return re.sub(r"([A-Za-z]+)-\n([a-z][A-Za-z]*)", _join, text)

# --- Regex cleanup maps ---

_LATEX_COMMAND_MAP = {
    r"\rightarrow": " leads to ",
    r"\leftarrow": " from ",
    r"\Rightarrow": " implies ",
    r"\Leftarrow": " is implied by ",
    r"\leftrightarrow": " if and only if ",
    r"\times": " times ",
    r"\cdot": " times ",
    r"\approx": " approximately ",
    r"\neq": " not equal to ",
    r"\leq": " less than or equal to ",
    r"\geq": " greater than or equal to ",
    r"\infty": " infinity ",
    r"\in": " in ",
    r"\notin": " not in ",
    r"\subset": " is a subset of ",
    r"\supset": " is a superset of ",
    r"\cup": " union ",
    r"\cap": " intersection ",
    r"\forall": " for all ",
    r"\exists": " there exists ",
    r"\nabla": " nabla ",
    r"\partial": " partial ",
    r"\pm": " plus or minus ",
    r"\mp": " minus or plus ",
    r"\propto": " is proportional to ",
    r"\sum": " sum of ",
    r"\prod": " product of ",
    r"\int": " integral of ",
}

_UNICODE_MATH_MAP = {
    "\u2264": " less than or equal to ",
    "\u2265": " greater than or equal to ",
    "\u2260": " not equal to ",
    "\u2248": " approximately ",
    "\u2192": " leads to ",
    "\u2190": " from ",
    "\u21D2": " implies ",
    "\u21D0": " is implied by ",
    "\u2194": " if and only if ",
    "\u2208": " in ",
    "\u2209": " not in ",
    "\u2282": " is a subset of ",
    "\u2283": " is a superset of ",
    "\u222A": " union ",
    "\u2229": " intersection ",
    "\u2200": " for all ",
    "\u2203": " there exists ",
    "\u221E": " infinity ",
    "\u2202": " partial ",
    "\u2211": " sum of ",
    "\u220F": " product of ",
    "\u222B": " integral of ",
    "\u00D7": " times ",
    "\u00B1": " plus or minus ",
    "\u221D": " is proportional to ",
    "\u2207": " nabla ",
    "\u2218": " composed with ",
    "\u2297": " tensor product ",
    "\u2295": " direct sum ",
    "\u22C5": " times ",
    "\u2261": " is identical to ",
    "\u221A": " square root of ",  # √
    "\u2248": " approximately ",
    "\u03F5": " epsilon ",          # ϵ (lunate epsilon, common in math)
    "\u03F5": " epsilon ",          # ϵ variant
    "\u03D5": " phi ",              # ϕ (phi symbol variant)
}

_GREEK_MAP = {
    "\u0391": "Alpha", "\u0392": "Beta", "\u0393": "Gamma", "\u0394": "Delta",
    "\u0395": "Epsilon", "\u0396": "Zeta", "\u0397": "Eta", "\u0398": "Theta",
    "\u0399": "Iota", "\u039A": "Kappa", "\u039B": "Lambda", "\u039C": "Mu",
    "\u039D": "Nu", "\u039E": "Xi", "\u039F": "Omicron", "\u03A0": "Pi",
    "\u03A1": "Rho", "\u03A3": "Sigma", "\u03A4": "Tau", "\u03A5": "Upsilon",
    "\u03A6": "Phi", "\u03A7": "Chi", "\u03A8": "Psi", "\u03A9": "Omega",
    "\u03B1": "alpha", "\u03B2": "beta", "\u03B3": "gamma", "\u03B4": "delta",
    "\u03B5": "epsilon", "\u03B6": "zeta", "\u03B7": "eta", "\u03B8": "theta",
    "\u03B9": "iota", "\u03BA": "kappa", "\u03BB": "lambda", "\u03BC": "mu",
    "\u03BD": "nu", "\u03BE": "xi", "\u03BF": "omicron", "\u03C0": "pi",
    "\u03C1": "rho", "\u03C2": "sigma", "\u03C3": "sigma", "\u03C4": "tau",
    "\u03C5": "upsilon", "\u03C6": "phi", "\u03C7": "chi", "\u03C8": "psi",
    "\u03C9": "omega",
}


# ---------------------------------------------------------------------------
# Pre-processing (raw PDF extraction artifacts)
# ---------------------------------------------------------------------------

def _preprocess_raw_text(text: str) -> str:
    """Fix raw PDF extraction artifacts before semantic processing.

    This must run before citation/math regexes so that hyphenated line-breaks
    are rejoined and ligatures are expanded to ASCII.
    """
    # 0. Normalize to NFC so decomposed accents (e.g. "c" + combining caron
    #    used by some PDF extractors for "č") become their precomposed form.
    #    This lets the citation regex match author names like "Veličković".
    text = unicodedata.normalize("NFC", text)

    # 1. Replace typographic ligatures (common in academic PDFs)
    for lig, repl in _LIGATURES.items():
        text = text.replace(lig, repl)

    # 2. Rejoin words split by a line-break hyphen. Keeps the hyphen for
    #    intentional compounds like "self-attention" and strips it for
    #    layout artifacts like "intro-\nduction".
    text = _rejoin_hyphenated_linebreaks(text)

    # 3. Join broken citations: "Smith et al.,\n2023" → "Smith et al., 2023"
    text = re.sub(r",\n(\d{4}[a-z]?\s*[;,)])", r", \1", text)

    # 3b. Join URLs split across lines (no hyphen, just a line break mid-URL)
    text = re.sub(r"(https?://\S*)\n(\S+)", r"\1\2", text)

    # 4. Remove email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "", text)

    # 5. Remove arXiv metadata lines (e.g., "arXiv:1803.02155v2 [cs.CL] 12 Apr 2018")
    text = re.sub(r"arXiv:\S+\s*\[[\w.]+\]\s*\d{1,2}\s+\w+\s+\d{4}", "", text)

    # 6. Remove References section and everything after it.
    #    Match "References" as a standalone section header.
    text = re.sub(r"\n\s*References\s*\n.*", "", text, flags=re.DOTALL)

    # 7. Remove inline footnote markers embedded in prose.
    #    e.g. "1The tensor2tensor library..." → "The tensor2tensor library..."
    text = re.sub(r"(?m)^(\d+)([A-Z])", r"\2", text)

    # 8. Remove isolated section/subsection number lines (e.g. standalone "1", "2.1")
    #    that come from PDF section-header extraction.
    text = re.sub(r"(?m)^\d+(?:\.\d+)?\s*$", "", text)

    return text


# ---------------------------------------------------------------------------
# Diagram / figure block cleanup
# ---------------------------------------------------------------------------

def _remove_diagram_blocks(text: str) -> str:
    """Remove paragraphs that are clearly extracted diagram/figure graphic content.

    Academic PDF figures are often extracted as sequences of short identifier tokens
    (x1, x2, aV, wK, etc.) that are meaningless as audio. This removes such blocks
    while keeping figure captions.
    """
    paragraphs = text.split("\n\n")
    kept = []
    for para in paragraphs:
        lines = [l.strip() for l in para.strip().split("\n") if l.strip()]
        if not lines:
            kept.append(para)
            continue

        # If the paragraph contains a "Figure N:" or "Table N:" caption line,
        # keep only from that line onward (drop preceding diagram content).
        caption_match = re.search(
            r"(?m)^(Figure|Table)\s+\d+[.:].+", para, re.IGNORECASE
        )
        if caption_match:
            kept.append(para[caption_match.start():].strip())
            continue

        # Remove small (1-3 line) paragraphs of pure math identifier tokens.
        # These are leftover diagram/equation fragments: "aK", "2,4=wK", "wV", etc.
        # Require at least one line with a digit, comma, or equals (clearly mathematical).
        if len(lines) <= 3:
            def _is_math_token(s: str) -> bool:
                return bool(re.match(r"^[a-zA-Z0-9,=.\-]{1,12}$", s)) and " " not in s
            has_math_char = any(re.search(r"[\d=,]", l) for l in lines)
            if all(_is_math_token(l) for l in lines) and has_math_char:
                continue  # Drop pure-identifier fragment

        # Detect larger paragraphs that are mostly short identifier tokens (diagram elements):
        # - 4+ lines, each ≤ 8 characters
        # - No words longer than 8 chars (would indicate real prose)
        # - No sentence-ending punctuation
        if len(lines) >= 4:
            short_lines = sum(1 for l in lines if len(l) <= 8)
            has_long_word = any(len(w) > 8 for l in lines for w in l.split())
            has_sentence_end = any(
                l.endswith((".", "!", "?", ":")) for l in lines
            )
            if short_lines / len(lines) > 0.75 and not has_long_word and not has_sentence_end:
                continue  # Drop this block

        kept.append(para)

    return "\n\n".join(kept)


# ---------------------------------------------------------------------------
# Regex cleanup
# ---------------------------------------------------------------------------

def _regex_cleanup(text: str) -> str:
    """Deterministic regex-based cleanup of TTS-unfriendly elements."""

    # 0. Remove standalone equation reference numbers on their own line: "(1)", "(2)" …
    text = re.sub(r"(?m)^\s*\(\d+\)\s*$", "", text)

    # 1. Remove citations
    # Bracketed numeric: [1], [1, 2], [1-3], [1; 2; 3]
    text = re.sub(r"\[(\d+[\s,;\-]*)+\]", "", text)
    # Author token allows Latin letters with diacritics (Č, é, ñ, …) so names
    # like "Veličković" and "Bañuls" are matched, not just ASCII names.
    _ANAME = r"[A-Z\u00C0-\u024F\u1E00-\u1EFF][a-z\u00C0-\u024F\u1E00-\u1EFF]+"
    # Author-year: (Smith et al., 2023), (Smith, 2023), (Smith and Jones, 2023)
    text = re.sub(
        rf"\(\s*{_ANAME}(?:\s+(?:and|&)\s+{_ANAME})*(?:\s+et\s+al\.?)?,?\s*\d{{4}}[a-z]?\s*\)",
        "",
        text,
    )
    # Multiple author-year in one paren: (Smith, 2020; Jones, 2021)
    text = re.sub(
        rf"\(\s*(?:{_ANAME}(?:\s+et\s+al\.?)?,?\s*\d{{4}}[a-z]?\s*[;,]\s*)+{_ANAME}(?:\s+et\s+al\.?)?,?\s*\d{{4}}[a-z]?\s*\)",
        "",
        text,
    )
    # Inline (no parens) author-year: "Veličković et al. (2017)" -> ""
    text = re.sub(
        rf"\b{_ANAME}(?:\s+(?:and|&)\s+{_ANAME})*(?:\s+et\s+al\.?)?\s*\(\s*\d{{4}}[a-z]?\s*\)",
        "",
        text,
    )

    # 2. Remove URLs (and orphaned "available at"/"can be found at" phrases
    #    that dangle after the URL is gone).
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(
        r"(?im)^[^\n]*\b(?:is\s+available|can\s+be\s+found|is\s+hosted)\s+at\s*\.?\s*$",
        "",
        text,
    )

    # 3. Replace known LaTeX commands with English
    for cmd, replacement in _LATEX_COMMAND_MAP.items():
        text = text.replace(cmd, replacement)
    # Remove remaining LaTeX commands (e.g., \mathbb, \frac, \textbf)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    # Remove stray dollar signs (math delimiters)
    text = re.sub(r"\$", "", text)
    # Remove curly braces
    text = re.sub(r"[{}]", "", text)

    # 4. Replace unicode math symbols with English
    for symbol, replacement in _UNICODE_MATH_MAP.items():
        text = text.replace(symbol, replacement)

    # 5. Replace Greek letters with spelled-out names
    for symbol, name in _GREEK_MAP.items():
        text = text.replace(symbol, f" {name} ")

    # 6. Handle subscript/superscript patterns
    # x_t -> x sub t, x_{t-1} -> x sub t-1 (braces already removed above)
    text = re.sub(r"(\w)_(\w+)", r"\1 sub \2", text)
    # x^2 -> x to the 2, x^n -> x to the n
    text = re.sub(r"(\w)\^(\w+)", r"\1 to the \2", text)

    # 7. Remove parenthetical figure/table references but keep inline ones
    # "(see Fig. 1)", "(cf. Table 2)", "(Figure 3)" -> remove
    text = re.sub(
        r"\(\s*(?:see|cf\.?|refer to)?\s*(?:Fig(?:ure)?|Table)\.?\s*\d+[a-z]?\s*\)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 8. Remove remaining unicode math operators (U+2200-U+22FF) not already mapped
    text = re.sub(r"[\u2200-\u22FF]", " ", text)
    # Remove misc math symbols (U+2300-U+23FF, U+27C0-U+27EF, U+2980-U+29FF)
    text = re.sub(r"[\u2300-\u23FF\u27C0-\u27EF\u2980-\u29FF]", " ", text)

    # 9. Remove diagram/figure graphic content blocks
    text = _remove_diagram_blocks(text)

    # 10. Whitespace normalization
    text = re.sub(r"[ \t]+", " ", text)  # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse multiple blank lines
    text = "\n".join(line.strip() for line in text.splitlines())  # strip each line

    return text


# ---------------------------------------------------------------------------
# Math detection and LLM rewriting
# ---------------------------------------------------------------------------

def _identify_math_passages(text: str) -> List[Tuple[int, str]]:
    """Identify paragraphs that contain significant math needing LLM rewriting."""
    paragraphs = text.split("\n\n")
    flagged = []

    # Patterns that indicate remaining math content after regex cleanup
    math_indicators = re.compile(
        r"("
        r"\bsub \w+"                           # subscript remnants: "x sub t"
        r"|\bto the \w+"                       # superscript remnants: "x to the 2"
        r"|\b(?:sum of|product of|integral of|partial|nabla)\b"
        r"|\b(?:equals?|=)\s*\w+\s*(?:times|plus|minus)"  # equation-like sequences
        r"|\b[a-z]\s+sub\s+"                   # variable with subscript
        r"|\b(?:less than or equal to|greater than or equal to|not equal to|approximately|implies|leads to)\b"
        # New: equation fragment patterns
        r"|\b[xyz][ij]\b"                      # matrix-indexed variables (xi, xj, zi)
        r"|W\s*[QKV]\b"                        # matrix notation W^Q, W^K, W^V (no leading \b since W often follows xi)
        r"|\ba[KV]\b"                          # edge weight matrices aK, aV
        r"|\b(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|tau|phi|omega)\s+[a-z]{1,3}\b"  # Greek + index
        r"|\bO\s*\([a-z]"                      # Big-O notation O(n)
        r"|\b[a-z]{1,2}\d[a-z]{1,2}\b"        # alphanumeric like n2da, bh
        r"|\bsqrt\b"                           # square root remnant
        r"|\bPn\b"                             # "Pn" rendered sum
        r"|\b[a-z]\s*=\s*n\s+X\b"             # "= n X" sum rendering
        r"|\beij\b|\baij\b|\bwij\b"            # common equation variables
        r")",
        re.IGNORECASE,
    )

    # Single-letter variable patterns (isolated letters that look like math variables)
    variable_pattern = re.compile(r"(?<!\w)[a-zA-Z](?:\s+sub\s+|\s+to the\s+)(?:\w+)")

    for i, para in enumerate(paragraphs):
        if len(para.strip()) < 20:
            continue

        indicator_matches = len(math_indicators.findall(para))
        variable_matches = len(variable_pattern.findall(para))
        word_count = len(para.split())

        if word_count == 0:
            continue

        math_density = (indicator_matches + variable_matches) / word_count

        # Flag if math density is high or there are many math patterns
        if math_density > 0.10 or indicator_matches >= 2:
            flagged.append((i, para))

    return flagged


# Transient errors raised by the OpenAI/Cerebras and Google client libraries.
# Matched by class name so we don't add hard imports for either SDK.
_RETRYABLE_ERROR_NAMES = frozenset({
    # openai SDK (used by langchain_openai → Cerebras)
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    # google.api_core (used by langchain_google_genai)
    "ResourceExhausted",
    "ServiceUnavailable",
    "DeadlineExceeded",
})


def _is_retryable(exc: BaseException) -> bool:
    return type(exc).__name__ in _RETRYABLE_ERROR_NAMES


def _invoke_with_retry(llm: BaseChatModel, messages, max_attempts: int = 5, base_delay: float = 2.0):
    """Call llm.invoke with exponential backoff on transient API errors.

    Cerebras in particular returns 429 'queue_exceeded' under load; a short
    backoff usually clears it. Non-retryable errors propagate immediately.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            if attempt == max_attempts or not _is_retryable(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1))
            print(
                f"      Transient LLM error ({type(exc).__name__}): {exc}. "
                f"Retrying in {delay:.0f}s ({attempt}/{max_attempts - 1})..."
            )
            time.sleep(delay)


def _llm_rewrite_math(
    llm: BaseChatModel, passages: List[Tuple[int, str]]
) -> Dict[int, str]:
    """Send math-heavy paragraphs individually to Gemini for plain-English rewriting."""
    instructions = (
        "You are cleaning up a single paragraph from an academic paper for text-to-speech. "
        "This paragraph contains mathematical notation that has been partially cleaned up "
        "but is still not suitable for spoken audio.\n\n"
        "Your task:\n"
        "- Describe what the math and formulas DO in plain English\n"
        "- Keep ALL non-math text EXACTLY as written, word for word\n"
        "- Do NOT summarize, add commentary, or restructure the paragraph\n"
        "- Do NOT add introductory phrases like 'This paragraph discusses...'\n"
        "- Output ONLY the cleaned paragraph, nothing else"
    )

    # Use a single human message (Gemma 3 does not support system messages)
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", instructions + "\n\nParagraph:\n\n{passage}"),
        ]
    )

    replacements: Dict[int, str] = {}
    total = len(passages)
    model_name = (
        getattr(llm, "model", None)
        or getattr(llm, "model_name", None)
        or type(llm).__name__
    )
    print(f"      Rewriting {total} paragraph(s) with {model_name}")

    batch_start = time.perf_counter()
    for n, (idx, passage) in enumerate(passages, start=1):
        print(
            f"      [{n}/{total}] paragraph #{idx + 1} ({len(passage)} chars)... ",
            end="",
            flush=True,
        )
        t0 = time.perf_counter()
        response = _invoke_with_retry(llm, prompt.format_messages(passage=passage))
        content = response.content
        if isinstance(content, list):
            # Handle thinking-model responses: find the text block, skip thinking blocks
            text_parts = [
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            result = " ".join(text_parts) if text_parts else str(content)
        else:
            result = str(content)
        result = result.strip()
        # Strip any HTML tags the LLM may have introduced (e.g., <sub>, <sup>)
        result = re.sub(r"<[^>]+>", "", result)
        elapsed = time.perf_counter() - t0

        # Length sanity check: warn if LLM output is suspiciously short (likely a failure)
        if len(result) < len(passage) * 0.3:
            print(
                f"SKIPPED ({elapsed:.1f}s, output {len(result)} chars too short — keeping original)",
                flush=True,
            )
            continue

        print(f"done ({elapsed:.1f}s, {len(result)} chars out)", flush=True)
        replacements[idx] = result

    batch_elapsed = time.perf_counter() - batch_start
    print(
        f"      LLM batch complete: {len(replacements)}/{total} rewritten in {batch_elapsed:.1f}s",
        flush=True,
    )
    return replacements


def _final_sanitize(text: str) -> str:
    """Last-pass cleanup for remaining non-speakable elements."""
    # Fix double periods
    text = re.sub(r"\.{2,}", ".", text)
    # Fix spacing around punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    # Remove isolated single characters that aren't words (a, I)
    text = re.sub(r"(?<!\w)\s+[^aAI\s]\s+(?!\w)", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Remove empty paragraphs
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip lines
    text = "\n".join(line.strip() for line in text.splitlines())

    return text.strip()


def clean_for_tts(text: str, llm: Optional[BaseChatModel] = None) -> str:
    """Orchestrate the full targeted-replacement cleanup pipeline.

    If llm is None, only regex-based cleanup is performed (no LLM calls).
    """
    # Pass 0: fix raw PDF extraction artifacts (ligatures, hyphenation, metadata)
    print("      Pass 1/4: preprocessing raw extraction artifacts...", flush=True)
    t0 = time.perf_counter()
    before = len(text)
    text = _preprocess_raw_text(text)
    print(
        f"               {before} -> {len(text)} chars ({time.perf_counter() - t0:.2f}s)",
        flush=True,
    )

    # Pass 1: deterministic regex cleanup
    print("      Pass 2/4: regex cleanup (citations, LaTeX, math symbols, diagrams)...", flush=True)
    t0 = time.perf_counter()
    before = len(text)
    para_before = text.count("\n\n") + 1
    text = _regex_cleanup(text)
    para_after = text.count("\n\n") + 1
    print(
        f"               {before} -> {len(text)} chars, "
        f"{para_before} -> {para_after} paragraphs ({time.perf_counter() - t0:.2f}s)",
        flush=True,
    )

    # Pass 2-3: identify and rewrite math-heavy paragraphs with LLM
    if llm is not None:
        print("      Pass 3/4: identifying math-heavy paragraphs...", flush=True)
        t0 = time.perf_counter()
        total_paragraphs = len([p for p in text.split("\n\n") if p.strip()])
        math_passages = _identify_math_passages(text)
        print(
            f"               scanned {total_paragraphs} paragraphs, "
            f"flagged {len(math_passages)} for LLM rewrite ({time.perf_counter() - t0:.2f}s)",
            flush=True,
        )
        if math_passages:
            replacements = _llm_rewrite_math(llm, math_passages)
            # Reconstruct text with replacements
            paragraphs = text.split("\n\n")
            for idx, new_text in replacements.items():
                paragraphs[idx] = new_text
            text = "\n\n".join(paragraphs)
        else:
            print("               no math-heavy paragraphs detected, skipping LLM step", flush=True)
    else:
        print("      Pass 3/4: skipped (LLM disabled)", flush=True)

    # Pass 4: final sanitization
    print("      Pass 4/4: final sanitization...", flush=True)
    t0 = time.perf_counter()
    before = len(text)
    text = _final_sanitize(text)
    print(
        f"               {before} -> {len(text)} chars ({time.perf_counter() - t0:.2f}s)",
        flush=True,
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a paper PDF into audio-friendly text using targeted replacements"
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--out",
        default=None,
        help="Output text filename (default: <pdf_basename>_audio_text.txt)",
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
            f"Model to use for math rewriting. "
            f"Google default: {DEFAULT_LLM_MODEL}. "
            f"Cerebras default: {DEFAULT_CEREBRAS_MODEL}."
        ),
    )
    args = parser.parse_args()

    # Resolve model default based on provider
    if args.llm_model is None:
        args.llm_model = (
            DEFAULT_CEREBRAS_MODEL if args.llm_provider == "cerebras" else DEFAULT_LLM_MODEL
        )

    # Load environment variables
    load_dotenv()
    if not args.no_llm:
        if args.llm_provider == "cerebras":
            if not os.getenv("CEREBRAS_API_KEY", "").strip():
                raise RuntimeError("Missing CEREBRAS_API_KEY in environment.")
        else:
            if not os.getenv("GOOGLE_API_KEY", "").strip():
                raise RuntimeError("Missing GOOGLE_API_KEY in environment.")

    # Determine output filename
    if args.out is None:
        pdf_basename = os.path.splitext(os.path.basename(args.pdf))[0]
        args.out = f"{pdf_basename}_audio_text.txt"

    print(f"[1/3] Loading PDF: {args.pdf}")
    paper_text = load_pdf_text(args.pdf)
    print(f"      Loaded {len(paper_text)} characters")

    if args.no_llm:
        print("[2/3] Cleaning text for TTS (regex-only, no LLM)...")
        llm = None
    else:
        print(
            f"[2/3] Cleaning text for TTS (provider={args.llm_provider}, "
            f"model={args.llm_model})..."
        )
        llm = build_llm(model=args.llm_model, provider=args.llm_provider)
    cleaned = clean_for_tts(paper_text, llm=llm)
    print(f"      Output: {len(cleaned)} characters")

    print(f"[3/3] Saving text to {args.out}...")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"Done. Saved audio-friendly text to {args.out}")


if __name__ == "__main__":
    main()
