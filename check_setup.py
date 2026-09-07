#!/usr/bin/env python3
"""Check only the selected local workflow; never make paid provider calls."""
import argparse
import os
import shutil
import sys


def check_import(module, package=None):
    try:
        __import__(module)
        print(f"OK: {package or module}")
        return True
    except Exception as exc:
        print(f"FAIL: {package or module}: {exc}")
        return False


def check_env_var(name):
    ok = bool(os.getenv(name, "").strip())
    print(f"{'OK' if ok else 'FAIL'}: {name} {'set' if ok else 'missing'}")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts-engine", choices=["none", "murf", "kokoro"], default="none")
    parser.add_argument("--llm-provider", choices=["none", "google", "cerebras"], default="none")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args(argv)
    ok = sys.version_info[:2] == (3, 12)
    print(f"{'OK' if ok else 'FAIL'}: Python {sys.version.split()[0]} (supported: 3.12)")
    for module in ["fitz", "pdfplumber", "num2words", "flask", "dotenv", "requests", "pydub"]:
        ok = check_import(module) and ok
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if args.dev:
        ok = check_import("pytest") and ok
    if args.llm_provider != "none":
        module, key = {"google": ("langchain_google_genai", "GOOGLE_API_KEY"),
                       "cerebras": ("langchain_openai", "CEREBRAS_API_KEY")}[args.llm_provider]
        ok = check_import(module) and ok
        ok = check_env_var(key) and ok
    if args.tts_engine != "none":
        ffmpeg = bool(shutil.which("ffmpeg"))
        print(f"{'OK' if ffmpeg else 'FAIL'}: ffmpeg")
        ok = ffmpeg and ok
        if args.tts_engine == "murf":
            ok = check_env_var("MURF_API_KEY") and ok
        else:
            for module in ["kokoro", "torch", "numpy", "soundfile", "en_core_web_sm"]:
                ok = check_import(module) and ok
            print("Kokoro model resources are checked when audio runs. English may work without espeak-ng; other phonemizers can require it.")
    print("Selected setup checks passed." if ok else "Resolve the failures above; see SETUP.md.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
