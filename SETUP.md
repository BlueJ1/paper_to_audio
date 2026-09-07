# Setup and reproducibility

Supported runtime: **Python 3.12** (tested with 3.12.10 on Apple Silicon). Python 3.13 is not currently supported because the pinned audio stack uses `audioop`. Linux Python 3.12 core checks are configured in CI; remote CI results are separate from local verification.

## Choose dependencies

Create a virtual environment, then install only the modes you use:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -c constraints.txt

# Optional local TTS, including its English pronunciation model
python -m pip install -r requirements-kokoro.txt -c constraints.txt

# Optional LLM providers; choose either or both
python -m pip install -r requirements-google.txt -c constraints.txt
python -m pip install -r requirements-cerebras.txt -c constraints.txt

# Development
python -m pip install -r requirements-dev.txt -c constraints.txt
```

Murf uses the lightweight dependencies in the core set. Deterministic processing imports no LangChain provider integrations or Kokoro/PyTorch. The standalone legacy `pdf_to_text.py --no-llm` also avoids those imports, but uses the old cleanup algorithm; prefer the shared pipeline.

Install FFmpeg for any audio generation. Kokoro's default English path uses `en_core_web_sm` 3.8.0, pinned to the [official spaCy model release](https://spacy.io/models/en/). Its weights and voice still need a first download from Hugging Face. After that, use `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to require cached speech resources. The app checks the pronunciation package before model loading, so a missing package produces an installation instruction instead of silently invoking pip. Some phonemizer/language configurations need `espeak-ng`; the tested default English voice works without a separate PATH installation.

## Keys and defaults

Copy `.env.example` to `.env` if needed; do not overwrite an existing file containing your keys.

| Setting | When needed / default |
| --- | --- |
| `GOOGLE_API_KEY` | Google LLM only |
| `CEREBRAS_API_KEY` | Cerebras LLM only |
| `MURF_API_KEY` | Murf TTS only |
| `MURF_VOICE_ID` | `marcus` |
| `MURF_FORMAT` | `mp3`; supported chunk formats: MP3/WAV/FLAC/OGG |
| `MURF_CHUNK_CHARS` | `2800`, positive |
| `KOKORO_VOICE` | `af_bella` (English) |
| `KOKORO_WORKERS` | `1`, positive; each worker loads a model |

Google construction explicitly selects the Gemini Developer API (`vertexai=False`); an extra Vertex AI environment override is no longer required. The CLI model defaults are `gemma-3-27b-it` for Google and `gpt-oss-120b` for Cerebras. The previously configured Cerebras Qwen model was deprecated; the replacement is listed in the [current Cerebras model catalog](https://inference-docs.cerebras.ai/models/overview). Model availability still depends on the provider/account.

Murf requests use uppercase format names as specified by the [synthesis API](https://murf.ai/api/docs/api-reference/text-to-speech/generate). The [official voice library](https://murf.ai/api/docs/voices-styles/voice-library) lists Marcus as `en-US-marcus`; the API also accepts actor names. Set a voice available to your account. Live cloud authentication, quotas, voice availability, and narration quality have not been tested by this repair. No paid smoke calls are part of setup checks or CI.

## Verify your selected mode

```bash
python check_setup.py                             # deterministic PDF/UI dependencies only
python check_setup.py --dev                       # also pytest
python check_setup.py --tts-engine kokoro         # also local TTS dependencies and FFmpeg
python check_setup.py --tts-engine murf           # also FFmpeg and Murf key
python check_setup.py --llm-provider google       # selected LLM import and key
python check_setup.py --llm-provider cerebras
python -m pip check
```

Import/key checks do not call providers or guarantee cached Kokoro weights exist. Use a short local conversion to check the actual engine:

```bash
python text_to_speech.py edited.txt --tts-engine kokoro --out sample.mp3
```

The old working environment reports a PyTorch wheel tag of `cp312-cp312-macosx_110_0_arm64`, which does not match this machine's supported wheel tags. Its actual Kokoro runtime works. A fresh install of the pinned dependencies passes `pip check`; do not alter a working environment merely to suppress the old metadata warning.

## Dependency updates

Direct dependencies are pinned in the requirements sets; `constraints.txt` pins installed transitive versions from the validated macOS Python 3.12 environment. Constraints do not install optional packages. This is a version snapshot, not a cross-platform hash lock: dependencies used only on another OS are resolved there.

For an intentional update, install all selected dependency sets in a disposable environment, run setup checks, the full suite, and a short audio smoke test. If all optional sets are installed, `python scripts/snapshot_constraints.py` refreshes the combined snapshot. Review version changes and re-check a clean installation. Never refresh corpus baselines solely to hide a text regression.
