"""Web UI for Paper to Audio conversion."""
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import traceback
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, send_file

from pipeline import run_pipeline, serialize, PipelineConfig
from pdf_to_text import build_llm

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

WORK_DIR = Path(tempfile.gettempdir()) / "paper_to_audio_ui"
WORK_DIR.mkdir(exist_ok=True)

PROJECT_DIR = Path(__file__).parent

_jobs: dict = {}
_jobs_lock = threading.Lock()


def _new_job() -> tuple[str, dict]:
    job_id = str(uuid.uuid4())
    job = {"status": "idle", "queue": queue.Queue(), "text": None, "audio_path": None}
    with _jobs_lock:
        _jobs[job_id] = job
    return job_id, job


def _wrap_langchain_llm(chat_model) -> callable:
    """Adapt a LangChain BaseChatModel to the pipeline's Callable[[str], str]."""
    def call(prompt: str) -> str:
        result = chat_model.invoke(prompt)
        if hasattr(result, "content"):
            return str(result.content)
        return str(result)
    return call


def _stream_process(proc: subprocess.Popen, job: dict) -> int:
    """Forward subprocess stdout to the job's queue."""
    for line in proc.stdout:
        stripped = line.rstrip()
        if stripped:
            job["queue"].put({"type": "log", "message": stripped})
    proc.wait()
    return proc.returncode


def _run_text_processing(job_id: str, use_llm: bool, llm_provider: str, llm_model: str) -> None:
    job = _jobs[job_id]
    pdf_path = str(WORK_DIR / f"{job_id}.pdf")

    try:
        llm = None
        if use_llm:
            job["queue"].put({"type": "log", "message": f"Building LLM ({llm_provider}:{llm_model})..."})
            llm = _wrap_langchain_llm(build_llm(model=llm_model, provider=llm_provider))

        job["queue"].put({"type": "log", "message": "Running pipeline (extract → classify → filter → polish)..."})
        doc, _ = run_pipeline(pdf_path, config=PipelineConfig(), llm=llm)

        job["queue"].put({"type": "log", "message": "Serializing text..."})
        text = serialize(doc)

        job["text"] = text
        job["queue"].put({"type": "done", "text": text})

    except Exception as exc:
        job["queue"].put({"type": "error", "message": traceback.format_exc()})


def _run_audio_generation(job_id: str, text: str, tts_engine: str) -> None:
    job = _jobs[job_id]
    text_path = WORK_DIR / f"{job_id}_edit.txt"
    audio_path = WORK_DIR / f"{job_id}.mp3"

    text_path.write_text(text, encoding="utf-8")

    cmd = [
        sys.executable,
        "-u",  # unbuffered stdout so logs stream live to the UI
        str(PROJECT_DIR / "text_to_speech.py"),
        str(text_path),
        "--out", str(audio_path),
        "--tts-engine", tts_engine,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered on the read side
            cwd=str(PROJECT_DIR),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        rc = _stream_process(proc, job)
        if rc != 0:
            job["queue"].put({"type": "error", "message": f"Audio generation failed (exit {rc})"})
            return

        job["audio_path"] = str(audio_path)
        job["queue"].put({"type": "done", "audio_url": f"/audio/{job_id}"})

    except Exception as exc:
        job["queue"].put({"type": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "pdf" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["pdf"]
    if not f.filename or not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a PDF"}), 400

    job_id, _ = _new_job()
    f.save(str(WORK_DIR / f"{job_id}.pdf"))
    return jsonify({"job_id": job_id, "filename": f.filename})


@app.route("/process-text/<job_id>", methods=["POST"])
def process_text(job_id: str):
    if job_id not in _jobs:
        return jsonify({"error": "Job not found"}), 404
    data = request.get_json() or {}
    use_llm = bool(data.get("use_llm", False))
    # The UI sends "provider:model" (e.g. "google:gemma-3-27b-it",
    # "cerebras:qwen-3-235b-a22b-instruct-2507"). Fall back to Google default.
    llm_choice = data.get("llm_model", "google:gemma-3-27b-it")
    if ":" in llm_choice:
        llm_provider, llm_model = llm_choice.split(":", 1)
    else:
        llm_provider, llm_model = "google", llm_choice
    job = _jobs[job_id]
    job["queue"] = queue.Queue()
    job["status"] = "processing_text"
    threading.Thread(
        target=_run_text_processing,
        args=(job_id, use_llm, llm_provider, llm_model),
        daemon=True,
    ).start()
    return jsonify({"ok": True})


@app.route("/generate-audio/<job_id>", methods=["POST"])
def generate_audio(job_id: str):
    if job_id not in _jobs:
        return jsonify({"error": "Job not found"}), 404
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    tts_engine = data.get("tts_engine", "kokoro")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    job = _jobs[job_id]
    job["queue"] = queue.Queue()
    job["status"] = "generating_audio"
    threading.Thread(
        target=_run_audio_generation, args=(job_id, text, tts_engine), daemon=True
    ).start()
    return jsonify({"ok": True})


@app.route("/stream/<job_id>")
def stream(job_id: str):
    if job_id not in _jobs:
        return jsonify({"error": "Job not found"}), 404
    job = _jobs[job_id]

    def generate():
        while True:
            try:
                msg = job["queue"].get(timeout=30)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["type"] in ("done", "error"):
                    break
            except queue.Empty:
                # Keep-alive ping
                yield 'data: {"type":"ping"}\n\n'

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/audio/<job_id>")
def serve_audio(job_id: str):
    if job_id not in _jobs:
        return jsonify({"error": "Not found"}), 404
    path = _jobs[job_id].get("audio_path")
    if not path or not os.path.exists(path):
        return jsonify({"error": "Audio not ready"}), 404
    as_attachment = request.args.get("download") == "1"
    return send_file(
        path,
        mimetype="audio/mpeg",
        as_attachment=as_attachment,
        download_name="paper_audio.mp3",
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
