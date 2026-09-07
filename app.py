"""Personal local web workflow with bounded, replayable processing attempts."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import fitz
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

from jobs import JobError, JobManager
from processing import process_pdf

load_dotenv()
app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=50 * 1024 * 1024, MAX_TEXT_CHARS=2_000_000,
                  AUDIO_TIMEOUT=1800)
WORK_DIR = Path(tempfile.gettempdir()) / "paper_to_audio_ui"
PROJECT_DIR = Path(__file__).resolve().parent
manager = JobManager(WORK_DIR)


@app.before_request
def expire_jobs():
    manager.cleanup()


@app.errorhandler(JobError)
def job_error(exc):
    return jsonify(error=str(exc)), exc.status


@app.errorhandler(HTTPException)
def http_error(exc):
    return jsonify(error=exc.description), exc.code


@app.errorhandler(Exception)
def server_error(exc):
    app.logger.exception("Request failed")
    return jsonify(error="Local operation failed; check available disk space and application logs."), 500


def payload():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise JobError("Expected a JSON object")
    return data


def start(job_id, operation, work):
    job, attempt = manager.begin(job_id, operation)
    def execute():
        try:
            result = work(job, attempt)
            manager.emit(job, attempt, {"type": "done", **result})
        except Exception as exc:
            app.logger.exception("Attempt %s failed", attempt.id)
            manager.emit(job, attempt, {"type": "error", "message": str(exc)})
    try:
        threading.Thread(target=execute, daemon=True).start()
    except Exception as exc:
        manager.emit(job, attempt, {"type": "error", "message": str(exc)})
        raise JobError("Could not start local worker", 503) from exc
    return jsonify(ok=True, attempt_id=attempt.id,
                   stream_url=f"/stream/{job_id}/{attempt.id}",
                   status_url=f"/status/{job_id}/{attempt.id}"), 202


def _run_text_processing(job, attempt, use_llm, provider, model):
    text, report = process_pdf(job.directory / "source.pdf", use_llm, provider, model)
    (job.directory / f"{attempt.id}.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    for notice in report["warnings"]:
        manager.emit(job, attempt, {"type": "log", "message": notice})
    return {"text": text, "warnings": report["warnings"],
            "inspection_url": f"/inspection/{job.id}/{attempt.id}"}


def _run_audio_generation(job, attempt, text, engine):
    text_path = job.directory / f"{attempt.id}.txt"
    audio_path = job.directory / f"{attempt.id}.mp3"
    proc = None
    timer = None
    timed_out = threading.Event()
    def stop_process():
        timed_out.set()
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        text_path.write_text(text, encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-u", str(PROJECT_DIR / "text_to_speech.py"), str(text_path),
             "--out", str(audio_path), "--tts-engine", engine],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            cwd=PROJECT_DIR, start_new_session=True)
        timer = threading.Timer(app.config["AUDIO_TIMEOUT"], stop_process)
        timer.daemon = True
        timer.start()
        for line in proc.stdout:
            if line.strip():
                manager.emit(job, attempt, {"type": "log", "message": line.rstrip()[:4000]})
        rc = proc.wait()
        if timed_out.is_set():
            raise RuntimeError("Audio generation timed out; try a shorter transcript or fewer workers")
        if rc != 0 or not audio_path.is_file() or not audio_path.stat().st_size:
            raise RuntimeError(f"Audio generation failed (exit {rc})")
        return {"audio_url": f"/audio/{job.id}/{attempt.id}"}
    finally:
        if timer:
            timer.cancel()
        if proc is not None:
            if proc.poll() is None:
                stop_process()
                proc.wait()
            if proc.stdout:
                proc.stdout.close()
        text_path.unlink(missing_ok=True)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/upload")
def upload():
    file = request.files.get("pdf")
    if not file or not file.filename or not file.filename.lower().endswith(".pdf"):
        raise JobError("Select a PDF file")
    job = manager.create()
    try:
        path = job.directory / "source.pdf"
        file.save(path)
        with fitz.open(path) as pdf:
            if not pdf.is_pdf or pdf.needs_pass or len(pdf) == 0:
                raise ValueError("Unreadable or password-protected PDF")
            # Load each page to catch broken page trees without requiring text.
            for page in pdf:
                _ = page.rect
    except Exception as exc:
        manager.delete(job.id)
        if isinstance(exc, OSError):
            raise
        raise JobError("File is not a readable, unencrypted PDF") from exc
    return jsonify(job_id=job.id, filename=file.filename)


@app.post("/process-text/<job_id>")
def process_text(job_id):
    manager.get(job_id)
    data = payload()
    use_llm = data.get("use_llm", False)
    choice = data.get("llm_model", "google:gemma-3-27b-it")
    if type(use_llm) is not bool or not isinstance(choice, str) or not choice.strip() or len(choice) > 200:
        raise JobError("use_llm must be a boolean and llm_model a nonempty string")
    provider, model = choice.split(":", 1) if ":" in choice else ("google", choice)
    if provider not in {"google", "cerebras"} or not model.strip():
        raise JobError("Invalid model/provider choice")
    return start(job_id, "text processing", lambda j, a: _run_text_processing(j, a, use_llm, provider, model))


@app.post("/generate-audio/<job_id>")
def generate_audio(job_id):
    manager.get(job_id)
    data = payload()
    text, engine = data.get("text"), data.get("tts_engine", "kokoro")
    if not isinstance(text, str) or not text.strip() or len(text) > app.config["MAX_TEXT_CHARS"]:
        raise JobError("Provide nonempty text within the 2,000,000 character limit")
    if not isinstance(engine, str) or engine not in {"kokoro", "murf"}:
        raise JobError("Invalid TTS engine")
    return start(job_id, "audio generation", lambda j, a: _run_audio_generation(j, a, text.strip(), engine))


@app.get("/status/<job_id>")
@app.get("/status/<job_id>/<attempt_id>")
def status(job_id, attempt_id=None):
    return jsonify(manager.snapshot(job_id, attempt_id))


@app.get("/stream/<job_id>/<attempt_id>")
def stream(job_id, attempt_id):
    job, attempt = manager.attempt(job_id, attempt_id)
    try:
        after = int(request.headers.get("Last-Event-ID", request.args.get("after", "0")))
        if not 0 <= after <= attempt.sequence:
            raise ValueError()
    except ValueError:
        raise JobError("Invalid event cursor")
    def generate():
        for sequence, event in manager.events(job, attempt, after):
            prefix = f"id: {sequence}\n" if sequence is not None else ""
            yield f"{prefix}data: {json.dumps(event)}\n\n"
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/audio/<job_id>/<attempt_id>")
def serve_audio(job_id, attempt_id):
    job, attempt = manager.attempt(job_id, attempt_id)
    path = job.directory / f"{attempt.id}.mp3"
    if attempt.status != "done" or not attempt.result.get("audio_url") or not path.is_file():
        raise JobError("This attempt has no completed audio", 404)
    return send_file(path, mimetype="audio/mpeg", as_attachment=request.args.get("download") == "1",
                     download_name="paper_audio.mp3")


@app.get("/inspection/<job_id>/<attempt_id>")
def inspection(job_id, attempt_id):
    job, attempt = manager.attempt(job_id, attempt_id)
    path = job.directory / f"{attempt.id}.json"
    if attempt.status != "done" or not path.is_file():
        raise JobError("Inspection not ready", 404)
    return send_file(path, mimetype="application/json", as_attachment=True)


@app.delete("/jobs/<job_id>")
def delete_job(job_id):
    manager.delete(job_id)
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=False, port=5000, threaded=True)
