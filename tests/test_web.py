"""Local workflow contract; no network calls or TTS models required."""
import io
import json
import threading
import time
from pathlib import Path

import fitz
import pytest

import app as web
from jobs import JobManager


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "manager", JobManager(tmp_path / "jobs"))
    web.app.config.update(TESTING=True)
    return web.app.test_client()


def pdf_bytes():
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 60), "A Short Paper", fontsize=18)
        page.insert_text((50, 110), "1 Introduction", fontsize=13)
        page.insert_text((50, 150), "This is a brief paper about faithful narration.", fontsize=10)
        return pdf.tobytes()


def upload(client):
    response = client.post("/upload", data={"pdf": (io.BytesIO(pdf_bytes()), "paper.pdf")})
    assert response.status_code == 200
    return response.json["job_id"]


def wait(job_id, attempt_id):
    _, attempt = web.manager.attempt(job_id, attempt_id)
    with web.manager.condition:
        assert web.manager.condition.wait_for(lambda: attempt.status != "running", timeout=5)
    return attempt


def test_upload_requires_readable_pdf(client):
    for content, name in [(b"not a PDF", "x.pdf"), (pdf_bytes(), "x.txt")]:
        assert client.post("/upload", data={"pdf": (io.BytesIO(content), name)}).status_code == 400
    assert not web.manager.jobs and not list(web.manager.root.iterdir())


def test_upload_413_is_json(client, monkeypatch):
    monkeypatch.setitem(web.app.config, "MAX_CONTENT_LENGTH", 10)
    response = client.post("/upload", data={"pdf": (io.BytesIO(pdf_bytes()), "x.pdf")})
    assert response.status_code == 413 and response.json["error"]


@pytest.mark.parametrize("route,data", [
    ("process-text", {"use_llm": "false"}), ("process-text", {"llm_model": 7}),
    ("process-text", {"llm_model": "unknown:model"}), ("process-text", []),
    ("generate-audio", {"text": 7}), ("generate-audio", {"text": "x", "tts_engine": "bad"}),
    ("generate-audio", {"text": "x", "tts_engine": []}), ("generate-audio", {"text": "  "}),
])
def test_payload_validation(client, route, data):
    job_id = upload(client)
    response = client.post(f"/{route}/{job_id}", json=data)
    assert response.status_code == 400 and not web.manager.get(job_id).attempts


def test_full_route_flow_and_replay(client, monkeypatch):
    job_id = upload(client)
    response = client.post(f"/process-text/{job_id}", json={"use_llm": False})
    assert response.status_code == 202
    text_attempt = wait(job_id, response.json["attempt_id"])
    assert text_attempt.status == "done" and "faithful narration" in text_attempt.result["text"]
    inspection = client.get(text_attempt.result["inspection_url"])
    assert inspection.status_code == 200 and json.loads(inspection.data)["stats"]
    inspection.close()
    # Two readers each see completion; reconnecting with an event cursor replays.
    for _ in range(2):
        events = client.get(response.json["stream_url"]).data.decode()
        assert '"type": "done"' in events
    replay = client.get(response.json["stream_url"], headers={"Last-Event-ID": "1"}).data.decode()
    assert '"type": "done"' in replay
    assert client.get(response.json["status_url"]).json["result"] == text_attempt.result

    def audio(job, attempt, text, engine):
        assert text.endswith("Edited.")
        (job.directory / f"{attempt.id}.mp3").write_bytes(b"fake test audio")
        return {"audio_url": f"/audio/{job.id}/{attempt.id}"}
    monkeypatch.setattr(web, "_run_audio_generation", audio)
    urls = []
    for _ in range(2):
        response = client.post(f"/generate-audio/{job_id}", json={"text": text_attempt.result["text"] + " Edited.", "tts_engine": "kokoro"})
        attempt = wait(job_id, response.json["attempt_id"])
        assert attempt.status == "done"
        urls.append(attempt.result["audio_url"])
        download = client.get(urls[-1] + "?download=1")
        assert download.status_code == 200 and "attachment" in download.headers["Content-Disposition"]
        download.close()
    assert urls[0] != urls[1]


def test_overlaps_and_local_concurrency_rejected(client, monkeypatch):
    release = threading.Event()
    def held(job, attempt, *args):
        assert release.wait(5)
        return {"text": "Done"}
    monkeypatch.setattr(web, "_run_text_processing", held)
    one, two, three = [upload(client) for _ in range(3)]
    started = []
    try:
        for key in [one, two]:
            result = client.post(f"/process-text/{key}", json={})
            assert result.status_code == 202
            started.append((key, result.json["attempt_id"]))
        assert client.post(f"/generate-audio/{one}", json={"text": "x"}).status_code == 409
        assert client.post(f"/process-text/{three}", json={}).status_code == 429
    finally:
        release.set()
        for key, attempt in started:
            wait(key, attempt)


@pytest.mark.parametrize("failure", ["disk", "subprocess"])
def test_worker_lifecycle_failures_are_terminal(client, monkeypatch, failure):
    job_id = upload(client)
    if failure == "disk":
        original = Path.write_text
        def broken(path, *args, **kwargs):
            if path.suffix == ".txt":
                raise OSError("disk full")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "write_text", broken)
    else:
        monkeypatch.setattr(web.subprocess, "Popen", lambda *a, **kw: (_ for _ in ()).throw(OSError("subprocess failed")))
    response = client.post(f"/generate-audio/{job_id}", json={"text": "Hello"})
    attempt = wait(job_id, response.json["attempt_id"])
    assert attempt.status == "error"
    assert failure in attempt.result["message"]
    assert client.get(f"/audio/{job_id}/{attempt.id}").status_code == 404
    assert not list(web.manager.get(job_id).directory.glob("*.txt"))


def test_failed_regeneration_has_no_stale_audio(client, monkeypatch):
    key = upload(client)
    job, old = web.manager.begin(key, "audio generation")
    path = job.directory / f"{old.id}.mp3"; path.write_bytes(b"old")
    web.manager.emit(job, old, {"type": "done", "audio_url": f"/audio/{key}/{old.id}"})
    monkeypatch.setattr(web, "_run_audio_generation", lambda *a: (_ for _ in ()).throw(ValueError("failed")))
    response = client.post(f"/generate-audio/{key}", json={"text": "new"})
    attempt = wait(key, response.json["attempt_id"])
    assert client.get(f"/audio/{key}/{attempt.id}").status_code == 404
    assert path.read_bytes() == b"old"


def test_empty_narration_is_error(client, monkeypatch):
    key = upload(client)
    monkeypatch.setattr(web, "process_pdf", lambda *a: (_ for _ in ()).throw(ValueError("No usable narration")))
    response = client.post(f"/process-text/{key}", json={})
    assert wait(key, response.json["attempt_id"]).status == "error"


def test_retention_attempt_bounds_and_delete(client):
    key = upload(client)
    job = web.manager.get(key)
    for _ in range(web.manager.max_attempts + 1):
        _, attempt = web.manager.begin(key, "test")
        (job.directory / f"{attempt.id}.json").write_text("{}")
        web.manager.emit(job, attempt, {"type": "done"})
    assert len(job.attempts) == web.manager.max_attempts
    assert len(list(job.directory.glob("*.json"))) == web.manager.max_attempts
    job.touched = time.time() - web.manager.retention - 1
    web.manager.cleanup()
    assert not job.directory.exists() and client.get(f"/status/{key}").status_code == 404
    another = upload(client)
    assert client.delete(f"/jobs/{another}").status_code == 200


def test_event_log_is_bounded_but_keeps_terminal(client):
    key = upload(client)
    job, attempt = web.manager.begin(key, "test")
    for i in range(1100):
        web.manager.emit(job, attempt, {"type": "log", "message": str(i)})
    web.manager.emit(job, attempt, {"type": "done", "text": "Done"})
    assert len(attempt.events) == 1000
    assert list(web.manager.events(job, attempt))[-1][1]["type"] == "done"
