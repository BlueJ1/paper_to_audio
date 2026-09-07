"""Bounded in-process jobs for a personal local UI, with replayable events."""
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import threading
import time
import uuid


@dataclass
class Attempt:
    id: str
    operation: str
    status: str = "running"
    events: deque = field(default_factory=lambda: deque(maxlen=1000))
    sequence: int = 0
    result: dict | None = None


@dataclass
class Job:
    id: str
    directory: Path
    touched: float = field(default_factory=time.time)
    attempts: dict = field(default_factory=dict)
    latest: str | None = None


class JobError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class JobManager:
    def __init__(self, root, max_jobs=32, max_active=2, retention=86400, max_attempts=8):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.condition = threading.Condition(threading.RLock())
        self.max_jobs, self.max_active = max_jobs, max_active
        self.retention, self.max_attempts = retention, max_attempts

    def cleanup(self):
        with self.condition:
            now = time.time()
            for key, job in list(self.jobs.items()):
                if not self._busy(job) and now - job.touched > self.retention:
                    shutil.rmtree(job.directory, ignore_errors=True)
                    del self.jobs[key]
            # Previous-process files cannot be reconnected; expire them as well.
            for directory in self.root.iterdir():
                if (directory.is_dir() and directory.name not in self.jobs
                        and now - directory.stat().st_mtime > self.retention):
                    shutil.rmtree(directory, ignore_errors=True)

    @staticmethod
    def _busy(job):
        return any(a.status == "running" for a in job.attempts.values())

    def create(self):
        with self.condition:
            self.cleanup()
            if len(self.jobs) >= self.max_jobs:
                raise JobError("Local job storage is full. Remove an old job before uploading.", 429)
            key = str(uuid.uuid4())
            directory = self.root / key
            directory.mkdir()
            job = Job(key, directory)
            self.jobs[key] = job
            return job

    def get(self, key):
        with self.condition:
            if key not in self.jobs:
                raise JobError("Job not found or expired", 404)
            return self.jobs[key]

    def delete(self, key):
        with self.condition:
            job = self.get(key)
            if self._busy(job):
                raise JobError("Job is still running", 409)
            shutil.rmtree(job.directory, ignore_errors=True)
            del self.jobs[key]

    def begin(self, key, operation):
        with self.condition:
            job = self.get(key)
            if self._busy(job):
                raise JobError("This job already has an operation running", 409)
            if sum(self._busy(j) for j in self.jobs.values()) >= self.max_active:
                raise JobError("Local processing limit reached; try again shortly", 429)
            while len(job.attempts) >= self.max_attempts:
                oldest = next(iter(job.attempts))
                del job.attempts[oldest]
                for path in job.directory.glob(f"{oldest}.*"):
                    path.unlink(missing_ok=True)
            attempt = Attempt(str(uuid.uuid4()), operation)
            job.attempts[attempt.id] = attempt
            job.latest = attempt.id
            job.touched = time.time()
            self.emit(job, attempt, {"type": "log", "message": f"Started {operation}"})
            return job, attempt

    def attempt(self, key, attempt_id=None):
        with self.condition:
            job = self.get(key)
            attempt = job.attempts.get(attempt_id or job.latest)
            if attempt is None:
                raise JobError("Attempt not found or expired", 404)
            return job, attempt

    def emit(self, job, attempt, event):
        with self.condition:
            if attempt.status != "running":
                return
            attempt.sequence += 1
            attempt.events.append((attempt.sequence, event))
            if event["type"] in {"done", "error"}:
                attempt.status = event["type"]
                attempt.result = event
            job.touched = time.time()
            self.condition.notify_all()

    def snapshot(self, key, attempt_id=None):
        with self.condition:
            job, attempt = self.attempt(key, attempt_id)
            return {"job_id": job.id, "attempt_id": attempt.id, "operation": attempt.operation,
                    "status": attempt.status, "result": attempt.result, "last_event_id": attempt.sequence}

    def events(self, job, attempt, after=0):
        while True:
            with self.condition:
                pending = [(n, e) for n, e in attempt.events if n > after]
                terminal = attempt.status != "running"
                if not pending and not terminal:
                    self.condition.wait(timeout=15)
                    pending = [(n, e) for n, e in attempt.events if n > after]
                    terminal = attempt.status != "running"
            for sequence, event in pending:
                after = sequence
                yield sequence, event
            if terminal:
                return
            if not pending:
                yield None, {"type": "ping"}
