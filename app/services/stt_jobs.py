from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
import pathlib
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from typing import Any

from fastapi import HTTPException

from app.core.config import settings
from app.core.logger import logger
from app.services.stt_service import STTService


@dataclass
class STTJob:
    job_id: str
    kind: str
    model: str
    input_path: pathlib.Path
    normalized_audio_path: pathlib.Path
    output_json_path: pathlib.Path
    output_txt_path: pathlib.Path
    output_srt_path: pathlib.Path
    output_vtt_path: pathlib.Path
    status: str
    created_at: datetime
    options: dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0
    message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    original_filename: str | None = None
    duration_seconds: float | None = None
    language: str | None = None
    language_probability: float | None = None
    text: str | None = None
    segments: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False
    process_pid: int | None = None


class STTJobManager:
    def __init__(self, jobs_dir: str | pathlib.Path | None = None):
        self.jobs_dir = pathlib.Path(jobs_dir or settings.STT_JOBS_DIR)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        self._jobs: dict[str, STTJob] = {}
        self._jobs_lock = threading.Lock()

        self._processes: dict[str, subprocess.Popen] = {}
        self._processes_lock = threading.Lock()

        self._queue: queue.Queue[str | None] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._stop_event = threading.Event()

    def start(self):
        if self._workers:
            return

        self._cleanup_all_storage()

        workers_count = max(1, int(settings.STT_WORKERS))
        for index in range(workers_count):
            worker = threading.Thread(
                target=self._run_worker,
                name=f"stt-job-worker-{index + 1}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

        logger.info("STT job workers started count=%s jobs_dir=%s", workers_count, self.jobs_dir)

    def stop(self):
        logger.warning("Stopping STT job manager. Cancelling running processes.")

        with self._jobs_lock:
            running_ids = [
                job_id
                for job_id, job in self._jobs.items()
                if job.status in {"queued", "processing"}
            ]

        for job_id in running_ids:
            self.cancel_job(job_id, reason="El servicio fue detenido.")

        if not self._workers:
            return

        self._stop_event.set()

        for _ in self._workers:
            self._queue.put(None)

        for worker in self._workers:
            worker.join(timeout=5)

        self._workers.clear()
        logger.info("STT job workers stopped")

    def submit_file(
        self,
        content: bytes,
        filename: str | None,
        content_type: str | None,
        options: dict[str, Any],
    ) -> STTJob:
        STTService.validate_upload_size(content)

        extension = STTService.resolve_extension(filename, content_type)
        model_name = STTService.validate_model_name(
            options.get("model") or settings.STT_DEFAULT_MODEL
        )

        if not STTService.model_exists_locally(model_name):
            raise HTTPException(
                status_code=409,
                detail="El modelo seleccionado todavía no está preparado. Presiona 'Preparar modelo' primero.",
            )

        kind = "video" if extension in {".mp4", ".webm"} and (content_type or "").startswith("video/") else "audio"

        job = self._create_job(
            kind=kind,
            model=model_name,
            input_suffix=extension,
            options=options,
            original_filename=filename,
        )

        job.input_path.write_bytes(content)

        logger.info(
            "STT job created job_id=%s kind=%s model=%s filename=%s size=%s",
            job.job_id,
            kind,
            model_name,
            filename,
            len(content),
        )

        self._append_event(
            job.job_id,
            {
                "event": "queued",
                "status": "queued",
                "progress": 0.0,
                "message": "Tarea recibida y puesta en cola.",
            },
        )

        self._queue.put(job.job_id)
        return job

    def cancel_job(self, job_id: str, reason: str = "Tarea cancelada por el usuario.") -> dict[str, Any]:
        job = self.get_job(job_id)

        if job.status in {"done", "failed", "cancelled"}:
            return self.get_status(job_id)

        logger.warning("Cancelling STT job job_id=%s status=%s reason=%s", job_id, job.status, reason)

        self._update_job(
            job_id,
            cancel_requested=True,
            status="cancelled",
            progress=100.0,
            message=reason,
            finished_at=datetime.now(timezone.utc),
            error=None,
        )

        with self._processes_lock:
            process = self._processes.get(job_id)

        if process and process.poll() is None:
            self._terminate_process(process)

        self._append_event(
            job_id,
            {
                "event": "cancelled",
                "status": "cancelled",
                "progress": 100.0,
                "message": reason,
            },
        )

        return self.get_status(job_id)

    def get_job(self, job_id: str) -> STTJob:
        with self._jobs_lock:
            job = self._jobs.get(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return job

    def get_status(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)

        return {
            "job_id": job.job_id,
            "status": job.status,
            "kind": job.kind,
            "model": job.model,
            "language": job.language,
            "language_probability": job.language_probability,
            "task": job.options.get("task"),
            "progress": job.progress,
            "message": job.message,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "error": job.error,
            "original_filename": job.original_filename,
            "duration_seconds": job.duration_seconds,
            "text": job.text,
            "segments": job.segments,
            "result_ready": job.status == "done" and job.output_json_path.exists(),
            "cancel_requested": job.cancel_requested,
            "process_pid": job.process_pid,
        }

    def get_events_since(self, job_id: str, after_index: int = 0) -> list[dict[str, Any]]:
        job = self.get_job(job_id)

        with self._jobs_lock:
            return job.events[after_index:]

    def get_result_path(self, job_id: str, result_type: str) -> pathlib.Path:
        job = self.get_job(job_id)

        if job.status == "failed":
            raise HTTPException(status_code=409, detail=job.error or "Job failed")

        if job.status == "cancelled":
            raise HTTPException(status_code=409, detail="Job cancelled")

        if job.status != "done":
            raise HTTPException(status_code=409, detail="Job result is not ready yet")

        mapping = {
            "json": job.output_json_path,
            "txt": job.output_txt_path,
            "srt": job.output_srt_path,
            "vtt": job.output_vtt_path,
        }

        path = mapping.get(result_type)
        if not path:
            raise HTTPException(status_code=404, detail="Unsupported result type")

        if not path.exists():
            raise HTTPException(status_code=404, detail="Result file not found")

        return path

    def cleanup_finished_jobs(self) -> dict[str, Any]:
        terminal_statuses = {"done", "failed", "cancelled", "completed"}

        with self._jobs_lock:
            removable_ids = [
                job_id
                for job_id, job in self._jobs.items()
                if job.status in terminal_statuses
            ]

        for job_id in removable_ids:
            self._delete_job(job_id)

        removed_dirs = 0

        # Limpieza defensiva de carpetas huérfanas de jobs que ya no estén en memoria.
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        with self._jobs_lock:
            active_ids = set(self._jobs.keys())

        for item in self.jobs_dir.iterdir():
            if not item.is_dir():
                continue

            if item.name not in active_ids:
                shutil.rmtree(item, ignore_errors=True)
                removed_dirs += 1

        logger.info(
            "Cleaned finished STT jobs removed_jobs=%s removed_orphan_dirs=%s",
            len(removable_ids),
            removed_dirs,
        )

        return {
            "removed_jobs": len(removable_ids),
            "removed_orphan_dirs": removed_dirs,
        }
    def _create_job(
        self,
        kind: str,
        model: str,
        input_suffix: str,
        options: dict[str, Any],
        original_filename: str | None,
    ) -> STTJob:
        job_id = uuid.uuid4().hex
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        job = STTJob(
            job_id=job_id,
            kind=kind,
            model=model,
            input_path=job_dir / f"input{input_suffix}",
            normalized_audio_path=job_dir / "normalized.wav",
            output_json_path=job_dir / "result.json",
            output_txt_path=job_dir / "result.txt",
            output_srt_path=job_dir / "result.srt",
            output_vtt_path=job_dir / "result.vtt",
            status="queued",
            created_at=datetime.now(timezone.utc),
            options=options,
            original_filename=original_filename,
        )

        with self._jobs_lock:
            self._jobs[job_id] = job

        return job

    def _run_worker(self):
        while not self._stop_event.is_set():
            job_id = self._queue.get()

            try:
                if job_id is None:
                    return

                job = self.get_job(job_id)
                if job.cancel_requested or job.status == "cancelled":
                    logger.info("Skipping cancelled STT job before processing job_id=%s", job_id)
                    continue

                self._process_job(job_id)
            finally:
                self._queue.task_done()

    def _process_job(self, job_id: str):
        job = self.get_job(job_id)

        logger.info("Starting STT job processing job_id=%s model=%s", job_id, job.model)

        self._update_job(
            job_id,
            status="processing",
            progress=5.0,
            message="Preparando archivo.",
            started_at=datetime.now(timezone.utc),
            error=None,
        )

        self._append_event(
            job_id,
            {
                "event": "processing",
                "status": "processing",
                "progress": 5.0,
                "message": "Preparando archivo.",
            },
        )

        try:
            self._append_event(
                job_id,
                {
                    "event": "normalizing",
                    "status": "processing",
                    "progress": 10.0,
                    "message": "Preparando audio para transcripción.",
                },
            )

            STTService.normalize_audio_to_wav(job.input_path, job.normalized_audio_path)

            if self.get_job(job_id).cancel_requested:
                raise RuntimeError("Tarea cancelada antes de iniciar transcripción.")

            self._update_job(
                job_id,
                progress=12.0,
                message="Audio preparado. Iniciando transcripción.",
            )

            return_code = self._run_transcription_process(job_id)

            latest_job = self.get_job(job_id)
            if latest_job.cancel_requested or latest_job.status == "cancelled":
                logger.warning("STT job cancelled job_id=%s", job_id)
                return

            if return_code != 0:
                raise RuntimeError(f"El proceso de transcripción terminó con código {return_code}.")

            if not job.output_json_path.exists():
                raise RuntimeError("La transcripción terminó sin generar resultado JSON.")

            result = json.loads(job.output_json_path.read_text(encoding="utf-8"))

            self._update_job(
                job_id,
                status="done",
                progress=100.0,
                message="Transcripción completada.",
                finished_at=datetime.now(timezone.utc),
                duration_seconds=result.get("duration_seconds"),
                language=result.get("language"),
                language_probability=result.get("language_probability"),
                text=result.get("text"),
                segments=result.get("segments", []),
            )

            self._append_event(
                job_id,
                {
                    "event": "completed",
                    "status": "done",
                    "progress": 100.0,
                    "message": "Transcripción completada.",
                    "result": {
                        "text": result.get("text"),
                        "language": result.get("language"),
                        "language_probability": result.get("language_probability"),
                        "duration_seconds": result.get("duration_seconds"),
                    },
                },
            )

            logger.info("STT job finished job_id=%s", job_id)

        except Exception as exc:
            latest_job = self.get_job(job_id)
            if latest_job.cancel_requested or latest_job.status == "cancelled":
                logger.warning("STT job ended after cancellation job_id=%s", job_id)
                return

            error_text = self._stringify_exception(exc)

            self._update_job(
                job_id,
                status="failed",
                progress=100.0,
                message="La transcripción falló.",
                finished_at=datetime.now(timezone.utc),
                error=error_text,
            )

            self._append_event(
                job_id,
                {
                    "event": "failed",
                    "status": "failed",
                    "progress": 100.0,
                    "message": error_text,
                },
            )

            logger.exception("STT job failed job_id=%s error=%s", job_id, error_text)

        finally:
            with self._processes_lock:
                self._processes.pop(job_id, None)

            self._update_job(job_id, process_pid=None)

    def _run_transcription_process(self, job_id: str) -> int:
        job = self.get_job(job_id)

        command = [
            sys.executable,
            "-m",
            "app.services.stt_transcribe_runner",
            "--model",
            job.model,
            "--audio-path",
            str(job.normalized_audio_path),
            "--output-json",
            str(job.output_json_path),
            "--output-txt",
            str(job.output_txt_path),
            "--output-srt",
            str(job.output_srt_path),
            "--output-vtt",
            str(job.output_vtt_path),
            "--models-dir",
            str(settings.MODELS_DIR),
            "--device",
            settings.STT_DEVICE,
            "--compute-type",
            settings.STT_DEFAULT_COMPUTE_TYPE,
            "--cpu-threads",
            str(settings.STT_DEFAULT_CPU_THREADS),
            "--num-workers",
            str(settings.STT_DEFAULT_NUM_WORKERS),
            "--task",
            job.options.get("task") or "transcribe",
            "--language",
            job.options.get("language") or "",
            "--beam-size",
            str(int(job.options.get("beam_size") or 3)),
            "--vad-filter",
            "true" if settings.STT_DEFAULT_VAD_FILTER else "false",
            "--word-timestamps",
            "true" if settings.STT_DEFAULT_WORD_TIMESTAMPS else "false",
        ]

        logger.info("Launching STT transcription subprocess job_id=%s command=%s", job_id, " ".join(command))

        process = subprocess.Popen(
            command,
            cwd=str(settings.BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        with self._processes_lock:
            self._processes[job_id] = process

        self._update_job(job_id, process_pid=process.pid)

        assert process.stdout is not None

        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue

            logger.info("STT subprocess output job_id=%s line=%s", job_id, line)

            if self.get_job(job_id).cancel_requested:
                logger.warning("Cancellation detected while reading subprocess output job_id=%s", job_id)
                self._terminate_process(process)
                return 130

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                self._append_event(
                    job_id,
                    {
                        "event": "log",
                        "status": "processing",
                        "progress": self.get_job(job_id).progress,
                        "message": line,
                    },
                )
                continue

            self._handle_process_event(job_id, payload)

        return_code = process.wait()

        logger.info("STT subprocess finished job_id=%s return_code=%s", job_id, return_code)
        return return_code

    def _handle_process_event(self, job_id: str, payload: dict[str, Any]) -> None:
        event = payload.get("event", "progress")
        progress = float(payload.get("progress", self.get_job(job_id).progress))
        message = payload.get("message") or "Procesando..."

        changes: dict[str, Any] = {
            "progress": progress,
            "message": message,
        }

        if payload.get("duration_seconds"):
            changes["duration_seconds"] = payload.get("duration_seconds")

        if payload.get("segment"):
            with self._jobs_lock:
                job = self._jobs.get(job_id)
                if job:
                    job.segments.append(payload["segment"])
                    current_text = "\n".join(
                        segment.get("text", "").strip()
                        for segment in job.segments
                        if segment.get("text", "").strip()
                    )
                    job.text = current_text

        if event == "error":
            changes["error"] = payload.get("message")

        self._update_job(job_id, **changes)

        mapped_event = {
            "done": "completed",
            "error": "failed",
        }.get(event, event)

        self._append_event(
            job_id,
            {
                "event": mapped_event,
                "status": "processing" if mapped_event not in {"failed", "completed"} else mapped_event,
                "progress": progress,
                "message": message,
                **({"segment": payload["segment"]} if payload.get("segment") else {}),
                **({"result": payload["result"]} if payload.get("result") else {}),
            },
        )

    def _update_job(self, job_id: str, **changes):
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                return

            for field_name, value in changes.items():
                setattr(job, field_name, value)

    def _append_event(self, job_id: str, event: dict[str, Any]):
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                return

            event_payload = {
                "index": len(job.events),
                "job_id": job_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                **event,
            }
            job.events.append(event_payload)

            if len(job.events) > 1000:
                job.events = job.events[-1000:]

    def _delete_job(self, job_id: str):
        with self._jobs_lock:
            job = self._jobs.pop(job_id, None)

        if not job:
            return

        with self._processes_lock:
            process = self._processes.pop(job_id, None)

        if process and process.poll() is None:
            self._terminate_process(process)

        job_dir = job.input_path.parent
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)

    def _cleanup_all_storage(self):
        removed = 0
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        for item in self.jobs_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
                removed += 1
            elif item.is_file():
                item.unlink(missing_ok=True)
                removed += 1

        if removed:
            logger.info("Removed stale STT job files on startup count=%s", removed)

    @staticmethod
    def _terminate_process(process: subprocess.Popen) -> None:
        try:
            if process.poll() is not None:
                return

            pid = process.pid
            logger.warning("Terminating STT subprocess pid=%s", pid)

            try:
                os.killpg(pid, signal.SIGTERM)
            except Exception:
                process.terminate()

            deadline = time.time() + settings.STT_PROCESS_TERMINATE_TIMEOUT_SECONDS
            while process.poll() is None and time.time() < deadline:
                time.sleep(0.2)

            if process.poll() is None:
                logger.warning("Force killing STT subprocess pid=%s", pid)
                try:
                    os.killpg(pid, signal.SIGKILL)
                except Exception:
                    process.kill()

        except Exception:
            logger.exception("Could not terminate STT subprocess cleanly")

    @staticmethod
    def _stringify_exception(exc: Exception) -> str:
        if isinstance(exc, HTTPException):
            return str(exc.detail)
        return str(exc) or exc.__class__.__name__