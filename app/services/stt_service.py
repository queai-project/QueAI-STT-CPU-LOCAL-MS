from __future__ import annotations

import asyncio
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

from fastapi import HTTPException

from app.core.config import settings
from app.core.logger import logger


class STTService:
    SUPPORTED_EXTENSIONS = {
        ".wav",
        ".mp3",
        ".m4a",
        ".ogg",
        ".webm",
        ".mp4",
        ".mpeg",
        ".mpga",
        ".flac",
    }

    SUPPORTED_MODELS = [
        {
            "id": "tiny",
            "label": "Básico",
            "public_label": "Básico",
            "description": "Más rápido y liviano. Ideal para empezar.",
            "ui_note": "Rápido",
            "multilingual": True,
            "english_only": False,
            "default": True,
        },
        {
            "id": "base",
            "label": "Recomendado",
            "public_label": "Recomendado",
            "description": "Mejor equilibrio entre calidad y velocidad.",
            "ui_note": "Equilibrado",
            "multilingual": True,
            "english_only": False,
            "default": False,
        },
        {
            "id": "small",
            "label": "Mejor",
            "public_label": "Mejor",
            "description": "Mayor precisión, pero más lento en CPU.",
            "ui_note": "Mayor precisión",
            "multilingual": True,
            "english_only": False,
            "default": False,
        },
    ]

    LANGUAGE_OPTIONS = [
        {"code": "", "label": "Detectar automáticamente"},
        {"code": "en", "label": "Inglés"},
        {"code": "es", "label": "Español"},
        {"code": "pt", "label": "Portugués"},
        {"code": "fr", "label": "Francés"},
        {"code": "de", "label": "Alemán"},
        {"code": "it", "label": "Italiano"},
        {"code": "nl", "label": "Neerlandés"},
        {"code": "ru", "label": "Ruso"},
        {"code": "zh", "label": "Chino"},
        {"code": "ja", "label": "Japonés"},
        {"code": "ko", "label": "Coreano"},
        {"code": "ar", "label": "Árabe"},
        {"code": "hi", "label": "Hindi"},
        {"code": "tr", "label": "Turco"},
        {"code": "pl", "label": "Polaco"},
        {"code": "uk", "label": "Ucraniano"},
        {"code": "vi", "label": "Vietnamita"},
        {"code": "id", "label": "Indonesio"},
        {"code": "sv", "label": "Sueco"},
        {"code": "no", "label": "Noruego"},
        {"code": "da", "label": "Danés"},
        {"code": "fi", "label": "Finés"},
        {"code": "el", "label": "Griego"},
        {"code": "he", "label": "Hebreo"},
        {"code": "ro", "label": "Rumano"},
        {"code": "cs", "label": "Checo"},
        {"code": "hu", "label": "Húngaro"},
        {"code": "th", "label": "Tailandés"},
    ]

    _download_lock: asyncio.Lock | None = None
    _download_jobs: dict[str, dict[str, Any]] = {}
    _download_processes: dict[str, asyncio.subprocess.Process] = {}
    _download_tasks: dict[str, asyncio.Task] = {}
    _download_cancelled: set[str] = set()

    def __init__(self, model_name: str | None = None):
        self.model_name = self.validate_model_name(model_name or settings.STT_DEFAULT_MODEL)

    @classmethod
    def _get_download_lock(cls) -> asyncio.Lock:
        if cls._download_lock is None:
            cls._download_lock = asyncio.Lock()
        return cls._download_lock

    @classmethod
    def validate_model_name(cls, model_name: str) -> str:
        model_name = (model_name or "").strip()
        valid = {item["id"] for item in cls.SUPPORTED_MODELS}

        if model_name not in valid:
            raise HTTPException(status_code=422, detail=f"Modelo no soportado: {model_name}")

        return model_name

    @classmethod
    def get_models(cls) -> dict[str, Any]:
        models = []

        for item in cls.SUPPORTED_MODELS:
            model_id = item["id"]
            available = cls.model_exists_locally(model_id)

            models.append(
                {
                    **item,
                    "available": available,
                    "download": cls.get_download_status_sync(model_id),
                }
            )

        return {
            "default_model": settings.STT_DEFAULT_MODEL,
            "languages": cls.LANGUAGE_OPTIONS,
            "models": models,
        }

    @classmethod
    def _safe_model_marker_name(cls, model_name: str) -> str:
        return model_name.replace("/", "_").replace(".", "_")

    @classmethod
    def _safe_model_folder_name(cls, model_name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name)

    @classmethod
    def _model_marker_path(cls, model_name: str) -> pathlib.Path:
        return settings.PREPARED_MODELS_DIR / f"{cls._safe_model_marker_name(model_name)}.json"

    @classmethod
    def _temporary_marker_path(cls, model_name: str) -> pathlib.Path:
        return settings.PREPARED_MODELS_DIR / f"{cls._safe_model_marker_name(model_name)}.preparing.json"

    @classmethod
    def _model_marker_exists(cls, model_name: str) -> bool:
        return cls._model_marker_path(model_name).exists()

    @classmethod
    def _candidate_model_dirs(cls, model_name: str) -> list[pathlib.Path]:
        safe_name = cls._safe_model_folder_name(model_name)
        marker_safe_name = cls._safe_model_marker_name(model_name)

        return [
            settings.MODELS_DIR / model_name,
            settings.MODELS_DIR / safe_name,
            settings.MODELS_DIR / marker_safe_name,
            settings.MODELS_DIR / f"models--Systran--faster-whisper-{model_name}",
            settings.MODELS_DIR / f"models--openai--whisper-{model_name}",
        ]

    @classmethod
    def model_exists_locally(cls, model_name: str) -> bool:
        model_name = cls.validate_model_name(model_name)

        # Solo confiamos en el marcador final.
        # Esto evita marcar como listo un modelo parcial o corrupto.
        return cls._model_marker_exists(model_name)

    @classmethod
    def _has_partial_model_files(cls, model_name: str) -> bool:
        if cls._temporary_marker_path(model_name).exists():
            return True

        for folder in cls._candidate_model_dirs(model_name):
            if folder.exists() and folder.is_dir() and any(folder.rglob("*")):
                return True

        return False

    @classmethod
    def _cleanup_partial_model(cls, model_name: str) -> None:
        logger.warning("Cleaning partial STT model files for model=%s", model_name)

        cls._temporary_marker_path(model_name).unlink(missing_ok=True)

        for folder in cls._candidate_model_dirs(model_name):
            if folder.exists() and folder.is_dir():
                shutil.rmtree(folder, ignore_errors=True)

    @classmethod
    def get_download_status_sync(cls, model_name: str) -> dict[str, Any]:
        model_name = cls.validate_model_name(model_name)

        current = cls._download_jobs.get(model_name)
        if current:
            return current

        available = cls.model_exists_locally(model_name)
        partial = (not available) and cls._has_partial_model_files(model_name)

        if available:
            return {
                "model": model_name,
                "status": "downloaded",
                "progress": 100.0,
                "message": "Modelo listo",
                "partial": False,
                "started_at": None,
                "finished_at": None,
            }

        if partial:
            return {
                "model": model_name,
                "status": "partial",
                "progress": 0.0,
                "message": "Modelo incompleto. Prepáralo nuevamente.",
                "partial": True,
                "started_at": None,
                "finished_at": None,
            }

        return {
            "model": model_name,
            "status": "idle",
            "progress": 0.0,
            "message": "Modelo no preparado",
            "partial": False,
            "started_at": None,
            "finished_at": None,
        }

    @classmethod
    async def get_download_status(cls, model_name: str) -> dict[str, Any]:
        model_name = cls.validate_model_name(model_name)
        return cls.get_download_status_sync(model_name)

    @classmethod
    async def prepare_model(cls, model_name: str) -> dict[str, Any]:
        model_name = cls.validate_model_name(model_name)

        async with cls._get_download_lock():
            current_task = cls._download_tasks.get(model_name)
            current_status = cls._download_jobs.get(model_name)

            if current_task and not current_task.done():
                logger.info("STT model preparation already running model=%s", model_name)
                return current_status or cls.get_download_status_sync(model_name)

            if cls.model_exists_locally(model_name):
                cls._download_jobs[model_name] = {
                    "model": model_name,
                    "status": "downloaded",
                    "progress": 100.0,
                    "message": "Modelo listo",
                    "partial": False,
                    "started_at": None,
                    "finished_at": time.time(),
                }
                return cls._download_jobs[model_name]

            if cls._has_partial_model_files(model_name):
                logger.warning("Partial model detected before prepare model=%s", model_name)
                cls._cleanup_partial_model(model_name)

            cls._download_cancelled.discard(model_name)

            cls._download_jobs[model_name] = {
                "model": model_name,
                "status": "running",
                "progress": 1.0,
                "message": "Preparando modelo. Esto puede tardar la primera vez.",
                "partial": False,
                "started_at": time.time(),
                "finished_at": None,
            }

            logger.info("Scheduling STT model preparation model=%s", model_name)

            cls._download_tasks[model_name] = asyncio.create_task(
                cls._prepare_model_process(model_name)
            )

            return cls._download_jobs[model_name]

    @classmethod
    async def cancel_model_download(cls, model_name: str) -> dict[str, Any]:
        model_name = cls.validate_model_name(model_name)

        logger.warning("Cancel model preparation requested model=%s", model_name)

        cls._download_cancelled.add(model_name)

        process = cls._download_processes.get(model_name)
        task = cls._download_tasks.get(model_name)

        if process and process.returncode is None:
            logger.warning(
                "Terminating STT model preparation process model=%s pid=%s",
                model_name,
                process.pid,
            )
            await cls._terminate_async_process(process)
        else:
            logger.warning(
                "No active process found for model=%s. Marking as cancelled anyway.",
                model_name,
            )

        if task and not task.done():
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                logger.warning("Model preparation task cancelled model=%s", model_name)
            except Exception:
                logger.exception("Model preparation task ended with error while cancelling model=%s", model_name)

        cls._cleanup_partial_model(model_name)

        cls._download_jobs[model_name] = {
            "model": model_name,
            "status": "cancelled",
            "progress": 100.0,
            "message": "Preparación cancelada.",
            "partial": False,
            "started_at": cls._download_jobs.get(model_name, {}).get("started_at"),
            "finished_at": time.time(),
        }

        cls._download_processes.pop(model_name, None)
        cls._download_tasks.pop(model_name, None)
        cls._download_cancelled.discard(model_name)

        logger.warning("Model preparation cancelled model=%s", model_name)

        return cls._download_jobs[model_name]

    @classmethod
    async def _prepare_model_process(cls, model_name: str) -> None:
        command = [
            sys.executable,
            "-m",
            "app.services.stt_model_prepare_runner",
            "--model",
            model_name,
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
        ]

        logger.info(
            "Starting STT model preparation process model=%s command=%s",
            model_name,
            " ".join(command),
        )

        process: asyncio.subprocess.Process | None = None

        try:
            if model_name in cls._download_cancelled:
                logger.warning("Model preparation cancelled before process start model=%s", model_name)
                cls._cleanup_partial_model(model_name)
                cls._download_jobs[model_name] = {
                    "model": model_name,
                    "status": "cancelled",
                    "progress": 100.0,
                    "message": "Preparación cancelada.",
                    "partial": False,
                    "started_at": cls._download_jobs.get(model_name, {}).get("started_at"),
                    "finished_at": time.time(),
                }
                return

            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(settings.BASE_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )

            cls._download_processes[model_name] = process

            logger.info(
                "STT model preparation process started model=%s pid=%s",
                model_name,
                process.pid,
            )

            assert process.stdout is not None

            while True:
                if model_name in cls._download_cancelled:
                    logger.warning("Cancellation detected while preparing model=%s", model_name)
                    await cls._terminate_async_process(process)
                    cls._cleanup_partial_model(model_name)
                    cls._download_jobs[model_name] = {
                        "model": model_name,
                        "status": "cancelled",
                        "progress": 100.0,
                        "message": "Preparación cancelada.",
                        "partial": False,
                        "started_at": cls._download_jobs.get(model_name, {}).get("started_at"),
                        "finished_at": time.time(),
                    }
                    return

                line = await process.stdout.readline()
                if not line:
                    break

                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue

                logger.info("STT model prepare output model=%s line=%s", model_name, text)

                try:
                    payload = json.loads(text)
                    cls._apply_model_prepare_event(model_name, payload)
                except json.JSONDecodeError:
                    current = cls._download_jobs.get(model_name, {})
                    cls._download_jobs[model_name] = {
                        **current,
                        "model": model_name,
                        "status": "running",
                        "message": text,
                    }

            return_code = await process.wait()

            logger.info(
                "STT model preparation process finished model=%s return_code=%s",
                model_name,
                return_code,
            )

            if model_name in cls._download_cancelled:
                cls._cleanup_partial_model(model_name)
                cls._download_jobs[model_name] = {
                    "model": model_name,
                    "status": "cancelled",
                    "progress": 100.0,
                    "message": "Preparación cancelada.",
                    "partial": False,
                    "started_at": cls._download_jobs.get(model_name, {}).get("started_at"),
                    "finished_at": time.time(),
                }
                return

            if return_code == 0 and cls.model_exists_locally(model_name):
                cls._download_jobs[model_name] = {
                    "model": model_name,
                    "status": "downloaded",
                    "progress": 100.0,
                    "message": "Modelo listo para usar.",
                    "partial": False,
                    "started_at": cls._download_jobs.get(model_name, {}).get("started_at"),
                    "finished_at": time.time(),
                }
                logger.info("STT model prepared successfully model=%s", model_name)
            else:
                cls._cleanup_partial_model(model_name)
                cls._download_jobs[model_name] = {
                    "model": model_name,
                    "status": "failed",
                    "progress": 100.0,
                    "message": f"No se pudo preparar el modelo. Código de salida: {return_code}",
                    "partial": False,
                    "started_at": cls._download_jobs.get(model_name, {}).get("started_at"),
                    "finished_at": time.time(),
                }
                logger.error(
                    "STT model preparation failed model=%s return_code=%s",
                    model_name,
                    return_code,
                )

        except asyncio.CancelledError:
            logger.warning("STT model preparation task cancelled model=%s", model_name)

            if process and process.returncode is None:
                await cls._terminate_async_process(process)

            cls._cleanup_partial_model(model_name)

            cls._download_jobs[model_name] = {
                "model": model_name,
                "status": "cancelled",
                "progress": 100.0,
                "message": "Preparación cancelada.",
                "partial": False,
                "started_at": cls._download_jobs.get(model_name, {}).get("started_at"),
                "finished_at": time.time(),
            }

            raise

        except Exception as exc:
            logger.exception("Unexpected error preparing STT model model=%s", model_name)

            cls._cleanup_partial_model(model_name)

            cls._download_jobs[model_name] = {
                "model": model_name,
                "status": "failed",
                "progress": 100.0,
                "message": str(exc) or exc.__class__.__name__,
                "partial": False,
                "started_at": cls._download_jobs.get(model_name, {}).get("started_at"),
                "finished_at": time.time(),
            }

        finally:
            cls._download_processes.pop(model_name, None)
            cls._download_tasks.pop(model_name, None)
            cls._download_cancelled.discard(model_name)

    @classmethod
    async def _terminate_async_process(cls, process: asyncio.subprocess.Process) -> None:
        try:
            pid = process.pid
            if pid is None:
                return

            if process.returncode is not None:
                return

            logger.warning("Terminating async process pid=%s", pid)

            try:
                os.killpg(pid, signal.SIGTERM)
            except Exception:
                process.terminate()

            try:
                await asyncio.wait_for(
                    process.wait(),
                    timeout=float(settings.STT_PROCESS_TERMINATE_TIMEOUT_SECONDS),
                )
                logger.warning("Async process terminated pid=%s returncode=%s", pid, process.returncode)
                return
            except asyncio.TimeoutError:
                logger.warning("Force killing async process pid=%s", pid)

            try:
                os.killpg(pid, signal.SIGKILL)
            except Exception:
                process.kill()

            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except asyncio.TimeoutError:
                logger.error("Async process did not exit after SIGKILL pid=%s", pid)

        except ProcessLookupError:
            logger.warning("Process already gone while terminating")
        except Exception:
            logger.exception("Could not terminate async process cleanly")

    @classmethod
    def _apply_model_prepare_event(cls, model_name: str, payload: dict[str, Any]) -> None:
        event = payload.get("event")
        current = cls._download_jobs.get(model_name, {})

        status = "running"
        if event == "done":
            status = "downloaded"
        elif event == "error":
            status = "failed"
        elif event == "cancelled":
            status = "cancelled"

        cls._download_jobs[model_name] = {
            **current,
            "model": model_name,
            "status": status,
            "progress": float(payload.get("progress", current.get("progress", 0.0))),
            "message": payload.get("message", current.get("message", "Preparando modelo.")),
            "partial": False,
        }

    @classmethod
    def resolve_extension(cls, filename: str | None, content_type: str | None = None) -> str:
        extension = pathlib.Path(filename or "").suffix.lower()

        if extension in cls.SUPPORTED_EXTENSIONS:
            return extension

        mime_to_extension = {
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "audio/mpeg": ".mp3",
            "audio/mp3": ".mp3",
            "audio/mp4": ".m4a",
            "audio/x-m4a": ".m4a",
            "audio/ogg": ".ogg",
            "audio/webm": ".webm",
            "video/webm": ".webm",
            "video/mp4": ".mp4",
            "audio/flac": ".flac",
        }

        guessed = mime_to_extension.get((content_type or "").lower())
        if guessed:
            return guessed

        supported = ", ".join(sorted(cls.SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=422,
            detail=f"Tipo de archivo no soportado. Formatos permitidos: {supported}",
        )

    @classmethod
    def validate_upload_size(cls, content: bytes) -> None:
        if not content:
            raise HTTPException(status_code=422, detail="El archivo está vacío")

        max_bytes = settings.STT_MAX_UPLOAD_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"El archivo supera el límite permitido de {settings.STT_MAX_UPLOAD_MB} MB",
            )

    @classmethod
    def ensure_ffmpeg_available(cls) -> None:
        if shutil.which("ffmpeg") is None:
            raise HTTPException(
                status_code=500,
                detail="FFmpeg no está instalado en el contenedor",
            )

    @classmethod
    def normalize_audio_to_wav(
        cls,
        input_path: pathlib.Path,
        output_path: pathlib.Path,
    ) -> pathlib.Path:
        cls.ensure_ffmpeg_available()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(output_path),
        ]

        logger.info("Normalizing audio command=%s", " ".join(command))

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip() or "No se pudo preparar el audio"
            logger.error("FFmpeg normalization failed detail=%s", detail)
            raise HTTPException(status_code=422, detail=detail) from exc

        if not output_path.exists() or output_path.stat().st_size <= 44:
            raise HTTPException(status_code=422, detail="El audio normalizado quedó vacío")

        logger.info(
            "Audio normalized input=%s output=%s output_size=%s",
            input_path,
            output_path,
            output_path.stat().st_size,
        )

        return output_path