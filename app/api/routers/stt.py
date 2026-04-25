from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse

from app.core.config import settings
from app.schemas.stt import (
    JobStatusResponse,
    JobSubmissionResponse,
    PrepareModelRequest,
    TranscriptionOptions,
)
from app.services.stt_service import STTService


router = APIRouter()


@router.get("/models")
async def get_models():
    return STTService.get_models()


@router.post("/models/prepare")
async def prepare_model(payload: PrepareModelRequest):
    return await STTService.prepare_model(model_name=payload.model)


@router.post("/models/downloads/{model_name:path}/cancel")
async def cancel_model_download_post(model_name: str):
    return await STTService.cancel_model_download(model_name)


@router.delete("/models/downloads/{model_name:path}")
async def cancel_model_download_delete(model_name: str):
    return await STTService.cancel_model_download(model_name)


@router.get("/models/downloads/{model_name:path}")
async def get_model_download_status(model_name: str):
    return await STTService.get_download_status(model_name)

@router.post(
    "/transcribe",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobSubmissionResponse,
)
async def enqueue_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
    language: str | None = Form(default=None),
    task: str = Form(default="transcribe"),
    beam_size: int = Form(default=3),
):
    try:
        content = await file.read()
    finally:
        await file.close()

    options = TranscriptionOptions(
        model=model,
        language=language or None,
        task=task,
        beam_size=beam_size,
    ).model_dump()

    job = request.app.state.stt_jobs.submit_file(
        content=content,
        filename=file.filename,
        content_type=file.content_type,
        options=options,
    )

    return {"job_id": job.job_id, "status": job.status}


@router.post(
    "/transcribe/options-json",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobSubmissionResponse,
)
async def enqueue_transcription_with_options_json(
    request: Request,
    file: UploadFile = File(...),
    options_json: str = Form(default="{}"),
):
    try:
        options_data = json.loads(options_json or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="Invalid options JSON") from exc

    options = TranscriptionOptions(**options_data).model_dump()

    try:
        content = await file.read()
    finally:
        await file.close()

    job = request.app.state.stt_jobs.submit_file(
        content=content,
        filename=file.filename,
        content_type=file.content_type,
        options=options,
    )

    return {"job_id": job.job_id, "status": job.status}


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_stt_job(job_id: str, request: Request):
    return request.app.state.stt_jobs.get_status(job_id)


@router.post("/jobs/{job_id}/cancel")
async def cancel_stt_job(job_id: str, request: Request):
    return request.app.state.stt_jobs.cancel_job(job_id)


@router.get("/jobs/{job_id}/events")
async def stream_stt_job_events(job_id: str, request: Request):
    request.app.state.stt_jobs.get_job(job_id)

    async def event_generator():
        last_index = 0

        while True:
            if await request.is_disconnected():
                break

            events = request.app.state.stt_jobs.get_events_since(job_id, last_index)

            for event in events:
                last_index = int(event["index"]) + 1
                yield f"event: {event.get('event', 'message')}\n"
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            status_data = request.app.state.stt_jobs.get_status(job_id)
            if status_data["status"] in {"done", "failed", "cancelled"} and not events:
                break

            await asyncio.sleep(0.7)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/jobs/{job_id}/result.json")
async def get_stt_result_json(job_id: str, request: Request):
    path = request.app.state.stt_jobs.get_result_path(job_id, "json")
    return FileResponse(path=path, media_type="application/json", filename=f"{job_id}.json")


@router.get("/jobs/{job_id}/result.txt")
async def get_stt_result_txt(job_id: str, request: Request):
    path = request.app.state.stt_jobs.get_result_path(job_id, "txt")
    return FileResponse(path=path, media_type="text/plain; charset=utf-8", filename=f"{job_id}.txt")


@router.get("/jobs/{job_id}/result.srt")
async def get_stt_result_srt(job_id: str, request: Request):
    path = request.app.state.stt_jobs.get_result_path(job_id, "srt")
    return FileResponse(path=path, media_type="application/x-subrip", filename=f"{job_id}.srt")


@router.get("/jobs/{job_id}/result.vtt")
async def get_stt_result_vtt(job_id: str, request: Request):
    path = request.app.state.stt_jobs.get_result_path(job_id, "vtt")
    return FileResponse(path=path, media_type="text/vtt", filename=f"{job_id}.vtt")


@router.delete("/jobs")
async def cleanup_finished_jobs(request: Request):
    return request.app.state.stt_jobs.cleanup_finished_jobs()


@router.get("/runtime")
async def get_runtime_info():
    return {
        "default_model": settings.STT_DEFAULT_MODEL,
        "device": settings.STT_DEVICE,
        "compute_type": settings.STT_DEFAULT_COMPUTE_TYPE,
        "workers": settings.STT_WORKERS,
        "cpu_threads": settings.STT_DEFAULT_CPU_THREADS,
        "num_workers": settings.STT_DEFAULT_NUM_WORKERS,
        "max_upload_mb": settings.STT_MAX_UPLOAD_MB,
        "vad_filter": settings.STT_DEFAULT_VAD_FILTER,
        "word_timestamps": settings.STT_DEFAULT_WORD_TIMESTAMPS,
        "models_dir": str(settings.MODELS_DIR),
    }