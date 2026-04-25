from __future__ import annotations

import argparse
import json
import pathlib
import sys
import traceback
from dataclasses import asdict, dataclass


@dataclass
class RunnerSegment:
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def format_vtt_time(seconds: float) -> str:
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:03}"


def build_srt(segments: list[dict]) -> str:
    blocks: list[str] = []

    for index, segment in enumerate(segments, start=1):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        start = format_srt_time(float(segment.get("start", 0.0)))
        end = format_srt_time(float(segment.get("end", 0.0)))
        blocks.append(f"{index}\n{start} --> {end}\n{text}")

    return "\n\n".join(blocks).strip() + ("\n" if blocks else "")


def build_vtt(segments: list[dict]) -> str:
    blocks = ["WEBVTT\n"]

    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        start = format_vtt_time(float(segment.get("start", 0.0)))
        end = format_vtt_time(float(segment.get("end", 0.0)))
        blocks.append(f"{start} --> {end}\n{text}")

    return "\n\n".join(blocks).strip() + "\n"


def probe_duration_seconds(path: pathlib.Path) -> float | None:
    import shutil
    import subprocess

    if shutil.which("ffprobe") is None:
        return None

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout or "{}")
        duration = payload.get("format", {}).get("duration")
        if duration is None:
            return None

        value = float(duration)
        return value if value > 0 else None
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-txt", required=True)
    parser.add_argument("--output-srt", required=True)
    parser.add_argument("--output-vtt", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--language", default="")
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--vad-filter", default="true")
    parser.add_argument("--word-timestamps", default="false")
    args = parser.parse_args()

    audio_path = pathlib.Path(args.audio_path)
    output_json_path = pathlib.Path(args.output_json)
    output_txt_path = pathlib.Path(args.output_txt)
    output_srt_path = pathlib.Path(args.output_srt)
    output_vtt_path = pathlib.Path(args.output_vtt)

    try:
        emit(
            {
                "event": "loading_model",
                "progress": 15.0,
                "message": "Cargando modelo de transcripción.",
            }
        )

        from faster_whisper import WhisperModel

        model = WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
            download_root=str(pathlib.Path(args.models_dir)),
            cpu_threads=max(1, int(args.cpu_threads)),
            num_workers=max(1, int(args.num_workers)),
        )

        duration = probe_duration_seconds(audio_path)

        emit(
            {
                "event": "transcribing",
                "progress": 22.0,
                "message": "Iniciando transcripción.",
                "duration_seconds": duration,
            }
        )

        kwargs = {
            "beam_size": max(1, int(args.beam_size)),
            "task": args.task,
            "vad_filter": str(args.vad_filter).lower() == "true",
            "word_timestamps": str(args.word_timestamps).lower() == "true",
        }

        if args.language:
            kwargs["language"] = args.language

        segments_iter, info = model.transcribe(str(audio_path), **kwargs)

        segments: list[dict] = []
        text_parts: list[str] = []

        for index, segment in enumerate(segments_iter):
            item = RunnerSegment(
                id=index,
                start=float(segment.start),
                end=float(segment.end),
                text=(segment.text or "").strip(),
                avg_logprob=getattr(segment, "avg_logprob", None),
                no_speech_prob=getattr(segment, "no_speech_prob", None),
                compression_ratio=getattr(segment, "compression_ratio", None),
            )

            payload = asdict(item)
            segments.append(payload)

            if item.text:
                text_parts.append(item.text)

            progress = 30.0
            if duration and duration > 0:
                progress = min(95.0, 30.0 + (item.end / duration) * 65.0)

            emit(
                {
                    "event": "segment",
                    "progress": progress,
                    "message": "Transcribiendo audio.",
                    "segment": payload,
                }
            )

        text = "\n".join(text_parts).strip()

        result = {
            "text": text,
            "segments": segments,
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "duration_seconds": duration,
            "model": args.model,
            "task": args.task,
        }

        output_json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_txt_path.write_text(text, encoding="utf-8")
        output_srt_path.write_text(build_srt(segments), encoding="utf-8")
        output_vtt_path.write_text(build_vtt(segments), encoding="utf-8")

        emit(
            {
                "event": "done",
                "progress": 100.0,
                "message": "Transcripción completada.",
                "result": {
                    "language": result["language"],
                    "language_probability": result["language_probability"],
                    "duration_seconds": result["duration_seconds"],
                    "text": text,
                },
            }
        )
        return 0

    except KeyboardInterrupt:
        emit(
            {
                "event": "cancelled",
                "progress": 100.0,
                "message": "Transcripción cancelada.",
            }
        )
        return 130

    except Exception as exc:
        emit(
            {
                "event": "error",
                "progress": 100.0,
                "message": str(exc) or exc.__class__.__name__,
                "traceback": traceback.format_exc(),
            }
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())