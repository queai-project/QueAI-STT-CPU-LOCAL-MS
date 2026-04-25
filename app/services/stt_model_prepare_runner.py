from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def safe_marker_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(".", "_")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    models_dir = pathlib.Path(args.models_dir)
    prepared_dir = models_dir / ".prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    marker_path = prepared_dir / f"{safe_marker_name(args.model)}.json"
    temporary_marker_path = prepared_dir / f"{safe_marker_name(args.model)}.preparing.json"

    if marker_path.exists():
        emit(
            {
                "event": "done",
                "progress": 100.0,
                "message": "Modelo ya estaba preparado.",
                "model": args.model,
            }
        )
        return 0

    if temporary_marker_path.exists():
        temporary_marker_path.unlink(missing_ok=True)

    try:
        emit(
            {
                "event": "start",
                "progress": 5.0,
                "message": "Iniciando preparación del modelo.",
                "model": args.model,
            }
        )

        temporary_marker_path.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "status": "preparing",
                    "started_at": time.time(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        emit(
            {
                "event": "loading",
                "progress": 20.0,
                "message": "Descargando y verificando modelo. Puede tardar la primera vez.",
                "model": args.model,
            }
        )

        from faster_whisper import WhisperModel

        WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
            download_root=str(models_dir),
            cpu_threads=max(1, int(args.cpu_threads)),
            num_workers=max(1, int(args.num_workers)),
        )

        marker_path.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "status": "downloaded",
                    "prepared_at": time.time(),
                    "device": args.device,
                    "compute_type": args.compute_type,
                    "cpu_threads": args.cpu_threads,
                    "num_workers": args.num_workers,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary_marker_path.unlink(missing_ok=True)

        emit(
            {
                "event": "done",
                "progress": 100.0,
                "message": "Modelo preparado correctamente.",
                "model": args.model,
            }
        )
        return 0

    except KeyboardInterrupt:
        temporary_marker_path.unlink(missing_ok=True)
        emit(
            {
                "event": "cancelled",
                "progress": 100.0,
                "message": "Preparación cancelada.",
                "model": args.model,
            }
        )
        return 130

    except Exception as exc:
        temporary_marker_path.unlink(missing_ok=True)
        emit(
            {
                "event": "error",
                "progress": 100.0,
                "message": str(exc) or exc.__class__.__name__,
                "traceback": traceback.format_exc(),
                "model": args.model,
            }
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())