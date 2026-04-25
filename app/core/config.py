from __future__ import annotations

from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ==========================================
    # PROJECT
    # ==========================================
    PROJECT_NAME: str = "STT Local CPU"
    PROJECT_SLUG: str = "stt_local_cpu"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    CORS_ORIGINS: list[str] = ["*"]

    BASE_PATH: str = "/api/stt_local_cpu"

    # ==========================================
    # STT DEFAULTS
    # ==========================================
    STT_DEFAULT_MODEL: str = "tiny"
    STT_DEVICE: str = "cpu"
    STT_DEFAULT_COMPUTE_TYPE: str = "int8"
    STT_DEFAULT_CPU_THREADS: int = 4
    STT_DEFAULT_NUM_WORKERS: int = 1
    STT_WORKERS: int = 1
    STT_MAX_UPLOAD_MB: int = 500

    # Opciones internas. No se muestran en la interfaz.
    STT_DEFAULT_VAD_FILTER: bool = True
    STT_DEFAULT_WORD_TIMESTAMPS: bool = False

    # Timeout suave para procesos.
    STT_PROCESS_TERMINATE_TIMEOUT_SECONDS: int = 8

    # ==========================================
    # PATHS
    # ==========================================
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    MODELS_DIR: Path = BASE_DIR / "models"
    PREPARED_MODELS_DIR: Path = MODELS_DIR / ".prepared"
    RUNTIME_DIR: Path = BASE_DIR / "runtime"
    STT_JOBS_DIR: Path = RUNTIME_DIR / "stt_jobs"

    # ==========================================
    # LOG
    # ==========================================
    LOG_LEVEL: str = "DEBUG"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_DIR: str = "logs"
    LOG_FILENAME: str = "file.log"

    # ==========================================
    # PLUGIN META
    # ==========================================
    DISPLAY_NAME: str = "STT Local CPU"
    DESCRIPTION: str = (
        "Módulo STT oficial para QueAI. Transcripción local de audio a texto "
        "basada en faster-whisper, optimizada para CPU, con modelos preparados "
        "bajo demanda, cola de tareas, progreso en vivo y cancelación segura."
    )
    AUTHOR: str = "Juana Iris Perez && Alejandro Fonseca"
    LICENSE: str = "MIT"
    LOGO: str = "stt_logo.png"

    @property
    def is_dev(self) -> bool:
        return self.ENVIRONMENT == "development"

    @computed_field
    @property
    def OPENAPI_PATH(self) -> str:
        return f"{self.BASE_PATH}/openapi.json"

    @computed_field
    @property
    def DOCS_PATH(self) -> str:
        return f"{self.BASE_PATH}/docs"

    @computed_field
    @property
    def REDOC_PATH(self) -> str:
        return f"{self.BASE_PATH}/redoc"

    @computed_field
    @property
    def UI_PATH(self) -> str:
        return f"{self.BASE_PATH}/ui"

    @computed_field
    @property
    def HEALTH_PATH(self) -> str:
        return f"{self.BASE_PATH}/health"

    @computed_field
    @property
    def CONFIG_PATH(self) -> str:
        return f"{self.BASE_PATH}/config"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        arbitrary_types_allowed=True,
    )


settings = Settings()