import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from app.core.config import settings


class CustomLogger:
    """
    Logging configuration for QueAI STT.
    Keeps application logs useful and suppresses noisy third-party DEBUG logs.
    """

    def __init__(self):
        self.log_dir = settings.LOG_DIR
        self.log_file = os.path.join(self.log_dir, settings.LOG_FILENAME)
        self._create_log_dir()
        self.logger = self._setup_logger()

    def _create_log_dir(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

    def _setup_logger(self):
        formatter = logging.Formatter(
            fmt=settings.LOG_FORMAT,
            datefmt=settings.LOG_DATETIME_FORMAT,
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        file_handler = RotatingFileHandler(
            filename=self.log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.setLevel(settings.LOG_LEVEL)

        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

        # Silenciar librerías demasiado ruidosas.
        noisy_loggers = [
            "python_multipart",
            "python_multipart.multipart",
            "multipart",
            "urllib3",
            "asyncio",
            "watchfiles",
            "uvicorn.access",
            "huggingface_hub",
            "faster_whisper",
            "ctranslate2",
        ]

        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        return root_logger


logger = CustomLogger().logger