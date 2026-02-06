from __future__ import annotations

from typing import Optional, Any, Dict
import logging
import os

from utils.config import ensure_dir, load_config


class Logger:
    def __init__(
        self,
        name: str = "ids",
        level: str | int = "INFO",
        log_file: Optional[str] = None,
        console: bool = True,
    ) -> None:
        self.name = name
        self.level = level
        self.log_file = log_file
        self.console = console
        self._logger = self._build_logger()

    @staticmethod
    def _normalize_level(level: str | int) -> int:
        if isinstance(level, int):
            return level
        return int(getattr(logging, str(level).upper(), logging.INFO))

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(self._normalize_level(self.level))
        logger.propagate = False
        if logger.handlers:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)

        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

        if self.console:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        if self.log_file:
            ensure_dir(os.path.dirname(self.log_file))
            file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def get_logger(self) -> logging.Logger:
        return self._logger


_LOGGER_CACHE: Dict[tuple[str, str | int, Optional[str], bool], logging.Logger] = {}


def get_logger(
    name: str = "ids",
    level: str | int | None = None,
    log_file: Optional[str] = None,
    console: bool = True,
    config: Optional[Dict[str, Any]] = None,
    config_path: str = "config.yaml",
) -> logging.Logger:
    if level is None or log_file is None:
        cfg = config or load_config(config_path)
        log_cfg = cfg.get("logging") or {}
        if level is None:
            level = log_cfg.get("level", "INFO")
        if log_file is None:
            log_file = log_cfg.get("log_file")
    cache_key = (name, level, log_file, console)
    if cache_key not in _LOGGER_CACHE:
        _LOGGER_CACHE[cache_key] = Logger(
            name=name,
            level=level,
            log_file=log_file,
            console=console,
        ).get_logger()
    return _LOGGER_CACHE[cache_key]
