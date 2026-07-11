"""Structured logger for the quant pipeline."""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with a stdout handler.

    Args:
        name: Logger name (usually __name__).
        level: Logging level (default INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def add_file_handler(
    log_dir: str = "logs",
    filename: str = "quantcode.log",
    max_bytes: int = 5_000_000,
    backup_count: int = 5,
    level: int = logging.INFO,
) -> logging.Handler:
    """Attach a rotating file handler to the root logger (idempotent).

    Since every `get_logger(name)` logger propagates to the root logger by
    default, this gives every module's logs a persistent, rotated file
    without touching each module's own logger — call once at process
    startup (e.g. in run_live.py).

    Returns the existing handler instead of adding a duplicate if one for
    the same file path is already attached.
    """
    log_path = Path(log_dir) / filename
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    for existing in root_logger.handlers:
        if getattr(existing, "baseFilename", None) == str(log_path.resolve()):
            return existing

    handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    handler.setLevel(level)
    root_logger.addHandler(handler)
    if root_logger.level == logging.NOTSET or root_logger.level > level:
        root_logger.setLevel(level)
    return handler
