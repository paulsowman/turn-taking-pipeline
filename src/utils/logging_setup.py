"""Logging configuration for the pipeline."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    level: str = "INFO",
    console: bool = True,
) -> logging.Logger:
    """
    Set up logging for the pipeline.

    Parameters
    ----------
    log_file : str, optional
        Path to log file. If None, only log to console.
    level : str
        Logging level ("DEBUG", "INFO", "WARNING", "ERROR").
    console : bool
        Whether to also log to console.

    Returns
    -------
    logger : logging.Logger
        Configured logger.
    """
    # Get root logger
    logger = logging.getLogger("turn_taking_pipeline")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "turn_taking_pipeline") -> logging.Logger:
    """Get existing logger or create new one."""
    return logging.getLogger(name)
