"""
Shared logging helpers for FoeGlass scripts.
"""
from __future__ import annotations

import contextlib
import logging
from io import StringIO
import sys
from pathlib import Path


_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str, log_path: Path, level: int = logging.INFO) -> logging.Logger:
    """Create a script logger that writes both to stdout and a file."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    return logger


@contextlib.contextmanager
def suppress_external_output(logger: logging.Logger, prefix: str) -> None:
    """Capture noisy stdout/stderr from third-party libs and redirect to debug logs."""

    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        yield

    for raw_line in stdout_buffer.getvalue().splitlines():
        line = raw_line.strip()
        if line:
            logger.debug("%s %s", prefix, line)
    for raw_line in stderr_buffer.getvalue().splitlines():
        line = raw_line.strip()
        if line:
            logger.debug("%s %s", prefix, line)
