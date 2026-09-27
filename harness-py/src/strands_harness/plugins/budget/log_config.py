"""Colored, prefixed log formatting for the strands_budget package.

Every log record from this package is rendered as:

    [strands-budget-warning] 2026-09-07 18:30:12 message...

with a color per level (when the stream is a terminal):
    DEBUG=cyan, INFO=green, WARNING=yellow, ERROR=red, CRITICAL=bold red

Only the ``strands_budget`` logger is configured — the root logger and the
host application's logging setup are left untouched.
"""

import logging
import sys

_RESET = "\033[0m"
_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",  # cyan
    logging.INFO: "\033[32m",  # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}


class _ColorFormatter(logging.Formatter):
    """Formats records as '[strands-budget-<level>] <timestamp> <message>'."""

    def __init__(self, use_color: bool) -> None:
        super().__init__()
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        prefix = f"[strands-budget-{record.levelname.lower()}]"
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        line = f"{prefix} {timestamp} {message}"
        if self._use_color:
            color = _LEVEL_COLORS.get(record.levelno, "")
            return f"{color}{line}{_RESET}"
        return line


def setup_logging() -> None:
    """Attach the colored formatter to the package logger (idempotent).

    Colors are only emitted when stderr is a real terminal, so log files
    and CI output stay free of ANSI escape codes.
    """
    package_logger = logging.getLogger("strands_budget")
    if package_logger.handlers:
        return  # already configured (or the application configured it)

    handler = logging.StreamHandler()
    use_color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    handler.setFormatter(_ColorFormatter(use_color=use_color))
    package_logger.addHandler(handler)
    # Don't double-print through the root logger's handlers.
    package_logger.propagate = False
