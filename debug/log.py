from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Color(str, Enum):
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    RESET = "\033[0m"


class Level(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass
class LogRecord:
    level: Level
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    name: str = ""
    extra: dict[str, Any] | None = None


_collector_stack: ContextVar[list[list[LogRecord]] | None] = ContextVar(
    "log_collector_stack", default=None
)


class Logger:
    LEVEL_COLORS: dict[Level, Color] = {
        Level.DEBUG: Color.GRAY,
        Level.INFO: Color.GREEN,
        Level.WARN: Color.YELLOW,
        Level.ERROR: Color.RED,
    }

    def __init__(self, name: str = ""):
        self._name = name

    def _log(self, level: Level, message: str, color: Color | None = None, **extra):
        record = LogRecord(
            level=level,
            message=message,
            name=self._name,
            extra=extra if extra else None,
        )

        # Collect to all collectors in current context
        stack = _collector_stack.get()
        if stack:
            for collector in stack:
                collector.append(record)

        self._print(record, color)

    def _print(self, record: LogRecord, color: Color | None = None):
        c = color or self.LEVEL_COLORS.get(record.level, Color.WHITE)
        ts = record.timestamp.strftime("%H:%M:%S")
        prefix = f"[{record.name}]" if record.name else ""
        print(f"{Color.GRAY.value}{ts}{Color.RESET.value} {c.value}{record.level.value:5}{Color.RESET.value} {prefix} {record.message}")

    def debug(self, message: str, **extra):
        self._log(Level.DEBUG, message, **extra)

    def info(self, message: str, color: Color | None = None, **extra):
        self._log(Level.INFO, message, color=color, **extra)

    def warn(self, message: str, **extra):
        self._log(Level.WARN, message, **extra)

    def error(self, message: str, **extra):
        self._log(Level.ERROR, message, **extra)


class LogCollector:

    def __init__(self):
        self._records: list[LogRecord] = []
        self._token = None

    def __enter__(self) -> "LogCollector":
        stack = _collector_stack.get()
        if stack is None:
            stack = []
            self._token = _collector_stack.set(stack)
        else:
            self._token = None
        stack.append(self._records)
        return self

    def __exit__(self, *args):
        stack = _collector_stack.get()
        if stack:
            stack.pop()
        if self._token is not None:
            _collector_stack.reset(self._token)

    @property
    def records(self) -> list[LogRecord]:
        return self._records

    def messages(self) -> list[str]:
        return [r.message for r in self._records]

    def to_text(self) -> str:
        return "\n".join(self.messages())


# Global default logger
log = Logger()