import config as cfg
import enum
from datetime import datetime


class LogLevel(enum.Enum):
    ALL = 6
    TRACE = 5
    DEBUG = 4
    INFO = 3
    WARN = 2
    ERROR = 1
    FATAL = 0
    OFF = -1

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


def timestamp() -> str:
    return datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")


def _log(level: LogLevel, *args, **kwargs):
    if cfg.LOG_LEVEL >= level:
        ts = timestamp()
        print(f"[{ts}] [{level.name}] ", end="")
        print(*args, **kwargs)


def _log_raw(level: LogLevel, *args, **kwargs):
    if cfg.LOG_LEVEL >= level:
        ts = timestamp()
        print(f"[{ts}] ", end="")
        print(*args, **kwargs)


def log_trace(*args, **kwargs): _log(LogLevel.TRACE, *args, **kwargs)
def log_debug(*args, **kwargs): _log(LogLevel.DEBUG, *args, **kwargs)
def log_info(*args, **kwargs): _log(LogLevel.INFO, *args, **kwargs)
def log_warn(*args, **kwargs): _log(LogLevel.WARN, *args, **kwargs)
def log_error(*args, **kwargs): _log(LogLevel.ERROR, *args, **kwargs)
def log_fatal(*args, **kwargs): _log(LogLevel.FATAL, *args, **kwargs)


def log_trace_raw(*args, **kwargs): _log_raw(LogLevel.TRACE, *args, **kwargs)
def log_debug_raw(*args, **kwargs): _log_raw(LogLevel.DEBUG, *args, **kwargs)
def log_info_raw(*args, **kwargs): _log_raw(LogLevel.INFO, *args, **kwargs)
def log_warn_raw(*args, **kwargs): _log_raw(LogLevel.WARN, *args, **kwargs)
def log_error_raw(*args, **kwargs): _log_raw(LogLevel.ERROR, *args, **kwargs)
def log_fatal_raw(*args, **kwargs): _log_raw(LogLevel.FATAL, *args, **kwargs)
