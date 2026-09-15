from threading import Lock
from typing import Any


_PRINT_LINE_LOCK = Lock()


def print_line(*values: Any, sep: str = " ", end: str = "\n") -> None:
    with _PRINT_LINE_LOCK:
        print(*values, sep=sep, end=end, flush=True)
