"""Atomic JSON publication with retries for transient Windows sharing locks."""

import json
from pathlib import Path
import tempfile
import time


def atomic_write_json(path: Path, value, attempts: int = 10) -> None:
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name+".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
    try:
        for attempt in range(attempts):
            try:
                temporary.replace(path)
                return
            except OSError as error:
                sharing_error = isinstance(error, PermissionError) or getattr(error, "winerror", None) in (5, 32, 33, 1224)
                if not sharing_error or attempt == attempts-1:
                    raise
                time.sleep(min(0.05*2**attempt, 0.5))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # A scanner can briefly retain the temporary file too. Do not turn
            # successful publication or the original error into a cleanup error.
            pass
