"""Low-level durable file and locking primitives used by PyNST storage."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

from ..data.model import json_default


class _RunLock:
    """Cross-platform non-blocking OS lock backed by a persistent file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any | None = None

    @property
    def acquired(self) -> bool:
        return self._handle is not None

    def acquire(self) -> None:
        if self.acquired:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(
                    handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
        except (OSError, IOError) as error:
            handle.close()
            raise RuntimeError(
                "PyNST run directory is locked by another active manager: "
                f"{self.path}"
            ) from error
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _fsync_file(path: Path) -> None:
    # Windows' CRT rejects ``fsync`` on a read-only descriptor with EBADF.
    # All callers own the file being committed, so opening it read/write is
    # both safe and portable while still leaving its contents untouched.
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_file(temporary: Path, target: Path) -> None:
    """Atomically replace a file, tolerating brief Windows scanner locks."""
    delays = (0.02, 0.05, 0.10, 0.25)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(temporary, target)
            return
        except PermissionError as error:
            transient_windows_lock = (
                os.name == "nt"
                and getattr(error, "winerror", None) in {5, 32}
            )
            if not transient_windows_lock or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(text.encode("utf-8"))
    _fsync_file(temporary)
    _replace_file(temporary, path)


def _json_text(payload: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            default=json_default,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, _json_text(payload))


def _lock_identity(path: Path) -> str:
    text = str(path.expanduser().resolve())
    return os.path.normcase(text) if os.name == "nt" else text
