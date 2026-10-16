"""Secure audio storage.

Files live outside the web root (``data/storage`` by default) under a folder
per Audio ID, with fixed names (never the user's filename). They are served
only through routes that check the user's permission for that record.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from flask import current_app


def root() -> Path:
    path = Path(current_app.config["STORAGE_DIR"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_dir(audio_id: str) -> Path:
    safe = "".join(ch for ch in audio_id if ch.isalnum() or ch in "-_")
    path = root() / "records" / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def tmp_dir() -> Path:
    path = root() / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel(path: Path) -> str:
    return str(Path(path).relative_to(root())).replace("\\", "/")


def resolve(relative: str | None) -> Path | None:
    if not relative:
        return None
    path = (root() / relative).resolve()
    if root().resolve() not in path.parents:  # path traversal guard
        return None
    return path if path.exists() else None


def delete_audio(record) -> None:
    """Remove audio files but keep the record (retention of audio only)."""
    for attr in ("stored_path", "processed_path"):
        p = resolve(getattr(record, attr))
        if p:
            p.unlink(missing_ok=True)
    record.audio_deleted = True


def delete_all(record) -> None:
    folder = root() / "records" / record.audio_id
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
