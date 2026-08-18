from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any


_LOCKS_LOCK = threading.Lock()
_LOCKS: dict[Path, threading.Lock] = {}


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with _lock_for(jsonl_path):
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def replace_jsonl_by_key(path: str | Path, record: dict[str, Any], *, key: str = "episode_id") -> None:
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    target = record[key]
    rows: list[dict[str, Any]] = []
    replaced = False
    with _lock_for(jsonl_path):
        if jsonl_path.exists():
            with jsonl_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    existing = json.loads(line)
                    if existing.get(key) == target:
                        if not replaced:
                            rows.append(record)
                            replaced = True
                        continue
                    rows.append(existing)
        if not replaced:
            rows.append(record)
        _rewrite_rows(jsonl_path, rows)


def _rewrite_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _lock_for(path: Path) -> threading.Lock:
    key = path.resolve()
    with _LOCKS_LOCK:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock
