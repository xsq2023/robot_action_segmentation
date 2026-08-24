from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_model(path: Path, model: BaseModel) -> None:
    write_json(path, model.model_dump(mode="json"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def deterministic_key(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return sha256_bytes(canonical.encode("utf-8"))


class JsonArtifactCache:
    """Simple file-backed JSON cache keyed by deterministic hashes."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = ensure_dir(base_dir)

    def path_for(self, namespace: str, key: str) -> Path:
        return self.base_dir / namespace / f"{key}.json"

    def load(self, namespace: str, key: str) -> Any | None:
        path = self.path_for(namespace, key)
        if not path.exists():
            return None
        return read_json(path)

    def save(self, namespace: str, key: str, data: Any) -> Path:
        path = self.path_for(namespace, key)
        write_json(path, data)
        return path

