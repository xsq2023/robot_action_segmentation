from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


CACHE_SCHEMA_VERSION = 1


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


def cache_metadata_path(artifact_path: Path) -> Path:
    return artifact_path.with_name(f"{artifact_path.name}.cache.json")


def build_cache_metadata(fingerprint_payload: Any) -> dict[str, Any]:
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": deterministic_key(fingerprint_payload),
        "fingerprint_payload": fingerprint_payload,
    }


def cache_matches(artifact_path: Path, fingerprint_payload: Any) -> bool:
    if not artifact_path.exists():
        return False
    metadata_path = cache_metadata_path(artifact_path)
    if not metadata_path.exists():
        return False
    try:
        metadata = read_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return False
    expected = build_cache_metadata(fingerprint_payload)
    return (
        metadata.get("cache_schema_version") == CACHE_SCHEMA_VERSION
        and metadata.get("fingerprint") == expected["fingerprint"]
    )


def write_cache_metadata(artifact_path: Path, fingerprint_payload: Any) -> None:
    write_json(cache_metadata_path(artifact_path), build_cache_metadata(fingerprint_payload))


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
