from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PipelineConfig(BaseModel):
    """Runtime configuration for the TAS pipeline."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(default="codex")
    model: str = Field(default="codex-local")
    sample_fps: float = Field(default=2.0, gt=0.0)
    window_size: int = Field(default=16, ge=2)
    window_stride: int = Field(default=8, ge=1)
    verification_radius: int = Field(default=3, ge=1)
    boundary_tolerance: int = Field(default=2, ge=0)
    min_segment_samples: int = Field(default=2, ge=1)
    min_boundary_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    global_edit_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    retry_limit: int = Field(default=2, ge=0)
    cache_api_calls: bool = True
    force: bool = False


def load_config(config_path: Path) -> PipelineConfig:
    """Load pipeline configuration from YAML."""

    data: dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if loaded:
            data = loaded
    return PipelineConfig.model_validate(data)


def apply_overrides(config: PipelineConfig, **overrides: Any) -> PipelineConfig:
    """Apply CLI overrides while ignoring values left unset."""

    filtered = {key: value for key, value in overrides.items() if value is not None}
    return config.model_copy(update=filtered)
