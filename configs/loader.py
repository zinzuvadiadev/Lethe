from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    name: str
    hf_repo: str
    served_model_name: str
    quantization: str
    native_context_length: int
    max_context_length: int


@dataclasses.dataclass(frozen=True)
class HardwareConfig:
    name: str
    gpu_name: str
    vram_gb: float


def load_model_config(path: str | Path) -> ModelConfig:
    data = yaml.safe_load(Path(path).read_text())
    return ModelConfig(**data)


def load_hardware_config(path: str | Path) -> HardwareConfig:
    data = yaml.safe_load(Path(path).read_text())
    return HardwareConfig(**data)
