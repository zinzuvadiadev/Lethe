from pathlib import Path

import pytest

from configs.loader import (
    HardwareConfig,
    ModelConfig,
    load_hardware_config,
    load_model_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_model_config_reads_qwen3_4b_awq():
    cfg = load_model_config(REPO_ROOT / "configs/models/qwen3-4b-instruct-2507-awq.yaml")
    assert cfg == ModelConfig(
        name="qwen3-4b-instruct-2507-awq",
        hf_repo="cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit",
        served_model_name="qwen3-4b-instruct-2507",
        quantization="awq",
        native_context_length=262144,
        max_context_length=6000,
    )


def test_load_hardware_config_reads_rtx5070():
    cfg = load_hardware_config(REPO_ROOT / "configs/hardware/rtx5070-laptop.yaml")
    assert cfg == HardwareConfig(
        name="rtx5070-laptop",
        gpu_name="NVIDIA GeForce RTX 5070 Laptop GPU",
        vram_gb=8,
    )


def test_load_model_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_model_config(REPO_ROOT / "configs/models/does-not-exist.yaml")
