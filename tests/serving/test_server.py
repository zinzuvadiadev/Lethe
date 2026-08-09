from pathlib import Path

from configs.loader import HardwareConfig, ModelConfig
from serving.server import build_serve_args, build_serve_env, find_pip_cuda_home

MODEL = ModelConfig(
    name="qwen3-4b-instruct-2507-awq",
    hf_repo="cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit",
    served_model_name="qwen3-4b-instruct-2507",
    quantization="awq",
    native_context_length=262144,
    max_context_length=6000,
)
SMALL_HW = HardwareConfig(name="rtx5070-laptop", gpu_name="RTX 5070 Laptop", vram_gb=8)
BIG_HW = HardwareConfig(name="rtx4090", gpu_name="RTX 4090", vram_gb=24)


def test_build_serve_args_includes_model_and_max_len():
    args = build_serve_args(MODEL, SMALL_HW, port=8000)
    assert args[:3] == ["vllm", "serve", "cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit"]
    assert args[args.index("--max-model-len") + 1] == "6000"
    assert args[args.index("--port") + 1] == "8000"
    assert args[args.index("--served-model-name") + 1] == "qwen3-4b-instruct-2507"


def test_build_serve_args_does_not_force_quantization_flag():
    # vLLM auto-detects quantization from the checkpoint's own config.json;
    # forcing a mismatched value makes it refuse to start (see commit message).
    args = build_serve_args(MODEL, SMALL_HW)
    assert "--quantization" not in args


def test_build_serve_args_lowers_gpu_util_for_small_vram():
    args = build_serve_args(MODEL, SMALL_HW)
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.75"


def test_build_serve_args_raises_gpu_util_for_large_vram():
    args = build_serve_args(MODEL, BIG_HW)
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.9"


def test_find_pip_cuda_home_locates_nvcc_under_site_packages(tmp_path: Path):
    nvcc = tmp_path / "nvidia" / "cu13" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    assert find_pip_cuda_home(tmp_path) == tmp_path / "nvidia" / "cu13"


def test_find_pip_cuda_home_returns_none_when_absent(tmp_path: Path):
    assert find_pip_cuda_home(tmp_path) is None


def test_build_serve_args_omits_eviction_flags_when_disabled():
    args = build_serve_args(MODEL, SMALL_HW)
    assert "--model-class-overrides" not in args
    assert "--scheduler-cls" not in args


def test_build_serve_args_includes_eviction_flags_when_enabled():
    args = build_serve_args(MODEL, SMALL_HW, sink_len=64, recent_window=256)
    assert "--model-class-overrides" in args
    override_value = args[args.index("--model-class-overrides") + 1]
    assert "serving.eviction.sink_model:SinkQwen3ForCausalLM" in override_value
    assert "Qwen3ForCausalLM" in override_value
    assert args[args.index("--scheduler-cls") + 1] == "serving.eviction.sink_scheduler.SinkScheduler"


def test_build_serve_env_sets_pythonpath_to_repo_root():
    env = build_serve_env(sink_len=None, recent_window=None, base_env={})
    assert str(Path(__file__).resolve().parents[2]) in env["PYTHONPATH"]


def test_build_serve_env_sets_flashinfer_sampler_disabled_by_default():
    env = build_serve_env(sink_len=None, recent_window=None, base_env={})
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


def test_build_serve_env_respects_existing_flashinfer_sampler_setting():
    env = build_serve_env(
        sink_len=None, recent_window=None,
        base_env={"VLLM_USE_FLASHINFER_SAMPLER": "1"},
    )
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "1"


def test_build_serve_env_omits_eviction_env_vars_when_disabled():
    env = build_serve_env(sink_len=None, recent_window=None, base_env={})
    assert "LETHE_SINK_LEN" not in env
    assert "LETHE_RSWA_WINDOW" not in env


def test_build_serve_env_sets_eviction_env_vars_when_enabled():
    env = build_serve_env(sink_len=64, recent_window=512, base_env={})
    assert env["LETHE_SINK_LEN"] == "64"
    assert env["LETHE_RSWA_WINDOW"] == "512"
