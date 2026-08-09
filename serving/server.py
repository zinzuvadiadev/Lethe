from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from configs.loader import HardwareConfig, ModelConfig, load_hardware_config, load_model_config


def find_pip_cuda_home(site_packages: Path) -> Path | None:
    """Locate a pip-installed CUDA toolkit (nvidia-cuda-nvcc) under site-packages.

    flashinfer JIT-compiles sampling kernels at runtime and looks for nvcc via
    CUDA_HOME (default /usr/local/cuda), which doesn't exist when CUDA was
    installed as a pip package (nvidia-cuda-nvcc) into the venv instead of
    system-wide. Returns the directory containing bin/nvcc, or None if not found.
    """
    for nvcc_path in sorted(site_packages.glob("nvidia/cu*/bin/nvcc")):
        return nvcc_path.parent.parent
    return None


def build_serve_args(model_cfg: ModelConfig, hw_cfg: HardwareConfig, port: int = 8000) -> list[str]:
    # --quantization is deliberately NOT passed here: vLLM auto-detects it
    # from the checkpoint's own config.json, and forcing a value that doesn't
    # exactly match the checkpoint's internal quant_method label (e.g. this
    # AWQ checkpoint declares itself as "compressed-tensors") makes vLLM
    # refuse to start. model_cfg.quantization stays as documentation/results
    # metadata rather than a CLI constraint.
    #
    # 0.75 (not 0.85) for small-VRAM cards: measured on the RTX 5070 laptop
    # (8GB, shared with a live desktop session), 0.85 leaves too thin a
    # margin between vLLM's KV-cache budget estimate and actual allocation
    # (desktop overhead + fragmentation) — observed a real CUDA OOM with
    # only 91MB free during activation-memory allocation at 0.85.
    gpu_util = 0.75 if hw_cfg.vram_gb <= 12 else 0.9
    return [
        "vllm",
        "serve",
        model_cfg.hf_repo,
        "--served-model-name", model_cfg.served_model_name,
        "--max-model-len", str(model_cfg.max_context_length),
        "--gpu-memory-utilization", str(gpu_util),
        "--port", str(port),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a config-driven vLLM server")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model_cfg = load_model_config(args.model_config)
    hw_cfg = load_hardware_config(args.hardware_config)
    argv = build_serve_args(model_cfg, hw_cfg, port=args.port)
    venv_bin = Path(sys.executable).parent
    # Resolve the console-script path explicitly: subprocess.run inherits our
    # PATH, which won't include .venv/bin unless the venv was activated (we
    # invoke .venv/bin/python directly instead), so a bare "vllm" lookup fails.
    argv[0] = str(venv_bin / "vllm")

    env = os.environ.copy()
    # .venv/bin also holds pip-installed build tools (ninja) that vLLM's
    # runtime JIT kernel compilation (flashinfer) shells out to by bare name;
    # without it on PATH those subprocess lookups fail with FileNotFoundError.
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    site_packages = venv_bin.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    cuda_home = find_pip_cuda_home(site_packages)
    if cuda_home is not None and "CUDA_HOME" not in env:
        env["CUDA_HOME"] = str(cuda_home)
        env["PATH"] = f"{cuda_home / 'bin'}{os.pathsep}{env['PATH']}"
        print(f"CUDA_HOME not set; using pip-installed CUDA toolkit at {cuda_home}")

    # FlashInfer's JIT-compiled top-k/top-p sampling kernel fails to build on
    # this GPU/CUDA-toolkit combination: its bundled CCCL headers reject the
    # pip-installed CUDA 13.3 nvcc + sm_120f (Blackwell) target with "CUDA
    # compiler and CUDA toolkit headers are incompatible". This falls back to
    # vLLM's PyTorch-native sampler, which doesn't hit that JIT path.
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    print(f"launching: {' '.join(argv)}")
    subprocess.run(argv, check=True, env=env)


if __name__ == "__main__":
    main()
