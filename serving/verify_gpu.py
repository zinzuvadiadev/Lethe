from __future__ import annotations

import torch


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"device: {name}")
    print(f"capability: {capability}")
    arch_list = torch.cuda.get_arch_list()
    print(f"arch list: {arch_list}")
    sm = f"sm_{capability[0]}{capability[1]}"
    if sm not in arch_list:
        raise SystemExit(f"{sm} not in torch arch list {arch_list}")
    print(f"OK: {sm} supported by installed torch")


if __name__ == "__main__":
    main()
