"""Executed in a model-specific environment by the lightweight frontend."""
import json
import os
from pathlib import Path
import sys


def main():
    request = json.loads(sys.argv[1])
    os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")
    cache = Path(request["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    import torch
    # CUDA numbers the GPUs exposed by CUDA_VISIBLE_DEVICES starting at zero.
    request["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    if request["command"] == "refold":
        from .refold import run
    elif request["model"] == "rfdiffusion":
        from .rf import run
    else:
        from .ipa import run
    run(request)


if __name__ == "__main__":
    main()

