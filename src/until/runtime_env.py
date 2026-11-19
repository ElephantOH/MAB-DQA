"""Runtime setup shared by the macOS and Linux entry points.

The retrieval and scoring code is the same on both platforms. This module only
prepares the process before torch is imported:

- Linux keeps every visible NVIDIA GPU and the paper's CUDA allocator setting.
  The retriever then uses cuda:0 and the VLM uses device_map "auto".
- macOS runs the same models on Apple's GPU. Ops that GPU does not implement
  fall back to CPU so the pipeline can finish.
"""
import os
import sys


def prepare_runtime_env() -> None:
    if sys.platform == "darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
