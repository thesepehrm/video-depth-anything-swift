#!/usr/bin/env python3
"""Convert the official Video Depth Anything Small checkpoint to the layout VideoDepthAnything loads.

Reads the PyTorch checkpoint published by the Video Depth Anything authors (Apache-2.0), transposes
the convolution kernels from PyTorch's (out, in, kh, kw) to MLX's channel-last (out, kh, kw, in), and
writes `model.safetensors` (float32) and `config.json` to the output directory. Every other tensor is
copied unchanged, with the checkpoint's own key names.

Only the two Apache-2.0 checkpoints are accepted:

    small         depth-anything/Video-Depth-Anything-Small         (relative depth, the default)
    metric-small  depth-anything/Metric-Video-Depth-Anything-Small  (metric depth)

The Base and Large checkpoints are licensed CC-BY-NC-4.0 and are not supported.

Usage:

    python scripts/convert.py --out weights/video-depth-anything-small
    python scripts/convert.py --variant metric-small --out weights/metric-video-depth-anything-small
    python scripts/convert.py --checkpoint path/to/video_depth_anything_vits.pth --out weights/...

Without `--checkpoint` the file is downloaded from the Hugging Face Hub at a pinned revision. Either
way its SHA-256 must match the pinned digest, so the output is reproducible byte for byte.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

VARIANTS = {
    "small": {
        "repo": "depth-anything/Video-Depth-Anything-Small",
        "revision": "256875362cff76724b920335dfb4b29dd611f66e",
        "filename": "video_depth_anything_vits.pth",
        "sha256": "13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609",
        "metric": False,
    },
    "metric-small": {
        "repo": "depth-anything/Metric-Video-Depth-Anything-Small",
        "revision": "273d090f2ce17df50c2872d82c8322c45da5b4dd",
        "filename": "metric_video_depth_anything_vits.pth",
        "sha256": "3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33",
        "metric": True,
    },
}

# ConvTranspose2d kernels are stored (in, out, kh, kw) by PyTorch; every other 4-D tensor is a
# Conv2d kernel (out, in, kh, kw).
CONV_TRANSPOSE_KEYS = ("head.resize_layers.0.weight", "head.resize_layers.1.weight")

EXPECTED_TENSORS = 351


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(state_dict: dict) -> dict:
    out = {}
    for key, tensor in state_dict.items():
        value = tensor.detach().to(torch.float32).contiguous().numpy()
        if value.ndim == 4:
            if key in CONV_TRANSPOSE_KEYS:
                value = value.transpose(1, 2, 3, 0)  # (in, out, kh, kw) -> (out, kh, kw, in)
            else:
                value = value.transpose(0, 2, 3, 1)  # (out, in, kh, kw) -> (out, kh, kw, in)
        out[key] = np.ascontiguousarray(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="small")
    parser.add_argument("--checkpoint", type=Path, help="local .pth file (skips the download)")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    if args.checkpoint is None:
        from huggingface_hub import hf_hub_download

        args.checkpoint = Path(
            hf_hub_download(variant["repo"], variant["filename"], revision=variant["revision"])
        )

    digest = sha256(args.checkpoint)
    if digest != variant["sha256"]:
        print(
            f"error: {args.checkpoint} has SHA-256 {digest}, expected {variant['sha256']} "
            f"({variant['repo']}@{variant['revision']})",
            file=sys.stderr,
        )
        return 1

    # weights_only=True loads tensors without running pickled code.
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if len(state_dict) != EXPECTED_TENSORS:
        print(f"error: expected {EXPECTED_TENSORS} tensors, found {len(state_dict)}", file=sys.stderr)
        return 1

    weights = convert(state_dict)
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(args.out / "model.safetensors"), metadata={"format": "mlx"})

    config = {
        "model_type": "video_depth_anything",
        "encoder": "vits",
        "metric": variant["metric"],
        "source": {
            "repo": variant["repo"],
            "revision": variant["revision"],
            "filename": variant["filename"],
            "sha256": variant["sha256"],
            "license": "apache-2.0",
        },
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    print(f"wrote {len(weights)} tensors to {args.out / 'model.safetensors'}")
    print(f"model.safetensors sha256 {sha256(args.out / 'model.safetensors')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
