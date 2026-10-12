#!/usr/bin/env python3
"""Generate the golden outputs the Swift parity tests compare against, using mlx-vlm as the reference.

Loads the weights written by scripts/convert.py into mlx-vlm's Video Depth Anything port
(`mlx_vlm.models.video_depth_anything`, MIT), decodes the clip with OpenCV, runs the reference
sliding-window inference on the GPU, and writes to `--out`:

    frames.npy        (T, H, W, 3) uint8   decoded RGB frames, fed to the Swift pipeline unchanged
    depth.npy         (T, H, W) float32    aligned per-frame depth at source resolution
    window0.npy       (32, h, w) float32   raw model output of the first window at inference resolution
    pixels0.npy       (h, w, 3) float32    preprocessed frame 0
    features.safetensors  layer-level tensors for a 4-frame input (backbone layers, head, output)
    manifest.json     versions, sizes and settings

Usage:

    python scripts/reference.py --weights weights/video-depth-anything-small \\
        --video Tests/VideoDepthAnythingTests/Resources/static-scene.mp4 --out parity/fixtures

The weights are never downloaded here. Pass `--checkpoint` with the original .pth to also assert that
the converted tensors equal mlx-vlm's own `Model.sanitize` of the PyTorch checkpoint.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np


def load_model(weights: Path):
    from mlx_vlm.models.video_depth_anything import Model, ModelConfig

    config = json.loads((weights / "config.json").read_text())
    model = Model(ModelConfig(encoder=config["encoder"], metric=config.get("metric", False)))
    tensors = mx.load(str(weights / "model.safetensors"))
    model.load_weights(list(tensors.items()), strict=True)
    mx.eval(model.parameters())
    return model, tensors


def check_against_checkpoint(tensors, checkpoint: Path):
    import torch
    from mlx_vlm.models.video_depth_anything import Model

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    raw = {k: mx.array(v.to(torch.float32).numpy()) for k, v in state.items()}
    sanitized = Model.sanitize(raw)
    assert set(sanitized) == set(tensors), "key sets differ"
    for k, v in sanitized.items():
        assert v.shape == tensors[k].shape, f"{k}: {v.shape} vs {tensors[k].shape}"
        assert mx.array_equal(v, tensors[k]).item(), f"{k}: values differ"
    print(f"converted weights equal mlx-vlm sanitize of {checkpoint.name} ({len(tensors)} tensors)")


def read_frames(path: Path, max_frames: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames), fps


def layer_features(model, pixels: mx.array):
    """Backbone layers, head output and final depth for one (1, T, h, w, 3) input."""
    cfg = model.config
    B, T, H, W, _ = pixels.shape
    ph, pw = H // cfg.patch_size, W // cfg.patch_size
    x = pixels.reshape(B * T, H, W, 3)
    out = {"input": x, "tokens": model.pretrained.prepare_tokens(x)}
    feats = model.pretrained.get_intermediate_layers(x, cfg.intermediate_layer_idx)
    for i, (patch, cls) in enumerate(feats):
        out[f"layer{i}"] = patch
        out[f"cls{i}"] = cls
    head = model.head(feats, ph, pw, T)
    out["head"] = head
    out["depth"] = model(pixels)[0]
    mx.eval(out)
    return {k: np.array(v) for k, v in out.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="directory written by convert.py")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--max-frames", type=int, default=90)
    parser.add_argument("--checkpoint", type=Path, help="original .pth, to verify the conversion")
    args = parser.parse_args()

    import mlx_vlm
    from mlx_vlm.models.video_depth_anything.generate import INFER_LEN, VideoDepthPredictor
    from mlx_vlm.models.video_depth_anything.processing_video_depth_anything import VideoDepthProcessor

    model, tensors = load_model(args.weights)
    if args.checkpoint:
        check_against_checkpoint(tensors, args.checkpoint)

    frames, fps = read_frames(args.video, args.max_frames)
    processor = VideoDepthProcessor(input_size=args.input_size)
    predictor = VideoDepthPredictor(model, processor)
    depth = predictor.infer(frames, progress=False)

    pixels = np.stack([processor.preprocess_frame(f) for f in frames[:INFER_LEN]]).astype(np.float32)
    window0 = np.array(model(mx.array(pixels)[None])[0])
    features = layer_features(model, mx.array(pixels[:4])[None])

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "frames.npy", frames)
    np.save(args.out / "depth.npy", depth.astype(np.float32))
    np.save(args.out / "window0.npy", window0.astype(np.float32))
    np.save(args.out / "pixels0.npy", pixels[0])
    mx.save_safetensors(str(args.out / "features.safetensors"), {k: mx.array(v) for k, v in features.items()})

    manifest = {
        "mlx_vlm": mlx_vlm.__version__,
        "mlx": mx.__version__,
        "opencv": cv2.__version__,
        "device": str(mx.default_device()),
        "video": args.video.name,
        "fps": fps,
        "frames": int(frames.shape[0]),
        "source_size": [int(frames.shape[1]), int(frames.shape[2])],
        "input_size": args.input_size,
        "inference_size": list(pixels.shape[1:3]),
        "metric": bool(model.config.metric),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
