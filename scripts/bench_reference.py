#!/usr/bin/env python3
"""Time mlx-vlm's reference sliding-window inference on a clip, for comparison with `vda-cli bench`.

Decoding is not timed: the reference reads the whole clip into memory first. Peak memory is MLX's
peak plus the process's maximum resident size.

    python scripts/bench_reference.py --weights weights/video-depth-anything-small \\
        --video parity/fixtures/bench-1080p30.mp4
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference import load_model, read_frames  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--max-frames", type=int, default=100000)
    parser.add_argument("--cache-limit-mb", type=float, help="mx.set_cache_limit, as vda-cli's flag")
    args = parser.parse_args()

    from mlx_vlm.models.video_depth_anything.generate import VideoDepthPredictor
    from mlx_vlm.models.video_depth_anything.processing_video_depth_anything import VideoDepthProcessor

    if args.cache_limit_mb is not None:
        mx.set_cache_limit(int(args.cache_limit_mb * 1024 * 1024))
    model, _ = load_model(args.weights)
    frames, fps = read_frames(args.video, args.max_frames)
    predictor = VideoDepthPredictor(model, VideoDepthProcessor(input_size=args.input_size))
    mx.reset_peak_memory()
    start = time.perf_counter()
    depth = predictor.infer(frames, progress=False)
    wall = time.perf_counter() - start
    result = {
        "frames": int(depth.shape[0]),
        "source": f"{frames.shape[2]}x{frames.shape[1]} @ {fps:g} fps",
        "wall_s": round(wall, 2),
        "fps": round(depth.shape[0] / wall, 2),
        "peak_mlx_gb": round(mx.get_peak_memory() / 1e9, 2),
        "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
