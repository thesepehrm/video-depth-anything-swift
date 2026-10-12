#!/usr/bin/env python3
"""Render the synthetic test clips: a static camera over a checkered floor, a back wall and spheres.

The clips are generated rather than downloaded so the repository carries no third-party footage.
Encoding needs `ffmpeg` on PATH (H.264, yuv420p).

    python scripts/make_fixtures.py parity  Tests/VideoDepthAnythingTests/Resources/static-scene.mp4
    python scripts/make_fixtures.py bench   parity/fixtures/bench-1080p30.mp4

`parity`: 72 frames, 480x272, 24 fps (3 s). One sphere rolls across the floor, the rest of the scene
is still, and every frame gets fresh sensor-like noise, so the model sees changing input while the
true depth of most pixels stays constant. Used by the parity and temporal consistency tests.

`bench`: 360 frames, 1920x1080, 30 fps (12 s), same scene with the camera slowly panning.
"""

import argparse
import subprocess
import sys

import numpy as np

SPHERES = [  # center (x, y, z), radius, rgb
    ((-1.6, 0.6, 5.0), 0.6, (0.85, 0.25, 0.2)),
    ((1.4, 0.9, 7.0), 0.9, (0.2, 0.45, 0.85)),
    ((0.3, 0.4, 3.6), 0.4, (0.9, 0.8, 0.25)),
]
ROLLING = (0.35, (0.25, 0.8, 0.35))  # radius, rgb
WALL_Z = 11.0
LIGHT = np.array([-0.4, 0.8, -0.45])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def render(width, height, t, yaw, rng, noise):
    aspect = width / height
    fov = np.deg2rad(60)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    px = (2 * (xs + 0.5) / width - 1) * np.tan(fov / 2) * aspect
    py = (1 - 2 * (ys + 0.5) / height) * np.tan(fov / 2)
    pitch = np.deg2rad(-8)
    d = np.stack([px, py, np.ones_like(px)], axis=-1)
    # pitch around x, then yaw around y
    cp, sp = np.cos(pitch), np.sin(pitch)
    d = d @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32).T
    cy, sy = np.cos(yaw), np.sin(yaw)
    d = d @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32).T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    origin = np.array([0.0, 1.4, 0.0], dtype=np.float32)

    far = np.full((height, width), np.inf, dtype=np.float32)
    color = np.zeros((height, width, 3), dtype=np.float32)
    normal = np.zeros((height, width, 3), dtype=np.float32)

    # floor y = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = -origin[1] / d[..., 1]
    hit = (tf > 0) & np.isfinite(tf)
    p = origin + d * tf[..., None]
    checker = ((np.floor(p[..., 0]) + np.floor(p[..., 2])) % 2 == 0)
    floor_rgb = np.where(checker[..., None], [0.78, 0.76, 0.72], [0.32, 0.30, 0.28])
    far = np.where(hit, tf, far)
    color = np.where(hit[..., None], floor_rgb, color)
    normal = np.where(hit[..., None], [0.0, 1.0, 0.0], normal)

    # back wall z = WALL_Z
    with np.errstate(divide="ignore", invalid="ignore"):
        tw = (WALL_Z - origin[2]) / d[..., 2]
    p = origin + d * tw[..., None]
    hit = (tw > 0) & (tw < far)
    stripe = (np.floor(p[..., 0] * 1.5) % 2 == 0)
    wall_rgb = np.where(stripe[..., None], [0.55, 0.62, 0.7], [0.45, 0.52, 0.6])
    far = np.where(hit, tw, far)
    color = np.where(hit[..., None], wall_rgb, color)
    normal = np.where(hit[..., None], [0.0, 0.0, -1.0], normal)

    roll_x = -3.0 + 6.0 * t
    spheres = SPHERES + [((roll_x, ROLLING[0], 4.2), ROLLING[0], ROLLING[1])]
    for center, radius, rgb in spheres:
        c = np.array(center, dtype=np.float32)
        oc = origin - c
        b = (d * oc).sum(-1)
        disc = b * b - (oc @ oc - radius * radius)
        ts = -b - np.sqrt(np.maximum(disc, 0))
        hit = (disc > 0) & (ts > 0) & (ts < far)
        p = origin + d * ts[..., None]
        n = (p - c) / radius
        far = np.where(hit, ts, far)
        color = np.where(hit[..., None], rgb, color)
        normal = np.where(hit[..., None], n, normal)

    sky = far == np.inf
    shade = 0.35 + 0.65 * np.clip((normal * LIGHT).sum(-1), 0, 1)
    img = color * shade[..., None]
    img = np.where(sky[..., None], [0.7, 0.8, 0.95], img)
    img = img + rng.normal(0, noise, img.shape)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def encode(frames, w, h, fps, path):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-movflags", "+faststart", path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["parity", "bench"])
    parser.add_argument("out")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    if args.kind == "parity":
        n, w, h, fps = 72, 480, 272, 24
        frames = [render(w, h, i / (n - 1), 0.0, rng, 0.02) for i in range(n)]
    else:
        n, w, h, fps = 360, 1920, 1080, 30
        frames = (render(w, h, i / (n - 1), np.deg2rad(-10 + 20 * i / (n - 1)), rng, 0.01) for i in range(n))
    encode(frames, w, h, fps, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
