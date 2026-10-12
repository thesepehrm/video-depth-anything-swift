# Contributing

This package is a port. Its job is to match mlx-vlm's Video Depth Anything output, so every change
is judged against the Python reference first and against taste second.

## Setup

- macOS 14+ on Apple silicon, Xcode 26 (Swift 6.2 or later). Swift 6.1 cannot build mlx-swift 0.31.4.
- Python, for weights and fixtures:
  `uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r scripts/requirements.txt`.
- Weights: `.venv/bin/python scripts/convert.py --out weights/video-depth-anything-small`.

`weights/`, `parity/fixtures/` and `.venv/` are gitignored. Never commit them, and never commit or
link weights from other sources. Only the two Apache-2.0 Small checkpoints are in scope.

`Package.swift` accepts mlx-swift 0.31.x from 0.31.4 (`.upToNextMinor(from: "0.31.4")`), the same
range as [sam31-swift](https://github.com/thesepehrm/sam31-swift), so an app can depend on both.
`Package.resolved` is committed on purpose: it pins the version the parity suite ran against.
Update it in its own commit, after a full parity run.

## Architecture

The package is a 1:1 port of mlx-vlm 0.7.6's `mlx_vlm/models/video_depth_anything` and the parts of
`mlx_vlm/models/dinov2` it uses. Module and parameter names follow the Python code, so the converted
checkpoint loads with the original PyTorch key names. The modules in `Sources/VideoDepthAnything/`:

- `Model/`: DINOv2 ViT-S/14 backbone, temporal motion modules, and the DPT head.
- `Processing/`: OpenCV-compatible bicubic preprocessing, the separable interpolation kernel, and
  the window aligner (scale/shift fit and overlap blending).
- `Config/`: the model configuration, decoded from `config.json`.
- `Loading/`: weight loading with full key and shape checks.
- `API/`: `VideoDepthModel`, `VideoDepthJob`, frame and result types, errors.
- `Video/`: `VideoFrameReader` (AVAssetReader), `DepthVideoWriter` (AVAssetWriter) and the Metal
  colormap kernel.

The reference runs every window, keeps all depth maps, and aligns them at the end. The Swift job
aligns each window as soon as it is inferred and releases every frame no later window can change.
The arithmetic is the same; only the order of the bookkeeping differs.

## Build and test

SwiftPM does not compile MLX's Metal shaders
([mlx-swift#488](https://github.com/ml-explore/mlx-swift/issues/488)). Build them once:

```bash
scripts/build-metallib.sh
```

It runs xcodebuild once and installs `.build/mlx.metallib`. The test suite copies it next to the test
binary before the first GPU call, so plain `swift test` works afterwards. Without it the GPU suites
are skipped, not failed. `xcodebuild test -scheme video-depth-anything-swift-Package -destination
'platform=macOS'` also works and needs no script.

Lint before committing. CI runs the same command:

```bash
swift format lint --recursive --strict Sources Tests
```

## Parity tests

The suites in `Tests/VideoDepthAnythingTests/Parity/` read `VDA_WEIGHTS` and `VDA_FIXTURES`. When
either is unset they are disabled, not failed.

```bash
.venv/bin/python scripts/reference.py --weights weights/video-depth-anything-small \
    --video Tests/VideoDepthAnythingTests/Resources/static-scene.mp4 --out parity/fixtures
VDA_WEIGHTS=$PWD/weights/video-depth-anything-small VDA_FIXTURES=$PWD/parity/fixtures swift test
```

Under xcodebuild, prefix the variables with `TEST_RUNNER_`.

`reference.py` decodes the clip with OpenCV and saves the decoded RGB frames, so the Swift side
runs on identical pixels. AVFoundation converts YUV to RGB differently from OpenCV, which is why the
movie-file test has a looser bound than the frame test.

Gates:

- Preprocessing: `rtol 1e-4, atol 1e-4` against OpenCV.
- Backbone layers: `rtol 1e-3, atol 1e-3`. Head and first window: mean relative error below 1e-3.
- Full pipeline: mean relative error at most 1% on every frame. It currently measures 1.4e-6.
- Temporal consistency: frame-to-frame change below 2% of the mean depth on the static clip, with a
  negative control that must exceed it.

## Every new op needs a parity assert

If you port a module, change math, or change an op's order, add an assert against the mlx-vlm
tensor at the same point (dump it in `scripts/reference.py` if it is not there yet). Keep Python's
order of ops, dtypes and epsilon values. Do not tidy the math.

## Commits

Conventional Commits (`feat:`, `fix:`, `perf:`, `test:`, `docs:`, `chore:`). Keep perf changes in
separate commits, each with before and after numbers and a passing parity run.
