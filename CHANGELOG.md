# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-12

First release: a Swift/MLX port of Video Depth Anything Small, verified against mlx-vlm 0.7.6.

### Added

- `VideoDepthModel` actor: loads `config.json` and `model.safetensors` from a directory, never
  downloads. Accepts Video Depth Anything Small and Metric Small (both Apache-2.0) and rejects Base
  and Large. `unload()` frees the weights.
- `VideoDepthJob`: a pull-based `AsyncSequence` of per-frame Float32 depth at source size, using the
  reference's 32-frame windows, keyframes, and scale/shift alignment. Memory is bounded by the
  window. Progress callback and cooperative cancellation.
- `VideoFrameReader`: `AVAssetReader` frame source for a URL, with an optional time range and target
  frame rate.
- `DepthVideoWriter`: H.264 or HEVC `.mov` output in grayscale, inferno or turbo (Metal lookup
  kernel), with the source frames' timing.
- `DepthOptions(inputSize:)`: network input size, 518 by default, any multiple of 14.
- `vda-cli` with `depth` and `bench` commands, and `scripts/run-cli.sh`.
- `scripts/convert.py` (reproducible conversion of the official checkpoints), `scripts/reference.py`
  (golden fixtures from mlx-vlm), `scripts/bench_reference.py`, `scripts/make_fixtures.py`,
  `scripts/build-metallib.sh`.
- Parity, temporal consistency, reader, writer and API test suites. Depends on mlx-swift 0.31.x from
  0.31.4.

### Performance

M4 Pro (24 GB), 12 s 1080p30 clip, input size 518, 1 GB MLX cache limit: 54.5 s including decoding,
4.17 GB peak MLX memory, 5.32 GB peak process footprint. mlx-vlm on the same clip: 64.3 s excluding
decoding, 8.2 GB max RSS.

### Known limits

- fp32 only.
- The track's preferred transform (rotation) is not applied.

[Unreleased]: https://github.com/thesepehrm/video-depth-anything-swift/compare/0.1.0...HEAD
[0.1.0]: https://github.com/thesepehrm/video-depth-anything-swift/releases/tag/0.1.0
