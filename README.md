# video-depth-anything-swift

[Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) Small in Swift on
Apple silicon. Built on MLX, no Python at runtime. Depth for videos of any length, consistent from
frame to frame, with memory bounded by one 32-frame window.

```swift
import VideoDepthAnything

let model = try VideoDepthModel.load(directory: URL(filePath: "weights/video-depth-anything-small"))
let job = try await model.depth(ofVideoAt: movieURL) { progress in
    print(progress.fraction ?? 0)
}
for try await frame in job {
    // frame.values: frame.width × frame.height Float32 relative depth, at source size
}

// Or render it straight to a movie.
try await DepthVideoWriter.write(
    try await model.depth(ofVideoAt: movieURL), to: outURL,
    options: DepthVideoOptions(codec: .hevc, colormap: .turbo))
```

On the 72-frame test clip the output matches the Python reference (mlx-vlm 0.7.6, same weights, GPU)
to a mean relative error of 1.4e-6 per frame. A 12 s 1080p30 clip takes 54.5 s on an M4 Pro.

---

## Requirements

- macOS 14 or later on Apple silicon.
- Xcode 26 (Swift 6.2 or later).
- mlx-swift 0.31.x from 0.31.4, the same range as
  [sam31-swift](https://github.com/thesepehrm/sam31-swift), so one app can use both.

## Install

```swift
.package(url: "https://github.com/thesepehrm/video-depth-anything-swift", from: "0.1.0"),
```

Then add `.product(name: "VideoDepthAnything", package: "video-depth-anything-swift")` to your
target.

## Weights and licence

Only the official Apache-2.0 Small checkpoints are supported:

| Variant         | Hugging Face repo                                  | Output                 |
| :-------------- | :------------------------------------------------- | :--------------------- |
| Small (default) | `depth-anything/Video-Depth-Anything-Small`        | relative inverse depth |
| Metric Small    | `depth-anything/Metric-Video-Depth-Anything-Small` | metric depth (m)       |

The Base and Large checkpoints are CC-BY-NC-4.0 and are rejected at load time. Do not use the
`mlx-community/*-MLX` conversions either: they carry a non-commercial label.

The package never downloads anything. Convert the official checkpoint yourself:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r scripts/requirements.txt
.venv/bin/python scripts/convert.py --out weights/video-depth-anything-small
```

`convert.py` fetches the `.pth` at a pinned revision, checks its SHA-256, transposes the convolution
kernels to channel-last, and writes `model.safetensors` (float32, 116 MB) and `config.json`. The
output is byte-for-byte reproducible. Pass `--variant metric-small` for the metric model, or
`--checkpoint file.pth` to convert a local copy.

Converted weights stay under Apache-2.0. Ship them with your app or download them in your app; this
package only reads a directory.

## API

### Load

`VideoDepthModel.load(directory:)` reads `config.json` and `model.safetensors`. It throws
`VideoDepthError.unsupportedModel` for anything but Small or Metric Small. `unload()` frees the
weights and MLX's buffer cache; jobs still running then throw `VideoDepthError.unloaded`.

### Depth jobs

`model.depth(of:options:progress:)` takes any `AsyncSequence` of `VideoFrame`s and returns a
`VideoDepthJob`, an `AsyncSequence` of `DepthFrame`s: one per input frame, in order, at the input's
size.

- **Frames in.** `VideoFrameReader` decodes a movie with `AVAssetReader`, optionally limited to a
  `CMTimeRange` and to a target frame rate (`.target(15)` keeps every n-th frame, as the reference
  does). Any other source can build `VideoFrame`s from BGRA or RGB bytes.
- **Depth out.** `DepthFrame.values` holds `width × height` Float32 values, row-major, plus the
  source frame's presentation time and duration. Relative depth is disparity-like (larger is
  nearer) with one scale per job.
- **Pull-based.** Work runs inside `next()`, a window at a time. Stop iterating and the job stops.
  Cancel the task and `next()` throws `CancellationError`.
- **Progress.** The callback gets `completedFrames` and, for a `VideoFrameReader`, an estimate of
  the total.

`DepthOptions(inputSize:)` sets the network's short side: 518 by default (the training size), any
multiple of 14. Depth is always resized back to the source size.

### Rendering

`DepthVideoWriter` turns depth frames into a QuickTime movie, H.264 or HEVC, with one video frame
per depth frame at the depth frame's timestamp. Fed from a `VideoFrameReader`, the movie has the
source range's frame count and timing, starting at zero.

- Colormaps: `.grayscale` (near is white), `.inferno`, `.turbo`. The lookup runs in a Metal compute
  kernel that writes straight into the encoder's pixel buffers.
- Normalization: `.running` (default, the range seen so far, which only widens, so no flicker),
  `.perFrame`, or `.fixed(range)`.

### Memory

Memory depends on the window and the frame size, not on the clip length: the same 4.17 GB MLX peak
at 2, 5 and 12 s of 1080p. MLX also keeps freed buffers in a cache. Uncapped, that cache brought the
1080p process footprint to 13.1 GB; with `setCacheLimit(bytes: 1 << 30)` it was 5.3 GB at the same
speed. The limit applies to the whole process, so the package leaves it to the host app.

## CLI

```bash
scripts/run-cli.sh depth --weights weights/video-depth-anything-small \
    --input clip.mov --output depth.mov --colormap turbo
scripts/run-cli.sh bench --weights weights/video-depth-anything-small --input clip.mov --cache-limit-mb 1024
```

Both take `--input-size`, `--fps`, `--start` and `--duration`. The script builds MLX's Metal kernels
once (see [Build and test](#build-and-test)).

## Performance

`vda-cli bench` on a MacBook Pro with an M4 Pro (24 GB), macOS 26.1, fp32, input size 518 (518×924).
The clip is 12 s of synthetic 1080p30 footage rendered by `scripts/make_fixtures.py bench`: 360
frames, 17 windows. Wall time includes decoding.

|                                 | Wall time | Throughput | Peak MLX memory | Peak process memory |
| :------------------------------ | --------: | ---------: | --------------: | ------------------: |
| Swift, cache limit 1 GB         |    54.5 s |    6.6 fps |         4.17 GB |             5.32 GB |
| Swift, no cache limit           |    56.1 s |    6.4 fps |         4.17 GB |            13.13 GB |
| mlx-vlm 0.7.6, cache limit 1 GB |    64.3 s |    5.6 fps |         4.51 GB |    8.2 GB (max RSS) |

The Python row excludes decoding, which it does up front, and its memory grows with the clip: it
holds every frame and every full-resolution depth map until the end. Without a cache limit it went
into swap on this machine and took 478 s for the first 150 frames.

Reproduce:

```bash
.venv/bin/python scripts/make_fixtures.py bench parity/fixtures/bench-1080p30.mp4
scripts/run-cli.sh bench --weights weights/video-depth-anything-small \
    --input parity/fixtures/bench-1080p30.mp4 --cache-limit-mb 1024
.venv/bin/python scripts/bench_reference.py --weights weights/video-depth-anything-small \
    --video parity/fixtures/bench-1080p30.mp4 --cache-limit-mb 1024
```

## How parity is verified

`scripts/reference.py` runs mlx-vlm's port on the converted weights and writes golden tensors for
`Tests/VideoDepthAnythingTests/Resources/static-scene.mp4`, a 3 s, 72-frame clip rendered by
`scripts/make_fixtures.py` (4 windows). It saves the decoded RGB frames too, so Swift runs on the
same pixels. Measured on the GPU:

| Check                                               | Gate      | Measured |
| :-------------------------------------------------- | :-------- | -------: |
| Preprocessing vs OpenCV bicubic                     | rtol 1e-4 |     pass |
| Backbone layers 2, 5, 8, 11                         | rtol 1e-3 |     pass |
| First window, mean relative error                   | < 1e-3    |   8.0e-7 |
| Full pipeline, worst frame mean relative error      | ≤ 1%      |   1.4e-6 |
| Metric Small, full pipeline                         | ≤ 1%      |   7.9e-6 |
| Same clip decoded by AVFoundation instead of OpenCV | < 3%      |     2.6% |

`reference.py --checkpoint file.pth` also asserts that the converted tensors equal mlx-vlm's own
`sanitize` of the PyTorch checkpoint.

The temporal test runs the clip through the full pipeline. The camera is still and one small sphere
moves, so depth should barely change. The largest frame-to-frame change is 1.2% of the mean depth,
under a 2% gate, including at the window boundaries. A negative control runs two independent
windows with no keyframes or alignment: their boundary jumps 3.9%, so the gate catches broken
alignment.

## Build and test

SwiftPM does not compile MLX's Metal shaders
([mlx-swift#488](https://github.com/ml-explore/mlx-swift/issues/488)). Build them once with:

```bash
scripts/build-metallib.sh
```

After that, `swift build` and `swift test` work as usual. The tests copy the kernels next to the test
binary; without them the GPU suites are skipped. Apps built with Xcode need none of this.

The parity suites also need weights and fixtures:

```bash
.venv/bin/python scripts/reference.py --weights weights/video-depth-anything-small \
    --video Tests/VideoDepthAnythingTests/Resources/static-scene.mp4 --out parity/fixtures
VDA_WEIGHTS=$PWD/weights/video-depth-anything-small VDA_FIXTURES=$PWD/parity/fixtures swift test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the porting rules and tolerances.

## Known limits

- fp32 only.
- Frames wider than about 16:9 get a smaller input size, as in the reference.
- `VideoFrameReader` ignores the track's preferred transform: rotated phone videos come out in
  their stored orientation.
- The reference's streaming mode (`video_depth_stream.py`) is not ported. Each job runs the offline
  windowed algorithm, emitting frames as soon as they are final.

## License and acknowledgements

MIT, see [LICENSE](LICENSE). The model and weights are Apache-2.0 by the Video Depth Anything
authors, and the Swift code is a port of mlx-vlm (MIT). [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)
has the credits and notices.
