# Acknowledgements

video-depth-anything-swift builds on other people's work. This file credits it and carries the
license notices that apply to code or data derived from it.

## Video Depth Anything (ByteDance)

The model architecture, the inference schedule (32-frame windows, keyframes, scale/shift alignment)
and the weights are from [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)
by Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng and Bingyi Kang
(CVPR 2025). The code and the Small checkpoints are released under the Apache License 2.0; a copy is
in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

This repository does not include or redistribute any weights. `scripts/convert.py` downloads
`depth-anything/Video-Depth-Anything-Small` (or `Metric-Video-Depth-Anything-Small`) from the
Hugging Face Hub at a pinned revision and rewrites its tensors to channel-last layout. Weights you
convert remain under Apache-2.0.

The Base and Large checkpoints are licensed CC-BY-NC-4.0. This package does not support them.

```bibtex
@article{video_depth_anything,
  title={Video Depth Anything: Consistent Depth Estimation for Super-Long Videos},
  author={Chen, Sili and Guo, Hengkai and Zhu, Shengnan and Zhang, Feihu and Huang, Zilong and Feng, Jiashi and Kang, Bingyi},
  journal={arXiv:2501.12375},
  year={2025}
}
```

## DINOv2 (Meta)

The backbone is [DINOv2](https://github.com/facebookresearch/dinov2) ViT-S/14 (Apache-2.0), as
embedded in the Video Depth Anything checkpoint.

## mlx-vlm

The Swift modules are a line-by-line port of mlx-vlm 0.7.6's Video Depth Anything implementation
(`mlx_vlm/models/video_depth_anything`, `mlx_vlm/models/dinov2`, and the interpolation helpers in
`mlx_vlm/models/interpolate.py`), including its `separable_interpolate` Metal kernel. The parity
fixtures come from running mlx-vlm itself, on weights converted by `scripts/convert.py`.
Source: [github.com/Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm).

```
MIT License

Copyright © 2025 Prince Canuma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## OpenCV

The frame preprocessing reproduces `cv::resize(..., INTER_CUBIC)` (half-pixel centres, a = -0.75,
replicated borders) so that it matches the reference, which resizes with OpenCV. No OpenCV code is
included. OpenCV 4.5+ is licensed under Apache-2.0 ([opencv.org/license](https://opencv.org/license/)).

## Colormaps

`Sources/VideoDepthAnything/Video/ColormapTables.swift` holds two 256-entry color tables, generated
from matplotlib by `scripts/colormaps.py`:

- **inferno** by Nathaniel J. Smith and Stéfan van der Walt, dedicated to the public domain under
  CC0 ([github.com/BIDS/colormap](https://github.com/BIDS/colormap)).
- **turbo** by Anton Mikhailov, Copyright 2019 Google LLC, licensed under the Apache License 2.0
  ([original data](https://gist.github.com/mikhailov-work/6a308c20e494d9e0ccc29036b28faa7a)).

## mlx-swift

The package runs on Apple's [mlx-swift](https://github.com/ml-explore/mlx-swift) (MIT), its only
runtime dependency. It is fetched by Swift Package Manager and not vendored here.

## Test footage

`Tests/VideoDepthAnythingTests/Resources/static-scene.mp4` is rendered by
`scripts/make_fixtures.py` (part of this repository, MIT). No third-party footage is included.
