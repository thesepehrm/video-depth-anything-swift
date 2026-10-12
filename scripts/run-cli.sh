#!/usr/bin/env bash
# Runs `swift run -c release vda-cli "$@"` with MLX's Metal kernels in place.
#
# SwiftPM does not compile mlx-swift's Metal shaders (mlx-swift#488), so a `swift build` binary
# fails with "Failed to load the default metallib". scripts/build-metallib.sh builds them once with
# xcodebuild; this script copies the result next to the SwiftPM binary as `mlx.metallib`, the first
# place MLX looks.
set -euo pipefail
cd "$(dirname "$0")/.."

scripts/build-metallib.sh
swift build -c release --product vda-cli >&2
bin=$(swift build -c release --show-bin-path)
cp .build/mlx.metallib "$bin/mlx.metallib"
exec "$bin/vda-cli" "$@"
