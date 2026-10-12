#!/usr/bin/env bash
# Builds MLX's Metal kernels once and installs them as `.build/mlx.metallib`.
#
# SwiftPM does not compile mlx-swift's Metal shaders (mlx-swift#488), so binaries built by
# `swift build` / `swift test` fail with "Failed to load the default metallib". xcodebuild does
# compile them. This script builds the metallib with xcodebuild and copies it to `.build/mlx.metallib`,
# where the test suite (Tests/VideoDepthAnythingTests/Support/Metallib.swift) and scripts/run-cli.sh
# pick it up. It rebuilds when Package.resolved changes.
set -euo pipefail
cd "$(dirname "$0")/.."

out=.build/mlx.metallib
xcode_lib=.build/xcode/Build/Products/Release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib

if [[ -f $out && ! Package.resolved -nt $out ]]; then
    echo "$out is up to date" >&2
    exit 0
fi
if [[ ! -f $xcode_lib || Package.resolved -nt $xcode_lib ]]; then
    echo "building MLX Metal kernels with xcodebuild (one time)..." >&2
    xcodebuild build -scheme VideoDepthAnything -configuration Release -destination 'platform=macOS' \
        -derivedDataPath .build/xcode -quiet >&2
fi
mkdir -p .build
cp "$xcode_lib" "$out"
echo "installed $out" >&2
