// Port of mlx_vlm/models/interpolate.py::{resize_bicubic_nhwc, _bicubic_weights_1d},
// mlx_vlm/models/kernels.py::{_cubic_weight, separable_interpolate} and
// mlx_vlm/models/video_depth_anything/dpt.py::upsample_bilinear (mlx-vlm 0.7.6), plus OpenCV's
// `cv::resize(INTER_CUBIC)` coefficients for the frame preprocessing.
//
// Every resize here is a separable tap table applied by `separableInterpolate`, mlx-vlm's Metal
// kernel, which contracts the width taps of each source row and then the height taps over those rows.
// That is also the order of OpenCV's horizontal-then-vertical passes.
import Foundation
import MLX
import MLXFast

/// Per-axis resampling taps, `(out, taps)` row-major: output position `o` reads source index
/// `indices[o * taps + t]` with weight `weights[o * taps + t]`.
struct ResampleTaps: Equatable {
    let outSize: Int
    let taps: Int
    var indices: [Int32]
    var weights: [Float]

    /// ATen `upsample_bicubic2d` (no antialias): a = -0.75 cubic sampled at the given `scale`, border
    /// taps clamped and not renormalized. Matches `_bicubic_weights_1d`, float32 throughout.
    static func atenBicubic(inSize: Int, outSize: Int, scale: Double) -> ResampleTaps {
        var t = ResampleTaps(outSize: outSize, taps: 4, indices: [], weights: [])
        let s = Float(scale)
        for o in 0..<outSize {
            let src = (Float(o) + 0.5) / s - 0.5
            let base = src.rounded(.down)
            for k in -1...2 {
                let tap = base + Float(k)
                t.weights.append(cubicWeight(src - tap, a: -0.75))
                t.indices.append(Int32(min(max(Int(tap), 0), inSize - 1)))
            }
        }
        return t
    }

    /// OpenCV `cv::resize(..., INTER_CUBIC)` on a float image: half-pixel centres, a = -0.75, the
    /// fourth coefficient is one minus the other three, and out-of-range taps are clamped (replicated).
    /// No antialiasing when downscaling.
    static func opencvCubic(inSize: Int, outSize: Int) -> ResampleTaps {
        var t = ResampleTaps(outSize: outSize, taps: 4, indices: [], weights: [])
        let scale = 1.0 / (Double(outSize) / Double(inSize))
        let a: Float = -0.75
        for o in 0..<outSize {
            var fx = Float((Double(o) + 0.5) * scale - 0.5)
            let sx = Int(fx.rounded(.down))
            fx -= Float(sx)
            let c0 = ((a * (fx + 1) - 5 * a) * (fx + 1) + 8 * a) * (fx + 1) - 4 * a
            let c1 = ((a + 2) * fx - (a + 3)) * fx * fx + 1
            let c2 = ((a + 2) * (1 - fx) - (a + 3)) * (1 - fx) * (1 - fx) + 1
            let c3 = 1 - c0 - c1 - c2
            for (k, c) in [c0, c1, c2, c3].enumerated() {
                t.weights.append(c)
                t.indices.append(Int32(min(max(sx + k - 1, 0), inSize - 1)))
            }
        }
        return t
    }

    /// Bilinear with `align_corners=True`, as mlx-vlm's `bilinear_interpolate`: positions are
    /// `linspace(0, in - 1, out)` in float32, the two taps are the floor and ceiling clamped to the
    /// input.
    static func bilinearAlignCorners(inSize: Int, outSize: Int) -> ResampleTaps {
        var t = ResampleTaps(outSize: outSize, taps: 2, indices: [], weights: [])
        let step: Float = outSize > 1 ? Float(Double(inSize - 1) / Double(outSize - 1)) : 0
        for o in 0..<outSize {
            let pos = outSize == 1 ? 0 : Float(o) * step
            let lo = Int(pos.rounded(.down))
            let w = pos - Float(lo)
            t.indices.append(Int32(min(max(lo, 0), inSize - 1)))
            t.indices.append(Int32(min(max(lo + 1, 0), inSize - 1)))
            t.weights.append(1 - w)
            t.weights.append(w)
        }
        return t
    }

    var indexTable: MLXArray { MLXArray(indices, [outSize, taps]) }
    var weightTable: MLXArray { MLXArray(weights, [outSize, taps]) }
}

/// The cubic convolution kernel (`_cubic_weight`).
func cubicWeight(_ t: Float, a: Float) -> Float {
    let at = abs(t)
    let at2 = at * at
    let at3 = at2 * at
    if at <= 1 { return (a + 2) * at3 - (a + 3) * at2 + 1 }
    if at < 2 { return a * at3 - 5 * a * at2 + 8 * a * at - 4 * a }
    return 0
}

/// Channel-last `F.interpolate(mode="bicubic")` with a scale factor: the output size is floored and
/// sampling uses the given scale, not the effective one (`resize_bicubic_nhwc(scale_factor=)`).
func resizeBicubic(_ x: MLXArray, scaleFactor: (h: Double, w: Double)) -> MLXArray {
    let (inH, inW) = (x.dim(1), x.dim(2))
    let (outH, outW) = (Int(Double(inH) * scaleFactor.h), Int(Double(inW) * scaleFactor.w))
    if outH == inH && outW == inW { return x }
    return separableInterpolate(
        x, y: .atenBicubic(inSize: inH, outSize: outH, scale: scaleFactor.h),
        x: .atenBicubic(inSize: inW, outSize: outW, scale: scaleFactor.w))
}

/// Bilinear upsample of `(N, H, W, C)` with `align_corners=True` (`upsample_bilinear`). Returns `x`
/// unchanged when the size already matches.
func upsampleBilinear(_ x: MLXArray, h: Int, w: Int) -> MLXArray {
    let (inH, inW) = (x.dim(1), x.dim(2))
    if inH == h && inW == w { return x }
    return separableInterpolate(
        x, y: TapCache.shared.bilinear(inH, h), x: TapCache.shared.bilinear(inW, w)
    ).asType(x.dtype)
}

/// Applies per-axis taps to `(B, H, W, C)` in float32. One thread per output element.
func separableInterpolate(_ x: MLXArray, y ty: ResampleTaps, x tx: ResampleTaps) -> MLXArray {
    separableInterpolate(
        x, iy: ty.indexTable, wy: ty.weightTable, ix: tx.indexTable, wx: tx.weightTable)
}

func separableInterpolate(_ x: MLXArray, iy: MLXArray, wy: MLXArray, ix: MLXArray, wx: MLXArray) -> MLXArray {
    let (b, c) = (x.dim(0), x.dim(3))
    let (outH, outW) = (iy.dim(0), ix.dim(0))
    return separableInterpolateKernel(
        [x.asType(.float32), iy, wy, ix, wx],
        grid: (outW * c, outH, b), threadGroup: (min(256, outW * c), 1, 1),
        outputShapes: [[b, outH, outW, c]], outputDTypes: [.float32])[0]
}

/// Bilinear tap tables by (input, output) size. The DPT head asks for the same few sizes on every
/// window.
final class TapCache: @unchecked Sendable {
    static let shared = TapCache()
    private let lock = NSLock()
    private var tables: [Pair: ResampleTaps] = [:]

    private struct Pair: Hashable {
        let a: Int
        let b: Int
    }

    func bilinear(_ inSize: Int, _ outSize: Int) -> ResampleTaps {
        lock.withLock {
            let key = Pair(a: inSize, b: outSize)
            if let hit = tables[key] { return hit }
            let taps = ResampleTaps.bilinearAlignCorners(inSize: inSize, outSize: outSize)
            tables[key] = taps
            return taps
        }
    }
}

/// mlx-vlm 0.7.6 `models/kernels.py::separable_interpolate`, verbatim.
private let separableInterpolateKernel = MLXFast.metalKernel(
    name: "separable_interpolate", inputNames: ["x", "iy", "wy", "ix", "wx"], outputNames: ["out"],
    source: """
            uint gx = thread_position_in_grid.x;
            uint y_out = thread_position_in_grid.y;
            uint b = thread_position_in_grid.z;

            int in_h = x_shape[1];
            int in_w = x_shape[2];
            int channels = x_shape[3];
            int out_h = iy_shape[0];
            int taps_y = iy_shape[1];
            int out_w = ix_shape[0];
            int taps_x = ix_shape[1];

            if (gx >= (uint)(out_w * channels) || y_out >= (uint)out_h)
                return;

            int x_out = gx / channels;
            size_t y_tap = (size_t)y_out * taps_y;
            size_t x_tap = (size_t)x_out * taps_x;
            size_t input_base = (size_t)b * in_h * in_w * channels + (gx % channels);

            // W taps are contracted inside the H loop to match ATen's summation order
            float result = 0.0f;
            for (int a = 0; a < taps_y; a++) {
                size_t row = input_base + (size_t)iy[y_tap + a] * in_w * channels;
                float row_result = 0.0f;
                for (int c = 0; c < taps_x; c++) {
                    row_result += x[row + (size_t)ix[x_tap + c] * channels] * wx[x_tap + c];
                }
                result += row_result * wy[y_tap + a];
            }

            out[((size_t)b * out_h + y_out) * out_w * channels + gx] = result;
        """)
