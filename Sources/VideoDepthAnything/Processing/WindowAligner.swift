// Port of mlx_vlm/models/video_depth_anything/generate.py::{compute_scale_and_shift,
// get_interpolate_frames, VideoDepthPredictor.infer} (mlx-vlm 0.7.6), restructured to stream: each
// window is aligned as soon as it is inferred, and only frames no later window can change are released.
import Foundation
import MLX

/// Inference settings from the reference implementation. Do not change.
enum Windowing {
    static let inferLen = 32
    static let overlap = 10
    static let keyframes = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
    static let interpLen = 8
    static var frameStep: Int { inferLen - overlap }
    /// Keyframes whose depth anchors the scale/shift fit of the next window (`kf_align_list`).
    static var alignKeyframes: [Int] { Array(keyframes.prefix(overlap - interpLen)) }

    /// Index of the first new (not keyframe) source frame in window `k`.
    static func firstNewFrame(window k: Int) -> Int {
        k == 0 ? 0 : k * frameStep + overlap
    }

    /// One past the last source frame of window `k`.
    static func endFrame(window k: Int) -> Int {
        k * frameStep + inferLen
    }
}

/// Least-squares scale and shift that map `prediction` onto `target` (`compute_scale_and_shift` with
/// an all-ones mask). Sums run in float32 on the GPU; the 2×2 solve is float32 scalar math, as numpy
/// does on `np.float32` sums.
func scaleAndShift(prediction: MLXArray, target: MLXArray) -> (scale: Float, shift: Float) {
    let p = prediction.asType(.float32)
    let t = target.asType(.float32)
    let sums = stacked([(p * p).sum(), p.sum(), (p * t).sum(), t.sum()])
    let s = sums.asArray(Float.self)
    let (a00, a01, b0, b1) = (s[0], s[1], s[2], s[3])
    let a11 = Float(p.size)
    var (x0, x1): (Float, Float) = (1, 0)
    let det = a00 * a11 - a01 * a01
    if det != 0 {
        x0 = (a11 * b0 - a01 * b1) / det
        x1 = (-a01 * b0 + a00 * b1) / det
    }
    return (x0, x1)
}

/// Aligns each window's depth onto the first window's scale and blends the overlaps.
///
/// Feed windows in order with ``add(_:)``. Frames before ``finalCount`` no longer change; the last
/// `interpLen` aligned frames are kept pending because the next window blends into them.
struct WindowAligner {
    let metric: Bool
    /// Aligned frames not yet taken, `(H, W)` each. `pending[0]` is source frame `taken`.
    private(set) var pending: [MLXArray] = []
    /// Frames already handed out with ``take(upTo:)``.
    private(set) var taken = 0
    private var refAlign: [MLXArray] = []
    private var windows = 0

    init(metric: Bool) {
        self.metric = metric
    }

    /// Source frames aligned so far, final or not.
    var alignedCount: Int { taken + pending.count }

    /// Frames that no later window can change.
    var finalCount: Int { max(taken, alignedCount - Windowing.interpLen) }

    /// Adds one window's depth: `inferLen` frames of `(H, W)`, in window order.
    mutating func add(_ depth: [MLXArray]) {
        precondition(depth.count == Windowing.inferLen)
        let kf = Windowing.alignKeyframes
        defer { windows += 1 }
        if windows == 0 {
            pending += depth
            refAlign = kf.map { depth[$0] }
            return
        }
        let currAlign = (0..<kf.count).map { depth[$0] }
        let (scale, shift): (Float, Float) =
            metric
            ? (1, 0)
            : scaleAndShift(prediction: concatenated(currAlign), target: concatenated(refAlign))
        let s = MLXArray(scale)
        let t = MLXArray(shift)
        func aligned(_ d: MLXArray) -> MLXArray { maximum(d * s + t, 0) }

        let n = Windowing.interpLen
        precondition(pending.count >= n, "the pending tail was taken")
        let pre = Array(pending.suffix(n))
        let post = (kf.count..<Windowing.overlap).map { aligned(depth[$0]) }
        pending.removeLast(n)
        pending += interpolateFrames(pre, post)
        for i in Windowing.overlap..<Windowing.inferLen {
            pending.append(aligned(depth[i]))
        }
        refAlign = [refAlign[0]] + kf.dropFirst().map { aligned(depth[$0]) }
        eval(pending + refAlign)
    }

    /// Removes and returns aligned frames up to source frame `end` (exclusive).
    mutating func take(upTo end: Int) -> [MLXArray] {
        let n = max(0, min(end, alignedCount) - taken)
        let out = Array(pending.prefix(n))
        pending.removeFirst(n)
        taken += n
        return out
    }
}

/// Linear blend from `pre` to `post` across the overlap (`get_interpolate_frames`).
func interpolateFrames(_ pre: [MLXArray], _ post: [MLXArray]) -> [MLXArray] {
    precondition(pre.count == post.count && pre.count > 1)
    let n = pre.count
    let step = 1.0 / Double(n - 1)
    let weights = [0.0] + (1..<(n - 1)).map { Double($0) * step } + [1.0]
    return weights.enumerated().map { i, w in
        pre[i] * Float(1 - w) + post[i] * Float(w)
    }
}
