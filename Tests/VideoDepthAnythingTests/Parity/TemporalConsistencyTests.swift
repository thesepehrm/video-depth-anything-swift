import Foundation
import MLX
import Testing

@testable import VideoDepthAnything

/// Frame-to-frame stability on the static-camera fixture clip. Only one small sphere moves, so the
/// depth of almost every pixel should stay put; broken motion modules, keyframes, or window
/// alignment show up as jumps, most of all across window boundaries (frames 32, 54).
@Suite(.enabled(if: Fixtures.weightsAvailable), .serialized)
struct TemporalConsistencyTests {
    /// Mean |d[t] - d[t-1]| over the frame, relative to the mean depth.
    static func change(_ a: MLXArray, _ b: MLXArray) -> Float {
        (abs(a - b).mean() / b.mean()).item(Float.self)
    }

    /// Measured on an M4 Pro: aligned clip at most 0.012, independent windows 0.039 at the boundary.
    static let threshold: Float = 0.02

    @Test func depthIsStableAcrossFramesAndWindows() async throws {
        let model = try await SharedModel.get()
        var previous: MLXArray?
        var changes: [Float] = []
        for try await frame in try await model.depth(ofVideoAt: Fixtures.clipURL) {
            let d = MLXArray(frame.values, [frame.height, frame.width])
            if let previous { changes.append(Self.change(d, previous)) }
            previous = d
        }
        let boundary = [31, 53].map { changes[$0] }  // change into frames 32 and 54
        print(
            "temporal: max change \(changes.max()!), mean \(changes.reduce(0, +) / Float(changes.count)), "
                + "window boundaries \(boundary)")
        #expect(changes.count == 71)
        #expect(changes.max()! < Self.threshold)
    }

    /// Negative control: the same clip as independent 32-frame windows, with no keyframes and no
    /// alignment, jumps across the boundary by more than the threshold. Shows the test above can fail.
    @Test func independentWindowsJumpAtTheBoundary() async throws {
        let model = try await SharedModel.get()
        var frames: [VideoFrame] = []
        for try await f in try await VideoFrameReader(url: Fixtures.clipURL) { frames.append(f) }
        let pre = FramePreprocessor(inputSize: 518)
        let taps = pre.taps(height: frames[0].height, width: frames[0].width)
        func window(_ range: Range<Int>) async throws -> MLXArray {
            let input = stacked(range.map { pre(rgbArray(frames[$0]), taps: taps) })
            return try await model.forward(UncheckedArray(expandedDimensions(input, axis: 0))).array[0]
        }
        // Frames 0..<32 and 32..<64 share no context and each window keeps its own scale.
        let a = try await window(0..<32)
        let b = try await window(32..<64)
        let jump = Self.change(b[0], a[31])
        let within = Self.change(a[31], a[30])
        print("temporal control: boundary jump \(jump), within-window \(within)")
        #expect(within < Self.threshold)
        #expect(jump > Self.threshold)
    }
}
