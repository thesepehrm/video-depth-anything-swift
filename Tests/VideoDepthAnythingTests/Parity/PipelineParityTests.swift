import CoreMedia
import Foundation
import MLX
import Testing

@testable import VideoDepthAnything

/// End-to-end parity with mlx-vlm's `VideoDepthPredictor.infer`: sliding windows, keyframes, and
/// scale/shift alignment, on the 72-frame fixture clip (4 windows).
@Suite(.enabled(if: Fixtures.available), .serialized)
struct PipelineParityTests {
    /// The required gate: at most 1% mean relative error on every frame, on the GPU.
    static let maxMeanRelativeError: Float = 0.01

    @Test func streamMatchesReferenceOnDecodedFrames() async throws {
        let model = try await SharedModel.get()
        let frames = try Fixtures.array("frames.npy")
        let golden = try Fixtures.array("depth.npy")
        let source = ArrayFrameSource(rgb: frames)

        Memory.peakMemory = 0
        let progress = ProgressLog()
        var count = 0
        var worst: (Float, Int) = (0, -1)
        for try await frame in model.depth(of: source, progress: { progress.record($0) }) {
            #expect(frame.index == count)
            #expect(frame.width == frames.dim(2) && frame.height == frames.dim(1))
            #expect(frame.presentationTime == CMTime(value: CMTimeValue(count), timescale: 24))
            let ours = MLXArray(frame.values, [frame.height, frame.width])
            let rel = meanRelativeError(ours, golden[count])
            if rel > worst.0 { worst = (rel, count) }
            count += 1
        }
        print("parity: pipeline worst per-frame mean rel \(worst.0) at frame \(worst.1)")
        print("parity: pipeline peak MLX memory \(Memory.peakMemory / 1_048_576) MB")
        #expect(count == frames.dim(0))
        #expect(worst.0 <= Self.maxMeanRelativeError)
        #expect(progress.last == DepthProgress(completedFrames: count, expectedFrames: count))
    }

    /// The same clip through `VideoFrameReader`. AVFoundation and OpenCV convert YUV to RGB slightly
    /// differently, so this is looser than the gate above. Relative model only: the metric model has
    /// no per-window scale fit, so the decoder difference shows up undamped in its absolute values.
    @Test(.enabled(if: Fixtures.available && !Fixtures.isMetric))
    func streamFromMovieFileStaysCloseToReference() async throws {
        let model = try await SharedModel.get()
        let golden = try Fixtures.array("depth.npy")
        var worst: Float = 0
        var count = 0
        for try await frame in try await model.depth(ofVideoAt: Fixtures.clipURL) {
            let ours = MLXArray(frame.values, [frame.height, frame.width])
            worst = max(worst, meanRelativeError(ours, golden[count]))
            count += 1
        }
        print("parity: movie decode worst per-frame mean rel \(worst)")
        #expect(count == golden.dim(0))
        #expect(worst < 0.03)
    }
}

final class ProgressLog: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [DepthProgress] = []
    func record(_ p: DepthProgress) { lock.withLock { values.append(p) } }
    var last: DepthProgress? { lock.withLock { values.last } }
    var all: [DepthProgress] { lock.withLock { values } }
}
