import CoreMedia
import Foundation
import MLX

@testable import VideoDepthAnything

/// Parity fixtures written by scripts/reference.py and the weights written by scripts/convert.py.
/// Suites that need them gate on `available`, so they are disabled (not failed) when VDA_FIXTURES or
/// VDA_WEIGHTS is unset.
enum Fixtures {
    static let env = ProcessInfo.processInfo.environment
    static let fixturesURL = env["VDA_FIXTURES"].map { URL(fileURLWithPath: $0) }
    static let weightsDir = env["VDA_WEIGHTS"].map { URL(fileURLWithPath: $0) }

    static var weightsAvailable: Bool {
        guard Metallib.available, let w = weightsDir else { return false }
        return FileManager.default.fileExists(atPath: w.appending(path: "model.safetensors").path)
    }

    static var available: Bool {
        guard weightsAvailable, let f = fixturesURL else { return false }
        return FileManager.default.fileExists(atPath: f.appending(path: "manifest.json").path)
    }

    static var weightsURL: URL { weightsDir! }

    /// Whether the fixtures were made with the metric checkpoint.
    static var isMetric: Bool { manifest["metric"] as? Bool ?? false }

    static func array(_ file: String) throws -> MLXArray {
        try loadArray(url: fixturesURL!.appending(path: file))
    }

    static func arrays(_ file: String) throws -> [String: MLXArray] {
        try loadArrays(url: fixturesURL!.appending(path: file))
    }

    static var manifest: [String: Any] {
        let data = try! Data(contentsOf: fixturesURL!.appending(path: "manifest.json"))
        return try! JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    /// The committed synthetic clip (scripts/make_fixtures.py parity).
    static var clipURL: URL {
        Bundle.module.url(forResource: "Resources/static-scene", withExtension: "mp4")!
    }
}

/// One model per test process.
enum SharedModel {
    private static let task = Task {
        let model = try VideoDepthModel.load(directory: Fixtures.weightsURL)
        // Suites run in parallel; an unbounded MLX buffer cache pushes the machine into swap.
        await model.setCacheLimit(bytes: 1 << 30)
        return model
    }

    static func get() async throws -> VideoDepthModel { try await task.value }
}

/// An in-memory frame source, for frames that did not come from a file.
struct ArrayFrameSource: AsyncSequence, Sendable, ExpectedFrameCount {
    typealias Element = VideoFrame
    let frames: [VideoFrame]

    var expectedFrameCount: Int? { frames.count }

    struct AsyncIterator: AsyncIteratorProtocol {
        var frames: ArraySlice<VideoFrame>
        mutating func next() async throws -> VideoFrame? { frames.popFirst() }
    }

    func makeAsyncIterator() -> AsyncIterator { AsyncIterator(frames: frames[...]) }
}

extension ArrayFrameSource {
    /// `(T, H, W, 3)` uint8 RGB frames at `fps`.
    init(rgb: MLXArray, fps: Int32 = 24) {
        let (t, h, w) = (rgb.dim(0), rgb.dim(1), rgb.dim(2))
        frames = (0..<t).map { i in
            VideoFrame(
                width: w, height: h, pixelFormat: .rgb8, pixels: Data(rgb[i].asArray(UInt8.self)),
                presentationTime: CMTime(value: CMTimeValue(i), timescale: fps),
                duration: CMTime(value: 1, timescale: fps))
        }
    }
}
