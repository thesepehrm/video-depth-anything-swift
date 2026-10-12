import CoreMedia
import Foundation
import MLX

/// Fixed properties of a loaded model.
public struct ModelSummary: Sendable, Hashable {
    /// Whether the checkpoint is Metric Video Depth Anything Small: depth in meters, and windows
    /// joined without scale/shift alignment.
    public let isMetric: Bool
    /// Frames per inference window (32).
    public let windowLength: Int
    /// New frames per window after the first (22).
    public let windowStride: Int
}

/// Video Depth Anything Small on MLX: temporally consistent depth for videos of any length.
///
/// Load the model once, then stream frames through ``depth(of:options:progress:)``. Frames are
/// processed in overlapping 32-frame windows, each aligned to the first, so memory is bounded by the
/// window and not by the clip. The actor serializes all GPU work.
///
/// ```swift
/// let model = try VideoDepthModel.load(directory: weightsDirectory)
/// let reader = try VideoFrameReader(url: movie)
/// for try await frame in model.depth(of: reader) {
///     // frame.values: frame.width × frame.height Float32 relative depth
/// }
/// ```
public actor VideoDepthModel {
    /// Fixed properties of this model.
    public nonisolated let configuration: ModelSummary

    let config: ModelConfig
    private var network: VideoDepthNetwork?

    /// Loads `config.json` and `model.safetensors` (as written by `scripts/convert.py`) from
    /// `directory`. Never downloads anything.
    ///
    /// - Throws: ``VideoDepthError/weightsNotFound(_:)`` when a file is missing,
    ///   ``VideoDepthError/unsupportedModel(_:)`` for a checkpoint other than Small or Metric Small,
    ///   or another ``VideoDepthError`` when the weights do not match the architecture.
    public static func load(directory: URL) throws -> VideoDepthModel {
        let config = try ModelConfig.load(from: directory)
        let network = VideoDepthNetwork(config)
        try loadWeights(into: network, from: directory, dtype: .float32)
        return VideoDepthModel(config: config, network: network)
    }

    init(config: ModelConfig, network: VideoDepthNetwork) {
        self.config = config
        self.network = network
        configuration = ModelSummary(
            isMetric: config.metric, windowLength: Windowing.inferLen, windowStride: Windowing.frameStep)
    }

    /// Whether the weights are loaded (false after ``unload()``).
    public var isLoaded: Bool { network != nil }

    /// Releases the weights and MLX's buffer cache. Jobs still running fail with
    /// ``VideoDepthError/unloaded``; load the model again to continue.
    public func unload() {
        network = nil
        Memory.clearCache()
    }

    /// Caps the memory MLX keeps cached for reuse after arrays are freed (`Memory.cacheLimit`).
    /// Applies to the whole process.
    public func setCacheLimit(bytes: Int) {
        Memory.cacheLimit = max(0, bytes)
    }

    /// Starts a depth job over every frame of `frames`: one ``DepthFrame`` per input frame, in order.
    ///
    /// Frames are pulled from `frames` only as results are consumed, a window (32 frames) at a time.
    /// Cancelling the consuming task stops the job before the next frame or window: `next()` throws
    /// `CancellationError`.
    ///
    /// - Parameters:
    ///   - frames: the frames, all the same size.
    ///   - options: inference settings.
    ///   - progress: called after each delivered frame, on an unspecified executor.
    /// - Returns: a job that throws ``VideoDepthError/invalidOptions(_:)`` for bad options,
    ///   ``VideoDepthError/invalidFrame(_:)`` when a frame's size differs from the first, and
    ///   ``VideoDepthError/unloaded`` after ``unload()``.
    public nonisolated func depth<S: AsyncSequence & Sendable>(
        of frames: S, options: DepthOptions = DepthOptions(),
        progress: (@Sendable (DepthProgress) -> Void)? = nil
    ) -> VideoDepthJob where S.Element == VideoFrame {
        let cursor = FrameCursor(frames.makeAsyncIterator())
        let expected = (frames as? any ExpectedFrameCount)?.expectedFrameCount
        let job = DepthJobState(options: options, metric: config.metric)
        let model = self
        // Pull-based: each window runs inside the consumer's `next()`, so cancelling the consumer
        // cancels the work and surfaces as `CancellationError` rather than a quiet end of stream.
        return VideoDepthJob(expectedFrames: expected) {
            while true {
                try Task.checkCancellation()
                if let frame = job.takeReady() {
                    progress?(DepthProgress(completedFrames: frame.index + 1, expectedFrames: expected))
                    return frame
                }
                if job.isFinished { return nil }
                for _ in 0..<job.framesNeeded {
                    try Task.checkCancellation()
                    guard let frame = try await cursor.next() else {
                        job.markSourceEnded()
                        break
                    }
                    try await model.jobAdd(job, frame)
                }
                try Task.checkCancellation()
                try await model.jobRunWindow(job)
            }
        }
    }

    /// Depth for every frame of a movie file: ``VideoFrameReader`` plus ``depth(of:options:progress:)``.
    public nonisolated func depth(
        ofVideoAt url: URL, timeRange: CMTimeRange? = nil, frameRate: VideoFrameReader.FrameRate = .source,
        options: DepthOptions = DepthOptions(), progress: (@Sendable (DepthProgress) -> Void)? = nil
    ) async throws -> VideoDepthJob {
        let reader = try await VideoFrameReader(url: url, timeRange: timeRange, frameRate: frameRate)
        return depth(of: reader, options: options, progress: progress)
    }

    // MARK: - Job steps (actor-isolated MLX work)

    func jobAdd(_ job: DepthJobState, _ frame: VideoFrame) throws {
        guard network != nil else { throw VideoDepthError.unloaded }
        try job.add(frame)
    }

    func jobRunWindow(_ job: DepthJobState) throws {
        guard let network else { throw VideoDepthError.unloaded }
        try job.runWindow(network)
    }

    /// The network's raw output for one `(1, T, h, w, 3)` input. Tests compare it with mlx-vlm.
    func forward(_ pixels: UncheckedArray) throws -> UncheckedArray {
        guard let network else { throw VideoDepthError.unloaded }
        let out = network(pixels.array)
        eval(out)
        return UncheckedArray(out)
    }
}

/// A running depth job: an `AsyncSequence` of ``DepthFrame``s, one per source frame, in order.
///
/// Iterate it once. Work happens only inside `next()`, a window at a time: a consumer that stops
/// iterating stops the job, and cancelling the iterating task makes `next()` throw
/// `CancellationError`. Memory stays bounded by one window whatever the clip length.
public struct VideoDepthJob: AsyncSequence, Sendable {
    public typealias Element = DepthFrame

    /// Frames the source expects to produce, when it knows.
    public let expectedFrames: Int?
    let produce: @Sendable () async throws -> DepthFrame?

    init(expectedFrames: Int?, produce: @escaping @Sendable () async throws -> DepthFrame?) {
        self.expectedFrames = expectedFrames
        self.produce = produce
    }

    public func makeAsyncIterator() -> Iterator { Iterator(produce: produce) }

    public struct Iterator: AsyncIteratorProtocol {
        let produce: @Sendable () async throws -> DepthFrame?
        private var done = false

        init(produce: @escaping @Sendable () async throws -> DepthFrame?) {
            self.produce = produce
        }

        public mutating func next() async throws -> DepthFrame? {
            guard !done else { return nil }
            do {
                guard let frame = try await produce() else {
                    done = true
                    return nil
                }
                return frame
            } catch {
                done = true
                throw error
            }
        }
    }
}

/// A frame source that knows roughly how many frames it will produce, for progress reporting.
public protocol ExpectedFrameCount {
    /// Frames the source expects to produce, or `nil` when unknown.
    var expectedFrameCount: Int? { get }
}

/// Walks a frame sequence for ``VideoDepthModel/depth(of:options:progress:)``.
///
/// `@unchecked Sendable`: `AsyncThrowingStream(unfolding:)` calls its producer for one element at a
/// time, so the iterator is never used concurrently.
final class FrameCursor<Iterator: AsyncIteratorProtocol>: @unchecked Sendable
where Iterator.Element == VideoFrame {
    private var iterator: Iterator

    init(_ iterator: Iterator) {
        self.iterator = iterator
    }

    func next() async throws -> VideoFrame? {
        var it = iterator
        let frame = try await it.next()
        iterator = it
        return frame
    }
}

/// The state of one depth job.
///
/// `@unchecked Sendable`: the MLX state is read and written only on the owning ``VideoDepthModel``
/// actor (`jobAdd`, `jobRunWindow`); the scalar bookkeeping and the ready queue are guarded by `lock`.
final class DepthJobState: @unchecked Sendable {
    let options: DepthOptions
    private let lock = NSLock()

    // Guarded by `lock`.
    private var ready: [DepthFrame] = []
    private var finished = false
    private var sourceEnded = false
    private var framesRead = 0
    private var window = 0

    // Model actor only.
    private var aligner: WindowAligner
    private var preprocessor: FramePreprocessor
    private var taps: (y: ResampleTaps, x: ResampleTaps)?
    private var sourceSize: (height: Int, width: Int)?
    /// Preprocessed new frames of the next window, `(h, w, 3)` each.
    private var newFrames: [MLXArray] = []
    /// The last preprocessed frame, repeated to pad the final window.
    private var lastPixels: MLXArray?
    /// The previous window's input at the keyframe positions, `(10, h, w, 3)`.
    private var keyframeInput: MLXArray?
    /// Timing of frames not delivered yet, by source frame index.
    private var timing: [Int: (CMTime, CMTime)] = [:]
    private var delivered = 0

    init(options: DepthOptions, metric: Bool) {
        self.options = options
        aligner = WindowAligner(metric: metric)
        preprocessor = FramePreprocessor(inputSize: options.inputSize)
    }

    // MARK: Consumer side

    func takeReady() -> DepthFrame? {
        lock.withLock { ready.isEmpty ? nil : ready.removeFirst() }
    }

    var isFinished: Bool { lock.withLock { finished && ready.isEmpty } }

    /// Frames to read before the next window can run.
    var framesNeeded: Int {
        lock.withLock {
            sourceEnded || finished ? 0 : max(0, Windowing.endFrame(window: window) - framesRead)
        }
    }

    func markSourceEnded() {
        lock.withLock { sourceEnded = true }
    }

    // MARK: Model actor side

    func add(_ frame: VideoFrame) throws {
        try options.validate()
        if let size = sourceSize {
            guard size == (frame.height, frame.width) else {
                throw VideoDepthError.invalidFrame(
                    "frame \(framesRead) is \(frame.width)×\(frame.height), expected \(size.width)×\(size.height)"
                )
            }
        } else {
            sourceSize = (frame.height, frame.width)
            taps = preprocessor.taps(height: frame.height, width: frame.width)
        }
        let pixels = preprocessor(rgbArray(frame), taps: taps!)
        eval(pixels)
        newFrames.append(pixels)
        lastPixels = pixels
        lock.withLock {
            timing[framesRead] = (frame.presentationTime, frame.duration)
            framesRead += 1
        }
    }

    /// Runs the next window if the source has frames for it, aligns it, and queues every frame that
    /// is now final. Finishes the job after the last window.
    func runWindow(_ network: VideoDepthNetwork) throws {
        let (k, read, ended) = lock.withLock { (window, framesRead, sourceEnded) }
        let start = k * Windowing.frameStep
        guard start < read, let lastPixels, let sourceSize else {
            // No frame left for another window.
            precondition(ended || read == 0, "runWindow called before the window's frames were read")
            finish(total: read)
            return
        }
        precondition(read >= Windowing.endFrame(window: k) || ended, "window \(k) is short of frames")

        let newCount = Windowing.inferLen - (k == 0 ? 0 : Windowing.overlap)
        var frames = newFrames
        while frames.count < newCount { frames.append(lastPixels) }
        newFrames.removeAll()
        var input = stacked(frames)
        if let keyframeInput {
            input = concatenated([keyframeInput, input], axis: 0)
        }
        keyframeInput = input[MLXArray(Windowing.keyframes.map(Int32.init))]

        let depth = network(expandedDimensions(input, axis: 0))[0]  // (32, h, w)
        let full = upsampleBilinear(
            expandedDimensions(depth, axis: 3), h: sourceSize.height, w: sourceSize.width)[
                0..., 0..., 0..., 0]
        eval(full, keyframeInput!)
        aligner.add((0..<Windowing.inferLen).map { full[$0] })

        lock.withLock { window = k + 1 }
        let nextStart = (k + 1) * Windowing.frameStep
        // Every frame up to `nextStart + overlap` exists by now unless the source ended, so `read`
        // decides whether there is another window.
        if nextStart < read {
            release(upTo: aligner.finalCount, total: nil)
        } else {
            precondition(ended, "another window may follow")
            finish(total: read)
        }
    }

    private func finish(total: Int) {
        release(upTo: total, total: total)
        lock.withLock { finished = true }
    }

    /// Queues aligned frames up to `end` (exclusive). With `total` set (the last call), aligned
    /// padding frames past the source's end are dropped.
    private func release(upTo end: Int, total: Int?) {
        let frames = aligner.take(upTo: end)
        if total != nil { _ = aligner.take(upTo: .max) }
        guard let (h, w) = sourceSize else { return }
        var out: [DepthFrame] = []
        for depth in frames {
            let index = delivered
            delivered += 1
            let (pts, duration) = lock.withLock { timing.removeValue(forKey: index) ?? (.invalid, .invalid) }
            out.append(
                DepthFrame(
                    index: index, width: w, height: h, values: depth.asArray(Float.self),
                    presentationTime: pts, duration: duration))
        }
        lock.withLock {
            if total != nil { timing.removeAll() }
            ready += out
        }
    }
}

/// `(H, W, 3)` uint8 RGB view of a frame's pixels.
func rgbArray(_ frame: VideoFrame) -> MLXArray {
    let bpp = frame.pixelFormat.bytesPerPixel
    let rows = MLXArray(
        frame.pixels.prefix(frame.bytesPerRow * frame.height), [frame.height, frame.bytesPerRow],
        type: UInt8.self)
    let pixels = rows[0..., ..<(frame.width * bpp)].reshaped(frame.height, frame.width, bpp)
    switch frame.pixelFormat {
    case .rgb8:
        return pixels
    case .bgra8:
        return pixels[0..., 0..., MLXArray([Int32(2), 1, 0])]
    }
}

extension VideoDepthModel {
    struct DebugLayers: @unchecked Sendable {
        let tokens: MLXArray
        let layers: [MLXArray]
        let head: MLXArray
        let depth: MLXArray
    }

    /// Intermediate tensors for a `(T, h, w, 3)` input, for the layer parity tests.
    func debugLayers(_ input: UncheckedArray) throws -> DebugLayers {
        guard let network else { throw VideoDepthError.unloaded }
        let x = input.array
        let (T, H, W) = (x.dim(0), x.dim(1), x.dim(2))
        let p = config.patchSize
        let tokens = network.pretrained.prepareTokens(x)
        let feats = network.pretrained.intermediateLayers(x, indices: config.intermediateLayerIdx)
        let head = network.head(feats, patchH: H / p, patchW: W / p, frameLength: T)
        let depth = network(expandedDimensions(x, axis: 0))[0]
        let out = DebugLayers(tokens: tokens, layers: feats.map(\.patch), head: head, depth: depth)
        eval(out.tokens, out.layers, out.head, out.depth)
        return out
    }
}

/// Hands an `MLXArray` across an actor boundary in tests. `@unchecked Sendable`: the sender does not
/// touch the array again until the call returns.
struct UncheckedArray: @unchecked Sendable {
    let array: MLXArray
    init(_ array: MLXArray) { self.array = array }
}
