@preconcurrency import AVFoundation
import Accelerate
import CoreMedia
import CoreVideo
import Foundation

/// Settings of a rendered depth video.
public struct DepthVideoOptions: Sendable, Equatable {
    /// Video codec of the `.mov` file.
    public enum Codec: Sendable, Equatable {
        case h264
        case hevc

        var avCodec: AVVideoCodecType {
            switch self {
            case .h264: .h264
            case .hevc: .hevc
            }
        }
    }

    /// Which depth values map to the ends of the colormap.
    public enum Normalization: Sendable, Equatable {
        /// The smallest and largest value seen so far in this video. The range only widens, so there
        /// is no per-frame flicker; early frames may use less of the colormap than later ones.
        case running
        /// Each frame's own minimum and maximum. Flickers when the scene's range changes.
        case perFrame
        /// A fixed range, for example the clip's global range from an earlier pass.
        case fixed(ClosedRange<Float>)
    }

    public var codec: Codec
    public var colormap: DepthColormap
    public var normalization: Normalization
    /// Frame time used when a ``DepthFrame`` carries no presentation time.
    public var fallbackFrameDuration: CMTime

    public init(
        codec: Codec = .hevc, colormap: DepthColormap = .grayscale, normalization: Normalization = .running,
        fallbackFrameDuration: CMTime = CMTime(value: 1, timescale: 30)
    ) {
        self.codec = codec
        self.colormap = colormap
        self.normalization = normalization
        self.fallbackFrameDuration = fallbackFrameDuration
    }
}

/// Renders ``DepthFrame``s to a QuickTime movie (H.264 or HEVC), one video frame per depth frame, at
/// the depth frames' own timestamps.
///
/// Frames from a ``VideoFrameReader`` keep their source timing, so the movie has the source range's
/// frame count and frame times; the first frame plays at time zero.
///
/// ```swift
/// let writer = try DepthVideoWriter(url: out, width: w, height: h, options: .init(colormap: .turbo))
/// for try await frame in model.depth(of: reader) { try await writer.append(frame) }
/// try await writer.finish()
/// ```
public actor DepthVideoWriter {
    public nonisolated let url: URL
    public nonisolated let width: Int
    public nonisolated let height: Int
    public nonisolated let options: DepthVideoOptions

    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private let renderer: ColormapRenderer
    private var started = false
    private var appended = 0
    private var running: ClosedRange<Float>?
    private var endTime: CMTime = .invalid

    /// Creates the movie file, replacing any file at `url`.
    ///
    /// - Throws: ``VideoDepthError/writer(_:)`` when the file or the encoder cannot be set up.
    public init(url: URL, width: Int, height: Int, options: DepthVideoOptions = DepthVideoOptions()) throws {
        guard width > 0, height > 0 else {
            throw VideoDepthError.writer("empty video size \(width)×\(height)")
        }
        self.url = url
        self.width = width
        self.height = height
        self.options = options
        try? FileManager.default.removeItem(at: url)
        do {
            writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        } catch {
            throw VideoDepthError.writer(
                "cannot create \(url.lastPathComponent): \(error.localizedDescription)")
        }
        input = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: options.codec.avCodec,
                AVVideoWidthKey: width,
                AVVideoHeightKey: height,
                AVVideoColorPropertiesKey: [
                    AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                    AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                    AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
                ],
            ])
        input.expectsMediaDataInRealTime = false
        adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferMetalCompatibilityKey as String: true,
                kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
            ])
        guard writer.canAdd(input) else { throw VideoDepthError.writer("cannot add a video track") }
        writer.add(input)
        renderer = try ColormapRenderer(colormap: options.colormap)
    }

    /// Frames appended so far.
    public var frameCount: Int { appended }

    /// Renders and appends one frame. Frames must come in presentation order.
    ///
    /// - Throws: ``VideoDepthError/writer(_:)`` for a frame of another size, a timestamp that does not
    ///   increase, or an encoder failure.
    public func append(_ frame: DepthFrame) async throws {
        guard frame.width == width, frame.height == height else {
            throw VideoDepthError.writer(
                "frame is \(frame.width)×\(frame.height), video is \(width)×\(height)")
        }
        let time =
            frame.presentationTime.isNumeric
            ? frame.presentationTime
            : CMTimeMultiply(options.fallbackFrameDuration, multiplier: Int32(appended))
        let duration = frame.duration.isNumeric ? frame.duration : options.fallbackFrameDuration
        if !started {
            guard writer.startWriting() else { throw failure("cannot start writing") }
            writer.startSession(atSourceTime: time)
            started = true
        }

        let range = displayRange(frame.values)
        guard let pool = adaptor.pixelBufferPool else { throw failure("no pixel buffer pool") }
        var buffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer) == kCVReturnSuccess, let buffer else {
            throw failure("cannot allocate a pixel buffer")
        }
        try renderer.render(frame.values, width: width, height: height, range: range, into: buffer)

        while !input.isReadyForMoreMediaData {
            guard writer.status == .writing else { throw failure("encoder stopped") }
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        guard adaptor.append(buffer, withPresentationTime: time) else {
            throw failure("cannot append frame \(appended)")
        }
        appended += 1
        endTime = CMTimeAdd(time, duration)
    }

    /// Finishes the file. The last frame is shown for its own duration.
    public func finish() async throws {
        guard started else { throw VideoDepthError.writer("no frames were appended") }
        input.markAsFinished()
        if endTime.isNumeric { writer.endSession(atSourceTime: endTime) }
        await writer.finishWriting()
        guard writer.status == .completed else { throw failure("cannot finish the movie") }
    }

    /// Stops writing and deletes the partial file.
    public func cancel() {
        if started { writer.cancelWriting() }
        try? FileManager.default.removeItem(at: url)
    }

    /// Writes every frame of `frames` to a new movie at `url` and finishes it. The size comes from the
    /// first frame. On error or cancellation the partial file is deleted.
    ///
    /// - Returns: the number of frames written.
    @discardableResult
    public static func write<S: AsyncSequence>(
        _ frames: S, to url: URL, options: DepthVideoOptions = DepthVideoOptions()
    ) async throws -> Int where S.Element == DepthFrame {
        var writer: DepthVideoWriter?
        do {
            for try await frame in frames {
                if writer == nil {
                    writer = try DepthVideoWriter(
                        url: url, width: frame.width, height: frame.height, options: options)
                }
                try await writer!.append(frame)
            }
            guard let writer else { throw VideoDepthError.writer("no frames to write") }
            try await writer.finish()
            return await writer.frameCount
        } catch {
            await writer?.cancel()
            throw error
        }
    }

    private func displayRange(_ values: [Float]) -> ClosedRange<Float> {
        switch options.normalization {
        case .fixed(let range):
            return range
        case .perFrame:
            return Self.minMax(values)
        case .running:
            let r = Self.minMax(values)
            let merged =
                running.map { min($0.lowerBound, r.lowerBound)...max($0.upperBound, r.upperBound) } ?? r
            running = merged
            return merged
        }
    }

    static func minMax(_ values: [Float]) -> ClosedRange<Float> {
        guard !values.isEmpty else { return 0...0 }
        return vDSP.minimum(values)...vDSP.maximum(values)
    }

    private func failure(_ message: String) -> VideoDepthError {
        let detail = writer.error.map { ": \($0.localizedDescription)" } ?? ""
        return VideoDepthError.writer(message + detail)
    }
}
