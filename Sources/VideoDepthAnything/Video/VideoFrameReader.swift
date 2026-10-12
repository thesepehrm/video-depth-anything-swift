@preconcurrency import AVFoundation
import CoreMedia
import CoreVideo
import Foundation

/// Decodes a movie's first video track to ``VideoFrame``s with `AVAssetReader`.
///
/// An `AsyncSequence`: each iteration opens its own reader and decodes lazily, one frame per
/// `next()`, so memory stays at one frame regardless of the clip length. Frames are 32BGRA at the
/// track's natural size (the preferred transform, such as a portrait rotation, is not applied).
///
/// ```swift
/// let reader = try VideoFrameReader(
///     url: movie, timeRange: CMTimeRange(start: .zero, duration: CMTime(seconds: 3, preferredTimescale: 600)),
///     frameRate: .target(15))
/// for try await frame in reader { ... }
/// ```
public struct VideoFrameReader: AsyncSequence, Sendable, ExpectedFrameCount {
    public typealias Element = VideoFrame

    /// Which frames to decode.
    public enum FrameRate: Sendable, Equatable {
        /// Every frame.
        case source
        /// At most this many frames per second: every `n`-th frame, `n = round(source fps / target)`,
        /// as the reference's `read_video_frames`. A target at or above the source rate keeps every
        /// frame. Kept frames keep their source timestamps.
        case target(Double)
    }

    public let url: URL
    /// The decoded range, clamped to the track; the whole track when `nil`.
    public let timeRange: CMTimeRange?
    public let frameRate: FrameRate
    /// Natural size of the video track, in pixels.
    public let size: CGSize
    /// The track's nominal frame rate.
    public let nominalFrameRate: Double
    /// Duration of one source frame, used when a sample carries none.
    public let frameDuration: CMTime
    /// Keep every `stride`-th decoded frame.
    public let stride: Int
    /// Estimated number of frames this reader produces (track frame rate × range duration ÷ stride).
    public let expectedFrameCount: Int?

    /// Opens `url` and reads its video track's properties.
    ///
    /// - Throws: ``VideoDepthError/invalidVideo(_:)`` when the file cannot be read or has no video,
    ///   and ``VideoDepthError/invalidOptions(_:)`` for a non-positive target frame rate.
    public init(url: URL, timeRange: CMTimeRange? = nil, frameRate: FrameRate = .source) async throws {
        let asset = AVURLAsset(url: url)
        let track = try await Self.videoTrack(asset)
        let (naturalSize, fps, trackRange, minFrameDuration): (CGSize, Float, CMTimeRange, CMTime)
        do {
            (naturalSize, fps, trackRange, minFrameDuration) = try await track.load(
                .naturalSize, .nominalFrameRate, .timeRange, .minFrameDuration)
        } catch {
            throw VideoDepthError.invalidVideo(
                "cannot read the video track of \(url.lastPathComponent): \(error)")
        }
        self.url = url
        self.timeRange = timeRange
        self.frameRate = frameRate
        size = naturalSize
        nominalFrameRate = Double(fps)
        frameDuration =
            minFrameDuration.isNumeric && minFrameDuration > .zero
            ? minFrameDuration
            : (fps > 0 ? CMTime(seconds: 1 / Double(fps), preferredTimescale: 600_000) : .invalid)
        switch frameRate {
        case .source:
            stride = 1
        case .target(let fps):
            guard fps > 0, fps.isFinite else {
                throw VideoDepthError.invalidOptions("target frame rate \(fps) must be positive")
            }
            stride = nominalFrameRate > fps ? Swift.max(1, Int((nominalFrameRate / fps).rounded())) : 1
        }
        let range = timeRange.map { $0.intersection(trackRange) } ?? trackRange
        if nominalFrameRate > 0, range.duration.isNumeric {
            let decoded = Int((range.duration.seconds * nominalFrameRate).rounded())
            expectedFrameCount = (decoded + stride - 1) / stride
        } else {
            expectedFrameCount = nil
        }
    }

    public func makeAsyncIterator() -> Iterator {
        Iterator(reader: self)
    }

    /// Decodes frames on demand. Not shareable between tasks.
    public struct Iterator: AsyncIteratorProtocol {
        let reader: VideoFrameReader
        private var state: ReaderState?
        private var decoded = 0

        init(reader: VideoFrameReader) {
            self.reader = reader
        }

        public mutating func next() async throws -> VideoFrame? {
            try Task.checkCancellation()
            if state == nil { state = try await ReaderState(reader) }
            guard let state else { return nil }
            while let sample = state.output.copyNextSampleBuffer() {
                defer { decoded += 1 }
                guard decoded % reader.stride == 0 else { continue }
                guard let buffer = CMSampleBufferGetImageBuffer(sample) else { continue }
                var duration = CMSampleBufferGetDuration(sample)
                if !duration.isNumeric { duration = reader.frameDuration }
                if duration.isNumeric && reader.stride > 1 {
                    duration = CMTimeMultiply(duration, multiplier: Int32(reader.stride))
                }
                return try frame(
                    buffer, time: CMSampleBufferGetPresentationTimeStamp(sample), duration: duration)
            }
            if state.reader.status == .failed {
                throw VideoDepthError.invalidVideo(
                    "decoding \(reader.url.lastPathComponent) failed: "
                        + (state.reader.error?.localizedDescription ?? "unknown error"))
            }
            return nil
        }
    }

    static func videoTrack(_ asset: AVURLAsset) async throws -> AVAssetTrack {
        let tracks: [AVAssetTrack]
        do {
            tracks = try await asset.loadTracks(withMediaType: .video)
        } catch {
            throw VideoDepthError.invalidVideo(
                "cannot open \(asset.url.lastPathComponent): \(error.localizedDescription)")
        }
        guard let track = tracks.first else {
            throw VideoDepthError.invalidVideo("\(asset.url.lastPathComponent) has no video track")
        }
        return track
    }
}

/// An open `AVAssetReader` and its track output.
///
/// `@unchecked Sendable`: owned by one iterator, which is used by one task at a time.
final class ReaderState: @unchecked Sendable {
    let reader: AVAssetReader
    let output: AVAssetReaderTrackOutput

    init(_ source: VideoFrameReader) async throws {
        let asset = AVURLAsset(url: source.url)
        let track = try await VideoFrameReader.videoTrack(asset)
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            throw VideoDepthError.invalidVideo(
                "cannot read \(source.url.lastPathComponent): \(error.localizedDescription)")
        }
        if let range = source.timeRange { reader.timeRange = range }
        output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw VideoDepthError.invalidVideo(
                "cannot decode the video track of \(source.url.lastPathComponent)")
        }
        reader.add(output)
        guard reader.startReading() else {
            throw VideoDepthError.invalidVideo(
                "cannot start reading \(source.url.lastPathComponent): "
                    + (reader.error?.localizedDescription ?? "unknown error"))
        }
    }

    deinit {
        if reader.status == .reading { reader.cancelReading() }
    }
}

/// Copies a 32BGRA pixel buffer into a ``VideoFrame``.
func frame(_ buffer: CVPixelBuffer, time: CMTime, duration: CMTime) throws -> VideoFrame {
    guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA else {
        throw VideoDepthError.invalidFrame("expected a 32BGRA pixel buffer")
    }
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let (w, h) = (CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer))
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    guard let base = CVPixelBufferGetBaseAddress(buffer) else {
        throw VideoDepthError.invalidFrame("pixel buffer has no base address")
    }
    let data = Data(bytes: base, count: rowBytes * h)
    return VideoFrame(
        width: w, height: h, pixelFormat: .bgra8, bytesPerRow: rowBytes, pixels: data, presentationTime: time,
        duration: duration)
}
