import CoreMedia
import Foundation

/// One decoded video frame, as 8-bit pixels in memory.
///
/// ``VideoFrameReader`` produces these from a file. Any other source (a camera, a test fixture, an
/// image sequence) can build them directly.
public struct VideoFrame: Sendable {
    /// Byte layout of ``VideoFrame/pixels``.
    public enum PixelFormat: Sendable, Equatable {
        /// 4 bytes per pixel, blue, green, red, alpha (`kCVPixelFormatType_32BGRA`). Alpha is ignored.
        case bgra8
        /// 3 bytes per pixel, red, green, blue.
        case rgb8
    }

    public let width: Int
    public let height: Int
    public let pixelFormat: PixelFormat
    /// Bytes per row, at least `width` times the pixel size.
    public let bytesPerRow: Int
    /// `height * bytesPerRow` bytes, top row first.
    public let pixels: Data
    /// Presentation time in the source.
    public let presentationTime: CMTime
    /// How long the frame is shown. With a reduced frame rate this covers the skipped frames too.
    public let duration: CMTime

    /// - Precondition: `pixels` holds at least `height * bytesPerRow` bytes and `bytesPerRow` fits one
    ///   row of `width` pixels.
    public init(
        width: Int, height: Int, pixelFormat: PixelFormat, bytesPerRow: Int? = nil, pixels: Data,
        presentationTime: CMTime = .invalid, duration: CMTime = .invalid
    ) {
        let rowBytes = bytesPerRow ?? width * pixelFormat.bytesPerPixel
        precondition(width > 0 && height > 0, "empty frame")
        precondition(rowBytes >= width * pixelFormat.bytesPerPixel, "bytesPerRow too small")
        precondition(pixels.count >= rowBytes * height, "pixel data too short")
        self.width = width
        self.height = height
        self.pixelFormat = pixelFormat
        self.bytesPerRow = rowBytes
        self.pixels = pixels
        self.presentationTime = presentationTime
        self.duration = duration
    }
}

extension VideoFrame.PixelFormat {
    var bytesPerPixel: Int {
        switch self {
        case .bgra8: 4
        case .rgb8: 3
        }
    }
}

/// Depth for one frame, at the frame's own resolution.
///
/// Values are relative inverse depth (disparity-like: larger is nearer) and non-negative. Their scale
/// is consistent across all frames of one job, but not between jobs. For the metric model they are
/// metric depth in meters.
public struct DepthFrame: Sendable {
    /// Position in the job's output, from 0.
    public let index: Int
    public let width: Int
    public let height: Int
    /// `height * width` values, row-major, top row first.
    public let values: [Float]
    /// The source frame's presentation time.
    public let presentationTime: CMTime
    /// The source frame's duration.
    public let duration: CMTime

    public init(
        index: Int, width: Int, height: Int, values: [Float], presentationTime: CMTime, duration: CMTime
    ) {
        precondition(values.count == width * height, "values must hold width * height entries")
        self.index = index
        self.width = width
        self.height = height
        self.values = values
        self.presentationTime = presentationTime
        self.duration = duration
    }

    /// The value at column `x`, row `y`.
    public subscript(x: Int, y: Int) -> Float {
        values[y * width + x]
    }

    /// Smallest and largest value.
    public var range: ClosedRange<Float> {
        DepthVideoWriter.minMax(values)
    }
}

/// How far a depth job has got.
public struct DepthProgress: Sendable, Equatable {
    /// Depth frames delivered so far.
    public let completedFrames: Int
    /// Frames the source is expected to produce, when it knows (``VideoFrameReader`` estimates it from
    /// the track's frame rate and duration).
    public let expectedFrames: Int?

    /// `completedFrames / expectedFrames`, clamped to 0...1, or `nil` without an estimate.
    public var fraction: Double? {
        guard let expectedFrames, expectedFrames > 0 else { return nil }
        return min(1, Double(completedFrames) / Double(expectedFrames))
    }
}

/// Settings of a depth job.
public struct DepthOptions: Sendable, Equatable {
    /// Short side of the network input in pixels, a multiple of 14 (default 518, the training
    /// resolution). Smaller is faster and uses less memory; depth is still returned at source size.
    public var inputSize: Int

    public init(inputSize: Int = 518) {
        self.inputSize = inputSize
    }

    func validate() throws {
        guard inputSize >= 14, inputSize % 14 == 0 else {
            throw VideoDepthError.invalidOptions("inputSize \(inputSize) must be a positive multiple of 14")
        }
    }
}
