import AVFoundation
import CoreMedia
import Foundation
import Testing

@testable import VideoDepthAnything

@Suite struct WriterTests {
    func ramp(index: Int, width: Int = 64, height: Int = 48) -> DepthFrame {
        let values = (0..<(width * height)).map { Float($0 % width) / Float(width - 1) * 10 }
        return DepthFrame(
            index: index, width: width, height: height, values: values,
            presentationTime: CMTime(value: CMTimeValue(index), timescale: 24),
            duration: CMTime(value: 1, timescale: 24))
    }

    func tempURL() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "vda-\(UUID().uuidString).mov")
    }

    @Test(arguments: [
        (DepthColormap.grayscale, DepthVideoOptions.Codec.h264), (.inferno, .hevc), (.turbo, .h264),
    ])
    func writesOneVideoFramePerDepthFrame(colormap: DepthColormap, codec: DepthVideoOptions.Codec)
        async throws
    {
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let frames = (0..<10).map { ramp(index: $0) }
        let stream = AsyncStream { c in
            for f in frames { c.yield(f) }
            c.finish()
        }
        let written = try await DepthVideoWriter.write(
            stream, to: url, options: DepthVideoOptions(codec: codec, colormap: colormap))
        #expect(written == 10)

        let asset = AVURLAsset(url: url)
        let track = try #require(try await asset.loadTracks(withMediaType: .video).first)
        let (size, duration) = try await (track.load(.naturalSize), asset.load(.duration))
        #expect(size == CGSize(width: 64, height: 48))
        #expect(duration.seconds.isApproximately(10.0 / 24, tolerance: 1e-3))
        var count = 0
        for try await _ in try await VideoFrameReader(url: url) { count += 1 }
        #expect(count == 10)
    }

    @Test func grayscaleMapsNearToWhite() async throws {
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let writer = try DepthVideoWriter(
            url: url, width: 64, height: 48, options: DepthVideoOptions(codec: .h264, colormap: .grayscale))
        try await writer.append(ramp(index: 0))
        try await writer.finish()
        var frames: [VideoFrame] = []
        for try await f in try await VideoFrameReader(url: url) { frames.append(f) }
        let f = try #require(frames.first)
        let row = 24 * f.bytesPerRow
        let left = f.pixels[row + 2 * 4 + 1]  // green of x = 2
        let right = f.pixels[row + 61 * 4 + 1]  // green of x = 61
        #expect(left < 30 && right > 225)
    }

    @Test func mismatchedSizeThrows() async throws {
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let writer = try DepthVideoWriter(url: url, width: 32, height: 32)
        await #expect(throws: VideoDepthError.self) { try await writer.append(ramp(index: 0)) }
        await writer.cancel()
    }
}
