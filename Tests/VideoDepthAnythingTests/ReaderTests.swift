import CoreMedia
import Foundation
import Testing

@testable import VideoDepthAnything

@Suite struct ReaderTests {
    @Test func readsEveryFrameWithSourceTiming() async throws {
        let reader = try await VideoFrameReader(url: Fixtures.clipURL)
        #expect(reader.size == CGSize(width: 480, height: 272))
        #expect(reader.expectedFrameCount == 72)
        var times: [CMTime] = []
        for try await frame in reader {
            #expect(frame.width == 480 && frame.height == 272 && frame.pixelFormat == .bgra8)
            times.append(frame.presentationTime)
        }
        #expect(times.count == 72)
        #expect(times[0] == .zero)
        #expect(times[1].seconds.isApproximately(1.0 / 24))
    }

    @Test func targetFrameRateKeepsEveryNthFrame() async throws {
        let reader = try await VideoFrameReader(url: Fixtures.clipURL, frameRate: .target(12))
        #expect(reader.stride == 2 && reader.expectedFrameCount == 36)
        var frames: [VideoFrame] = []
        for try await frame in reader { frames.append(frame) }
        #expect(frames.count == 36)
        #expect(frames[1].presentationTime.seconds.isApproximately(2.0 / 24))
        #expect(frames[0].duration.seconds.isApproximately(2.0 / 24))
    }

    @Test func timeRangeLimitsTheFrames() async throws {
        let range = CMTimeRange(
            start: CMTime(value: 1, timescale: 1), duration: CMTime(value: 1, timescale: 2))
        let reader = try await VideoFrameReader(url: Fixtures.clipURL, timeRange: range)
        var count = 0
        var first: CMTime?
        for try await frame in reader {
            first = first ?? frame.presentationTime
            count += 1
        }
        #expect(reader.expectedFrameCount == 12)
        #expect(count == 12)
        #expect(first?.seconds.isApproximately(1) == true)
    }

    @Test func missingFileThrows() async {
        await #expect(throws: VideoDepthError.self) {
            _ = try await VideoFrameReader(url: URL(fileURLWithPath: "/nonexistent/clip.mov"))
        }
    }
}

extension Double {
    func isApproximately(_ other: Double, tolerance: Double = 1e-6) -> Bool { abs(self - other) <= tolerance }
}
