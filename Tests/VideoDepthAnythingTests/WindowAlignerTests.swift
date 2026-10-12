import MLX
import Testing

@testable import VideoDepthAnything

@Suite(.enabled(if: Metallib.available)) struct WindowAlignerTests {
    @Test func windowScheduleMatchesReference() {
        #expect(Windowing.frameStep == 22)
        #expect(Windowing.alignKeyframes == [0, 12])
        #expect(Windowing.firstNewFrame(window: 0) == 0 && Windowing.endFrame(window: 0) == 32)
        #expect(Windowing.firstNewFrame(window: 1) == 32 && Windowing.endFrame(window: 1) == 54)
        #expect(Windowing.firstNewFrame(window: 2) == 54 && Windowing.endFrame(window: 2) == 76)
    }

    @Test func scaleAndShiftRecoversAnAffineMap() {
        let p = MLXRandom.uniform(low: 0.5, high: 3, [64, 48])
        let (s, t) = scaleAndShift(prediction: p, target: p * 2.5 + 0.75)
        #expect(abs(s - 2.5) < 1e-3 && abs(t - 0.75) < 1e-3)
        let constant = scaleAndShift(prediction: MLXArray.ones([4, 4]), target: MLXArray.ones([4, 4]))
        #expect(constant == (1, 0))  // singular system: identity
    }

    @Test func secondWindowIsMappedOntoTheFirstScale() {
        // Window 1 sees the same scene at half the scale and an offset of 1: alignment undoes it.
        let frame = MLXRandom.uniform(low: 1, high: 2, [8, 8])
        var aligner = WindowAligner(metric: false)
        aligner.add(Array(repeating: frame, count: 32))
        #expect(aligner.finalCount == 24)
        aligner.add(Array(repeating: frame * 0.5 + 1, count: 32))
        #expect(aligner.alignedCount == 54 && aligner.finalCount == 46)
        let out = aligner.take(upTo: 54)
        #expect(out.count == 54)
        for f in out { #expect(abs(f - frame).max().item(Float.self) < 1e-4) }
    }

    @Test func metricWindowsAreNotRescaled() {
        let frame = MLXArray.ones([4, 4])
        var aligner = WindowAligner(metric: true)
        aligner.add(Array(repeating: frame, count: 32))
        aligner.add(Array(repeating: frame * 2, count: 32))
        let out = aligner.take(upTo: 54)
        #expect(out[53].max().item(Float.self) == 2)
        // The overlap blends from the first window's value to the second's.
        #expect(out[24].max().item(Float.self) == 1 && out[31].max().item(Float.self) == 2)
    }

    @Test func blendWeightsRunFromZeroToOne() {
        let pre = Array(repeating: MLXArray.zeros([1]), count: 8)
        let post = Array(repeating: MLXArray.ones([1]), count: 8)
        let w = interpolateFrames(pre, post).map { $0.item(Float.self) }
        #expect(w.first == 0 && w.last == 1)
        #expect(abs(w[1] - 1.0 / 7) < 1e-6)
    }
}
