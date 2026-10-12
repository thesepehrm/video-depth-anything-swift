import Testing

@testable import VideoDepthAnything

@Suite struct PreprocessTests {
    let pre = FramePreprocessor(inputSize: 518)

    @Test func targetSizeMatchesReference() {
        // Values from mlx-vlm's VideoDepthProcessor.target_size.
        #expect(pre.targetSize(height: 1080, width: 1920) == (518, 924))
        #expect(pre.targetSize(height: 272, width: 480) == (518, 910))
        #expect(pre.targetSize(height: 1920, width: 1080) == (924, 518))
        #expect(pre.targetSize(height: 518, width: 518) == (518, 518))
        #expect(pre.targetSize(height: 1080, width: 1440) == (518, 686))
        #expect(pre.targetSize(height: 84, width: 112) == (518, 686))
        // Wider than 16:9: the input size shrinks.
        #expect(pre.targetSize(height: 800, width: 1920) == (378, 910))
    }

    @Test func opencvCubicTapsSumToOne() {
        for (i, o) in [(1080, 518), (480, 910), (7, 3)] {
            let t = ResampleTaps.opencvCubic(inSize: i, outSize: o)
            #expect(t.indices.allSatisfy { $0 >= 0 && $0 < i })
            for row in 0..<o {
                let sum = t.weights[(row * 4)..<(row * 4 + 4)].reduce(0, +)
                #expect(abs(sum - 1) < 1e-5)
            }
        }
    }

    @Test func alignCornersHitsBothEnds() {
        let t = ResampleTaps.bilinearAlignCorners(inSize: 37, outSize: 1080)
        #expect(t.indices[0] == 0 && t.weights[0] == 1)
        let last = (1080 - 1) * 2
        #expect(t.indices[last] == 36 && t.weights[last] > 0.999)
    }

    @Test func optionsRejectNonMultiplesOf14() {
        #expect(throws: VideoDepthError.self) { try DepthOptions(inputSize: 500).validate() }
        #expect(throws: VideoDepthError.self) { try DepthOptions(inputSize: 0).validate() }
        #expect(throws: Never.self) { try DepthOptions(inputSize: 280).validate() }
    }
}
