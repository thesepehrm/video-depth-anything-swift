import Foundation
import MLX
import Testing

@testable import VideoDepthAnything

/// Layer-by-layer parity with mlx-vlm on the fixtures from scripts/reference.py.
@Suite(.enabled(if: Fixtures.available), .serialized)
struct ModelParityTests {
    @Test func preprocessingMatchesOpenCV() async throws {
        let frames = try Fixtures.array("frames.npy")
        let golden = try Fixtures.array("pixels0.npy")
        let pre = FramePreprocessor(inputSize: 518)
        let taps = pre.taps(height: frames.dim(1), width: frames.dim(2))
        let pixels = pre(frames[0], taps: taps)
        // cv2 resizes in float32 with SIMD; the kernel sums the same taps in the same order.
        assertClose(pixels, golden, rtol: 1e-4, atol: 1e-4, "pixels0")
    }

    @Test func layersMatchReference() async throws {
        let model = try await SharedModel.get()
        let golden = try Fixtures.arrays("features.safetensors")
        let input = golden["input"]!
        let results = try await model.debugLayers(UncheckedArray(input))
        assertClose(results.tokens, golden["tokens"]!, rtol: 1e-4, atol: 1e-4, "tokens")
        for i in 0..<4 {
            // GPU matmuls: allow a little drift through the blocks.
            assertClose(results.layers[i], golden["layer\(i)"]!, rtol: 1e-3, atol: 1e-3, "layer\(i)")
        }
        let headRel = meanRelativeError(results.head, golden["head"]!)
        let depthRel = meanRelativeError(results.depth, golden["depth"]!)
        print("parity: head mean rel \(headRel), depth mean rel \(depthRel)")
        #expect(headRel < 1e-3)
        #expect(depthRel < 1e-3)
    }

    @Test func firstWindowMatchesReference() async throws {
        let model = try await SharedModel.get()
        let frames = try Fixtures.array("frames.npy")
        let golden = try Fixtures.array("window0.npy")
        let pre = FramePreprocessor(inputSize: 518)
        let taps = pre.taps(height: frames.dim(1), width: frames.dim(2))
        let input = stacked((0..<32).map { pre(frames[$0], taps: taps) })
        let depth = try await model.forward(UncheckedArray(expandedDimensions(input, axis: 0))).array[0]
        var worst: Float = 0
        for t in 0..<32 { worst = max(worst, meanRelativeError(depth[t], golden[t])) }
        print("parity: window0 worst per-frame mean rel \(worst)")
        #expect(worst < 1e-3)
    }
}
