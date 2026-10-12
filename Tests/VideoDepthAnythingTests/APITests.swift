import Foundation
import MLX
import Testing

@testable import VideoDepthAnything

/// Job behaviour on the real model: cancellation, unload, short clips, bad input.
@Suite(.enabled(if: Fixtures.weightsAvailable), .serialized)
struct APITests {
    func frames(_ n: Int, width: Int = 112, height: Int = 84) -> ArrayFrameSource {
        let rgb = MLXRandom.randInt(low: 0, high: 256, [n, height, width, 3]).asType(.uint8)
        return ArrayFrameSource(rgb: rgb)
    }

    @Test(arguments: [1, 5, 22, 32, 33, 54, 55]) func everyFrameGetsDepth(count: Int) async throws {
        let model = try await SharedModel.get()
        var indices: [Int] = []
        for try await f in model.depth(of: frames(count), options: DepthOptions(inputSize: 112)) {
            #expect(f.width == 112 && f.height == 84 && f.values.allSatisfy { $0 >= 0 && $0.isFinite })
            indices.append(f.index)
        }
        #expect(indices == Array(0..<count))
    }

    @Test func emptySourceYieldsNothing() async throws {
        let model = try await SharedModel.get()
        var n = 0
        for try await _ in model.depth(of: ArrayFrameSource(frames: [])) { n += 1 }
        #expect(n == 0)
    }

    @Test func frameSizeChangeThrows() async throws {
        let model = try await SharedModel.get()
        let mixed = ArrayFrameSource(frames: frames(3).frames + frames(3, width: 98).frames)
        await #expect(throws: VideoDepthError.self) {
            for try await _ in model.depth(of: mixed, options: DepthOptions(inputSize: 112)) {}
        }
    }

    @Test func invalidInputSizeThrows() async throws {
        let model = try await SharedModel.get()
        await #expect(
            throws: VideoDepthError.invalidOptions("inputSize 100 must be a positive multiple of 14")
        ) {
            for try await _ in model.depth(of: frames(2), options: DepthOptions(inputSize: 100)) {}
        }
    }

    @Test func cancellationStopsTheJob() async throws {
        let model = try await SharedModel.get()
        let source = frames(80)
        let task = Task {
            var n = 0
            for try await _ in model.depth(of: source, options: DepthOptions(inputSize: 112)) {
                n += 1
                if n == 3 { withUnsafeCurrentTask { $0?.cancel() } }
            }
            return n
        }
        await #expect(throws: CancellationError.self) { _ = try await task.value }
    }

    @Test func unloadFailsLaterJobs() async throws {
        let model = try VideoDepthModel.load(directory: Fixtures.weightsURL)
        #expect(await model.isLoaded)
        await model.unload()
        #expect(await !model.isLoaded)
        await #expect(throws: VideoDepthError.unloaded) {
            for try await _ in model.depth(of: frames(2), options: DepthOptions(inputSize: 112)) {}
        }
    }
}
