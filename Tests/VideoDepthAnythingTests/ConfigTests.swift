import Foundation
import Testing

@testable import VideoDepthAnything

@Suite struct ConfigTests {
    @Test func convertedConfigDecodesToSmallPreset() throws {
        let json = #"""
            {"model_type": "video_depth_anything", "encoder": "vits", "metric": false,
             "source": {"repo": "depth-anything/Video-Depth-Anything-Small"}}
            """#
        let c = try JSONDecoder().decode(ModelConfig.self, from: Data(json.utf8))
        try c.validate()
        #expect(c.embedDim == 384 && c.depth == 12 && c.numHeads == 6)
        #expect(c.features == 64 && c.outChannels == [48, 96, 192, 384])
        #expect(c.intermediateLayerIdx == [2, 5, 8, 11])
        #expect(c.numFrames == 32 && c.numAttentionHeads == 8 && c.interpolateOffset == 0.1)
        #expect(!c.metric)
    }

    @Test func metricFlagIsRead() throws {
        let c = try JSONDecoder().decode(ModelConfig.self, from: Data(#"{"metric": true}"#.utf8))
        #expect(c.metric)
    }

    @Test(arguments: ["vitb", "vitl"]) func baseAndLargeAreRejected(encoder: String) throws {
        let c = try JSONDecoder().decode(ModelConfig.self, from: Data(#"{"encoder": "\#(encoder)"}"#.utf8))
        #expect(throws: VideoDepthError.self) { try c.validate() }
    }

    @Test func missingDirectoryThrowsWeightsNotFound() throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        #expect(throws: VideoDepthError.weightsNotFound(dir)) { try ModelConfig.load(from: dir) }
        #expect(throws: VideoDepthError.weightsNotFound(dir)) { try VideoDepthModel.load(directory: dir) }
    }
}
