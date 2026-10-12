// Port of mlx_vlm/models/video_depth_anything/config.py (mlx-vlm 0.7.6), Small ("vits") preset only.
//
// Every field defaults to the Python dataclass default after `__post_init__` resolves the encoder
// preset. A key that is missing or `null` in the JSON keeps its default, and unknown keys (such as the
// `source` block scripts/convert.py writes) are ignored, which matches `BaseModelConfig.from_dict`.
import Foundation

extension KeyedDecodingContainer {
    /// Decodes `key`, falling back to `fallback` when the key is absent or `null`.
    fileprivate func decode<T: Decodable>(_ key: Key, or fallback: T) throws -> T {
        try decodeIfPresent(T.self, forKey: key) ?? fallback
    }
}

struct ModelConfig: Codable, Sendable, Equatable {
    var modelType = "video_depth_anything"
    /// Only `"vits"` (Video Depth Anything Small) is supported; Base and Large are CC-BY-NC.
    var encoder = "vits"
    /// Metric (absolute) depth head: windows are not scale/shift aligned.
    var metric = false

    // DPT head
    var features = 64
    var outChannels = [48, 96, 192, 384]
    var useBn = false
    var useClstoken = false

    // Temporal module
    var numFrames = 32
    var pe = "ape"
    var numAttentionHeads = 8
    var numTransformerBlock = 1
    var numAttentionBlocks = 2
    var normNumGroups = 32

    // DINOv2 backbone (DINOV2_PRESETS["vits14"])
    var embedDim = 384
    var depth = 12
    var numHeads = 6
    var ffn = "mlp"
    var intermediateLayerIdx = [2, 5, 8, 11]
    var imgSize = 518
    var patchSize = 14
    var mlpRatio: Float = 4.0
    var layerNormEps: Float = 1e-6
    var interpolateOffset: Float = 0.1
    var interpolateAntialias = false

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case encoder
        case metric
        case features
        case outChannels = "out_channels"
        case useBn = "use_bn"
        case useClstoken = "use_clstoken"
        case numFrames = "num_frames"
        case pe
        case numAttentionHeads = "num_attention_heads"
        case numTransformerBlock = "num_transformer_block"
        case numAttentionBlocks = "num_attention_blocks"
        case normNumGroups = "norm_num_groups"
        case embedDim = "embed_dim"
        case depth
        case numHeads = "num_heads"
        case ffn
        case intermediateLayerIdx = "intermediate_layer_idx"
        case imgSize = "img_size"
        case patchSize = "patch_size"
        case mlpRatio = "mlp_ratio"
        case layerNormEps = "layer_norm_eps"
        case interpolateOffset = "interpolate_offset"
        case interpolateAntialias = "interpolate_antialias"
    }

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(.modelType, or: modelType)
        encoder = try c.decode(.encoder, or: encoder)
        metric = try c.decode(.metric, or: metric)
        features = try c.decode(.features, or: features)
        outChannels = try c.decode(.outChannels, or: outChannels)
        useBn = try c.decode(.useBn, or: useBn)
        useClstoken = try c.decode(.useClstoken, or: useClstoken)
        numFrames = try c.decode(.numFrames, or: numFrames)
        pe = try c.decode(.pe, or: pe)
        numAttentionHeads = try c.decode(.numAttentionHeads, or: numAttentionHeads)
        numTransformerBlock = try c.decode(.numTransformerBlock, or: numTransformerBlock)
        numAttentionBlocks = try c.decode(.numAttentionBlocks, or: numAttentionBlocks)
        normNumGroups = try c.decode(.normNumGroups, or: normNumGroups)
        embedDim = try c.decode(.embedDim, or: embedDim)
        depth = try c.decode(.depth, or: depth)
        numHeads = try c.decode(.numHeads, or: numHeads)
        ffn = try c.decode(.ffn, or: ffn)
        intermediateLayerIdx = try c.decode(.intermediateLayerIdx, or: intermediateLayerIdx)
        imgSize = try c.decode(.imgSize, or: imgSize)
        patchSize = try c.decode(.patchSize, or: patchSize)
        mlpRatio = try c.decode(.mlpRatio, or: mlpRatio)
        layerNormEps = try c.decode(.layerNormEps, or: layerNormEps)
        interpolateOffset = try c.decode(.interpolateOffset, or: interpolateOffset)
        interpolateAntialias = try c.decode(.interpolateAntialias, or: interpolateAntialias)
    }

    /// Reads `<directory>/config.json` and checks that it describes a supported model.
    ///
    /// - Throws: ``VideoDepthError/weightsNotFound(_:)`` when the file is missing,
    ///   ``VideoDepthError/invalidConfig(_:)`` when it cannot be decoded, and
    ///   ``VideoDepthError/unsupportedModel(_:)`` for anything but the Small encoder.
    static func load(from directory: URL) throws -> ModelConfig {
        let file = directory.appending(path: "config.json")
        guard let data = FileManager.default.contents(atPath: file.path) else {
            throw VideoDepthError.weightsNotFound(directory)
        }
        let config: ModelConfig
        do {
            config = try JSONDecoder().decode(ModelConfig.self, from: data)
        } catch {
            throw VideoDepthError.invalidConfig("config.json: \(error)")
        }
        try config.validate()
        return config
    }

    func validate() throws {
        guard modelType == "video_depth_anything" else {
            throw VideoDepthError.unsupportedModel("model_type \(modelType)")
        }
        guard encoder == "vits" else {
            throw VideoDepthError.unsupportedModel(
                "encoder \(encoder): only the Small (vits) checkpoint is supported")
        }
        guard pe == "ape", ffn == "mlp", !useBn, !useClstoken else {
            throw VideoDepthError.unsupportedModel(
                "pe \(pe), ffn \(ffn), use_bn \(useBn), use_clstoken \(useClstoken)")
        }
        guard outChannels.count == 4, intermediateLayerIdx.count == 4, patchSize > 0 else {
            throw VideoDepthError.invalidConfig("out_channels and intermediate_layer_idx need 4 entries")
        }
    }
}
