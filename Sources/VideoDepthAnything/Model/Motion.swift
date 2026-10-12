// Port of mlx_vlm/models/video_depth_anything/motion.py (mlx-vlm 0.7.6): the AnimateDiff-style
// temporal motion module.
import Foundation
import MLX
import MLXFast
import MLXNN

/// Adds the checkpoint's sinusoidal position table (`pe`, (1, max_len, d_model)).
final class PositionalEncoding: Module {
    @ParameterInfo(key: "pe") var pe: MLXArray

    init(dModel: Int, maxLen: Int) {
        _pe.wrappedValue = sinusoidalTable(dModel: dModel, maxLen: maxLen)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + pe[0..., ..<x.dim(1)].asType(x.dtype)
    }
}

/// Sinusoidal position table of shape (1, max_len, d_model). The checkpoint stores the same table, so
/// this is only the initial value.
func sinusoidalTable(dModel: Int, maxLen: Int) -> MLXArray {
    let position = expandedDimensions(MLXArray(0..<maxLen).asType(.float32), axis: 1)
    let divTerm = exp(
        MLXArray(stride(from: 0, to: dModel, by: 2)).asType(.float32) * Float(-log(10000.0) / Double(dModel)))
    let angles = position * divTerm  // (max_len, d_model / 2)
    // Interleave: even channels sin, odd channels cos.
    return stacked([sin(angles), cos(angles)], axis: -1).reshaped(1, maxLen, dModel)
}

/// Self-attention along the temporal axis.
final class TemporalAttention: Module {
    let heads: Int
    let scale: Float

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    /// `[Linear, Dropout(0)]` in Python; only index 0 carries weights.
    @ModuleInfo(key: "to_out") var toOut: [UnaryLayer]
    @ModuleInfo(key: "pos_encoder") var posEncoder: PositionalEncoding

    init(queryDim: Int, heads: Int, dimHead: Int, temporalMaxLen: Int) {
        let innerDim = heads * dimHead
        self.heads = heads
        scale = pow(Float(dimHead), -0.5)
        _toQ.wrappedValue = Linear(queryDim, innerDim, bias: false)
        _toK.wrappedValue = Linear(queryDim, innerDim, bias: false)
        _toV.wrappedValue = Linear(queryDim, innerDim, bias: false)
        _toOut.wrappedValue = [Linear(innerDim, queryDim, bias: true), Dropout(p: 0)]
        _posEncoder.wrappedValue = PositionalEncoding(dModel: queryDim, maxLen: temporalMaxLen)
    }

    /// - Parameter hiddenStates: (B*T, D, C) with D spatial positions.
    func callAsFunction(_ hiddenStates: MLXArray, videoLength: Int) -> MLXArray {
        let (BT, D, C) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))
        let batch = BT / videoLength
        // (batch*T, D, C) -> (batch*D, T, C)
        var x = hiddenStates.reshaped(batch, videoLength, D, C).transposed(0, 2, 1, 3)
        x = x.reshaped(batch * D, videoLength, C)

        // The position table is added before the projections; queries use the encoded states.
        x = posEncoder(x)

        let q = toQ(x)
        let k = toK(x)
        let v = toV(x)

        func splitHeads(_ t: MLXArray) -> MLXArray {
            t.reshaped(batch * D, videoLength, heads, C / heads).transposed(0, 2, 1, 3)
        }

        var out = MLXFast.scaledDotProductAttention(
            queries: splitHeads(q), keys: splitHeads(k), values: splitHeads(v), scale: scale, mask: nil)
        out = out.transposed(0, 2, 1, 3).reshaped(batch * D, videoLength, C)
        out = toOut[0](out)

        // (batch*D, T, C) -> (batch*T, D, C)
        out = out.reshaped(batch, D, videoLength, C).transposed(0, 2, 1, 3)
        return out.reshaped(BT, D, C)
    }
}

final class GEGLU: Module, UnaryLayer {
    @ModuleInfo(key: "proj") var proj: Linear

    init(dimIn: Int, dimOut: Int) {
        _proj.wrappedValue = Linear(dimIn, dimOut * 2, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(proj(x), parts: 2, axis: -1)
        return parts[0] * gelu(parts[1])
    }
}

final class FeedForward: Module {
    /// `[GEGLU, Dropout(0), Linear]`: keys `net.0` and `net.2`.
    @ModuleInfo(key: "net") var net: [UnaryLayer]

    init(dim: Int, mult: Int = 4) {
        let innerDim = dim * mult
        _net.wrappedValue = [GEGLU(dimIn: dim, dimOut: innerDim), Dropout(p: 0), Linear(innerDim, dim)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for module in net {
            x = module(x)
        }
        return x
    }
}

final class TemporalTransformerBlock: Module {
    @ModuleInfo(key: "attention_blocks") var attentionBlocks: [TemporalAttention]
    @ModuleInfo(key: "norms") var norms: [LayerNorm]
    @ModuleInfo(key: "ff") var ff: FeedForward
    @ModuleInfo(key: "ff_norm") var ffNorm: LayerNorm

    init(
        dim: Int, numAttentionHeads: Int, attentionHeadDim: Int, numAttentionBlocks: Int, temporalMaxLen: Int
    ) {
        _attentionBlocks.wrappedValue = (0..<numAttentionBlocks).map { _ in
            TemporalAttention(
                queryDim: dim, heads: numAttentionHeads, dimHead: attentionHeadDim,
                temporalMaxLen: temporalMaxLen)
        }
        _norms.wrappedValue = (0..<numAttentionBlocks).map { _ in LayerNorm(dimensions: dim) }
        _ff.wrappedValue = FeedForward(dim: dim)
        _ffNorm.wrappedValue = LayerNorm(dimensions: dim)
    }

    func callAsFunction(_ hiddenStates: MLXArray, videoLength: Int) -> MLXArray {
        var h = hiddenStates
        for (attentionBlock, norm) in zip(attentionBlocks, norms) {
            h = attentionBlock(norm(h), videoLength: videoLength) + h
        }
        return ff(ffNorm(h)) + h
    }
}

final class TemporalTransformer3DModel: Module {
    @ModuleInfo(key: "norm") var norm: GroupNorm
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [TemporalTransformerBlock]
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(
        inChannels: Int, numAttentionHeads: Int, attentionHeadDim: Int, numLayers: Int,
        numAttentionBlocks: Int, normNumGroups: Int, temporalMaxLen: Int
    ) {
        let innerDim = numAttentionHeads * attentionHeadDim
        _norm.wrappedValue = GroupNorm(
            groupCount: normNumGroups, dimensions: inChannels, eps: 1e-6, pytorchCompatible: true)
        _projIn.wrappedValue = Linear(inChannels, innerDim)
        _transformerBlocks.wrappedValue = (0..<numLayers).map { _ in
            TemporalTransformerBlock(
                dim: innerDim, numAttentionHeads: numAttentionHeads, attentionHeadDim: attentionHeadDim,
                numAttentionBlocks: numAttentionBlocks, temporalMaxLen: temporalMaxLen)
        }
        _projOut.wrappedValue = Linear(innerDim, inChannels)
    }

    /// - Parameter hiddenStates: (B, T, H, W, C) channel-last.
    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let (B, T, H, W, C) = (
            hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2), hiddenStates.dim(3),
            hiddenStates.dim(4)
        )
        var x = hiddenStates.reshaped(B * T, H, W, C)
        let residual = x

        x = norm(x)
        x = x.reshaped(B * T, H * W, C)
        x = projIn(x)

        for block in transformerBlocks {
            x = block(x, videoLength: T)
        }

        x = projOut(x)
        x = x.reshaped(B * T, H, W, C)
        return (x + residual).reshaped(B, T, H, W, C)
    }
}

final class TemporalModule: Module {
    @ModuleInfo(key: "temporal_transformer") var temporalTransformer: TemporalTransformer3DModel

    init(inChannels: Int, _ config: ModelConfig) {
        _temporalTransformer.wrappedValue = TemporalTransformer3DModel(
            inChannels: inChannels, numAttentionHeads: config.numAttentionHeads,
            attentionHeadDim: inChannels / config.numAttentionHeads, numLayers: config.numTransformerBlock,
            numAttentionBlocks: config.numAttentionBlocks, normNumGroups: config.normNumGroups,
            temporalMaxLen: config.numFrames)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        temporalTransformer(x)
    }
}
