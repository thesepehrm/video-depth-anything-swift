// Port of mlx_vlm/models/dinov2/dinov2.py::{PatchEmbed, Attention, Mlp, LayerScale, Block, DINOv2}
// (mlx-vlm 0.7.6), the subset Video Depth Anything uses: no registers, no masks, MLP FFN.
import Foundation
import MLX
import MLXFast
import MLXNN

/// 2D image to patch embedding: (B, H, W, C) -> (B, N, D).
final class PatchEmbed: Module {
    let patchSize: Int
    let numPatches: Int
    @ModuleInfo(key: "proj") var proj: Conv2d

    init(imgSize: Int, patchSize: Int, inChans: Int, embedDim: Int) {
        self.patchSize = patchSize
        numPatches = (imgSize / patchSize) * (imgSize / patchSize)
        _proj.wrappedValue = Conv2d(
            inputChannels: inChans, outputChannels: embedDim, kernelSize: .init(patchSize),
            stride: .init(patchSize))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        precondition(x.dim(1) % patchSize == 0 && x.dim(2) % patchSize == 0)
        let y = proj(x)  // (B, H/p, W/p, D)
        return y.reshaped(y.dim(0), -1, y.dim(-1))
    }
}

final class DINOAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    init(dim: Int, numHeads: Int, qkvBias: Bool = true) {
        self.numHeads = numHeads
        headDim = dim / numHeads
        scale = pow(Float(headDim), -0.5)
        _qkv.wrappedValue = Linear(dim, dim * 3, bias: qkvBias)
        _proj.wrappedValue = Linear(dim, dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, N, C) = (x.dim(0), x.dim(1), x.dim(2))
        let qkv = qkv(x).reshaped(B, N, 3, numHeads, headDim).transposed(2, 0, 3, 1, 4)
        let (q, k, v) = (qkv[0], qkv[1], qkv[2])
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        return proj(out.transposed(0, 2, 1, 3).reshaped(B, N, C))
    }
}

final class DINOMlp: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(dim: Int, hiddenDim: Int) {
        _fc1.wrappedValue = Linear(dim, hiddenDim, bias: true)
        _fc2.wrappedValue = Linear(hiddenDim, dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

final class LayerScale: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(dim: Int) {
        _gamma.wrappedValue = MLXArray.ones([dim])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * gamma
    }
}

final class DINOBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: DINOAttention
    @ModuleInfo(key: "ls1") var ls1: LayerScale
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: DINOMlp
    @ModuleInfo(key: "ls2") var ls2: LayerScale

    init(_ config: ModelConfig) {
        let dim = config.embedDim
        let hidden = Int(Float(dim) * config.mlpRatio)
        _norm1.wrappedValue = LayerNorm(dimensions: dim, eps: config.layerNormEps)
        _attn.wrappedValue = DINOAttention(dim: dim, numHeads: config.numHeads)
        _ls1.wrappedValue = LayerScale(dim: dim)
        _norm2.wrappedValue = LayerNorm(dimensions: dim, eps: config.layerNormEps)
        _mlp.wrappedValue = DINOMlp(dim: dim, hiddenDim: hidden)
        _ls2.wrappedValue = LayerScale(dim: dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x + ls1(attn(norm1(x)))
        x = x + ls2(mlp(norm2(x)))
        return x
    }
}

final class DINOv2: Module {
    let embedDim: Int
    let patchSize: Int
    let interpolateOffset: Float

    @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray
    @ParameterInfo(key: "mask_token") var maskToken: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [DINOBlock]
    @ModuleInfo(key: "norm") var norm: LayerNorm

    /// Interpolated position embeddings by input pixel size. A class, so module reflection skips it.
    private let posCache = PositionCache()

    init(_ config: ModelConfig) {
        embedDim = config.embedDim
        patchSize = config.patchSize
        interpolateOffset = config.interpolateOffset
        let patchEmbed = PatchEmbed(
            imgSize: config.imgSize, patchSize: config.patchSize, inChans: 3, embedDim: config.embedDim)
        _patchEmbed.wrappedValue = patchEmbed
        _clsToken.wrappedValue = MLXArray.zeros([1, 1, config.embedDim])
        _posEmbed.wrappedValue = MLXArray.zeros([1, patchEmbed.numPatches + 1, config.embedDim])
        _maskToken.wrappedValue = MLXArray.zeros([1, config.embedDim])
        _blocks.wrappedValue = (0..<config.depth).map { _ in DINOBlock(config) }
        _norm.wrappedValue = LayerNorm(dimensions: config.embedDim, eps: config.layerNormEps)
    }

    /// The position table for an `h`×`w` pixel input with `npatch` patch tokens.
    ///
    /// Uses the original repository's scale-factor bicubic resize (`interpolate_offset` 0.1). Cached
    /// per size: the weights do not change after loading.
    func interpolatePosEncoding(npatch: Int, h: Int, w: Int, dtype: DType) -> MLXArray {
        let numPatches = posEmbed.dim(1) - 1
        if npatch == numPatches && w == h { return posEmbed }
        if let hit = posCache.tables[[h, w]] { return hit.asType(dtype) }
        let dim = posEmbed.dim(-1)
        let classPosEmbed = posEmbed[0..., ..<1]
        var patchPosEmbed = posEmbed[0..., 1...]
        let sqrtN = Double(numPatches).squareRoot()
        let side = Int(sqrtN)
        patchPosEmbed = patchPosEmbed.reshaped(1, side, side, dim)
        precondition(interpolateOffset != 0, "only the scale-factor (offset) path is ported")
        // Small offset to avoid floating point error in the interpolation. The reference derives w0
        // from the pixel width and h0 from the height, and scales (sy, sx) accordingly.
        let w0 = Double(Float(w / patchSize) + interpolateOffset)
        let h0 = Double(Float(h / patchSize) + interpolateOffset)
        let (sx, sy) = (w0 / sqrtN, h0 / sqrtN)
        patchPosEmbed = resizeBicubic(patchPosEmbed.asType(.float32), scaleFactor: (sy, sx))
        patchPosEmbed = patchPosEmbed.reshaped(1, -1, dim)
        let out = concatenated([classPosEmbed, patchPosEmbed], axis: 1)
        eval(out)
        posCache.tables[[h, w]] = out
        return out.asType(dtype)
    }

    func prepareTokens(_ x: MLXArray) -> MLXArray {
        let (B, H, W) = (x.dim(0), x.dim(1), x.dim(2))
        var x = patchEmbed(x)
        let cls = broadcast(clsToken, to: [B, 1, embedDim])
        x = concatenated([cls, x], axis: 1)
        return x + interpolatePosEncoding(npatch: x.dim(1) - 1, h: H, w: W, dtype: x.dtype)
    }

    /// Runs the backbone and returns the normed patch tokens and cls token after each block in
    /// `indices`.
    func intermediateLayers(_ x: MLXArray, indices: [Int]) -> [(patch: MLXArray, cls: MLXArray)] {
        var x = prepareTokens(x)
        var outputs: [(patch: MLXArray, cls: MLXArray)] = []
        for (i, block) in blocks.enumerated() {
            x = block(x)
            if indices.contains(i) {
                let out = norm(x)
                outputs.append((out[0..., 1...], out[0..., 0]))
            }
        }
        precondition(outputs.count == indices.count)
        return outputs
    }
}

/// Holder for ``DINOv2``'s interpolated position tables. Only touched on the model's actor.
final class PositionCache {
    var tables: [[Int]: MLXArray] = [:]
}
