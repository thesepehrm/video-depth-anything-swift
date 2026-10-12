// Port of mlx_vlm/models/video_depth_anything/dpt.py::{ResidualConvUnit, FeatureFusionBlock, Scratch,
// DPTHeadTemporal} (mlx-vlm 0.7.6)
import Foundation
import MLX
import MLXNN

final class ResidualConvUnit: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv2") var conv2: Conv2d

    init(features: Int) {
        _conv1.wrappedValue = Conv2d(
            inputChannels: features, outputChannels: features, kernelSize: 3, stride: 1, padding: 1)
        _conv2.wrappedValue = Conv2d(
            inputChannels: features, outputChannels: features, kernelSize: 3, stride: 1, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv1(relu(x))
        out = conv2(relu(out))
        return out + x
    }
}

final class FeatureFusionBlock: Module {
    @ModuleInfo(key: "out_conv") var outConv: Conv2d
    // Checkpoint spelling ("resConfUnit").
    @ModuleInfo(key: "resConfUnit1") var resConfUnit1: ResidualConvUnit
    @ModuleInfo(key: "resConfUnit2") var resConfUnit2: ResidualConvUnit

    init(features: Int) {
        _outConv.wrappedValue = Conv2d(
            inputChannels: features, outputChannels: features, kernelSize: 1, stride: 1, padding: 0)
        _resConfUnit1.wrappedValue = ResidualConvUnit(features: features)
        _resConfUnit2.wrappedValue = ResidualConvUnit(features: features)
    }

    /// - Parameter size: output (h, w); `nil` upsamples by 2.
    func callAsFunction(_ x: MLXArray, _ res: MLXArray? = nil, size: (Int, Int)? = nil) -> MLXArray {
        var output = x
        if let res {
            output = output + resConfUnit1(res)
        }
        output = resConfUnit2(output)
        let (h, w) = size ?? (output.dim(1) * 2, output.dim(2) * 2)
        output = upsampleBilinear(output, h: h, w: w)
        return outConv(output)
    }
}

/// Container matching the checkpoint's `head.scratch.*` keys.
final class Scratch: Module {
    @ModuleInfo(key: "layer1_rn") var layer1Rn: Conv2d
    @ModuleInfo(key: "layer2_rn") var layer2Rn: Conv2d
    @ModuleInfo(key: "layer3_rn") var layer3Rn: Conv2d
    @ModuleInfo(key: "layer4_rn") var layer4Rn: Conv2d
    @ModuleInfo(key: "refinenet1") var refinenet1: FeatureFusionBlock
    @ModuleInfo(key: "refinenet2") var refinenet2: FeatureFusionBlock
    @ModuleInfo(key: "refinenet3") var refinenet3: FeatureFusionBlock
    @ModuleInfo(key: "refinenet4") var refinenet4: FeatureFusionBlock
    @ModuleInfo(key: "output_conv1") var outputConv1: Conv2d
    /// `[Conv2d, ReLU, Conv2d, ReLU]`: keys `output_conv2.0` and `output_conv2.2`.
    @ModuleInfo(key: "output_conv2") var outputConv2: [UnaryLayer]

    init(outChannels: [Int], features: Int) {
        func rn(_ c: Int) -> Conv2d {
            Conv2d(
                inputChannels: c, outputChannels: features, kernelSize: 3, stride: 1, padding: 1, bias: false)
        }
        _layer1Rn.wrappedValue = rn(outChannels[0])
        _layer2Rn.wrappedValue = rn(outChannels[1])
        _layer3Rn.wrappedValue = rn(outChannels[2])
        _layer4Rn.wrappedValue = rn(outChannels[3])
        _refinenet1.wrappedValue = FeatureFusionBlock(features: features)
        _refinenet2.wrappedValue = FeatureFusionBlock(features: features)
        _refinenet3.wrappedValue = FeatureFusionBlock(features: features)
        _refinenet4.wrappedValue = FeatureFusionBlock(features: features)
        _outputConv1.wrappedValue = Conv2d(
            inputChannels: features, outputChannels: features / 2, kernelSize: 3, stride: 1, padding: 1)
        _outputConv2.wrappedValue = [
            Conv2d(inputChannels: features / 2, outputChannels: 32, kernelSize: 3, stride: 1, padding: 1),
            ReLU(),
            Conv2d(inputChannels: 32, outputChannels: 1, kernelSize: 1, stride: 1, padding: 0),
            ReLU(),
        ]
    }
}

final class DPTHeadTemporal: Module {
    let patchSize: Int

    @ModuleInfo(key: "projects") var projects: [Conv2d]
    /// `[ConvTranspose2d, ConvTranspose2d, Identity, Conv2d]`.
    @ModuleInfo(key: "resize_layers") var resizeLayers: [UnaryLayer]
    @ModuleInfo(key: "scratch") var scratch: Scratch
    @ModuleInfo(key: "motion_modules") var motionModules: [TemporalModule]

    init(_ config: ModelConfig) {
        patchSize = config.patchSize
        let inChannels = config.embedDim
        let features = config.features
        let out = config.outChannels

        _projects.wrappedValue = out.map {
            Conv2d(inputChannels: inChannels, outputChannels: $0, kernelSize: 1, stride: 1, padding: 0)
        }
        _resizeLayers.wrappedValue = [
            ConvTransposed2d(
                inputChannels: out[0], outputChannels: out[0], kernelSize: 4, stride: 4, padding: 0),
            ConvTransposed2d(
                inputChannels: out[1], outputChannels: out[1], kernelSize: 2, stride: 2, padding: 0),
            Identity(),
            Conv2d(inputChannels: out[3], outputChannels: out[3], kernelSize: 3, stride: 2, padding: 1),
        ]
        _scratch.wrappedValue = Scratch(outChannels: out, features: features)
        _motionModules.wrappedValue = [
            TemporalModule(inChannels: out[2], config),
            TemporalModule(inChannels: out[3], config),
            TemporalModule(inChannels: features, config),
            TemporalModule(inChannels: features, config),
        ]
    }

    /// - Parameters:
    ///   - outFeatures: the backbone's four (patch tokens (B*T, N, C), cls token) pairs.
    ///   - microBatchSize: frames per chunk of the full-resolution output stage. Chunks are evaluated
    ///     one at a time to bound memory, as in the reference.
    /// - Returns: (B*T, patch_h * 14, patch_w * 14, 1) float32.
    func callAsFunction(
        _ outFeatures: [(patch: MLXArray, cls: MLXArray)], patchH: Int, patchW: Int, frameLength: Int,
        microBatchSize: Int = 4
    ) -> MLXArray {
        var out: [MLXArray] = []
        for (i, feature) in outFeatures.enumerated() {
            // (B*T, N, C) -> (B*T, patch_h, patch_w, C) channel-last
            var x = feature.patch.reshaped(-1, patchH, patchW, feature.patch.dim(-1))
            x = projects[i](x)
            x = resizeLayers[i](x)
            out.append(x)
        }

        var (layer1, layer2, layer3, layer4) = (out[0], out[1], out[2], out[3])
        let batch = layer1.dim(0) / frameLength

        func motion(_ i: Int, _ x: MLXArray) -> MLXArray {
            // (B*T, H, W, C) -> (B, T, H, W, C) -> temporal module -> back
            let y = motionModules[i](x.reshaped([batch, frameLength] + Array(x.shape.dropFirst())))
            return y.reshaped([batch * frameLength] + Array(y.shape.dropFirst(2)))
        }

        layer3 = motion(0, layer3)
        layer4 = motion(1, layer4)

        let layer1Rn = scratch.layer1Rn(layer1)
        let layer2Rn = scratch.layer2Rn(layer2)
        let layer3Rn = scratch.layer3Rn(layer3)
        let layer4Rn = scratch.layer4Rn(layer4)

        var path4 = scratch.refinenet4(layer4Rn, size: (layer3Rn.dim(1), layer3Rn.dim(2)))
        path4 = motion(2, path4)
        var path3 = scratch.refinenet3(path4, layer3Rn, size: (layer2Rn.dim(1), layer2Rn.dim(2)))
        path3 = motion(3, path3)

        func head(_ path3: MLXArray, _ layer2Rn: MLXArray, _ layer1Rn: MLXArray) -> MLXArray {
            let path2 = scratch.refinenet2(path3, layer2Rn, size: (layer1Rn.dim(1), layer1Rn.dim(2)))
            let path1 = scratch.refinenet1(path2, layer1Rn)

            var out = scratch.outputConv1(path1)
            out = upsampleBilinear(out, h: patchH * patchSize, w: patchW * patchSize)
            // The reference runs the output head in full precision.
            out = out.asType(.float32)
            for layer in scratch.outputConv2 {
                out = layer(out)
            }
            return out
        }

        let batchSize = layer1Rn.dim(0)
        if batchSize <= microBatchSize || batchSize % microBatchSize != 0 {
            return head(path3, layer2Rn, layer1Rn)
        }

        var chunks: [MLXArray] = []
        for i in stride(from: 0, to: batchSize, by: microBatchSize) {
            let sl = i..<(i + microBatchSize)
            let chunk = head(path3[sl], layer2Rn[sl], layer1Rn[sl])
            eval(chunk)
            chunks.append(chunk)
        }
        return concatenated(chunks, axis: 0)
    }
}
