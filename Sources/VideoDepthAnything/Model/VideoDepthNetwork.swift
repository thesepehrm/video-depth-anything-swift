// Port of mlx_vlm/models/video_depth_anything/video_depth_anything.py::Model (mlx-vlm 0.7.6)
import Foundation
import MLX
import MLXNN

/// DINOv2 backbone plus the temporal DPT head. Parameter keys match the original checkpoint
/// (`pretrained.*`, `head.*`).
final class VideoDepthNetwork: Module {
    let config: ModelConfig

    @ModuleInfo(key: "pretrained") var pretrained: DINOv2
    @ModuleInfo(key: "head") var head: DPTHeadTemporal

    init(_ config: ModelConfig) {
        self.config = config
        _pretrained.wrappedValue = DINOv2(config)
        _head.wrappedValue = DPTHeadTemporal(config)
    }

    /// - Parameter x: (B, T, H, W, 3) channel-last, ImageNet-normalized frames. H and W must be
    ///   multiples of the patch size (14).
    /// - Returns: (B, T, H, W) relative (or metric) disparity-like depth, non-negative.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, T, H, W) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let (patchH, patchW) = (H / config.patchSize, W / config.patchSize)
        let features = pretrained.intermediateLayers(
            x.reshaped(B * T, H, W, 3), indices: config.intermediateLayerIdx)
        var depth = head(features, patchH: patchH, patchW: patchW, frameLength: T)
        // (B*T, H, W, 1) -> resize to input resolution
        depth = upsampleBilinear(depth, h: H, w: W)
        depth = relu(depth)
        return depth.reshaped(B, T, H, W)
    }
}
