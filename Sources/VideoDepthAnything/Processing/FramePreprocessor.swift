// Port of mlx_vlm/models/video_depth_anything/processing_video_depth_anything.py::VideoDepthProcessor
// (mlx-vlm 0.7.6): aspect-preserving resize to a multiple of 14 with OpenCV's bicubic filter, then
// ImageNet normalization. Runs on the GPU.
import Foundation
import MLX

struct FramePreprocessor {
    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let std: [Float] = [0.229, 0.224, 0.225]

    let inputSize: Int
    let multiple: Int

    init(inputSize: Int, multiple: Int = 14) {
        self.inputSize = inputSize
        self.multiple = multiple
    }

    /// Output (height, width), a multiple of 14, aspect ratio preserved (`target_size`).
    ///
    /// Frames wider than about 16:9 get a smaller input size to save memory, as in the reference.
    func targetSize(height: Int, width: Int) -> (height: Int, width: Int) {
        var inputSize = inputSize
        let ratio = Double(max(height, width)) / Double(min(height, width))
        if ratio > 1.78 {
            inputSize = Int(Double(inputSize) * 1.777 / ratio)
            inputSize = Int((Double(inputSize) / 14).rounded(.toNearestOrEven)) * 14
        }
        // lower_bound: the short side reaches input_size
        let scale = max(Double(inputSize) / Double(height), Double(inputSize) / Double(width))
        let m = Double(multiple)
        let newH = max(Int((Double(height) * scale / m).rounded(.toNearestOrEven)) * multiple, inputSize)
        let newW = max(Int((Double(width) * scale / m).rounded(.toNearestOrEven)) * multiple, inputSize)
        return (newH, newW)
    }

    /// Resizes and normalizes `rgb`, an `(H, W, 3)` RGB array of 0...255 values (any dtype), to an
    /// `(h, w, 3)` float32 network input.
    func callAsFunction(_ rgb: MLXArray, taps: (y: ResampleTaps, x: ResampleTaps)) -> MLXArray {
        let frame = rgb.asType(.float32) / 255.0
        let resized = separableInterpolate(expandedDimensions(frame, axis: 0), y: taps.y, x: taps.x)[0]
        return (resized - MLXArray(Self.mean)) / MLXArray(Self.std)
    }

    /// The OpenCV bicubic taps from `height`×`width` to the target size.
    func taps(height: Int, width: Int) -> (y: ResampleTaps, x: ResampleTaps) {
        let (h, w) = targetSize(height: height, width: width)
        return (
            ResampleTaps.opencvCubic(inSize: height, outSize: h),
            ResampleTaps.opencvCubic(inSize: width, outSize: w)
        )
    }
}
