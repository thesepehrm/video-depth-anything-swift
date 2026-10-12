// Depth to BGRA through a 256-entry lookup table, on the GPU with a Metal compute kernel. The mapping
// follows the upstream Video Depth Anything `save_video`: normalize to the range, scale to 0...255,
// truncate to an index, look up the color.
import CoreVideo
import Foundation
import Metal

/// How depth is shown in a rendered video.
public enum DepthColormap: Sendable, Equatable {
    /// Near is white, far is black.
    case grayscale
    /// matplotlib's inferno: near is bright yellow, far is black.
    case inferno
    /// Google's turbo: near is dark red, far is dark blue.
    case turbo

    /// 256 × RGB.
    var table: [UInt8] {
        switch self {
        case .grayscale: (0..<256).flatMap { [UInt8($0), UInt8($0), UInt8($0)] }
        case .inferno: ColormapTables.inferno
        case .turbo: ColormapTables.turbo
        }
    }
}

/// Renders depth values into 32BGRA pixel buffers.
final class ColormapRenderer {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let lut: MTLBuffer
    private let textureCache: CVMetalTextureCache

    init(colormap: DepthColormap) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw VideoDepthError.writer("no Metal device")
        }
        self.device = device
        self.queue = queue
        do {
            let library = try device.makeLibrary(source: Self.source, options: nil)
            guard let function = library.makeFunction(name: "depth_colormap") else {
                throw VideoDepthError.writer("colormap kernel missing")
            }
            pipeline = try device.makeComputePipelineState(function: function)
        } catch let error as VideoDepthError {
            throw error
        } catch {
            throw VideoDepthError.writer("cannot build the colormap kernel: \(error)")
        }
        let rgb = colormap.table
        var rgba = [UInt8](repeating: 255, count: 256 * 4)
        for i in 0..<256 {
            rgba[i * 4] = rgb[i * 3]
            rgba[i * 4 + 1] = rgb[i * 3 + 1]
            rgba[i * 4 + 2] = rgb[i * 3 + 2]
        }
        guard let lut = device.makeBuffer(bytes: rgba, length: rgba.count, options: .storageModeShared) else {
            throw VideoDepthError.writer("cannot allocate the colormap table")
        }
        self.lut = lut
        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(nil, nil, device, nil, &cache) == kCVReturnSuccess, let cache else {
            throw VideoDepthError.writer("cannot create a Metal texture cache")
        }
        textureCache = cache
    }

    /// Writes `values` (`width × height`, row-major) into `pixelBuffer` (32BGRA, same size), mapping
    /// `range.lowerBound` to the first table entry and `range.upperBound` to the last.
    func render(
        _ values: [Float], width: Int, height: Int, range: ClosedRange<Float>, into pixelBuffer: CVPixelBuffer
    )
        throws
    {
        precondition(values.count == width * height)
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            nil, textureCache, pixelBuffer, nil, .bgra8Unorm, width, height, 0, &cvTexture)
        guard status == kCVReturnSuccess, let cvTexture, let texture = CVMetalTextureGetTexture(cvTexture)
        else {
            throw VideoDepthError.writer("cannot map the pixel buffer to a Metal texture (\(status))")
        }
        guard
            let depth = values.withUnsafeBytes({
                device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            }),
            let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder()
        else {
            throw VideoDepthError.writer("cannot encode the colormap pass")
        }
        let span = range.upperBound - range.lowerBound
        var params = Params(
            width: UInt32(width), height: UInt32(height), lower: range.lowerBound,
            scale: span > 0 ? 255 / span : 0)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(depth, offset: 0, index: 0)
        encoder.setBuffer(lut, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
        encoder.setTexture(texture, index: 0)
        let group = MTLSize(width: 16, height: 16, depth: 1)
        encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1), threadsPerThreadgroup: group)
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error {
            throw VideoDepthError.writer("colormap pass failed: \(error)")
        }
    }

    private struct Params {
        var width: UInt32
        var height: UInt32
        var lower: Float
        var scale: Float
    }

    private static let source = """
        #include <metal_stdlib>
        using namespace metal;

        struct Params {
            uint width;
            uint height;
            float lower;
            float scale;
        };

        kernel void depth_colormap(
            device const float* depth [[buffer(0)]],
            device const uchar4* lut [[buffer(1)]],
            constant Params& p [[buffer(2)]],
            texture2d<float, access::write> out [[texture(0)]],
            uint2 gid [[thread_position_in_grid]])
        {
            if (gid.x >= p.width || gid.y >= p.height) return;
            float d = depth[gid.y * p.width + gid.x];
            // (d - lo) / (hi - lo) * 255, truncated, as numpy's astype(uint8) after the clamp.
            float n = clamp((d - p.lower) * p.scale, 0.0f, 255.0f);
            uchar4 c = lut[uint(n)];
            out.write(float4(c) / 255.0f, gid);
        }
        """
}
