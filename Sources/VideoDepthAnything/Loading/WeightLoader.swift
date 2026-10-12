// Swift counterpart of mlx_vlm/utils.py::load_model's weight loading (mlx-vlm 0.7.6):
// load safetensors -> cast -> update(verify:). There is no sanitize step: scripts/convert.py already
// wrote the convolution kernels channel-last.
import Foundation
import MLX
import MLXNN

/// Loads `<directory>/model.safetensors` into `module`.
func loadWeights(into module: Module, from directory: URL, dtype: DType) throws {
    let file = directory.appending(path: "model.safetensors")
    guard FileManager.default.fileExists(atPath: file.path) else {
        throw VideoDepthError.weightsNotFound(directory)
    }
    let raw: [String: MLXArray]
    do {
        raw = try loadArrays(url: file)
    } catch {
        throw VideoDepthError.invalidConfig("\(file.lastPathComponent): \(error)")
    }
    try loadWeights(into: module, weights: raw, dtype: dtype)
}

/// Loads `weights` into `module` with full verification: every parameter must be set, every tensor
/// consumed, and every shape must match. Floating-point tensors are cast to `dtype`.
func loadWeights(into module: Module, weights all: [String: MLXArray], dtype: DType) throws {
    var weights: [String: MLXArray] = [:]
    for (key, value) in all {
        weights[key] = value.dtype.isFloatingPoint && value.dtype != dtype ? value.asType(dtype) : value
    }

    // MLXNN reports unused keys one module at a time and relative to that module; check up front so
    // the error lists every unused key with its full path.
    let expected = Set(module.parameters().flattened().map(\.0))
    let unused = weights.keys.filter { !expected.contains($0) }.sorted()
    if !unused.isEmpty {
        throw VideoDepthError.unusedParameters(unused)
    }

    do {
        try module.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    } catch let error as UpdateError {
        throw VideoDepthError(error)
    }
    eval(Array(weights.values))
}

extension VideoDepthError {
    /// Maps an MLXNN update failure to the matching error, with dotted parameter keys.
    fileprivate init(_ error: UpdateError) {
        switch error {
        case .keyNotFound(let path, _):
            self = .missingParameter(path.joined(separator: "."))
        case .mismatchedSize(let path, _, let expectedShape, let actualShape):
            self = .shapeMismatch(path.joined(separator: "."), expected: expectedShape, got: actualShape)
        case .unhandledKeys(let path, _, let keys):
            self = .unusedParameters(keys.map { (path + [$0]).joined(separator: ".") })
        default:
            self = .invalidConfig("weights do not match the module structure: \(error.localizedDescription)")
        }
    }
}
