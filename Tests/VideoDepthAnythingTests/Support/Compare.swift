import MLX
import Testing

/// Largest amount by which |a-b| exceeds atol + rtol*|b|; <= 0 means within tolerance.
func maxRelativeViolation(_ a: MLXArray, _ b: MLXArray, rtol: Float, atol: Float) -> Float {
    let a32 = a.asType(.float32)
    let b32 = b.asType(.float32)
    return (abs(a32 - b32) - (atol + rtol * abs(b32))).max().item(Float.self)
}

func assertClose(
    _ a: MLXArray, _ b: MLXArray, rtol: Float = 1e-3, atol: Float = 1e-4, _ label: String,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    #expect(a.shape == b.shape, "\(label): shape \(a.shape) vs \(b.shape)", sourceLocation: sourceLocation)
    guard a.shape == b.shape else { return }
    let v = maxRelativeViolation(a, b, rtol: rtol, atol: atol)
    let maxAbs = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    #expect(
        v <= 0, "\(label): max|Δ|=\(maxAbs) exceeds rtol \(rtol) atol \(atol)",
        sourceLocation: sourceLocation)
}

/// Mean of |a - b| / |b| over all elements with |b| above `floor` (relative error of a depth map).
func meanRelativeError(_ a: MLXArray, _ b: MLXArray, floor: Float = 1e-6) -> Float {
    let a32 = a.asType(.float32)
    let b32 = b.asType(.float32)
    let mask = abs(b32) .> floor
    let rel = which(mask, abs(a32 - b32) / maximum(abs(b32), floor), 0)
    return (rel.sum() / maximum(mask.sum(), 1)).item(Float.self)
}
