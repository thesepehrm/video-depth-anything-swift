import Foundation

/// Errors thrown by VideoDepthAnything.
public enum VideoDepthError: Error, Equatable, Sendable {
    /// The weights directory, `config.json`, or `model.safetensors` is missing.
    case weightsNotFound(URL)
    /// `config.json` could not be decoded, or the weights do not fit the module structure.
    case invalidConfig(String)
    /// The checkpoint is not Video Depth Anything Small (relative or metric).
    case unsupportedModel(String)
    /// A model parameter has no matching tensor in the checkpoint (dotted key).
    case missingParameter(String)
    /// A checkpoint tensor's shape differs from the model parameter's shape (dotted key).
    case shapeMismatch(String, expected: [Int], got: [Int])
    /// Checkpoint tensors that no model parameter consumes (dotted keys, sorted).
    case unusedParameters([String])
    /// ``VideoDepthModel/unload()`` was called; load the model again to use it.
    case unloaded
    /// An option is out of range (for example an input size that is not a multiple of 14).
    case invalidOptions(String)
    /// A frame cannot be read (unsupported pixel format, size change mid-stream, or a reader failure).
    case invalidFrame(String)
    /// The video file cannot be opened or has no video track.
    case invalidVideo(String)
    /// Writing the depth video failed.
    case writer(String)
}
