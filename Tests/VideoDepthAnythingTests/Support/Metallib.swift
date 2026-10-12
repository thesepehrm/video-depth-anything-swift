import Foundation

/// Makes MLX's Metal kernels available to a `swift test` run.
///
/// SwiftPM does not compile mlx-swift's shaders (mlx-swift#488). MLX first looks for `mlx.metallib`
/// next to the binary that contains it, which for `swift test` is the test bundle's executable. This
/// copies the metallib that `scripts/build-metallib.sh` installs at `.build/mlx.metallib` (or the one
/// `VDA_METALLIB` names) there, before the first GPU call. Under xcodebuild the kernels come with the
/// build and nothing is copied.
///
/// Suites that run MLX gate on ``available``, so without a metallib they are skipped, not crashed.
enum Metallib {
    static let available: Bool = install()

    private final class Anchor {}

    private static func install() -> Bool {
        let fm = FileManager.default
        guard let binary = Bundle(for: Anchor.self).executableURL?.resolvingSymlinksInPath() else {
            return false
        }
        let binDir = binary.deletingLastPathComponent()
        let colocated = binDir.appending(path: "mlx.metallib")
        if fm.fileExists(atPath: colocated.path) { return true }

        // xcodebuild: the kernels ship in the Cmlx resource bundle, which MLX finds on its own.
        let bundled =
            Bundle.allBundles.compactMap(\.resourceURL) + [binDir, binDir.deletingLastPathComponent()]
        for dir in bundled + bundled.map({ $0.deletingLastPathComponent() }) {
            let lib = dir.appending(path: "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib")
            if fm.fileExists(atPath: lib.path) { return true }
        }

        var candidates: [URL] = []
        if let env = ProcessInfo.processInfo.environment["VDA_METALLIB"] {
            candidates.append(URL(fileURLWithPath: env))
        }
        var dir = binDir
        while dir.path != "/" {
            if dir.lastPathComponent == ".build" {
                candidates.append(dir.appending(path: "mlx.metallib"))
                break
            }
            dir = dir.deletingLastPathComponent()
        }
        for source in candidates where fm.fileExists(atPath: source.path) {
            if (try? fm.copyItem(at: source, to: colocated)) != nil { return true }
        }
        print(
            "MLX Metal kernels not found: GPU suites are skipped. Run scripts/build-metallib.sh, or set "
                + "VDA_METALLIB to a metallib.")
        return false
    }
}
