// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "video-depth-anything-swift",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "VideoDepthAnything", targets: ["VideoDepthAnything"]),
        .executable(name: "vda-cli", targets: ["vda-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
    ],
    targets: [
        .target(
            name: "VideoDepthAnything",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "vda-cli",
            dependencies: ["VideoDepthAnything", .product(name: "MLX", package: "mlx-swift")]
        ),
        .testTarget(
            name: "VideoDepthAnythingTests",
            dependencies: ["VideoDepthAnything"],
            resources: [.copy("Resources")]
        ),
    ],
    swiftLanguageModes: [.v6]
)
