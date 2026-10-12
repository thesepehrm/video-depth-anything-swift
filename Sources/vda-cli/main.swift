import CoreMedia
import Darwin
import Foundation
import MLX
import VideoDepthAnything

let usage = """
    usage: vda-cli <command> [options]

    commands:
      depth   Render a depth video.
                --weights DIR --input MOVIE --output OUT.mov
                [--colormap grayscale|inferno|turbo] [--codec hevc|h264] [--input-size 518]
                [--fps N] [--start SECONDS] [--duration SECONDS] [--cache-limit-mb MB]
      bench   Time depth estimation without writing a file; prints wall time and peak memory.
                --weights DIR --input MOVIE [--input-size 518] [--fps N] [--start S] [--duration S]
                [--cache-limit-mb MB]
    """

struct UsageError: Error { let message: String }

/// `--key value` pairs.
struct Options {
    var values: [String: String] = [:]

    init(_ args: ArraySlice<String>, allowed: Set<String>) throws {
        var it = args.makeIterator()
        while let arg = it.next() {
            guard arg.hasPrefix("--") else { throw UsageError(message: "unexpected argument '\(arg)'") }
            let key = String(arg.dropFirst(2))
            guard allowed.contains(key) else { throw UsageError(message: "unknown option --\(key)") }
            guard let value = it.next() else { throw UsageError(message: "--\(key) needs a value") }
            values[key] = value
        }
    }

    func url(_ key: String) throws -> URL {
        guard let v = values[key] else { throw UsageError(message: "--\(key) is required") }
        return URL(fileURLWithPath: v)
    }

    func double(_ key: String) throws -> Double? {
        guard let v = values[key] else { return nil }
        guard let d = Double(v) else { throw UsageError(message: "--\(key) must be a number") }
        return d
    }
}

func job(_ o: Options) async throws -> (VideoDepthModel, VideoFrameReader, DepthOptions) {
    let model = try VideoDepthModel.load(directory: try o.url("weights"))
    if let mb = try o.double("cache-limit-mb") { await model.setCacheLimit(bytes: Int(mb * 1_048_576)) }
    var range: CMTimeRange?
    if o.values["start"] != nil || o.values["duration"] != nil {
        let start = CMTime(seconds: try o.double("start") ?? 0, preferredTimescale: 600)
        let duration =
            try o.double("duration").map { CMTime(seconds: $0, preferredTimescale: 600) } ?? .positiveInfinity
        range = CMTimeRange(start: start, duration: duration)
    }
    let rate: VideoFrameReader.FrameRate = try o.double("fps").map { .target($0) } ?? .source
    let reader = try await VideoFrameReader(url: try o.url("input"), timeRange: range, frameRate: rate)
    let size = try o.double("input-size").map(Int.init) ?? 518
    return (model, reader, DepthOptions(inputSize: size))
}

func progressPrinter() -> @Sendable (DepthProgress) -> Void {
    { p in
        let total = p.expectedFrames.map(String.init) ?? "?"
        FileHandle.standardError.write(Data("\r\(p.completedFrames)/\(total) frames".utf8))
    }
}

/// Peak physical footprint of this process (what Activity Monitor calls Memory), in bytes.
func peakFootprint() -> Int64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? info.ledger_phys_footprint_peak : 0
}

func runDepth(_ o: Options) async throws {
    let (model, reader, options) = try await job(o)
    let colormap: DepthColormap =
        switch o.values["colormap"] ?? "grayscale" {
        case "grayscale": .grayscale
        case "inferno": .inferno
        case "turbo": .turbo
        default: throw UsageError(message: "--colormap must be grayscale, inferno or turbo")
        }
    let codec: DepthVideoOptions.Codec =
        switch o.values["codec"] ?? "hevc" {
        case "hevc": .hevc
        case "h264": .h264
        default: throw UsageError(message: "--codec must be hevc or h264")
        }
    let out = try o.url("output")
    let start = Date()
    let n = try await DepthVideoWriter.write(
        model.depth(of: reader, options: options, progress: progressPrinter()), to: out,
        options: DepthVideoOptions(codec: codec, colormap: colormap))
    print(
        "\nwrote \(n) frames to \(out.path) in \(String(format: "%.1f", Date().timeIntervalSince(start))) s")
}

func runBench(_ o: Options) async throws {
    let loadStart = Date()
    let (model, reader, options) = try await job(o)
    let load = Date().timeIntervalSince(loadStart)
    Memory.peakMemory = 0
    let start = Date()
    var n = 0
    var firstFrame: Double?
    for try await _ in model.depth(of: reader, options: options, progress: progressPrinter()) {
        n += 1
        if firstFrame == nil { firstFrame = Date().timeIntervalSince(start) }
    }
    let wall = Date().timeIntervalSince(start)
    print("")
    print(
        "source: \(Int(reader.size.width))x\(Int(reader.size.height)) @ \(reader.nominalFrameRate) fps, stride \(reader.stride)"
    )
    print("input size: \(options.inputSize)")
    print("frames: \(n)")
    print("load s: \(String(format: "%.2f", load))")
    print("first frame s: \(String(format: "%.2f", firstFrame ?? 0))")
    print("wall s: \(String(format: "%.2f", wall))")
    print("fps: \(String(format: "%.2f", Double(n) / wall))")
    print("peak MLX memory GB: \(String(format: "%.2f", Double(Memory.peakMemory) / 1e9))")
    print("peak process footprint GB: \(String(format: "%.2f", Double(peakFootprint()) / 1e9))")
}

func run(_ arguments: [String]) async -> Int32 {
    guard let command = arguments.first, !["-h", "--help", "help"].contains(command) else {
        print(usage)
        return arguments.isEmpty ? 64 : 0
    }
    let common: Set<String> = [
        "weights", "input", "input-size", "fps", "start", "duration", "cache-limit-mb",
    ]
    do {
        switch command {
        case "depth":
            try await runDepth(
                Options(arguments.dropFirst(), allowed: common.union(["output", "colormap", "codec"])))
        case "bench": try await runBench(Options(arguments.dropFirst(), allowed: common))
        default: throw UsageError(message: "unknown command '\(command)'")
        }
        return 0
    } catch let error as UsageError {
        FileHandle.standardError.write(Data("error: \(error.message)\n\n\(usage)\n".utf8))
        return 64
    } catch {
        FileHandle.standardError.write(Data("error: \(error)\n".utf8))
        return 1
    }
}

exit(await run(Array(CommandLine.arguments.dropFirst())))
