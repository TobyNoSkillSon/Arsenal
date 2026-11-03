// SPDX-License-Identifier: MIT
import CryptoKit
import Darwin
import Flux2Klein
import Flux3Klein
import Foundation
import MLX

/// What the runner needs from an engine; the real one is `KleinPipeline`, tests use a fake.
public protocol ImageEngine: AnyObject {
    func image(prompt: String, width: Int, height: Int, steps: Int, seed: UInt64, step: (Int, Int) throws -> Void) throws -> (pixels: [UInt8], width: Int, height: Int)
}

extension KleinPipeline: ImageEngine {
    public func image(prompt: String, width: Int, height: Int, steps: Int, seed: UInt64, step: (Int, Int) throws -> Void) throws -> (pixels: [UInt8], width: Int, height: Int) {
        let out = try generate(prompt: prompt, width: width, height: height, steps: steps, seed: seed, step: step)
        return (out.pixels, out.width, out.height)
    }
}

/// The request contract of ops/generate.input.json, validated the way the Python runner did.
struct ImageRequest: Equatable, Sendable {
    var prompt: String, width: Int, height: Int, steps: Int, n: Int, seed: UInt32, outDir: URL

    static func parse(_ r: JSONObject, support: URL, randomSeed: () -> UInt32 = { UInt32.random(in: 0...UInt32.max) }) throws -> ImageRequest {
        guard let prompt = r["prompt"]?.string, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, prompt.count <= 32768 else {
            throw invalid("prompt must contain 1 to 32768 characters.")
        }
        var width: Int?, height: Int?
        if let resolution = r["resolution"] {
            guard let value = resolution.string, ["512x512", "768x768", "1024x1024"].contains(value) else {
                throw invalid("Resolution must be 512x512, 768x768, or 1024x1024.")
            }
            let side = Int(value.split(separator: "x")[0])!
            width = side; height = side
        }
        width = try integer(r, "width", default: width ?? 1024, 64...2048)
        height = try integer(r, "height", default: height ?? 1024, 64...2048)
        guard width! % 16 == 0, height! % 16 == 0 else { throw invalid("Image width and height must be multiples of 16.") }
        let steps = try integer(r, "steps", default: 4, 1...100)
        let n = try integer(r, "n", default: 1, 1...8)
        let seed = try integer(r, "seed", default: Int(randomSeed()), 0...Int(UInt32.max))
        if let guidance = r["guidance"] {
            guard let g = guidance.number, g.isFinite, g == 1 else { throw invalid("FLUX.2 Klein uses fixed guidance 1.") }
        }
        guard let out = r["out_dir"]?.string, out.hasPrefix("/") else { throw invalid("out_dir must be an absolute supervisor job directory.") }
        let dir = resolved(URL(fileURLWithPath: out)), jobs = resolved(support.appendingPathComponent("jobs"))
        guard dir.path.hasPrefix(jobs.path + "/") else { throw invalid("Image output must be inside an Arsenal job directory.") }
        return ImageRequest(prompt: prompt, width: width!, height: height!, steps: steps, n: n, seed: UInt32(seed), outDir: dir)
    }

    /// Like Python's Path.resolve(strict=False): symlinks resolved on the longest existing prefix, ".." removed.
    static func resolved(_ url: URL) -> URL {
        var head = url.standardizedFileURL, tail: [String] = []
        while !FileManager.default.fileExists(atPath: head.path), head.path != "/" { tail.insert(head.lastPathComponent, at: 0); head.deleteLastPathComponent() }
        return tail.reduce(head.resolvingSymlinksInPath()) { $0.appendingPathComponent($1) }
    }
    static func invalid(_ message: String) -> WorkerFailure { WorkerFailure(code: "invalid_request", message: message) }
    static func integer(_ r: JSONObject, _ key: String, default fallback: Int, _ range: ClosedRange<Int>) throws -> Int {
        guard let value = r[key] else { return fallback }
        guard let v = value.number, v.isFinite, v.rounded() == v, range.contains(Int(exactly: v) ?? Int.min) else {
            throw invalid("\(key) must be an integer from \(range.lowerBound) to \(range.upperBound).")
        }
        return Int(v)
    }
}

final class Runner {
    let wire: WorkerChannel
    let jobs: Jobs
    let loader: (URL, String) throws -> any ImageEngine
    var engine: (any ImageEngine)?
    var flux3: (any Flux3Engine)?
    var dtype: String?
    let supportDirectory: URL
    var flux3Loader: (URL, @escaping () -> Bool) throws -> any Flux3Engine = { root, cancelled in try Flux3Pipeline.load(snapshot: root, cancelled: cancelled) }
    var model: String { flux3 != nil ? Flux3Serving.architecture : "flux2-klein-4b" }

    init(wire: WorkerChannel, jobs: Jobs, support: URL = Runner.support, loader: @escaping (URL) throws -> any ImageEngine = { try KleinPipeline.load(snapshot: $0) }) {
        self.wire = wire; self.jobs = jobs; self.loader = { url, _ in try loader(url) }; self.supportDirectory = support
    }

    convenience init(wire: WorkerChannel, jobs: Jobs, support: URL, precisionLoader: @escaping (URL, String) throws -> any ImageEngine) {
        self.init(wire: wire, jobs: jobs, support: support)
        self.precisionLoader = precisionLoader
    }
    var precisionLoader: ((URL, String) throws -> any ImageEngine)?

    static var support: URL {
        ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_DIR"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Arsenal")
    }

    func memory() -> JSONValue {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return [
            "footprint_mb": .number(result == KERN_SUCCESS ? Double(info.phys_footprint) / 1e6 : 0),
            "footprint_peak_mb": .number(result == KERN_SUCCESS ? Double(info.ledger_phys_footprint_peak) / 1e6 : 0),
            "mlx_active_mb": .number(engine == nil && flux3 == nil ? 0 : Double(Memory.activeMemory) / 1e6),
            "mlx_cache_mb": .number(engine == nil && flux3 == nil ? 0 : Double(Memory.cacheMemory) / 1e6)
        ]
    }

    func status() -> JSONObject {
        [
            "loaded": .bool(engine != nil || flux3 != nil), "dtype": dtype.map(JSONValue.string) ?? .null, "engine": "mlx-swift", "model": .string(model),
            "fast_exact": false, "memory": memory()
        ]
    }

    func pushStatus() throws { try wire.send(["event": "status", "memory": memory()]) }

    /// Returns true when the process should exit (after unload).
    func run(_ r: JSONObject) throws -> Bool {
        let id = r["id"] ?? .null, number = id.number ?? -1
        let op = r["op"]?.string ?? ""
        if ["load", "generate", "edit", "convert"].contains(op) {
            Memory.peakMemory = 0
            try wire.scheduled(id, phase: op == "load" ? "loading" : "step", peak: { Double(Memory.peakMemory) / 1e6 })
        }
        switch op {
        case "load":
            guard engine == nil, flux3 == nil else { throw WorkerFailure(code: "already_loaded", message: "Unload this runner before loading another snapshot.") }
            let dtype = r["dtype"]?.string ?? "bf16"
            guard ["bf16", "fp16", "int8", "int4"].contains(dtype) else {
                throw WorkerFailure(code: "unsupported_dtype", message: "Use BF16, FP16, int8 or int4 after intake qualification.")
            }
            guard let path = r["weights_dir"]?.string, path.hasPrefix("/") else {
                throw WorkerFailure(
                    code: "invalid_weights", message: "Load requires an absolute local weights directory; call arsenal_download(model: flux2-klein-4b) through MCP.")
            }
            let root = URL(fileURLWithPath: path).resolvingSymlinksInPath()
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory), isDirectory.boolValue else {
                throw WorkerFailure(code: "missing_weights", message: "Weights are not installed; call arsenal_download(model: flux2-klein-4b) through MCP.")
            }
            if r["config"]?["architecture"]?.string == Flux3Serving.architecture {
                let start = Date()
                Memory.cacheLimit = 256 << 20
                if jobs.cancelled(number) { throw KleinRequestCancelled() }
                let loaded = try loadFlux3(root, dtype: dtype) { self.jobs.cancelled(number) }
                if jobs.cancelled(number) { Memory.clearCache(); throw KleinRequestCancelled() }
                flux3 = loaded; self.dtype = dtype
                try pushStatus()
                try wire.reply(
                    id,
                    [
                        "load_s": .number(Date().timeIntervalSince(start)), "engine": "mlx-swift", "dtype": .string(dtype), "memory": memory(),
                        "engine_reason": "Native Swift/MLX pipeline in the release's BF16; the float32 VAE is cast to BF16."
                    ])
                return false
            }
            if let missing = KleinWeights.requiredFiles.first(where: { !FileManager.default.fileExists(atPath: root.appendingPathComponent($0).path) }) {
                throw WorkerFailure(code: "missing_weights", message: "The local snapshot is incomplete: \(missing); run Get again.")
            }
            let start = Date()
            Memory.cacheLimit = 256 << 20
            if jobs.cancelled(number) { throw KleinRequestCancelled() }
            let loaded: any ImageEngine
            if let precisionLoader {
                loaded = try precisionLoader(root, dtype)
            } else if dtype == "bf16" {
                loaded = try loader(root, dtype)
            } else {
                let half: DType = dtype == "fp16" ? .float16 : try KleinWeights.storedHalf(snapshot: root)
                loaded = try KleinPipeline.load(snapshot: root, precision: KleinPrecision(weights: half, encoder: half, activations: half, vae: half))
            }
            if jobs.cancelled(number) { Memory.clearCache(); throw KleinRequestCancelled() }
            engine = loaded; self.dtype = dtype
            try pushStatus()
            try wire.reply(
                id,
                [
                    "load_s": .number(Date().timeIntervalSince(start)), "engine": "mlx-swift", "dtype": .string(dtype), "memory": memory(),
                    "engine_reason": "Native Swift/MLX pipeline; stored precision covers transformer and prompt encoder; VAE stays half."
                ])
        case "generate", "edit":
            if let flux3 {
                try generateFlux3(r, operation: op, engine: flux3); try pushStatus()
            } else {
                guard op == "generate" else { throw WorkerFailure(code: "unsupported_op", message: "FLUX.2 Klein 4B generates from a prompt only.") }
                guard let engine else { throw WorkerFailure(code: "not_loaded", message: "Load a FLUX model before generating an image.") }
                try generate(r, engine: engine); try pushStatus()
            }
        case "measurement_components":
            guard let pipeline = engine as? KleinPipeline, let fixture = r["fixture_dir"]?.string,
                let output = r["output_dir"]?.string, let capture = r["capture"]?.bool
            else { throw KleinLoadError("Component capture requires a loaded pipeline and paths") }
            try pipeline.intakeComponents(fixture: URL(fileURLWithPath: fixture), output: URL(fileURLWithPath: output), capture: capture)
            try wire.reply(id, ["output_dir": .string(output), "memory": memory()])
        case "status":
            try wire.reply(id, status())
        case "unload":
            engine = nil; flux3 = nil; dtype = nil
            Memory.clearCache()
            try wire.reply(id, ["unloaded": true]); return true
        case "convert":
            guard engine == nil, let from = r["src_dir"]?.string, from.hasPrefix("/"),
                let to = r["dst_dir"]?.string, to.hasPrefix("/"), let dtype = r["dtype"]?.string,
                let recipe = r["recipe"]?.string,
                (dtype == "fp16" && recipe == "cast:flux2-klein-4b")
                    || (["int8", "int4"].contains(dtype) && recipe == "mlx-affine-g64:flux2-klein-4b" && r["group_size"]?.number == 64)
            else { throw WorkerFailure(code: "unsupported_recipe", message: "Unknown FLUX conversion recipe or invalid paths") }
            let destination = URL(fileURLWithPath: to)
            guard try FileManager.default.contentsOfDirectory(atPath: to).isEmpty else { throw KleinLoadError("Conversion destination must be empty") }
            try Device.withDefaultDevice(.cpu) { try KleinWeights.convert(source: URL(fileURLWithPath: from), destination: destination, dtype: dtype) }
            var files: [JSONValue] = []
            for case let file as URL in FileManager.default.enumerator(at: destination, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey])! {
                let attrs = try file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
                guard attrs.isRegularFile == true else { continue }
                let handle = try FileHandle(forReadingFrom: file); defer { try? handle.close() }
                var hash = SHA256()
                while try autoreleasepool(invoking: { () throws -> Bool in
                    guard let chunk = try handle.read(upToCount: 4 << 20), !chunk.isEmpty else { return false }
                    hash.update(data: chunk); return true
                }) {}
                files.append([
                    "path": .string(String(file.path.dropFirst(to.count + 1))), "bytes": .number(Double(attrs.fileSize ?? 0)),
                    "sha256": .string(hash.finalize().map { String(format: "%02x", $0) }.joined())
                ])
            }
            try wire.reply(id, ["files": .array(files)])
        default:
            throw WorkerFailure(code: "unsupported_op", message: "Unsupported operation.")
        }
        return false
    }
    func generate(_ r: JSONObject, engine: any ImageEngine) throws {
        let id = r["id"] ?? .null, number = id.number ?? -1
        let request = try ImageRequest.parse(r, support: supportDirectory)
        try FileManager.default.createDirectory(at: request.outDir, withIntermediateDirectories: true)
        var images: [JSONValue] = [], seeds: [JSONValue] = [], written: [URL] = []
        let start = Date()
        do {
            for i in 0..<request.n {
                if jobs.cancelled(number) { throw KleinRequestCancelled() }
                let seed = UInt32(truncatingIfNeeded: UInt64(request.seed) + UInt64(i))
                let total = request.steps * request.n
                let out = try engine.image(prompt: request.prompt, width: request.width, height: request.height, steps: request.steps, seed: UInt64(seed)) { step, steps in
                    let done = i * steps + step
                    try? self.wire.send([
                        "id": id, "event": "progress", "phase": "step", "step": .number(Double(done)), "total": .number(Double(total)),
                        "fraction": .number(Double(done) / Double(total))
                    ])
                    if self.jobs.cancelled(number) { throw KleinRequestCancelled() }
                }
                if jobs.cancelled(number) { throw KleinRequestCancelled() }
                let file = request.outDir.appendingPathComponent("image-\(i + 1)-\(String(format: "%016llx", UInt64.random(in: 0...UInt64.max))).png")
                try PNG.write(out.pixels, width: out.width, height: out.height, to: file)
                written.append(file)
                let data = try Data(contentsOf: file)
                images.append([
                    "path": .string(file.path), "width": .number(Double(out.width)), "height": .number(Double(out.height)),
                    "bytes": .number(Double(data.count)), "sha256": .string(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
                ])
                seeds.append(.number(Double(seed)))
            }
        } catch {
            for file in written { try? FileManager.default.removeItem(at: file) }
            throw error
        }
        try wire.reply(id, ["images": .array(images), "seed": .number(Double(request.seed)), "seeds": .array(seeds), "seconds": .number(Date().timeIntervalSince(start))])
    }
}
