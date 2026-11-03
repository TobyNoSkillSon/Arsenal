// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import CoreGraphics
import CryptoKit
import MLX
import MLXNN
import MLXBiRefNet
import ArsenalWire

// CoreGraphics stores premultiplied RGB. Scale every channel, including existing alpha.
func applyAlphaMask(_ rgba: inout [UInt8], mask: [UInt8]) {
    precondition(rgba.count == mask.count * 4)
    for i in mask.indices {
        for channel in 0..<4 { rgba[i * 4 + channel] = UInt8(Int(rgba[i * 4 + channel]) * Int(mask[i]) / 255) }
    }
}

struct Failure: Error { let code: String; let message: String }
final class Runner: @unchecked Sendable {
    let output: FileHandle
    let writeLock = NSLock(), stateLock = NSLock()
    let queue = DispatchQueue(label: "birefnet.gpu")
    var cancelled: Set<Int> = [], pending: Set<Int> = []
    var pipeline: BiRefNetPipeline?
    var precision = "fp16"
    init(output: FileHandle) { self.output = output }
    func send(_ value: JSONObject) {
        writeLock.lock(); defer { writeLock.unlock() }
        try? FrameIO.write(.json(value), to: output)
    }
    func check(_ id: Int) throws {
        stateLock.lock(); let stop = cancelled.contains(id); stateLock.unlock()
        if stop { throw Failure(code: "cancelled", message: "Request cancelled.") }
    }
    func fail(_ id: Int, _ error: Error) {
        let f = error as? Failure ?? Failure(code: "runner", message: String(describing: error))
        send(["id": .number(Double(id)), "ok": false, "error": ["code": .string(f.code), "message": .string(f.message)]])
    }
    func path(_ raw: String, under directory: URL) throws -> URL {
        let url = URL(fileURLWithPath: raw).standardizedFileURL
        let resolved = url.resolvingSymlinksInPath()
        guard raw.hasPrefix("/"), url.path == resolved.path, resolved.path.hasPrefix(directory.resolvingSymlinksInPath().path + "/") else {
            throw Failure(code: "invalid_request", message: "Path must be a regular file inside the Arsenal job or inbox directory.")
        }
        return resolved
    }
    func run(_ request: JSONObject, _ id: Int, _ op: String) {
        defer { stateLock.lock(); pending.remove(id); cancelled.remove(id); stateLock.unlock() }
        let watchdog = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        let started = ProcessInfo.processInfo.systemUptime
        watchdog.schedule(deadline: .now() + 10, repeating: 10)
        watchdog.setEventHandler {
            // Serving permits macOS swap after Load Anyway. Intake owns its zero-growth guard.
            if ProcessInfo.processInfo.systemUptime - started > 180 {
                FileHandle.standardError.write(Data("BiRefNet stopped: request exceeded its 180-second bound.\n".utf8)); exit(75)
            }
        }
        watchdog.resume(); defer { watchdog.cancel() }
        do {
            try check(id)
            var result: JSONObject = [:]
            switch op {
            case "ping": break
            case "status":
                result = ["dtype": .string(precision), "memory": ["footprint_peak_mb": .number(Double(Self.footprintPeak()) / 1e6)]]
            case "convert":
                guard let source = request["src_dir"]?.string, let destination = request["dst_dir"]?.string,
                    request["recipe"]?.string == BiRefNetPrecision.recipe, request["group_size"]?.number == 64,
                    let dtype = request["dtype"]?.string, ["int8", "int4"].contains(dtype)
                else { throw Failure(code: "invalid_request", message: "Expected BiRefNet affine g64 int8/int4 derivation.") }
                let target = URL(fileURLWithPath: destination)
                try BiRefNetPrecision.convert(source: URL(fileURLWithPath: source), destination: target, bits: dtype == "int8" ? 8 : 4)
                result = [
                    "files": .array(
                        try ["model.safetensors", "config.json", "coverage.json", "component-coverage.json"].map { name in
                            let file = target.appendingPathComponent(name)
                            return [
                                "path": .string(name), "bytes": .number(Double(try file.resourceValues(forKeys: [.fileSizeKey]).fileSize!)),
                                "sha256": .string(try Self.digest(file))
                            ]
                        })
                ]
            case "parameter_inventory":
                guard ProcessInfo.processInfo.environment["ARSENAL_NATIVE_LAB"] == "1", let pipeline else {
                    throw Failure(code: "invalid_request", message: "Load an isolated lab worker first.")
                }
                result = [
                    "parameters": .object(
                        Dictionary(
                            uniqueKeysWithValues: pipeline.model.parameters().flattened().filter { $0.1.dtype.isFloatingPoint || $0.1.dtype == .uint32 }.map { name, value in
                                (name, JSONValue.object(["shape": .array(value.shape.map { .number(Double($0)) }), "dtype": .string(String(describing: value.dtype))]))
                            }))
                ]
            case "measurement_components":
                guard let pipeline, let fixture = request["fixture_dir"]?.string, let output = request["output_dir"]?.string else {
                    throw Failure(code: "invalid_request", message: "Load first and supply fixture_dir/output_dir.")
                }
                try pipeline.model.intakeComponents(fixture: URL(fileURLWithPath: fixture), output: URL(fileURLWithPath: output), capture: request["capture"]?.bool == true)
            case "load":
                guard pipeline == nil, let dtype = request["dtype"]?.string, ["fp16", "int8", "int4"].contains(dtype), let directory = request["weights_dir"]?.string else {
                    throw Failure(code: "invalid_request", message: "Load an FP16 or stored affine int8/int4 checkpoint after unloading any resident model.")
                }
                send(["id": .number(Double(id)), "event": "progress", "phase": "loading"])
                if request["config"]?["random_weights"]?.bool == true {
                    guard ProcessInfo.processInfo.environment["ARSENAL_NATIVE_LAB"] == "1" else {
                        throw Failure(code: "invalid_request", message: "Random weights require an isolated lab worker.")
                    }
                    let model = BiRefNet()
                    let values = model.parameters().flattened().map { ($0.0, $0.1.dtype.isFloatingPoint ? $0.1.asType(.float16) : $0.1) }
                    try model.update(parameters: ModuleParameters.unflattened(values), verify: .none)
                    if dtype == "int8" || dtype == "int4" {
                        quantize(model: model, groupSize: 64, bits: dtype == "int8" ? 8 : 4) { _, layer in
                            guard let linear = layer as? Linear else { return false }
                            return linear.weight.dim(1) % 64 == 0
                        }
                    }
                    model.train(false)
                    pipeline = BiRefNetPipeline(model: model, dtype: .float16, inputSize: (64, 64))
                } else {
                    pipeline = try BiRefNetPipeline.fromPretrained(directory, dtype: .float16, bits: dtype == "int8" ? 8 : dtype == "int4" ? 4 : nil)
                }
                precision = dtype
                result = ["engine": "mlx-swift", "dtype": .string(dtype)]
            case "unload": pipeline = nil; Memory.clearCache()
            case "remove_background":
                guard let pipeline else { throw Failure(code: "not_loaded", message: "Load BiRefNet first.") }
                guard Set(request.keys).isSubset(of: ["id", "op", "out_dir", "image"]),
                    let image = request["image"]?.object, Set(image.keys) == ["path"], let raw = image["path"]?.string,
                    let support = ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_DIR"], let out = request["out_dir"]?.string
                else {
                    throw Failure(code: "invalid_request", message: "Supply an uploaded image: {path: ...}.")
                }
                let root = URL(fileURLWithPath: support)
                let input: URL
                if let p = try? path(raw, under: root.appendingPathComponent("inbox")) { input = p } else { input = try path(raw, under: root.appendingPathComponent("jobs")) }
                let destination = try path(out + "/foreground.png", under: root.appendingPathComponent("jobs")).deletingLastPathComponent()
                guard ["png", "jpg", "jpeg"].contains(input.pathExtension.lowercased()), let cg = loadCGImage(url: input), cg.width * cg.height <= 16_000_000 else {
                    throw Failure(code: "invalid_request", message: "Supply a PNG or JPEG with at most 16 megapixels.")
                }
                send(["id": .number(Double(id)), "event": "progress", "phase": "step", "step": 0, "total": 1])
                let mask = try pipeline(cg).maskCGImage()
                try check(id)
                let w = cg.width, h = cg.height
                var gray = [UInt8](repeating: 0, count: w * h)
                var rgba = [UInt8](repeating: 0, count: w * h * 4)
                guard
                    let g = CGContext(
                        data: &gray, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue),
                    let c = CGContext(
                        data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
                else { throw BiRefNetPipelineError.maskRenderFailed }
                let rect = CGRect(x: 0, y: 0, width: w, height: h)
                g.draw(mask, in: rect); c.draw(cg, in: rect)
                applyAlphaMask(&rgba, mask: gray)
                guard let foreground = c.makeImage() else { throw BiRefNetPipelineError.maskRenderFailed }
                try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
                var files: [JSONValue] = []
                for (order, pair) in [("foreground", foreground), ("mask", mask)].enumerated() {
                    try check(id)
                    let url = destination.appendingPathComponent(pair.0 + ".png")
                    guard !FileManager.default.fileExists(atPath: url.path) else { throw Failure(code: "output_exists", message: "Output already exists.") }
                    try saveCGImage(pair.1, url: url)
                    let bytes = try Data(contentsOf: url)
                    files.append([
                        "id": .string(pair.0), "path": .string(url.path), "mime": "image/png", "width": .number(Double(w)), "height": .number(Double(h)),
                        "order": .number(Double(order)), "label": .string(order == 0 ? "Foreground" : "Alpha mask"), "bytes": .number(Double(bytes.count)),
                        "sha256": .string(SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined())
                    ])
                }
                result = [
                    "files": .array(files),
                    "metadata": ["model": "birefnet", "revision": "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4", "dtype": .string(precision), "engine": "mlx-swift"]
                ]
            default: throw Failure(code: "unsupported_op", message: "Unsupported operation.")
            }
            try check(id)
            result["id"] = .number(Double(id)); result["ok"] = true; send(result)
            if op == "unload" { exit(0) }
        } catch { fail(id, error) }
    }
    static func digest(_ file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file); defer { try? handle.close() }
        var hash = SHA256()
        while let data = try autoreleasepool(invoking: { try handle.read(upToCount: 4 << 20) }), !data.isEmpty { hash.update(data: data) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
    static func footprintPeak() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return status == KERN_SUCCESS ? UInt64(info.ledger_phys_footprint_peak) : 0
    }
    func serve() throws {
        send(
            try WorkerHello(
                worker: "birefnet", runner: "native", kinds: ["images"],
                ops: ["load", "unload", "ping", "status", "cancel", "convert", "measurement_components", "parameter_inventory", "remove_background"]
            ).json)
        while let frame = try FrameIO.read(from: .standardInput) {
            guard case .json(let request) = frame, let number = request["id"]?.number, number.rounded() == number, number > 0, number < Double(Int.max),
                let op = request["op"]?.string
            else { throw WireError.malformedFrame }
            let id = Int(number)
            if op == "cancel" {
                stateLock.lock()
                if let target = request["target"]?.number, target > 0, target < Double(Int.max), target.rounded() == target, pending.contains(Int(target)) {
                    cancelled.insert(Int(target))
                }
                stateLock.unlock()
                send(["id": .number(number), "ok": true]); continue
            }
            stateLock.lock()
            guard pending.count < 64, !pending.contains(id) else { stateLock.unlock(); fail(id, Failure(code: "busy", message: "Runner queue full or duplicate id.")); continue }
            pending.insert(id); stateLock.unlock()
            queue.async { self.run(request, id, op) }
        }
        stateLock.lock(); cancelled.formUnion(pending); stateLock.unlock()
        queue.sync {}
    }
}
public enum BiRefNetWorker {
    public static func run() -> Never {
        setenv("MLX_ENABLE_TF32", "0", 1)
        if let lib = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] {
            GPU.metallib = URL(fileURLWithPath: lib)
        } else {
            let library = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent(
                "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib")
            if FileManager.default.fileExists(atPath: library.path) { GPU.metallib = library }
        }
        // Serving admission belongs to Arsenal, including the user's Load Anyway choice. Do not impose
        // free-RAM, allocator-budget or system-swap stops here. Intake owns its separate bounded guard.
        Memory.cacheLimit = 256 * 1024 * 1024
        let output = FileHandle(fileDescriptor: dup(1), closeOnDealloc: true)
        dup2(2, 1)
        do { try Runner(output: output).serve() } catch { FileHandle.standardError.write(Data("BiRefNet: \(error)\n".utf8)); exit(2) }
        exit(0)
    }
}
