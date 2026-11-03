// SPDX-License-Identifier: MIT
// Arsenal new-runner kit: the runner protocol (docs/RUNNERS.md). Do not edit; write your model in Model.swift.
import Foundation
import MLX

final class Wire: @unchecked Sendable {
    private let outputLock = NSLock()
    private let stateLock = NSLock()
    private var cancelled: Set<Int> = []
    private var model: Model?
    private let work = DispatchQueue(label: "runner-work")
    /// Frames go to the original stdout; fd 1 is pointed at stderr so a print() in Model.swift cannot corrupt the protocol.
    private let frames: FileHandle = {
        let original = dup(1)
        dup2(2, 1)
        return FileHandle(fileDescriptor: original)
    }()

    /// Arsenal refuses a frame over 16 MiB as protocol corruption and stops the worker, so a reply past this size
    /// (the cap less 64 KiB of headroom) goes out as an invalid_request error instead.
    static let replyLimit = 16 * 1024 * 1024 - 64 * 1024

    func send(_ value: [String: Any]) {
        outputLock.lock(); defer { outputLock.unlock() }
        guard var data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]) else {
            FileHandle.standardError.write("runner: reply is not JSON\n".data(using: .utf8)!); exit(1)
        }
        if data.count + 1 > Wire.replyLimit {
            let size = String(format: "%.1f", Double(data.count) / 1_048_576)
            let message = "The reply would be \(size) MiB, over the 16 MiB limit: ask for less at once "
                + "(fewer residues, series, quantile levels or items, or a shorter horizon)."
            let refusal: [String: Any] = ["id": value["id"] as? Int ?? 0, "ok": false, "error": ["code": "invalid_request", "message": message]]
            data = (try? JSONSerialization.data(withJSONObject: refusal, options: [.sortedKeys])) ?? Data()
        }
        var packet = withUnsafeBytes(of: UInt32(data.count + 1).bigEndian) { Data($0) }
        packet.append(1); packet.append(data)
        do { try frames.write(contentsOf: packet) } catch { exit(1) }
    }

    /// The first NaN or infinite number in a reply, by its path. Failure must not look like an answer.
    static func nonFinite(_ value: Any, at path: String = "") -> String? {
        switch value {
        case let number as Double: return number.isFinite ? nil : path
        case let number as Float: return number.isFinite ? nil : path
        case let array as [Any]:
            for (index, item) in array.enumerated() { if let bad = nonFinite(item, at: "\(path)[\(index)]") { return bad } }
            return nil
        case let object as [String: Any]:
            for (key, item) in object { if let bad = nonFinite(item, at: path.isEmpty ? key : "\(path).\(key)") { return bad } }
            return nil
        default: return nil
        }
    }

    func isCancelled(_ id: Int) -> Bool { stateLock.lock(); defer { stateLock.unlock() }; return cancelled.contains(id) }
    func cancel(_ id: Int) { stateLock.lock(); cancelled.insert(id); stateLock.unlock() }
    func finished(_ id: Int) { stateLock.lock(); cancelled.remove(id); stateLock.unlock() }

    func memory() -> [String: Any] {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        let footprint = result == KERN_SUCCESS ? Double(info.phys_footprint) / 1_000_000 : 0
        return ["footprint_mb": footprint, "mlx_active_mb": Double(Memory.activeMemory) / 1_000_000, "mlx_cache_mb": Double(Memory.cacheMemory) / 1_000_000]
    }

    func perform(_ request: [String: Any]) {
        let id = request["id"] as? Int ?? 0
        func reply(_ result: [String: Any] = [:]) {
            send(["event": "status", "memory": memory()])
            send(result.merging(["id": id, "ok": true]) { _, new in new })
        }
        func failure(_ code: String, _ message: String) { send(["id": id, "ok": false, "error": ["code": code, "message": message]]) }
        let op = request["op"] as? String ?? ""
        defer { finished(id) }
        do {
            switch op {
            case "ping", "status": reply()
            case "load":
                guard let weights = request["weights_dir"] as? String, let folder = request["model_dir"] as? String else {
                    failure("invalid_request", "load needs weights_dir and model_dir."); return
                }
                let start = Date()
                model = nil
                Memory.clearCache()
                let settings = (request["config"] as? [String: Any])?["settings"] as? [String: Any] ?? [:]
                model = try Model(weightsDir: URL(fileURLWithPath: weights), modelDir: URL(fileURLWithPath: folder), settings: settings)
                reply(["load_s": Date().timeIntervalSince(start), "engine": "arsenal-kit-mlx"])
            case "unload":
                model = nil; Memory.clearCache(); reply(); exit(0)
            case "convert":
                reply(["files": try Convert.run(request)])
            case "selftest":
                reply(SelfTest.run())
            default:
                guard modelOps.contains(op) else { failure("unsupported_op", "Unknown operation \(op)."); return }
                guard let model else { failure("not_loaded", "Load the model first."); return }
                if isCancelled(id) { failure("cancelled", "Cancelled"); return }
                var params = request
                params.removeValue(forKey: "id"); params.removeValue(forKey: "op")
                let result = try model.run(op: op, params: Params(params, cancelled: { [unowned self] in self.isCancelled(id) }))
                Memory.clearCache()
                if let bad = Wire.nonFinite(result) {
                    failure("engine", "Output \(bad) is NaN or infinite, usually a float16 overflow: compute that step in float32 (x.asType(.float32)).")
                    return
                }
                reply(result)
            }
        } catch let error as RunnerError {
            switch error {
            case .invalid(let message): failure("invalid_request", message)
            case .failed(let message): failure("engine", message)
            case .cancelled: failure("cancelled", "Cancelled")
            }
        } catch {
            FileHandle.standardError.write("runner: \(error)\n".data(using: .utf8)!)
            failure(op == "load" ? "load_failed" : "engine", "\(error)")
        }
    }

    private static func exact(_ count: Int) throws -> Data? {
        var data = Data()
        while data.count < count {
            guard let part = try FileHandle.standardInput.read(upToCount: count - data.count), !part.isEmpty else {
                if data.isEmpty { return nil }; throw CocoaError(.fileReadCorruptFile)
            }
            data.append(part)
        }
        return data
    }

    func serve() -> Never {
        send([
            "event": "hello", "proto": 1, "worker": "arsenal-kit", "runner": "kit", "pid": Int(ProcessInfo.processInfo.processIdentifier),
            "build": "arsenal-kit-1", "kinds": [modelKind], "ops": ["load", "ping", "status", "cancel", "unload", "convert"] + modelOps,
        ])
        do {
            while let header = try Wire.exact(4) {
                let length = header.reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
                guard length > 0, length <= 16 * 1024 * 1024, let body = try Wire.exact(Int(length)), body.first == 1,
                    let request = try JSONSerialization.jsonObject(with: body.dropFirst()) as? [String: Any]
                else { throw CocoaError(.fileReadCorruptFile) }
                if request["op"] as? String == "cancel" {
                    cancel(request["target"] as? Int ?? 0)
                    send(["id": request["id"] as? Int ?? 0, "ok": true])
                } else {
                    let bytes = Data(body.dropFirst())
                    work.async { if let request = (try? JSONSerialization.jsonObject(with: bytes)) as? [String: Any] { self.perform(request) } }
                }
            }
            work.sync {}
            exit(0)
        } catch {
            FileHandle.standardError.write("runner: invalid frame\n".data(using: .utf8)!); exit(1)
        }
    }
}

/// `convert`: Get casts the pinned weights to 16-bit through this runner ([derive] fp16 = "cast:arsenal-kit-half-v1").
/// Every floating tensor in every .safetensors file becomes the requested dtype. A PyTorch archive (.pth, .pt) becomes
/// `<name>.safetensors` the same way, read without running its pickle (Checkpoints.swift); a NumPy array file (.npy,
/// .npz) becomes `<name>.safetensors` with its arrays unchanged. Other files are copied unchanged.
enum Convert {
    static func run(_ request: [String: Any]) throws -> [[String: Any]] {
        guard let src = request["src_dir"] as? String, let dst = request["dst_dir"] as? String else {
            throw RunnerError.invalid("convert needs src_dir and dst_dir.")
        }
        let target: DType = (request["dtype"] as? String) == "bf16" ? .bfloat16 : .float16
        let source = URL(fileURLWithPath: src).resolvingSymlinksInPath(), destination = URL(fileURLWithPath: dst)
        var files: [[String: Any]] = []
        for relative in try FileManager.default.subpathsOfDirectory(atPath: source.path).sorted() {
            let resolved = source.appendingPathComponent(relative).resolvingSymlinksInPath()
            guard (try? resolved.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { continue }
            let suffix = (relative as NSString).pathExtension
            let converted = ["pth", "pt", "npy", "npz"].contains(suffix)
            let name = converted ? (relative as NSString).deletingPathExtension + ".safetensors" : relative
            let output = destination.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
            guard !FileManager.default.fileExists(atPath: output.path) else {
                throw RunnerError.failed("\(relative) and another source both convert to \(name).")
            }
            func cast(_ value: MLXArray, _ key: String) -> MLXArray {
                guard value.dtype.isFloatingPoint else { return value }
                return target == .float16 && !Weights.fitsHalf(value, name: key) ? value.asType(.float32) : value.asType(target)
            }
            if suffix == "safetensors" {
                let (arrays, metadata) = try loadArraysAndMetadata(url: resolved)
                var out: [String: MLXArray] = [:]
                for (key, value) in arrays { out[key] = cast(value, key) }
                eval(Array(out.values))
                try save(arrays: out, metadata: metadata, url: output)
            } else if suffix == "pth" || suffix == "pt" {
                let checkpoint = try TorchCheckpoint(resolved)
                var out: [String: MLXArray] = [:]
                for (key, tensor) in try checkpoint.tensors() {
                    out[key] = cast(try checkpoint.array(tensor), key)
                    eval(out[key]!)
                }
                guard !out.isEmpty else { throw RunnerError.failed("\(relative) holds no tensors.") }
                try save(arrays: out, metadata: ["format": "pt", "arsenal_source": relative], url: output)
            } else if converted {
                try NumpyArrays.writeSafetensors(try NumpyArrays.read(resolved), metadata: ["format": "np", "arsenal_source": relative], to: output)
            } else {
                try FileManager.default.copyItem(at: resolved, to: output)
            }
            files.append(["path": name, "bytes": try Files.size(output), "sha256": try Files.sha256(output)])
        }
        return files
    }
}

extension DType {
    var isFloatingPoint: Bool { [.float32, .float16, .bfloat16, .float64].contains(self) }
}

/// `selftest` (kit build only, not advertised): the GPU, with the MLX shaders copied from Arsenal.app, against the CPU on
/// small fixed tensors. A shader file from another MLX version fails to load or disagrees here, before any model runs.
enum SelfTest {
    static func run() -> [String: Any] {
        let a = sin(MLXArray(0 ..< 4096).asType(.float32) * 0.01).reshaped([64, 64])
        let image = cos(MLXArray(0 ..< (2 * 9 * 9 * 3)).asType(.float32) * 0.07).reshaped([2, 9, 9, 3])
        let kernel = sin(MLXArray(0 ..< (4 * 3 * 3 * 3)).asType(.float32) * 0.13).reshaped([4, 3, 3, 3])
        let scale = MLXArray.ones([64]), shift = MLXArray.zeros([64])
        func graph(_ stream: StreamOrDevice) -> [MLXArray] {
            [
                matmul(a, a.transposed(), stream: stream),
                conv2d(image, kernel, stride: 1, padding: 1, stream: stream),
                softmax(a, axis: -1, stream: stream),
                MLXFast.layerNorm(a, weight: scale, bias: shift, eps: 1e-5, stream: stream),
                matmul(a.asType(.float16), a.asType(.float16), stream: stream).asType(.float32),
            ]
        }
        let gpu = graph(.gpu), cpu = graph(.cpu)
        eval(gpu + cpu)
        var worst: Float = 0
        for (g, c) in zip(gpu, cpu) {
            let scale = max(1, abs(c).max().item(Float.self))
            worst = max(worst, abs(g - c).max().item(Float.self) / scale)
        }
        return ["max_relative_difference": Double(worst), "checks": gpu.count]
    }
}
