// SPDX-License-Identifier: MIT
// Request validation and the decide/judge ops, as the folder's earlier Python adapter (runners/mlx/backend.py, git
// history) did them: same checks, codes, statuses and messages, so HTTP and MCP callers see no change.
import D1Engine
import Foundation
import MLX
import Metal
import CryptoKit
import Darwin

struct WorkerError: Error {
    let code: String
    let message: String
    let status: Int?
    init(_ code: String, _ message: String, status: Int? = nil) {
        self.code = code
        self.message = message.split(whereSeparator: \.isNewline).joined(separator: " ")
        self.status = status
    }
}

/// One clean line from an error for a per-item message.
func brief(_ error: Error) -> String {
    let text = "\(error)".trimmingCharacters(in: .whitespacesAndNewlines)
    let line = text.split(whereSeparator: \.isNewline).first.map(String.init) ?? String(describing: type(of: error))
    return String(line.prefix(200))
}

protocol DecisionEngine: AnyObject {
    var convKernelOK: Bool { get }
    func loadPictures(_ urls: [URL], budget: inout PictureBudget) throws -> [Picture]
    func tokens(_ state: PyJSON?, _ questions: [Question]) throws -> Int
    func imageTokens(_ state: PyJSON?, _ questions: [Question], _ pictures: [Picture]) throws -> Int
    func run(_ requests: [Request], cancelled: () -> Bool) throws -> [Outcome]
}
extension SystemOne: DecisionEngine {
    var convKernelOK: Bool { model.convKernelOK }
}

final class Backend: @unchecked Sendable {
    static let model = "d1-3b"
    static let context = 32768
    /// Bounds MLX's buffer cache between requests (the Python runner's 512 MB: 256 MB-2 GB measured the same speed).
    static let cacheLimit = 512 * 1024 * 1024
    /// Items per runtime call; pictures have a separate request byte budget; inside, passes of up to 4,096 tokens.
    static let group = 32

    private(set) var engine: (any DecisionEngine)?
    private var precision = "bf16"
    init(engine: (any DecisionEngine)? = nil, precision: String = "bf16") { self.engine = engine; self.precision = precision }

    // MARK: load

    func load(_ request: PyJSON) throws -> [(String, PyJSON)] {
        if engine != nil { throw WorkerError("already_loaded", "d1-3B is already loaded.") }
        let dtype = request["dtype"]?.string ?? "bf16"
        guard ["bf16", "int8", "int4"].contains(dtype) else {
            throw WorkerError("unsupported_dtype", "d1-3B supports bf16, int8 and int4.", status: 422)
        }
        if let kernels = request["config"]?["kernels"], kernels.truthy { throw WorkerError("unsupported_kernel", "This runner does not accept folder kernels.") }
        guard let path = request["weights_dir"]?.string else { throw WorkerError("missing_weights", "Get the verified d1-3B checkpoint first.") }
        let weights = URL(fileURLWithPath: path).resolvingSymlinksInPath()
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: weights.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw WorkerError("missing_weights", "Get the verified d1-3B checkpoint first.")
        }
        guard MTLCreateSystemDefaultDevice() != nil else { throw WorkerError("unsupported_device", "d1-3B requires Apple Silicon Metal in this runner.") }
        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: weights.appendingPathComponent("config.json"))) as? [String: Any]
        let quant = config?["quantization"] as? [String: Any]
        guard
            (["int4", "int8"].contains(dtype) && quant?["bits"] as? Int == (dtype == "int8" ? 8 : 4) && quant?["group_size"] as? Int == 64)
                || (dtype == "bf16" && quant == nil)
        else { throw WorkerError("unsupported_dtype", "Weights do not match the requested d1 precision.", status: 422) }
        let started = Date()
        do {
            Memory.cacheLimit = Self.cacheLimit
            let engine = try SystemOne(weights: weights)
            try engine.warm()
            self.engine = engine; precision = dtype
        } catch {
            engine = nil
            Memory.clearCache()
            throw error
        }
        var reply: [(String, PyJSON)] = [
            ("load_s", .number(Date().timeIntervalSince(started))), ("engine", .string("mlx-swift/d1-hybrid")), ("memory", memory())
        ]
        if !engine!.convKernelOK {
            reply.append(("engine_reason", .string("Short-convolution kernel failed its load self-test; stock MLX ops run instead.")))
        }
        return reply
    }

    func memory() -> PyJSON {
        .object([
            ("footprint_peak_mb", .number(peakFootprintMB())), ("footprint_mb", .number(footprintMB())), ("mlx_active_mb", .number(Double(Memory.activeMemory) / 1e6)),
            ("mlx_cache_mb", .number(Double(Memory.cacheMemory) / 1e6))
        ])
    }

    func unload() -> [(String, PyJSON)] {
        engine = nil
        Memory.clearCache()
        return [("memory", memory())]
    }

    func convert(_ request: PyJSON) throws -> [(String, PyJSON)] {
        guard engine == nil, request["recipe"]?.string == "mlx-affine-g64:d1-3b", ["int8", "int4"].contains(request["dtype"]?.string ?? ""),
            request["group_size"]?.int == 64, let src = request["src_dir"]?.string, let dst = request["dst_dir"]?.string
        else { throw WorkerError("unsupported_recipe", "Conversion requires unloaded d1 int8/int4 with affine group 64.") }
        let destination = URL(fileURLWithPath: dst)
        try Device.withDefaultDevice(.cpu) {
            try D1Precision.convert(source: URL(fileURLWithPath: src), destination: destination, bits: request["dtype"]?.string == "int8" ? 8 : 4)
        }
        var files: [PyJSON] = []
        let walker = FileManager.default.enumerator(atPath: dst)!
        while let name = walker.nextObject() as? String {
            let file = destination.appendingPathComponent(name)
            let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
            guard attributes[.type] as? FileAttributeType == .typeRegular else { continue }
            let handle = try FileHandle(forReadingFrom: file); defer { try? handle.close() }
            var hash = SHA256()
            while let chunk = try autoreleasepool(invoking: { try handle.read(upToCount: 4 << 20) }), !chunk.isEmpty { hash.update(data: chunk) }
            let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
            files.append(.object([("path", .string(name)), ("bytes", .number((attributes[.size] as? NSNumber)?.doubleValue ?? 0)), ("sha256", .string(digest))]))
        }
        return [("files", .array(files))]
    }

    // MARK: request checks

    private func body(_ request: PyJSON) throws -> [(String, PyJSON)] {
        var body: PyJSON
        if let raw = request["body"] {
            guard let text = raw.string, let parsed = try? PyJSONParser.parse(Data(text.utf8)) else {
                throw WorkerError("invalid_request", "Send a JSON object.", status: 422)
            }
            body = parsed
        } else {
            body = .object((request.fields ?? []).filter { $0.0 != "id" && $0.0 != "op" })
        }
        guard let fields = body.fields else { throw WorkerError("invalid_request", "Send a JSON object.", status: 422) }
        if let model = body["model"], model.string != Self.model { throw WorkerError("wrong_model", "Select model d1-3b explicitly.", status: 422) }
        if let merge = body["merge"], merge.bool != true {
            throw WorkerError("unsupported_merge", "d1-3B uses the vendor shared-state tree; merge must be true.", status: 422)
        }
        if let bits = body["bits"], !bits.isNull, bits.int != 0 && bits.int != (precision == "int4" ? 4 : precision == "int8" ? 8 : 16) {
            throw WorkerError("precision_conflict", "d1-3B is loaded at \(precision); this request cannot change it.", status: 409)
        }
        if engine == nil { throw WorkerError("not_loaded", "Load d1-3B before deciding.") }
        return fields
    }

    private struct Invalid: Error, CustomStringConvertible { let description: String }

    /// Named questions, normalised as the adapter did (instructions default to "", batch label lists become criteria).
    private func questions(_ raw: PyJSON?, batch: Bool = false) throws -> [(String, Question)] {
        guard let fields = raw?.fields, !fields.isEmpty, fields.count <= 64 else {
            throw WorkerError("invalid_request", "questions must contain 1–64 named questions.", status: 422)
        }
        do {
            return try fields.map { name, q in (name, try question(q, batch: batch)) }
        } catch let e as Invalid {
            throw WorkerError("invalid_request", e.description, status: 422)
        }
    }

    private func question(_ q: PyJSON, batch: Bool) throws -> Question {
        guard q.isObject, let type = q["type"]?.string, let kind = QuestionKind(rawValue: type) else {
            throw Invalid(description: "Each question needs type noul, choice or score.")
        }
        let instructions = q["instructions"] ?? .string("")
        switch instructions {
        case .string, .object, .array: break
        default: throw Invalid(description: "instructions must be text, an object or an array.")
        }
        var criteria = q["criteria"]
        if batch, kind == .choice, let labels = criteria?.array {
            guard labels.allSatisfy({ $0.string != nil }) else { throw Invalid(description: "Choice label arrays must contain strings.") }
            var fields: [(String, PyJSON)] = []
            for label in labels.compactMap(\.string) where !fields.contains(where: { $0.0.utf8.elementsEqual(label.utf8) }) {
                fields.append((label, .null))
            }
            criteria = .object(fields)
        }
        switch kind {
        case .choice:
            guard let options = criteria?.fields, !options.isEmpty, options.count <= 64 else { throw Invalid(description: "Choice needs 1–64 named criteria.") }
        case .score:
            guard let levels = criteria?.array, !levels.isEmpty, levels.count <= 10 else {
                throw Invalid(description: "Score needs 1–10 ordered criteria (single-token digit readout).")
            }
        case .noul:
            if let c = criteria, !c.isNull {
                guard let f = c.fields, f.allSatisfy({ $0.0 == "true" || $0.0 == "false" }) else { throw Invalid(description: "Noul criteria accepts only true/false.") }
            }
        }
        let descriptions = criteria?.fields?.map(\.1) ?? criteria?.array ?? []
        for d in descriptions {
            switch d {
            case .string, .object, .array: continue
            case .scalar("null"): continue
            default: throw Invalid(description: "Answer descriptions must be text, an object, an array or null.")
            }
        }
        switch kind {
        case .choice: return Question(kind: .choice, instructions: instructions, options: criteria!.fields!)
        case .score: return Question(kind: .score, instructions: instructions, levels: criteria!.array!)
        case .noul: return Question(kind: .noul, instructions: instructions, noulCriteria: criteria.flatMap { $0.isNull ? nil : $0 })
        }
    }

    private func pictures(_ raw: PyJSON?, budget: inout PictureBudget) throws -> [Picture] {
        guard let values = raw?.array, values.count <= 10 else {
            throw WorkerError("invalid_request", "images must be a list of at most 10 local path objects.", status: 422)
        }
        let urls = try values.map { value in
            guard let f = value.fields, f.count == 1, f[0].0 == "path", let path = f[0].1.string else {
                throw WorkerError("invalid_image", "Each image must be {\"path\": absolute local path}.", status: 422)
            }
            var isDirectory: ObjCBool = false
            guard path.hasPrefix("/"), FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory), !isDirectory.boolValue else {
                throw WorkerError("invalid_image", "Image path must be an existing absolute local file.", status: 422)
            }
            return URL(fileURLWithPath: path)
        }
        do { return try engine!.loadPictures(urls, budget: &budget) } catch { throw WorkerError("invalid_image", brief(error), status: 422) }
    }

    private func validate(_ state: PyJSON?, _ qs: [Question], _ pics: [Picture]) throws {
        guard let engine else { throw WorkerError("not_loaded", "Load d1-3B before deciding.") }
        if state == nil && pics.isEmpty { throw WorkerError("invalid_request", "Supply a text/JSON state or at least one image.", status: 422) }
        if let f = state?.fields, f.contains(where: { ["image", "images", "audio", "video", "videos"].contains($0.0) }) {
            throw WorkerError("unsupported_media", "Use the top-level images path list for d1-3B; audio/video are unsupported.", status: 422)
        }
        if try engine.tokens(state, qs) > Self.context {
            throw WorkerError("context_limit", "State plus question exceeds d1-3B\u{2019}s 32,768-token context.", status: 422)
        }
        if !pics.isEmpty {
            let count: Int
            do { count = try engine.imageTokens(state, qs, pics) } catch {
                throw WorkerError("invalid_image", "The pictures could not be prepared with this state: " + brief(error), status: 422)
            }
            if count > Self.context { throw WorkerError("context_limit", "Images plus state/question exceed 32,768 tokens.", status: 422) }
        }
    }

    private func answers(_ qs: [(String, Question)], _ outcome: (probabilities: [[Double]], read: Int)) throws -> [(String, PyJSON)] {
        guard outcome.probabilities.allSatisfy({ $0.allSatisfy(\.isFinite) }) else { throw WorkerError("non_finite", "d1-3B returned a non-finite answer.") }
        let named = zip(qs, outcome.probabilities).map { ($0.0, Readout.answer($0.1, $1)) }
        return [("answers", .object(named)), ("usage", .object([("input_tokens", .int(outcome.read)), ("output_tokens", .int(0))]))]
    }

    // MARK: ops

    func decide(_ request: PyJSON, cancelled: () -> Bool) throws -> [(String, PyJSON)] {
        let body = PyJSON.object(try self.body(request))
        guard let stateValue = body["state"] else { throw WorkerError("invalid_request", "state is required (null is allowed with images).", status: 422) }
        let state: PyJSON? = stateValue.isNull ? nil : stateValue
        let qs = try questions(body["questions"])
        var pictureBudget = PictureBudget()
        let pics = try pictures(body["images"] ?? .array([]), budget: &pictureBudget)
        try validate(state, qs.map(\.1), pics)
        if cancelled() { throw WorkerError("cancelled", "Request cancelled.") }
        let outcome: Outcome
        do { outcome = try engine!.run([Request(state: state, questions: qs.map(\.1), pictures: pics)], cancelled: cancelled)[0] } catch is Cancelled {
            throw WorkerError("cancelled", "Request cancelled.")
        }
        let result = try answers(qs, try outcome.get())
        if cancelled() { throw WorkerError("cancelled", "Request cancelled.") }
        return result + [("model", .string(Self.model)), ("bits", .int(precision == "int4" ? 4 : precision == "int8" ? 8 : 16))]
    }

    func judge(_ request: PyJSON, cancelled: () -> Bool) throws -> [(String, PyJSON)] {
        let body = PyJSON.object(try self.body(request))
        guard let items = body["items"]?.array, !items.isEmpty, items.count <= 256 else {
            throw WorkerError("invalid_request", "items must contain 1–256 states.", status: 400)
        }
        let qs = try questions(body["questions"], batch: true)
        let images = body["images"] ?? .array(items.map { _ in .array([]) })
        guard let imageLists = images.array, imageLists.count == items.count else {
            throw WorkerError("invalid_request", "Batch images must contain one image list per item.", status: 422)
        }
        func failed(_ message: String) -> PyJSON { .object([("error", .string(message)), ("model", .null), ("ms", .int(0))]) }
        var pictureBudget = PictureBudget()
        var results: [PyJSON] = []
        for offset in stride(from: 0, to: items.count, by: Self.group) {
            if cancelled() { throw WorkerError("cancelled", "Request cancelled.") }
            var valid: [Request] = []
            var indexes: [Int] = []
            var part: [PyJSON?] = []
            for k in offset..<min(offset + Self.group, items.count) {
                let state: PyJSON? = items[k].isNull ? nil : items[k]
                do {
                    var itemBudget = pictureBudget
                    let pics = try pictures(imageLists[k], budget: &itemBudget)
                    try validate(state, qs.map(\.1), pics)
                    pictureBudget = itemBudget
                    indexes.append(part.count)
                    valid.append(Request(state: state, questions: qs.map(\.1), pictures: pics))
                    part.append(nil)
                } catch let e as WorkerError {
                    part.append(failed(e.message))
                } catch {
                    FileHandle.standardError.write(Data("item \(k): \(error)\n".utf8))
                    part.append(failed("This item could not be prepared: " + brief(error)))
                }
            }
            if !valid.isEmpty {
                let start = Date()
                let outcomes: [Outcome]
                do { outcomes = try engine!.run(valid, cancelled: cancelled) } catch is Cancelled { throw WorkerError("cancelled", "Request cancelled.") }
                let perMS = Date().timeIntervalSince(start) * 1000 / Double(valid.count)
                for (index, outcome) in zip(indexes, outcomes) {
                    do {
                        part[index] = .object(try answers(qs, try outcome.get()) + [("model", .string(Self.model)), ("ms", .number(perMS))])
                    } catch let e as WorkerError {
                        part[index] = failed(e.message)
                    } catch {
                        FileHandle.standardError.write(Data("item: \(error)\n".utf8))
                        part[index] = failed("Decision failed for this item: " + brief(error))
                    }
                }
            }
            results += part.map { $0! }
        }
        if cancelled() { throw WorkerError("cancelled", "Request cancelled.") }
        return [("results", .array(results)), ("bits", .int(precision == "int4" ? 4 : precision == "int8" ? 8 : 16))]
    }
}

/// The process's physical footprint (task_vm_info.phys_footprint), decimal MB.
func footprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    return result == KERN_SUCCESS ? Double(info.phys_footprint) / 1e6 : 0
}

private func peakFootprintMB() -> Double {
    var info = rusage_info_v4()
    let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) }
    }
    return status == 0 ? Double(info.ri_lifetime_max_phys_footprint) / 1e6 : 0
}

extension Backend {
    func intakeComponents(_ request: PyJSON) throws -> [(String, PyJSON)] {
        guard let system = engine as? SystemOne, let fixture = request["fixture_dir"]?.string, let output = request["output_dir"]?.string else {
            throw WorkerError("invalid_request", "Load d1 and supply fixture_dir/output_dir")
        }
        try system.intakeComponents(fixture: URL(fileURLWithPath: fixture), output: URL(fileURLWithPath: output), capture: request["capture"]?.bool == true)
        return [("captured", .bool(true))]
    }
}
