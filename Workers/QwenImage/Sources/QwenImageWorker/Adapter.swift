// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN
import MLXRandom
import ArsenalWire
import ModelPrecision
import QwenImageNative

/// Qwen-Image-2.1 and Qwen-Image-2.1-Turbo: `generate` (text to RGBA image) and `edit` (prompt plus 1–10 reference
/// images: edits, multi-reference composition, masks or annotations drawn into a reference). Both checkpoints share this
/// adapter; Turbo differs only in its saved 8-step sigma schedule, which the snapshot's model_index.json carries.
final class QwenImageAdapter: NativeModel {
    static let id = "qwen-image-2.1"
    let architecture = QwenImageAdapter.id
    let operations: Set<String> = ["generate", "edit"]
    private(set) var pipeline: QwenImagePipeline?
    var peakBytes: Int { Memory.peakMemory }
    var parameters: [String: MLXArray] {
        guard let pipeline else { return [:] }
        var result: [String: MLXArray] = [:]
        let modules: [(String, Module?)] = [("text_encoder.", pipeline.encoder), ("transformer.", pipeline.transformer), ("vae.", pipeline.vae)]
        for (prefix, module) in modules {
            for (name, value) in module?.parameters().flattened() ?? [] { result[prefix + name] = value }
        }
        return result
    }

    /// The selected snapshot's half format: the served base, or the base an int8/int4 tier was derived from.
    static func half(_ context: NativeLoadContext) throws -> DType {
        switch context.precision {
        case "fp16": return .float16
        case "bf16": return .bfloat16
        default:
            if let serving = context.config["serving_dtype"]?.string { return serving == "bf16" ? .bfloat16 : .float16 }
            if context.randomWeights { return QwenImageWorker.servingHalf }
            // A stored pack keeps its scales and norms in the base half; read the dtype from a transformer header.
            let directory = context.weights.appendingPathComponent("transformer")
            for url in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) where url.pathExtension == "safetensors" {
                let archive = try TensorArchive(url)
                let dtypes: [DType] = archive.entries.values.map(\.dtype)
                if let floating = dtypes.first(where: { $0 == .bfloat16 || $0 == .float16 }) { return floating }
            }
            return QwenImageWorker.servingHalf
        }
    }

    func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        try cancellation.check()
        let half = try Self.half(context)
        let tier = try NativePrecisionTier(context.precision, half: half)
        let encoder: Qwen3VLEncoder, transformer: QwenImageTransformer, vae: QwenImageVAE
        let tokenize: (String) -> [Int], scheduler: SchedulerConfig
        if context.randomWeights {
            encoder = Qwen3VLEncoder(SmallConfigs.encoder); transformer = QwenImageTransformer(SmallConfigs.transformer); vae = QwenImageVAE(SmallConfigs.vae)
            for module in [encoder as Module, transformer, vae] {
                try NativeParameters.initializeRandom(module, half: half)
                if let bits = tier.bits {
                    quantize(model: module, groupSize: 64, bits: bits) { name, layer in
                        guard let weight = (layer as? Linear)?.weight ?? (layer as? Embedding)?.weight else { return false }
                        return module !== vae && QwenImageWeights.eligible(name + ".weight", weight.shape)
                    }
                }
            }
            let bytes = ByteTokens(config: SmallConfigs.encoder)
            tokenize = { bytes.encode($0) }
            var s = SchedulerConfig()
            if context.config["turbo"]?.bool == true { s.dynamicShifting = false; s.shiftTerminal = nil; s.sampleSigmas = [1.0, 0.8, 0.4] }
            scheduler = s
        } else {
            let root = context.weights
            func json(_ path: String) throws -> [String: Any] { try ConfigFile.json(root.appendingPathComponent(path)) }
            encoder = Qwen3VLEncoder(try Qwen3VLConfig(json: json("text_encoder/config.json")))
            transformer = QwenImageTransformer(try TransformerConfig(json: json("transformer/config.json")))
            vae = QwenImageVAE(try VAEConfig(json: json("vae/config.json")))
            scheduler = try SchedulerConfig(json: json("scheduler/scheduler_config.json"), pipeline: json("model_index.json"))
            let tokenizer = try LocalTokenizer.load(root.appendingPathComponent("processor"))
            tokenize = { tokenizer.encode(text: $0, addSpecialTokens: false) }
            func archives(_ directory: String) throws -> [URL] {
                try FileManager.default.contentsOfDirectory(at: root.appendingPathComponent(directory), includingPropertiesForKeys: nil)
                    .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
            }
            _ = try tier.load(
                encoder, archives: archives("text_encoder"), map: { key, value in QwenImageWeights.encoderName(key).map { ($0, value) } },
                eligible: QwenImageWeights.eligible)
            try cancellation.check()
            _ = try tier.load(
                transformer, archives: archives("transformer"), map: { key, value in (QwenImageWeights.transformerName(key), value) },
                eligible: QwenImageWeights.eligible)
            try cancellation.check()
            _ = try tier.load(
                vae, archives: archives("vae"),
                map: { key, value in
                    QwenImageWeights.vaeName(key).map { ($0, QwenImageWeights.vaeTensor($0, value)) }
                }, eligible: { _, _ in false })
        }
        for module in [encoder as Module, transformer, vae] { module.train(false) }
        pipeline = QwenImagePipeline(encoder: encoder, transformer: transformer, vae: vae, tokenize: tokenize, scheduler: scheduler)
    }

    func unload() { pipeline = nil; Memory.clearCache() }

    func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard let pipeline else { throw WorkerFailure(code: "not_loaded", message: "Load Qwen-Image before generating.") }
        let request = try ImageRequest(operation: operation, input: input, turbo: pipeline.scheduler.sampleSigmas != nil)
        var generation = GenerationRequest(prompt: request.prompt)
        generation.negativePrompt = request.negativePrompt; generation.guidance = request.guidance
        generation.width = request.width; generation.height = request.height; generation.steps = request.steps
        generation.images = try request.images.map { try ImageOps.read($0) }
        let total = try pipeline.schedule(generation).steps * request.count
        var images: [JSONValue] = [], seeds: [JSONValue] = [], written: [URL] = []
        let start = Date()
        do {
            for index in 0..<request.count {
                try cancellation.check()
                let seed = UInt32(truncatingIfNeeded: UInt64(request.seed) + UInt64(index))
                generation.seed = UInt64(seed)
                let image = try pipeline.generate(generation) { step, steps in
                    let done = index * steps + step
                    try emit(
                        .json([
                            "event": "progress", "phase": "step", "step": .number(Double(done)), "total": .number(Double(total)),
                            "fraction": .number(Double(done) / Double(total))
                        ]))
                    try cancellation.check()
                }
                try cancellation.check()
                let url = request.output.appendingPathComponent("image-\(index + 1)-\(String(format: "%08x", seed)).png")
                try ImageOps.writePNG(image, to: url); written.append(url)
                var file = try NativeMedia.file(url, root: request.output, mime: "image/png", id: "image-\(index + 1)")
                file["alpha"] = .bool(image.transparent)
                images.append(.object(file)); seeds.append(.number(Double(seed)))
            }
        } catch {
            for url in written { try? FileManager.default.removeItem(at: url) }
            throw error
        }
        return ["files": .array(images), "seed": .number(Double(request.seed)), "seeds": .array(seeds), "seconds": .number(Date().timeIntervalSince(start))]
    }
}

/// The request contract of ops/generate.input.json and ops/edit.input.json, rechecked at the worker boundary.
struct ImageRequest {
    var prompt: String, negativePrompt: String?, guidance: Float = 1
    var width: Int?, height: Int?, steps: Int, seed: UInt32, count: Int
    var images: [URL]
    var output: URL

    /// The maker's transparent-image prompt format (Qwen-Image README, "Transparent Image Generation (RGBA)"):
    /// "This is an RGBA image with transparency. <description>. The image has alpha channel and the background is transparent."
    static let transparencyOpening = "This is an RGBA image with transparency."
    static let transparencyClosing = "The image has alpha channel and the background is transparent."
    static func transparent(_ prompt: String) -> String {
        var body = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if body.hasPrefix(transparencyOpening) { body = String(body.dropFirst(transparencyOpening.count)).trimmingCharacters(in: .whitespacesAndNewlines) }
        if body.hasSuffix(transparencyClosing) { body = String(body.dropLast(transparencyClosing.count)).trimmingCharacters(in: .whitespacesAndNewlines) }
        if let last = body.last, !".!?".contains(last) { body += "." }
        return [transparencyOpening, body, transparencyClosing].filter { !$0.isEmpty }.joined(separator: " ")
    }

    init(operation: String, input: JSONObject, turbo: Bool, randomSeed: () -> UInt32 = { UInt32.random(in: 0...UInt32.max) }) throws {
        func invalid(_ message: String) -> WorkerFailure { WorkerFailure(code: "invalid_request", message: message) }
        func integer(_ key: String, _ range: ClosedRange<Int>) throws -> Int? {
            guard let value = input[key] else { return nil }
            guard let v = value.number, v.isFinite, v.rounded() == v, range.contains(Int(exactly: v) ?? Int.min) else {
                throw invalid("\(key) must be an integer from \(range.lowerBound) to \(range.upperBound).")
            }
            return Int(v)
        }
        guard ["generate", "edit"].contains(operation) else { throw WorkerFailure(code: "unsupported_op", message: "Unsupported operation.") }
        guard let prompt = input["prompt"]?.string, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, prompt.count <= 32768 else {
            throw invalid("prompt must contain 1 to 32768 characters.")
        }
        self.prompt = prompt
        if let value = input["transparent"] {
            guard operation == "generate", let flag = value.bool else { throw invalid("transparent is a generate option: true or false.") }
            if flag { self.prompt = Self.transparent(prompt) }
        }
        width = try integer("width", 256...2752); height = try integer("height", 256...2752)
        for side in [width, height].compactMap({ $0 }) where side % 32 != 0 { throw invalid("Width and height must be multiples of 32.") }
        if operation == "generate" { width = width ?? 1024; height = height ?? 1024 }
        // Native sizes only: up to 2048 × 2048 pixels (4.19 MP), plus 2752 × 1536, the two shapes the model card documents.
        if let width, let height, width * height > 2048 * 2048, (width, height) != (2752, 1536) {
            throw invalid("Width × height must be at most 2048 × 2048 pixels (or 2752 × 1536); choose a smaller size.")
        }
        if turbo {
            if input["steps"] != nil { throw invalid("Qwen-Image-2.1-Turbo always runs its saved 8-step schedule.") }
            if input["guidance"] != nil || input["negative_prompt"] != nil { throw invalid("Qwen-Image-2.1-Turbo runs without guidance.") }
            steps = 8
        } else {
            steps = try integer("steps", 1...100) ?? 40
            if let g = input["guidance"] {
                guard let value = g.number, value.isFinite, (1...10).contains(value) else { throw invalid("guidance must be from 1 to 10.") }
                guidance = Float(value)
            }
            if let negative = input["negative_prompt"] {
                guard let text = negative.string, text.count <= 32768 else { throw invalid("negative_prompt must be text.") }
                negativePrompt = text
            }
        }
        seed = UInt32(try integer("seed", 0...Int(UInt32.max)) ?? Int(randomSeed()))
        count = try integer("n", 1...4) ?? 1
        if operation == "edit" {
            guard let list = input["images"]?.array, (1...10).contains(list.count) else { throw invalid("edit needs 1 to 10 reference images.") }
            images = try list.map { item in
                guard let path = item["path"]?.string, path.hasPrefix("/") else { throw invalid("Each reference image needs an absolute path.") }
                return URL(fileURLWithPath: path)
            }
        } else {
            guard input["images"] == nil else { throw invalid("generate takes no images; use edit.") }
            images = []
        }
        guard let out = input["out_dir"]?.string, out.hasPrefix("/") else { throw invalid("out_dir must be an absolute job directory.") }
        output = URL(fileURLWithPath: out)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: output.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw invalid("out_dir must be an existing job directory.")
        }
    }
}
