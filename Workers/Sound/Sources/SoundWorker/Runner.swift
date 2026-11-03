// SPDX-License-Identifier: MIT
import Darwin
import Foundation
import MLX
import ArsenalWire
import WorkerTransport
import SoundEngine
import VireoCore

final class Runner {
    let wire: WorkerChannel
    var loaded: (any AudioModel)?
    var dtype: String?
    init(wire: WorkerChannel) { self.wire = wire }
    func memory() -> JSONValue {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return [
            "footprint_peak_mb": .number(result == KERN_SUCCESS ? Double(info.ledger_phys_footprint_peak) / 1_000_000 : 0),
            "footprint_mb": .number(result == KERN_SUCCESS ? Double(info.phys_footprint) / 1_000_000 : 0),
            "mlx_active_mb": .number(loaded == nil ? 0 : Double(Memory.activeMemory) / 1_000_000),
            "mlx_cache_mb": .number(loaded == nil ? 0 : Double(Memory.cacheMemory) / 1_000_000)
        ]
    }
    func status() -> JSONObject {
        var s: JSONObject = ["memory": memory(), "loaded": .bool(loaded != nil)]
        if let loaded {
            s["model"] = .string(loaded.id); s["dtype"] = dtype.map(JSONValue.string); s["engine"] = .string(loaded.engineLabel); s["engine_reason"] = .string(loaded.engineReason)
        }
        return s
    }
    func pushStatus() throws { var s = status(); s["event"] = "status"; try wire.send(s) }
    func run(_ r: JSONObject, control: GenerationControl) async throws -> Bool {
        let id = r["id"] ?? .null
        if let op = r["op"]?.string, ["load", "generate_sound", "generate_music", "enhance", "convert"].contains(op) {
            Memory.peakMemory = 0
            try wire.scheduled(id, phase: op == "load" ? "loading" : "step", peak: { Double(Memory.peakMemory) / 1e6 })
        }
        switch r["op"]?.string {
        case "load":
            guard loaded == nil else { throw WorkerFailure(code: "already_loaded", message: "Unload before loading another model") }
            let folder = try string(r, "model_dir"), weights = try string(r, "weights_dir")
            let config = r["config"]?.object ?? [:]
            let modelID = config["model_id"]?.string ?? URL(fileURLWithPath: folder).lastPathComponent
            let dtype = try string(r, "dtype")
            try checkDtype(dtype, model: modelID)
            // Arsenal always sends fast; the no-weights contract check (model-check smoke) omits it and means Fast.
            guard let fast = r["fast"] == nil && config["random_weights"]?.bool == true ? true : r["fast"]?.bool else {
                throw EngineError("load requires fast as a boolean")
            }
            var options: [String: String] = [:]
            if modelID == "stable-audio-3-small-sfx" { options["variant"] = "sm-sfx" }
            if let seconds = config["settings"]?["chunk_seconds"]?.number ?? config["chunk_seconds"]?.number {
                guard seconds.isFinite, (0.1...30).contains(seconds) else { throw EngineError("chunk_seconds must be 0.1–30") }
                options["chunk_seconds"] = String(seconds)
            }
            if dtype == "int8" || dtype == "int4" { options["quantized_directory"] = weights; options["quantize_bits"] = dtype == "int8" ? "8" : "4" }
            if config["random_weights"]?.bool == true {
                // No-weights contract check (scripts/model-check.py smoke): isolated lab workers only.
                guard ProcessInfo.processInfo.environment["ARSENAL_NATIVE_LAB"] == "1", SoundEngines.randomWeightModels.contains(modelID) else {
                    throw WorkerFailure(code: "invalid_request", message: "Random weights require an isolated ARSENAL_NATIVE_LAB worker and a model that supports them.")
                }
                options["random_weights"] = "1"
            }
            let cell = ModelCell(tier: dtype == "int8" ? "8" : dtype == "int4" ? "4" : "16", flavour: "optimized", mode: fast ? "fast" : "exact")
            let source = ModelSource(
                id: modelID, engine: SoundEngines.engine(for: modelID), directory: URL(fileURLWithPath: weights), precision: dtype,
                options: options, overrides: FastPathComponents.overrides(cell))
            RuntimeCaches.useBundledMetallib(); RuntimeCaches.prepare(cacheLimitMB: 128); RuntimeCaches.resetPeak()
            let start = Date(); let model = try await SoundEngines.load(source)
            if control.isCancelled { await model.unload(); RuntimeCaches.retire(); throw CancellationError() }
            loaded = model; self.dtype = dtype
            try wire.reply(
                id, ["load_s": .number(Date().timeIntervalSince(start)), "memory": memory(), "engine": .string(model.engineLabel), "engine_reason": .string(model.engineReason)])
            try pushStatus()
        case "generate_sound", "generate_music":
            guard let model = loaded as? any SoundModel else { throw WorkerFailure(code: "not_loaded", message: "Load a sound model before generating audio") }
            try await generate(r, model: model, control: control); try pushStatus()
        case "enhance":
            guard let model = loaded as? any EnhanceModel else { throw WorkerFailure(code: "not_loaded", message: "Load an enhancement model before enhancing audio") }
            try await enhance(r, model: model, control: control); try pushStatus()
        case "parameter_inventory":
            guard ProcessInfo.processInfo.environment["ARSENAL_NATIVE_LAB"] == "1",
                let inventory = (loaded as? MossFormer2Model)?.parameterInventory() ?? (loaded as? UniPASEModel)?.parameterInventory()
                    ?? (loaded as? DeepFilterNet3Model)?.parameterInventory()
            else {
                throw WorkerFailure(code: "invalid_request", message: "Parameter inventory requires a loaded isolated lab worker.")
            }
            let parameters = Dictionary(
                uniqueKeysWithValues: inventory.map { name, shape, dtype in
                    (name, JSONValue.object(["shape": .array(shape.map { .number(Double($0)) }), "dtype": .string(String(describing: dtype))]))
                })
            try wire.reply(id, ["parameters": .object(parameters)])
        case "convert":
            guard loaded == nil else { throw EngineError("Convert requires an unloaded worker") }
            let recipe = try string(r, "recipe"), dtype = try string(r, "dtype")
            if let half = SoundEngines.halfCastRecipes[recipe] {
                // Whole-model casts whose only output is one half dtype (no quantized tiers).
                guard dtype == half.dtype else { throw WorkerFailure(code: "unsupported_recipe", message: "\(recipe) makes \(half.dtype) only") }
                let from = URL(fileURLWithPath: try string(r, "src_dir")), to = URL(fileURLWithPath: try string(r, "dst_dir"))
                guard try !FileManager.default.fileExists(atPath: to.path) || FileManager.default.contentsOfDirectory(atPath: to.path).isEmpty else {
                    throw EngineError("Conversion destination must be empty")
                }
                RuntimeCaches.useBundledMetallib()
                try FileManager.default.createDirectory(at: to, withIntermediateDirectories: true)
                try Device.withDefaultDevice(.cpu) { try half.convert(from, to) }
                if control.isCancelled { throw CancellationError() }
                try wire.reply(id, ["files": .array(try convertedFiles(to))])
                return false
            }
            let baseRecipe = recipe.components(separatedBy: ";")[0]
            let model = String(
                baseRecipe.replacingOccurrences(of: "mlx-affine-g64:", with: "").replacingOccurrences(of: "cast:", with: "").replacingOccurrences(of: "intake-v1:", with: "")
                    .replacingOccurrences(of: "intake-v2:", with: ""))
            guard ["re-use", "stable-audio-3-medium", "stable-audio-3-small-sfx"].contains(model),
                ["cast:re-use", "cast:stable-audio-3-medium", "cast:stable-audio-3-small-sfx"].contains(recipe)
                    || (recipe.hasPrefix("mlx-affine-g64:") || recipe.hasPrefix("intake-v1:") || recipe == "intake-v2:stable-audio-3-small-sfx")
            else { throw WorkerFailure(code: "unsupported_recipe", message: "Unknown sound conversion recipe") }
            guard
                (dtype == "fp16" || dtype == "bf16" && model == "re-use") && ["cast:re-use", "cast:stable-audio-3-medium", "cast:stable-audio-3-small-sfx"].contains(recipe)
                    || (dtype == "int8" || dtype == "int4") && r["group_size"]?.number == 64
            else {
                throw EngineError("Conversion requires a supported dtype and affine group_size=64")
            }
            let from = URL(fileURLWithPath: try string(r, "src_dir")), to = URL(fileURLWithPath: try string(r, "dst_dir"))
            guard try !FileManager.default.fileExists(atPath: to.path) || FileManager.default.contentsOfDirectory(atPath: to.path).isEmpty else {
                throw EngineError("Conversion destination must be empty")
            }
            let fallback = recipe.components(separatedBy: ";half=").dropFirst().first ?? ""
            let source = ModelSource(
                id: model, engine: model == "re-use" ? "re-use" : "stable-audio-3", directory: from, precision: ["fp16", "bf16"].contains(dtype) ? "fp32" : "fp16",
                options: model == "stable-audio-3-small-sfx"
                    ? ["variant": "sm-sfx"]
                    : recipe.hasPrefix("intake-v1:") || recipe == "cast:stable-audio-3-medium"
                        ? ["intake": "1", "intake_half_modules": fallback] : ["intake_half_modules": fallback])
            // MLX creates its stream objects on first use, even for a CPU cast. Configure the
            // bundled library before that first use, and keep the conversion graph on the CPU.
            RuntimeCaches.useBundledMetallib()
            try Device.withDefaultDevice(.cpu) { try SoundEngines.convert(source, dtype: dtype, into: to) }
            if control.isCancelled { throw CancellationError() }
            try wire.reply(id, ["files": .array(try convertedFiles(to))])
        case "measurement_source":
            guard loaded == nil else { throw EngineError("Source casts need an unloaded worker") }
            try ReUseEngine.castSource(from: URL(fileURLWithPath: string(r, "src_dir")), to: URL(fileURLWithPath: string(r, "dst_dir")), truncate: r["truncate"]?.bool == true)
            try wire.reply(id)
        case "measurement_components":
            let fixture = URL(fileURLWithPath: try string(r, "fixture_dir")), output = URL(fileURLWithPath: try string(r, "output_dir"))
            if let model = loaded as? StableAudio3SoundModel {
                try model.intakeComponents(fixture: fixture, output: output, capture: r["capture"]?.bool == true)
            } else if let model = loaded as? ReUseModel {
                try await model.intakeComponents(fixture: fixture, output: output, capture: r["capture"]?.bool == true)
            } else {
                throw EngineError("Load a sound or enhancement model before component capture")
            }
            try wire.reply(id, ["memory": memory()])
        case "status": try wire.reply(id, status())
        case "unload":
            let hadModel = loaded != nil
            if let loaded { await loaded.unload() }; loaded = nil; dtype = nil
            if hadModel { RuntimeCaches.retire() }
            try wire.reply(id); try pushStatus(); return true
        default: throw WorkerFailure(code: "unsupported_op", message: "Unsupported sound worker operation")
        }
        return false
    }
    func generate(_ r: JSONObject, model: any SoundModel, control: GenerationControl) async throws {
        let request = try soundRequest(r, model: model.id, maxSeconds: model.maxSeconds)
        try checkOutput(r)
        let start = Date(); let audio = try await model.generate(request, control: control)
        if control.isCancelled { throw CancellationError() }
        try output(audio, r: r, elapsed: Date().timeIntervalSince(start), seed: request.seed)
    }
    func enhance(_ r: JSONObject, model: any EnhanceModel, control: GenerationControl) async throws {
        try checkOutput(r)
        guard let path = r["audio"]?["path"]?.string, !path.isEmpty else { throw invalid("enhance requires audio.path") }
        let mode = r["mode"]?.string ?? "denoise"
        guard ["denoise", "restore", "upscale"].contains(mode) else { throw invalid("mode must be denoise, restore or upscale") }
        let rate = r["rate"]?.number
        if let rate { guard rate.isFinite, [8000, 16000, 22050, 24000, 32000, 44100, 48000].contains(rate) else { throw invalid("Unsupported enhancement sample rate") } }
        var options = rate.map { ["rate": String(Int($0))] } ?? [:]
        // MossFormer2's decode controls (seconds); the model checks their ranges.
        for key in ["one_time_decode_length", "decode_window"] {
            guard let value = r[key] else { continue }
            guard let n = value.number, n.isFinite, n.rounded() == n, (1...60).contains(n) else { throw invalid("\(key) must be a whole number of seconds") }
            options[key] = String(Int(n))
        }
        // UniPASE's packet-loss concealment switch.
        if let value = r["enable_plc"] {
            guard let on = value.bool else { throw invalid("enable_plc must be a boolean") }
            options["enable_plc"] = on ? "1" : "0"
        }
        // DeepFilterNet3's attenuation limit (dB) and post filter.
        if let value = r["atten_lim_db"] {
            guard let db = value.number, db.isFinite, (0...100).contains(db) else { throw invalid("atten_lim_db must be 0–100 dB") }
            options["atten_lim_db"] = String(db)
        }
        if let value = r["post_filter"] {
            guard let on = value.bool else { throw invalid("post_filter must be a boolean") }
            options["post_filter"] = on ? "1" : "0"
        }
        let input = try PCM.input(URL(fileURLWithPath: path)), start = Date()
        let audio = try await model.enhance(input, request: EnhanceRequest(mode: mode, options: options), control: control)
        if control.isCancelled { throw CancellationError() }
        try output(audio, r: r, elapsed: Date().timeIntervalSince(start))
    }
    private func checkOutput(_ r: JSONObject) throws {
        if let stream = r["stream"], stream.bool == nil { throw invalid("stream must be a boolean") }
        guard r["stream"]?.bool == true || r["out_dir"]?.string?.isEmpty == false else { throw invalid("Audio request requires stream=true or out_dir") }
    }
    private func output(_ audio: AudioClip, r: JSONObject, elapsed: Double, seed: UInt64? = nil) throws {
        guard audio.frames > 0 else { throw EngineError("Model produced empty audio") }
        let pcm = try PCM.encode(audio.samples), id = r["id"] ?? .null
        if r["stream"]?.bool == true {
            let stream = UInt32.random(in: 1...UInt32.max)
            try wire.send([
                "id": id, "event": "audio_begin", "stream": .number(Double(stream)), "sample_rate": .number(Double(audio.sampleRate)), "channels": .number(Double(audio.channels)),
                "encoding": "pcm_s16le"
            ])
            try wire.audio(stream: stream, pcm: pcm)
        }
        var result: JSONObject = [
            "duration_s": .number(audio.seconds), "gen_s": .number(elapsed), "seconds": .number(elapsed), "rtf": .number(elapsed / audio.seconds),
            "sample_rate": .number(Double(audio.sampleRate)), "channels": .number(Double(audio.channels))
        ]
        if let seed { result["seed"] = .number(Double(seed)) }
        if let directory = r["out_dir"]?.string {
            let sink = try WaveSink(directory: URL(fileURLWithPath: directory), rate: audio.sampleRate, channels: audio.channels)
            try sink.append(pcm); try sink.finish()
            result["path"] = .string(sink.file.path); result["bytes"] = .number(Double(pcm.count + 44)); result["sha256"] = .string(try PCM.digest(sink.file))
            // The same file in the generic file-output form, which schema-3 validation and the no-weights check read.
            result["files"] = [["id": "audio", "path": .string(sink.file.path), "mime": "audio/wav"]]
        }
        try wire.reply(id, result)
    }
}

/// Every regular file under a conversion output, with its size and SHA-256.
func convertedFiles(_ directory: URL) throws -> [JSONValue] {
    let walker = FileManager.default.enumerator(atPath: directory.path)!
    var files: [JSONValue] = []
    while let entry = walker.nextObject() {
        guard let relative = entry as? String else { continue }
        let url = directory.appendingPathComponent(relative), attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard attrs[.type] as? FileAttributeType == .typeRegular else { continue }
        files.append(["path": .string(relative), "bytes": .number((attrs[.size] as? NSNumber)?.doubleValue ?? 0), "sha256": .string(try PCM.digest(url))])
    }
    return files
}

func string(_ r: JSONObject, _ key: String) throws -> String {
    guard let s = r[key]?.string, !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw invalid("\(key) must be a nonempty string") }; return s
}
func checkDtype(_ dtype: String, model: String) throws {
    if let only = SoundEngines.halfOnlyModels[model] {
        guard dtype == only else { throw EngineError("Unsupported dtype for \(model)") }
        return
    }
    guard ["re-use", "stable-audio-3-medium", "stable-audio-3-small-sfx"].contains(model) else { throw EngineError("Unknown sound model") }
    guard ["fp16", "int8", "int4"].contains(dtype) || model == "re-use" && dtype == "bf16" else { throw EngineError("Unsupported dtype for \(model)") }
}
func soundRequest(_ r: JSONObject, model: String, maxSeconds: Double) throws -> SoundRequest {
    let music = r["op"]?.string == "generate_music"
    guard !music || model == "stable-audio-3-medium" else { throw WorkerFailure(code: "unsupported_op", message: "Small SFX generates sound effects only") }
    let prompt = try string(r, "prompt")
    guard prompt.count <= 10_000 else { throw invalid("prompt exceeds 10000 characters") }
    let duration = try number(r, "duration", default: 5)
    guard duration.isFinite, duration > 0, duration <= maxSeconds else { throw invalid("duration exceeds the model limit") }
    let steps = try number(r, "steps", default: 8)
    guard steps.isFinite, steps.rounded() == steps, (1...100).contains(steps) else { throw invalid("steps must be an integer from 1 to 100") }
    let seed = try number(r, "seed", default: Double(UInt32.random(in: 0...UInt32.max)))
    guard seed.isFinite, seed.rounded() == seed, (0...Double(UInt32.max)).contains(seed) else { throw invalid("seed must be a uint32") }
    let guidance = try number(r, "guidance", default: 1)
    guard guidance.isFinite, (0...20).contains(guidance) else { throw invalid("guidance must be 0–20") }
    return SoundRequest(
        kind: music ? .music : .sfx, prompt: prompt, negativePrompt: r["negative_prompt"]?.string, seconds: duration, seed: UInt64(seed), steps: Int(steps), guidance: guidance)
}
func number(_ r: JSONObject, _ key: String, default fallback: Double) throws -> Double {
    guard let value = r[key] else { return fallback }
    guard let number = value.number else { throw invalid("\(key) must be a number") }
    return number
}

/// A caller mistake in the request: invalid_request (HTTP 422). EngineError is the worker's own failure (500).
func invalid(_ message: String) -> WorkerFailure { WorkerFailure(code: "invalid_request", message: message) }
