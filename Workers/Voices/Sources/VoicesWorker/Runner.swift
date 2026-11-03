// SPDX-License-Identifier: MIT
import Darwin
import Foundation
import MLX
import FishDrama3
import MLXAudioTTS
import ArsenalWire
import WorkerTransport
import VireoCore
import VireoEngine
import VoxCPM2

protocol WorkerSpeechModel: SpeechModel {
    func setSpeed(_ speed: Float) throws
}
extension MLXSpeech: WorkerSpeechModel {}
extension Drama3Speech: WorkerSpeechModel {
    func setSpeed(_ speed: Float) throws {
        guard speed == 1 else { throw WorkerFailure(code: "invalid_request", message: "Drama 3 does not support speed control") }
    }
}

extension VoxCPM2Speech: WorkerSpeechModel {
    func setSpeed(_ speed: Float) throws {
        guard speed == 1 else { throw WorkerFailure(code: "invalid_request", message: "VoxCPM2 has no speed control; describe the pace in instructions") }
    }
}

final class Runner {
    let wire: WorkerChannel
    var loaded: (any WorkerSpeechModel)?
    var dtype: String?
    var modelRevision: String?
    var fast = false
    let support: URL?
    init(wire: WorkerChannel, support: URL? = nil) {
        RuntimeCaches.useBundledMetallib()
        self.wire = wire; self.support = support
    }

    static func recipeModel(_ recipe: String) -> String {
        if recipe.hasPrefix("intake-v2:") || recipe.hasPrefix("intake-v3:") { return recipe.split(separator: ":").dropFirst().first.map(String.init) ?? "" }
        if recipe.hasPrefix("cast:") { return recipe.split(separator: ":").dropFirst().first.map(String.init) ?? "" }
        for prefix in ["mlx-affine-g64:", "intake-v1:"] where recipe.hasPrefix(prefix) { return String(recipe.dropFirst(prefix.count)) }
        return recipe
    }
    func memory() -> JSONValue {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return [
            "footprint_peak_mb": .number(Self.peakFootprintMB()),
            "footprint_mb": .number(result == KERN_SUCCESS ? Double(info.phys_footprint) / 1_000_000 : 0),
            "mlx_active_mb": .number(loaded == nil ? 0 : Double(Memory.activeMemory) / 1_000_000),
            "mlx_cache_mb": .number(loaded == nil ? 0 : Double(Memory.cacheMemory) / 1_000_000)
        ]
    }
    static func peakFootprintMB() -> Double {
        var info = rusage_info_v4()
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) }
        }
        return status == 0 ? Double(info.ri_lifetime_max_phys_footprint) / 1e6 : 0
    }
    func status() -> JSONObject {
        VireoFolderKernels.finalize(optimized: loaded?.engineLabel == "optimized")
        let kernels: [JSONValue] = VireoFolderKernels.wire.compactMap { entry in
            guard let data = try? JSONSerialization.data(withJSONObject: entry), let object = try? WireJSON.decode(data) else { return nil }
            return .object(object)
        }
        var s: JSONObject = ["memory": memory(), "loaded": .bool(loaded != nil), "kernels": .array(kernels)]
        if let loaded {
            s["model"] = .string(loaded.id); s["dtype"] = dtype.map(JSONValue.string); s["engine"] = .string(loaded.engineLabel); s["engine_reason"] = .string(loaded.engineReason)
        }
        return s
    }
    func pushStatus() throws { var s = status(); s["event"] = "status"; try wire.send(s) }

    func run(_ r: JSONObject, control: GenerationControl) async throws -> Bool {
        let id = r["id"] ?? .null
        if let op = r["op"]?.string, ["load", "speak", "design_voice", "clone_voice", "convert"].contains(op) {
            Memory.peakMemory = 0
            try wire.scheduled(id, phase: op == "load" ? "loading" : "step", peak: { Double(Memory.peakMemory) / 1e6 })
        }
        switch r["op"]?.string {
        case "load":
            guard loaded == nil else { throw WorkerFailure(code: "already_loaded", message: "Unload this runner before loading another model") }
            let folder = try string(r, "model_dir"), weights = try string(r, "weights_dir")
            let config = r["config"]?.object ?? [:]
            let modelID = config["model_id"]?.string ?? URL(fileURLWithPath: folder).lastPathComponent
            if modelID == Drama3Speech.modelID {
                try await loadDrama(r, weights: URL(fileURLWithPath: weights, isDirectory: true), control: control)
                return false
            }
            if modelID == VoxCPM2Speech.modelID {
                try await loadVoxCPM2(r, weights: URL(fileURLWithPath: weights, isDirectory: true), control: control)
                return false
            }
            let spec = try SpeechCatalog.model(modelID)
            let precision = try string(r, "dtype")
            if !["fp16", "bf16"].contains(precision) { try SpeechCatalog.checkPrecision(precision, spec: spec) }
            guard let fast = r["fast"]?.bool else { throw EngineError("load requires fast as a boolean") }
            let assets = (r["assets"]?.object ?? config["assets"]?.object ?? [:]).compactMapValues { $0.string.map { URL(fileURLWithPath: $0) } }
            let directory = URL(fileURLWithPath: weights, isDirectory: true)
            // Arsenal verified this small receipt before handing the snapshot to the worker.
            let receipt = try? WireJSON.decode(Data(contentsOf: StoredNames.snapshotURL(in: directory)))
            var options: [String: String] = [:]
            var base = directory
            let intakeSnapshot = StoredVoiceWeights.isIntake((try? Data(contentsOf: directory.appendingPathComponent("config.json"))) ?? Data())
            if modelID == "pocket-tts", !intakeSnapshot, precision == "int8" || precision == "int4" {
                if let basePath = config["base_weights_dir"]?.string { base = URL(fileURLWithPath: basePath) }
                options[Engines.quantizedDirectoryOption] = directory.path
            }
            if precision == "int8" || precision == "int4", modelID != "pocket-tts" {
                let settings = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("config.json"))) as? [String: Any]
                let quant = (settings?["quantization"] ?? settings?["quantization_config"]) as? [String: Any]
                guard let bits = quant?["bits"] as? Int, bits == (precision == "int8" ? 8 : 4) else {
                    throw EngineError("Weights do not match requested dtype \(precision)")
                }
                if modelID == "breeze-tts-2", !intakeSnapshot, precision == "int4", quant?["mode"] as? String != "mxfp4" {
                    throw EngineError("Breeze int4 requires the vendor MXFP4 checkpoint")
                }
            }
            let cell = ModelCell(tier: precision == "int8" ? "8" : precision == "int4" ? "4" : "16", flavour: "optimized", mode: fast ? "fast" : "exact")
            let source = ModelSource(
                id: modelID, engine: "mlx-audio", directory: base, precision: precision, assets: assets,
                options: options, overrides: FastPathComponents.overrides(cell))
            let start = Date()
            RuntimeCaches.useBundledMetallib(); RuntimeCaches.prepare(cacheLimitMB: 128); RuntimeCaches.resetPeak()
            let kernelConfig = try config["kernels"].map { try JSONSerialization.jsonObject(with: JSONEncoder().encode($0)) }
            try VireoFolderKernels.configure(kernelConfig, model: modelID, dtype: precision)
            guard let speech = try await MLXSpeech.load(source) as? MLXSpeech else { throw EngineError("Loader returned a non-speech model") }
            if control.isCancelled { await speech.unload(); throw CancellationError() }
            loaded = speech; dtype = precision; modelRevision = receipt?["revision"]?.string; self.fast = fast
            VireoFolderKernels.finalize(optimized: speech.engineLabel == "optimized")
            try wire.reply(
                id,
                [
                    "load_s": .number(Date().timeIntervalSince(start)), "memory": memory(), "engine": .string(speech.engineLabel), "engine_reason": .string(speech.engineReason),
                    "kernels": status()["kernels"] ?? .array([])
                ])
            try pushStatus()
        case "measurement_voice_patch":
            guard let model = loaded as? MLXSpeech else { throw EngineError("Voice diagnosis requires a loaded native model") }
            let keys = r["keeping"]?.array?.compactMap(\.string) ?? []
            let matrices = try await model.patchVoiceMatrices(
                halfDirectory: URL(fileURLWithPath: try string(r, "half_dir")),
                coverage: URL(fileURLWithPath: try string(r, "coverage")), components: Set(r["components"]?.array?.compactMap(\.string) ?? []), keeping: Set(keys))
            try wire.reply(id, ["matrix_groups": .array(matrices.map { .array($0.map(JSONValue.string)) })])
        case "measurement_components":
            guard let model = loaded as? MLXSpeech else { throw EngineError("Voice component capture requires a loaded native model") }
            let count = try model.measurementComponents(
                fixture: URL(fileURLWithPath: try string(r, "fixture_dir")),
                output: URL(fileURLWithPath: try string(r, "output_dir")), create: r["capture"]?.bool == true,
                only: r["diagnostic_components"]?.array.map { Set($0.compactMap(\.string)) },
                reference: r["diagnostic_reference"]?.string.map { URL(fileURLWithPath: $0) })
            try wire.reply(id, ["components": .number(Double(count))])
        case "speak":
            try SpeechOperation.check(r, modelID: loaded?.id)
            try wire.reply(id, try await speak(r, control: control)); try pushStatus()
        case "design_voice":
            try SpeechOperation.check(r, modelID: loaded?.id)
            guard let loaded else { throw WorkerFailure(code: "not_loaded", message: "Load a voice model before designing a voice") }
            let description = try string(r, "description")
            guard let modelRevision else { throw WorkerFailure(code: "integrity", message: "Voice design needs a verified model revision") }
            var preview = r
            preview["op"] = "speak"; preview["text"] = r["preview_text"] ?? "Hello. This is a preview of your new voice."
            preview["instructions"] = .string(description); preview["stream"] = false
            preview["seed"] = r["seed"] ?? .number(Double(UInt32.random(in: 0...UInt32.max)))
            var design: JSONObject = [
                "seed": preview["seed"]!, "model_revision": .string(modelRevision), "dtype": .string(dtype!), "fast": .bool(fast),
                "preview_text": preview["text"]!, "language": .string(SpeechOperation.language(r["language"]?.string) ?? "english")
            ]
            let generation = preview.filter { ["temperature", "top_p", "top_k", "repetition_penalty", "guidance"].contains($0.key) }
            if !generation.isEmpty { design["generation"] = .object(generation) }
            guard preview["out_dir"]?.string != nil else { throw WorkerFailure(code: "invalid_request", message: "Voice previews need an Arsenal job directory") }
            let audio = try await speak(preview, control: control)
            try wire.reply(id, try SavedVoices(model: loaded.id, support: support).designResult(request: r, preview: audio, design: design)); try pushStatus()
        case "clone_voice":
            try SpeechOperation.check(r, modelID: loaded?.id)
            guard let loaded else { throw WorkerFailure(code: "not_loaded", message: "Load a voice model before cloning a voice") }
            let audio = URL(fileURLWithPath: try string(r["audio"]?.object ?? [:], "path"))
            if loaded.id == Drama3Speech.modelID {
                // A transcript the prompt would refuse must not become a saved voice that fails every later request.
                do { try Drama3Text.validate(try string(r, "transcript"), location: "transcript") } catch let error as Drama3Error {
                    throw WorkerFailure(code: "invalid_request", message: error.message)
                }
            }
            let clip = try PCM.reference(audio, rate: loaded.id == VoxCPM2Speech.modelID ? VoxCPM2Speech.referenceRate : loaded.sampleRate)
            if let vox = loaded as? VoxCPM2Speech {
                try Self.checkVoxClip(clip)
                // A transcript whose prompt leaves no room for speech must not become a voice every later speak refuses.
                do { try vox.checkVoice(samples: clip.samples.count, transcript: r["transcript"]?.string) } catch let error as VoxCPM2Error {
                    throw WorkerFailure(code: "invalid_request", message: error.message)
                }
            }
            if control.isCancelled { throw CancellationError() }
            let saved = try SavedVoices(model: loaded.id, support: support).save(
                name: try string(r, "name"), audio: audio, transcript: r["transcript"]?.string, language: r["language"]?.string)
            try wire.reply(id, ["voice_id": .string(saved.id)]); try pushStatus()
        case "voices":
            let specs = loaded.map { speech in SpeechCatalog.all.filter { $0.id == speech.id } } ?? SpeechCatalog.all
            // Drama 3 has no preset voices; only the voices cloned for it.
            let savedOnlyModels = [Drama3Speech.modelID, VoxCPM2Speech.modelID]
            let savedOnly = loaded.map { savedOnlyModels.contains($0.id) ? [$0.id] : [] } ?? savedOnlyModels
            var voices: [JSONValue] = specs.flatMap { spec in
                spec.voices.map { voice in
                    let british = voice.hasPrefix("b"), female = voice.count > 1 && Array(voice)[1] == "f"
                    return [
                        "id": .string(voice), "name": .string(voice), "language": .string(spec.id == "kokoro-82m" ? (british ? "en-GB" : "en-US") : "en"),
                        "model": .string(spec.id), "gender": .string(female ? "female" : "unspecified")
                    ]
                }
            }
            for model in specs.map(\.id) + savedOnly { voices += try SavedVoices(model: model, support: support).list().map(\.json) }
            try wire.reply(id, ["voices": .array(voices)])
        case "convert": try await convert(r, control: control); try pushStatus()
        case "status": try wire.reply(id, status())
        case "unload":
            if let loaded { await loaded.unload() }; loaded = nil; dtype = nil; modelRevision = nil; VireoFolderKernels.reset()
            try wire.reply(id); try pushStatus(); return true
        default: throw WorkerFailure(code: "unsupported_op", message: "Unsupported voice worker operation")
        }
        return false
    }
    /// Drama 3 loads its pinned snapshot as stored: the vendor mixed 4/8-bit affine checkpoint (format `int4`), codec cast to
    /// FP16 in memory. It needs none of the Vireo engine's tiers, fast paths or folder kernels.
    func loadDrama(_ r: JSONObject, weights: URL, control: GenerationControl) async throws {
        let id = r["id"] ?? .null
        let precision = try string(r, "dtype")
        guard precision == "int4" else { throw WorkerFailure(code: "unsupported_format", message: "Drama 3 runs only its vendor 4/8-bit checkpoint (int4)") }
        let receipt = try? WireJSON.decode(Data(contentsOf: StoredNames.snapshotURL(in: weights)))
        let start = Date()
        RuntimeCaches.useBundledMetallib(); RuntimeCaches.prepare(cacheLimitMB: 1024); RuntimeCaches.resetPeak()
        let speech = try await Drama3Speech.load(directory: weights)
        if control.isCancelled { await speech.unload(); throw CancellationError() }
        loaded = speech; dtype = precision; modelRevision = receipt?["revision"]?.string; fast = r["fast"]?.bool ?? false
        try wire.reply(
            id,
            [
                "load_s": .number(Date().timeIntervalSince(start)), "memory": memory(), "engine": .string(speech.engineLabel), "engine_reason": .string(speech.engineReason),
                "kernels": .array([])
            ])
        try pushStatus()
    }
    /// VoxCPM2 loads its FP16 Get cast (`cast:voxcpm2`); no Vireo tiers, fast paths or folder kernels.
    func loadVoxCPM2(_ r: JSONObject, weights: URL, control: GenerationControl) async throws {
        let id = r["id"] ?? .null
        let precision = try string(r, "dtype")
        guard precision == "fp16" else { throw WorkerFailure(code: "unsupported_format", message: "VoxCPM2 runs only its FP16 cast (fp16)") }
        let receipt = try? WireJSON.decode(Data(contentsOf: StoredNames.snapshotURL(in: weights)))
        let start = Date()
        RuntimeCaches.useBundledMetallib(); RuntimeCaches.prepare(cacheLimitMB: 512); RuntimeCaches.resetPeak()
        let speech = try await VoxCPM2Speech.load(directory: weights)
        if control.isCancelled { await speech.unload(); throw CancellationError() }
        loaded = speech; dtype = precision; modelRevision = receipt?["revision"]?.string; fast = r["fast"]?.bool ?? false
        try wire.reply(
            id,
            [
                "load_s": .number(Date().timeIntervalSince(start)), "memory": memory(), "engine": .string(speech.engineLabel), "engine_reason": .string(speech.engineReason),
                "kernels": .array([])
            ])
        try pushStatus()
    }
    /// VoxCPM2's own request controls (maker names): cfg_value, inference_timesteps, clone_mode, retry_badcase.
    static func voxOptions(_ r: JSONObject) throws -> VoxCPM2Options {
        var options = VoxCPM2Options()
        if let value = r["cfg_value"] {
            guard let n = value.number, n.isFinite, (0.1...10).contains(n) else {
                throw WorkerFailure(code: "invalid_request", message: "cfg_value must be a number from 0.1 to 10")
            }
            options.cfgValue = Float(n)
        }
        if let value = r["inference_timesteps"] {
            guard let n = value.number, n.rounded() == n, (1...100).contains(n) else {
                throw WorkerFailure(code: "invalid_request", message: "inference_timesteps must be an integer from 1 to 100")
            }
            options.inferenceTimesteps = Int(n)
        }
        if let value = r["clone_mode"] {
            guard let mode = value.string.flatMap(VoxCPM2CloneMode.init(rawValue:)) else {
                throw WorkerFailure(code: "invalid_request", message: "clone_mode must be reference, continuation or both")
            }
            options.cloneMode = mode
        }
        if let value = r["retry_badcase"] {
            guard let flag = value.bool else { throw WorkerFailure(code: "invalid_request", message: "retry_badcase must be a boolean") }
            options.retryBadcase = flag
        }
        return options
    }
    /// A request VoxCPM2 cannot serve is the caller's error (invalid_request, HTTP 422), refused before any audio is read.
    static func checkVoxRequest(_ r: JSONObject, options: VoxCPM2Options) throws {
        let clip = r["reference"]?["path"]?.string.map { !$0.isEmpty } ?? false
        if !clip, let voice = r["voice"]?.string, !voice.isEmpty {
            throw WorkerFailure(code: "invalid_request", message: "Unknown voice \(voice): VoxCPM2 has no preset voices")
        }
        guard let text = r["text"]?.string else { throw WorkerFailure(code: "invalid_request", message: "text must be a string") }
        do {
            _ = try VoxCPM2Speech.cloneMode(hasClip: clip, transcript: r["reference"]?["text"]?.string, requested: options.cloneMode)
            _ = try VoxCPM2Speech.text(text, instructions: r["instructions"]?.string ?? r["description"]?.string)
        } catch let error as VoxCPM2Error {
            throw WorkerFailure(code: "invalid_request", message: error.message)
        }
    }
    /// Reads a request's reference clip. A saved voice's own recording that is missing or unreadable is Arsenal's fault,
    /// not the caller's: it is an engine error (500), not invalid_request (422).
    static func readReference(_ path: String?, rate: Int, fromSavedVoice: Bool) throws -> AudioClip? {
        guard let path else { return nil }
        do { return try PCM.reference(URL(fileURLWithPath: path), rate: rate) } catch let failure as WorkerFailure where fromSavedVoice && failure.code == "invalid_request" {
            throw EngineError("The saved voice's recording is missing or unreadable (\(failure.message)); clone the voice again")
        }
    }
    /// A reference clip VoxCPM2 can use: at least half a second (the reader already caps it at 30 s).
    static func checkVoxClip(_ clip: AudioClip) throws {
        guard clip.seconds >= 0.5 else { throw WorkerFailure(code: "invalid_request", message: "VoxCPM2 needs a reference clip of at least half a second") }
    }
    func speak(_ input: JSONObject, control: GenerationControl) async throws -> JSONObject {
        guard let model = loaded else { throw WorkerFailure(code: "not_loaded", message: "Load a voice model before speaking") }
        let r = try SavedVoices(model: model.id, support: support).resolve(input, modelRevision: modelRevision)
        let vox = model as? VoxCPM2Speech
        let voxOptions = try vox.map { _ in try Self.voxOptions(r) }
        if let voxOptions { try Self.checkVoxRequest(r, options: voxOptions) }
        let text = try string(r, "text")
        guard text.count <= 100_000 else { throw WorkerFailure(code: "invalid_request", message: "text exceeds 100000 characters") }
        // A reference clip wins over the default preset; Pocket otherwise quietly uses Alba instead of cloning.
        let referencePath = r["reference"]?["path"]?.string.flatMap { $0.isEmpty ? nil : $0 }
        let requestedVoice = referencePath == nil ? (r["voice"]?.string ?? (model.id == "kokoro-82m" ? "af_heart" : model.id == "pocket-tts" ? "alba" : nil)) : nil
        let voice = model.id == "kokoro-82m" && requestedVoice == "assistant" ? "af_heart" : requestedVoice
        if model.id == Drama3Speech.modelID, let voice, !voice.isEmpty {
            throw WorkerFailure(code: "invalid_request", message: "Unknown voice \(voice): Drama 3 has no preset voices")
        }
        if model.id != Drama3Speech.modelID, vox == nil {
            let spec = try SpeechCatalog.model(model.id)
            if let voice, !spec.voices.isEmpty, !spec.voices.contains(voice) {
                throw WorkerFailure(code: "invalid_request", message: "Unknown preset voice \(voice) for \(model.id)")
            }
        }
        try model.setSpeed(Float(r["speed"]?.number ?? 1))
        let ref = r["reference"]?.object
        if model.id == "qwen3-tts-base", ref?["path"]?.string == nil || ref?["text"]?.string?.isEmpty != false {
            throw WorkerFailure(code: "invalid_request", message: "Qwen3 Base requires reference.path and reference.text")
        }
        // VoxCPM2 encodes its reference at the AudioVAE's 16 kHz, not at its 48 kHz output rate.
        let referenceRate = vox == nil ? model.sampleRate : VoxCPM2Speech.referenceRate
        let reference = try Self.readReference(referencePath, rate: referenceRate, fromSavedVoice: referencePath != input["reference"]?["path"]?.string)
        if vox != nil, let reference { try Self.checkVoxClip(reference) }
        let description = r["instructions"]?.string ?? r["description"]?.string
        if model.id == "qwen3-tts-voicedesign", description?.isEmpty != false {
            throw WorkerFailure(code: "invalid_request", message: "VoiceDesign requires instructions describing the voice")
        }
        let request = SpeechRequest(
            text: text, preset: voice, description: description,
            reference: reference, referenceText: ref?["text"]?.string, language: SpeechOperation.language(r["language"]?.string),
            seed: r["seed"]?.number.flatMap { $0 >= 0 && $0 <= Double(UInt32.max) ? UInt64($0) : nil },
            maxTokens: r["max_tokens"]?.number.map(Int.init), temperature: r["temperature"]?.number.map(Float.init),
            topP: r["top_p"]?.number.map(Float.init), topK: r["top_k"]?.number.map(Int.init),
            repetitionPenalty: r["repetition_penalty"]?.number.map(Float.init), guidance: r["guidance"]?.number.map(Float.init))
        let stream = r["stream"]?.bool ?? false
        let out = try r["out_dir"]?.string.map { try WaveSink(directory: URL(fileURLWithPath: $0), rate: model.sampleRate) }
        guard stream || out != nil else { throw WorkerFailure(code: "invalid_request", message: "speak requires stream=true or out_dir") }
        let id = r["id"] ?? .null, streamID = UInt32.random(in: 1...UInt32.max)
        let start = Date(); var first: Double?; var samples = 0
        if stream {
            try wire.send([
                "id": id, "event": "audio_begin", "stream": .number(Double(streamID)), "sample_rate": .number(Double(model.sampleRate)), "channels": 1, "encoding": "pcm_s16le"
            ])
        }
        let emit: ([Float]) async throws -> Void = { chunk in
            guard !control.isCancelled else { return }
            if first == nil { first = Date().timeIntervalSince(start) }
            let pcm = try PCM.encode(chunk); samples += chunk.count
            try self.wire.send(["id": id, "event": "progress", "phase": "step", "audio_samples": .number(Double(samples))])
            try out?.append(pcm)
            if stream { try self.wire.audio(stream: streamID, pcm: pcm) }
        }
        if let vox, let voxOptions {
            do { try await vox.speak(request, options: voxOptions, control: control, emit: emit) } catch let error as VoxCPM2Error {
                throw WorkerFailure(code: "invalid_request", message: error.message)
            }
        } else {
            try await model.speak(request, control: control, emit: emit)
        }
        if control.isCancelled { throw CancellationError() }
        guard samples > 0 else { throw EngineError("Model produced empty audio") }
        let gen = Date().timeIntervalSince(start), duration = Double(samples) / Double(model.sampleRate)
        var result: JSONObject = [
            "duration_s": .number(duration), "gen_s": .number(gen), "first_audio_s": .number(first ?? gen), "rtf": .number(gen / duration),
            "sample_rate": .number(Double(model.sampleRate)), "channels": 1
        ]
        if let out {
            try out.finish(); let attrs = try FileManager.default.attributesOfItem(atPath: out.file.path)
            result["path"] = .string(out.file.path); result["bytes"] = .number(Double((attrs[.size] as? NSNumber)?.intValue ?? 0))
            result["sha256"] = .string(try PCM.digest(out.file))
        }
        return result
    }
    private func convert(_ r: JSONObject, control: GenerationControl) async throws {
        guard loaded == nil else { throw EngineError("Convert requires an unloaded worker") }
        let from = URL(fileURLWithPath: try string(r, "src_dir")), to = URL(fileURLWithPath: try string(r, "dst_dir"))
        let recipe = try string(r, "recipe"), dtype = try string(r, "dtype")
        if recipe == VoxCPM2Cast.recipe {
            try convertVoxCPM2(r, from: from, to: to, dtype: dtype, control: control)
            return
        }
        let recipeModel = Self.recipeModel(recipe)
        let canonical: [String: String] = ["kokoro": "kokoro-82m", "qwen3-tts-talker": "qwen3-tts-voicedesign"]
        let modelID =
            r["model"]?.string ?? r["config"]?["model_id"]?.string
            ?? (recipe == "cast:kokoro" || recipe == "kokoro-fp32-to-fp16" ? "kokoro-82m" : canonical[recipeModel] ?? recipeModel)
        guard let spec = SpeechCatalog.all.first(where: { $0.id == modelID }) else {
            throw WorkerFailure(code: "unsupported_recipe", message: "Unknown speech conversion recipe")
        }
        guard try !FileManager.default.fileExists(atPath: to.path) || FileManager.default.contentsOfDirectory(atPath: to.path).isEmpty else {
            throw EngineError("Conversion destination must be empty")
        }
        try FileManager.default.createDirectory(at: to, withIntermediateDirectories: true)
        let source = ModelSource(id: modelID, engine: "mlx-audio", directory: from, precision: spec.tier16_dtype)
        let report: QuantReport
        if recipe.hasPrefix("intake-v3:" + modelID + ":") || recipe == "cast:" + modelID + ":bf16-truncation" || recipe.hasPrefix("intake-v2:" + modelID + ":")
            || recipe == "intake-v1:" + modelID || (recipe == "cast:" + modelID && WholeVoiceConversion.models.contains(modelID))
        {
            let configObject = (try? JSONSerialization.jsonObject(with: Data(contentsOf: from.appendingPathComponent("config.json")))) as? [String: Any]
            let legacyIntake = configObject?["mobius_voice_intake"] // legacy-compat: voice packs written before the rename
            let sourceConfig = (configObject?["arsenal_voice_intake"] ?? legacyIntake) as? [String: Any]
            report = try Engines.convertWholeVoice(
                source, files: spec.files, dtype: dtype,
                half: r["half"]?.string ?? (["fp16", "bf16"].contains(dtype) ? dtype : sourceConfig?["half"] as? String ?? spec.tier16_dtype),
                keeping: (recipe.hasPrefix("intake-v2:") || recipe.hasPrefix("intake-v3:")) ? try WholeVoiceConversion.keeping(recipe: recipe, model: modelID, dtype: dtype) : [],
                truncateSource: recipe.hasSuffix(":bf16-truncation"), matrixPolicy: recipe.hasPrefix("intake-v3:"), into: to)
        } else if ["cast:kokoro", "kokoro-fp32-to-fp16"].contains(recipe), dtype == "fp16" {
            report = try await Engines.castTo16(source, files: spec.files, into: to)
        } else {
            guard dtype == "int8" || dtype == "int4", r["group_size"]?.number == 64 else { throw EngineError("Conversion requires int8/int4 with group_size=64") }
            try SpeechCatalog.checkPrecision(dtype, spec: spec)
            guard modelID != "breeze-tts-2" || dtype != "int4" else { throw EngineError("Breeze int4 requires the pinned vendor MXFP4 checkpoint") }
            guard recipeModel == modelID || canonical[recipeModel] == modelID || recipe == "mlx-affine-g64" || recipeModel == "qwen3-tts-talker" else {
                throw WorkerFailure(code: "unsupported_recipe", message: "Unknown speech conversion recipe")
            }
            report = try await Engines.convertQuantized(source, bits: dtype == "int8" ? 8 : 4, into: to)
            if modelID == "pocket-tts" {
                for path in spec.files where path != "model.safetensors" {
                    let target = to.appendingPathComponent(path); try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try FileManager.default.copyItem(at: from.appendingPathComponent(path).resolvingSymlinksInPath(), to: target)
                }
            }
        }
        // Get accepts only regular files; Vireo's checkpoint converter links unchanged voice files.
        // Materialize those sources before reporting a standalone snapshot to Arsenal's blob store.
        for (relative, sourcePath) in report.links {
            let target = to.appendingPathComponent(relative)
            let staging = target.deletingLastPathComponent().appendingPathComponent(".materialize-" + UUID().uuidString)
            try FileManager.default.copyItem(at: URL(fileURLWithPath: sourcePath).resolvingSymlinksInPath(), to: staging)
            guard rename(staging.path, target.path) == 0 else { throw EngineError("Cannot materialize converted snapshot file") }
        }
        if control.isCancelled { throw CancellationError() }
        let data = try JSONSerialization.data(withJSONObject: report.manifest, options: [.sortedKeys]); try data.write(to: to.appendingPathComponent("conversion.json"))
        guard let walker = FileManager.default.enumerator(atPath: to.path) else { throw EngineError("Cannot enumerate conversion output") }
        var files: [JSONValue] = []
        while let entry = walker.nextObject() {
            guard let relative = entry as? String else { continue }
            let url = to.appendingPathComponent(relative); var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), !isDir.boolValue else { continue }
            let a = try FileManager.default.attributesOfItem(atPath: url.resolvingSymlinksInPath().path)
            files.append(["path": .string(relative), "bytes": .number(Double((a[.size] as? NSNumber)?.intValue ?? 0)), "sha256": .string(try PCM.digest(url))])
        }
        try wire.reply(r["id"] ?? .null, ["files": .array(files)])
    }
}
extension Runner {
    /// Get's FP16 cast of the VoxCPM2 release (BF16 LMs, FP32 AudioVAE), file by file, with an FP16 range check.
    func convertVoxCPM2(_ r: JSONObject, from: URL, to: URL, dtype: String, control: GenerationControl) throws {
        guard dtype == "fp16" else { throw WorkerFailure(code: "unsupported_recipe", message: "cast:voxcpm2 writes fp16 only") }
        guard try !FileManager.default.fileExists(atPath: to.path) || FileManager.default.contentsOfDirectory(atPath: to.path).isEmpty else {
            throw EngineError("Conversion destination must be empty")
        }
        try FileManager.default.createDirectory(at: to, withIntermediateDirectories: true)
        // Only the folder's pinned files; everything written is removed again on failure or cancellation.
        let sources = try VoxCPM2Cast.manifestFiles(modelDirectory: URL(fileURLWithPath: try string(r, "model_dir")))
        let report = try VoxCPM2Cast.convert(source: from, files: sources, into: to, isCancelled: { control.isCancelled })
        let manifest: [String: Any] = [
            "recipe": VoxCPM2Cast.recipe, "dtype": "fp16", "cast_tensors": report.castTensors, "kept_tensors": report.keptTensors,
            "largest_magnitude": Double(report.largestMagnitude), "files": report.files
        ]
        do {
            if control.isCancelled { throw CancellationError() }
            try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys]).write(to: to.appendingPathComponent("conversion.json"))
        } catch {
            for path in report.files + ["conversion.json"] { try? FileManager.default.removeItem(at: to.appendingPathComponent(path)) }
            throw error
        }
        var files: [JSONValue] = []
        for relative in report.files + ["conversion.json"] {
            let url = to.appendingPathComponent(relative)
            let a = try FileManager.default.attributesOfItem(atPath: url.path)
            files.append(["path": .string(relative), "bytes": .number(Double((a[.size] as? NSNumber)?.intValue ?? 0)), "sha256": .string(try PCM.digest(url))])
        }
        try wire.reply(r["id"] ?? .null, ["files": .array(files)])
    }
}
func string(_ r: JSONObject, _ name: String) throws -> String {
    guard let s = r[name]?.string, !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw WorkerFailure(code: "invalid_request", message: "\(name) must be a nonempty string")
    }; return s
}
