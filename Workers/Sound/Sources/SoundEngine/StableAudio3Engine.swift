// SPDX-License-Identifier: MIT
import Foundation
import MLX
import ModelPrecision
import MLXRandom
import VireoCore

import StableAudio3


// Vireo engine adapter for Stable Audio 3 Medium (sound effects + instrumental music) and Small SFX (sound effects).
// Integration: copy this file together with Sources/StableAudio3/*.swift into
// Sources/VireoEngine/StableAudio3/ (the SA3_STANDALONE import disappears there)
// and register `StableAudio3Engine.key → StableAudio3Engine.load` in Engines.swift.
//
// Catalog: engine "stable-audio-3", repo stabilityai/stable-audio-3-optimized,
// files = SA3Checkpoint.officialMediumFiles, precision "f16" (DiT F16, T5Gemma F16,
// legacy SAME-L codec FP32; intake snapshots use half/packed storage), sample_rate 44100, channels 2, max_seconds 380.
// engine_options: "layout": "official" (default) | "aufklarer" (int8 safetensors bundle, medium only);
// "variant": "medium" (default) | "sm-sfx" (Small SFX: MLX/dit_sm-sfx_f16.npz + MLX/same_s_decoder_f32.npz,
// files = SA3Variant.smallSFX.officialFiles, max_seconds 120).

public enum StableAudio3Engine {
    public static let key = "stable-audio-3"
    public static let defaultSteps = 8

    public static let load: ModelLoader = { source in
        let layout = source.options["layout"] ?? "official"
        let variant = try SA3Variant(option: source.options["variant"])
        guard layout == "official" || variant == .medium else { throw EngineError("Stable Audio 3: the aufklarer layout is medium only") }
        var checkpoint = layout == "aufklarer"
            ? SA3Checkpoint.aufklarerBundle(source.directory)
            : SA3Checkpoint.official(snapshot: source.directory, variant: variant)
        // 8-/4-bit: the DiT converted at Get (QuantRules.stableAudio3) replaces the F16 DiT file; the SAME-L
        // codec stays FP32 and T5Gemma F16 from the base snapshot. sa3LoadDiTMedium installs the quantized layers
        // from the stored scales (strict key match), so this loads like the base.
        // An intake snapshot contains all three converted components, never references the source snapshot.
        for component in ["dit", "decoder", "t5gemma"] {
            let original = component == "dit" ? checkpoint.dit : component == "decoder" ? checkpoint.decoder : checkpoint.t5gemma
            let converted = original.deletingPathExtension().appendingPathExtension("safetensors")
            if FileManager.default.fileExists(atPath: converted.path) {
                if component == "dit" { checkpoint.dit = converted }
                else if component == "decoder" { checkpoint.decoder = converted }
                else { checkpoint.t5gemma = converted }
            }
        }
        let ditPrecision: String?
        var expectedBits: Int? = nil
        if let quantized = SoundEngines.quantizedDirectory(source) {
            guard layout == "official" else { throw EngineError("Stable Audio 3: converted precisions need the official layout") }
            let intakeDiT = checkpoint.dit
            let legacyDiT = quantized.appendingPathComponent(quantizedDiTFile(variant))
            checkpoint.dit = FileManager.default.fileExists(atPath: intakeDiT.path) && intakeDiT.pathExtension == "safetensors" ? intakeDiT : legacyDiT
            ditPrecision = nil
            expectedBits = source.options["quantize_bits"].flatMap(Int.init)
        } else {
            // Official export is F16; "bf16"/"fp32" convert the DiT at load. "8bit" without a converted DiT needs
            // the aufklarer int8 bundle (quantization is detected from its tensors).
            switch source.precision.lowercased() {
            case "f16", "fp16", "": ditPrecision = nil
            case "bf16", "fp32": ditPrecision = source.precision
            case "8bit":
                guard layout == "aufklarer" else {
                    throw EngineError("Stable Audio 3: 8-bit needs a converted DiT (vireo get) or the aufklarer int8 bundle")
                }
                ditPrecision = nil
            default: throw EngineError("Stable Audio 3: unsupported precision \(source.precision)")
            }
        }
        let model = try StableAudio3(checkpoint: checkpoint, variant: variant, ditPrecision: ditPrecision)
        if let expectedBits, model.ditQuantBits != expectedBits {
            throw EngineError("Stable Audio 3: converted DiT is \(model.ditQuantBits.map { "\($0)-bit" } ?? "unquantized"), expected \(expectedBits)-bit")
        }
        let sound = StableAudio3SoundModel(id: source.id, model: model)
        var configuration = FastPath(source: source)
        if checkpoint.decoder.pathExtension == "safetensors" && checkpoint.t5gemma.pathExtension == "safetensors" {
            configuration = FastPath(environment: configuration.switches, overrides: ["VIREO_SA3_F32_WEIGHTS": "0"])
        }
        sound.gate(weights: checkpoint.dit, decoderWeights: checkpoint.decoder, configuration: configuration)
        return sound
    }

    /// File name of the converted DiT inside the precision's directory (medium keeps its original name, so DiTs
    /// converted before the small variant existed still load).
    public static func quantizedDiTFile(_ variant: SA3Variant) -> String { variant.ditComponent + ".safetensors" }
    public static let quantizedDiTFile = quantizedDiTFile(.medium)

    /// Converts the official F16 DiT with `QuantRules.stableAudio3`. File keys stay the official ones
    /// (`ff.ff.0.proj`), so `sa3LoadDiTMedium` reads the result exactly like an int8 bundle.
    static func convertDiT(_ source: ModelSource, bits: Int, into directory: URL, report: inout QuantReport) throws {
        guard (source.options["layout"] ?? "official") == "official" else { throw EngineError("Stable Audio 3: only the official layout is converted") }
        let variant = try SA3Variant(option: source.options["variant"])
        let checkpoint = SA3Checkpoint.official(snapshot: source.directory, variant: variant)
        let rule = report.rule
        try QuantizedWeights.convertFile(source: checkpoint.dit, output: directory.appendingPathComponent(quantizedDiTFile(variant)),
                                         skeleton: DiTMedium(variant.dit), rule: rule, bits: bits, report: &report,
                                         read: sa3ReadTensors, modulePath: sa3DiTModuleKey) { saved, expected in
            try checkConvertedDiT(saved, config: variant.dit, expected: expected)
        }
    }

    /// The same key handling as `sa3LoadDiTMedium`, on a fresh DiT, without evaluating any weight.
    static func checkConvertedDiT(_ saved: [String: MLXArray], config: DiTMediumConfig = .medium, expected: Int) throws {
        var raw = saved
        for key in ["cond.padding_embedding", "cond.seconds_total_weight", "cond.seconds_total_bias"] {
            guard raw.removeValue(forKey: key) != nil else { throw EngineError("Converted DiT lacks \(key)") }
        }
        for key in sa3DiTIgnoredKeys { raw.removeValue(forKey: key) }
        try QuantizedWeights.strictCheck(DiTMedium(config), sa3RewriteDiTMediumKeys(raw), expected: expected)
    }
}

public final class StableAudio3SoundModel: SoundModel {
    public let id: String
    public let sampleRate = SA3Audio.sampleRate
    public let maxSeconds: Double
    /// Sound effects and music (medium); sound effects only (small SFX).
    public let kinds: Set<AudioKind>
    private var model: StableAudio3?
    public private(set) var engineLabel = "mlx"
    public private(set) var engineReason = "stock MLX (reference numerics: fp32 residual stream over fp16 weights)"
    /// The verdict keys of the optimized paths in use (a runtime failure makes each of them sticky stock).
    private var gateKeys: [(version: String, weights: URL)] = []
    /// The model-path configuration the gate ran under.
    private var configuration = FastPath.current

    public init(id: String, model: StableAudio3) {
        self.id = id
        self.model = model
        self.maxSeconds = model.variant.maxSeconds
        self.kinds = model.variant == .medium ? [.sfx, .music] : [.sfx]
    }

    /// Fast paths (FastPath.swift switches). The DiT, in order of preference:
    /// - VIREO_SA3_HALF (default on): fp16 residual stream (the zero local conditioning and the timestep features
    ///   in the latent dtype), so fp16 GEMMs run on fp16 operands. Inexact: gated on log-mel distance to stock.
    /// - VIREO_SA3_F32_WEIGHTS (default on, used when the fp16 stream is off or fails its self-test): fp32 copies of
    ///   exactly the weights stock promotes; bit-identical, +2.4 GB for the fp16 DiT. Not for quantized DiTs.
    /// The SAME-L decoder (medium), independently:
    /// - VIREO_SA3_SWA_VIEWS (default on): the sliding-window differential attention feeds SDPA strided views instead
    ///   of transposed window copies and concatenated heads (`DiffSWA.swaViews`). Same kernels and per-row arithmetic;
    ///   gated on bit-identity with the stock decode (a tolerance only if it ever is not).
    static let fastPathVersion = "sa3-fast-1"
    static let swaViewsVersion = "sa3-swa-views-1"
    /// Self-test bound for the fp16 stream (log-mel distance, `AudioMetrics.melDistance`, dB). M5 Max: see
    /// lab/perf/NOTES.md; a different seed is 7-13 dB away on the probe prompts.
    static let halfStreamMaxMelDB = 1.0
    /// Self-test bound for a decoder change that is not bit-identical (same metric, same latents on both paths).
    static let decoderMaxMelDB = 1.0

    func gate(weights: URL, decoderWeights: URL? = nil, configuration: FastPath = .current) {
        guard let model else { return }
        model.profile = FastPath.profiling("VIREO_SA3_PROFILE")
        // One configuration (this load's cell over the environment) selects the path and keys the verdicts.
        self.configuration = configuration
        // Small SFX's Get recipe is the ordinary half path, not an optional fast path.
        // Do not execute the legacy FP32 self-test or create a promoted-weight fallback.
        if model.variant == .smallSFX, decoderWeights?.pathExtension == "safetensors", let dit = model.dit {
            Self.setHalfStream(dit, true)
            engineReason = "FP16 intake stream; SAME-S half/packed weights; saved-input qualification is recorded separately"
            return
        }
        if configuration.forceStock { engineReason = "stock MLX (VIREO_FORCE_STOCK)"; return }
        var paths = [String](), failures = [String]()
        gateDiT(model, weights: weights, configuration: configuration, paths: &paths, failures: &failures)
        if configuration.enabled("VIREO_SA3_SWA_VIEWS"), let samel = model.decoder as? SAMELDecoder {
            let decoderWeights = decoderWeights ?? weights
            let verdict = KernelGate.inProcessVerdict(engine: id, kernelVersion: Self.swaViewsVersion, weights: decoderWeights, configuration: configuration) {
                try Self.decoderSelfTest(model, label: "SWA views", exact: configuration.exact) { samel.viewWindows = $0 }
            }
            if verdict.passed {
                samel.viewWindows = true
                paths.append("SWA views in the decoder (self-test passed: \(verdict.detail))")
                gateKeys.append((Self.swaViewsVersion, decoderWeights))
            } else {
                failures.append("SWA views: \(verdict.detail)")
            }
        }
        if paths.isEmpty {
            engineReason = failures.isEmpty ? "stock MLX (fast paths switched off)" : "stock MLX (fast-path self-test failed: \(failures.joined(separator: "; ")))"
        } else {
            engineLabel = "optimized"
            engineReason = paths.joined(separator: " + ") + (failures.isEmpty ? "" : "; " + failures.joined(separator: "; "))
        }
    }

    /// The DiT's path: the fp16 stream, else the exact fp32 weight copies, else stock.
    private func gateDiT(_ model: StableAudio3, weights: URL, configuration: FastPath, paths: inout [String], failures: inout [String]) {
        guard let dit = model.dit else { return }
        if configuration.enabled("VIREO_SA3_HALF"), model.computeDType == .float16 {
            let version = Self.fastPathVersion + " fp16 stream"
            let verdict = KernelGate.inProcessVerdict(engine: id, kernelVersion: version, weights: weights, configuration: configuration) {
                try Self.halfStreamSelfTest(model)
            }
            if verdict.passed {
                Self.setHalfStream(dit, true)
                paths.append("fp16 residual stream (self-test passed: \(verdict.detail))")
                gateKeys.append((version, weights))
                return
            }
            failures.append("fp16 stream: \(verdict.detail)")
        }
        if configuration.enabled("VIREO_SA3_F32_WEIGHTS"), model.ditQuantBits == nil {
            let version = Self.fastPathVersion + " f32 weights"
            let verdict = KernelGate.inProcessVerdict(engine: id, kernelVersion: version, weights: weights, configuration: configuration) {
                try Self.promotedWeightsSelfTest(model)
            }
            if verdict.passed {
                dit.promoteWeights()
                paths.append("fp32 copies of the promoted weights (self-test passed: \(verdict.detail))")
                gateKeys.append((version, weights))
                return
            }
            failures.append("fp32 weights: \(verdict.detail)")
        }
    }

    static func setHalfStream(_ dit: DiTMedium, _ on: Bool) {
        dit.localCondDType = on ? .float16 : .float32
        dit.halfTimestepFeatures = on
    }

    static let testPrompt = "rain on a tin roof"

    /// Exact change: a 1.5 s, 2-step render with a fixed seed must be bit-identical with and without the fp32
    /// weight copies (the model is left on stock afterwards; `gate` applies the verdict).
    static func promotedWeightsSelfTest(_ model: StableAudio3) throws -> String {
        guard let dit = model.dit else { throw EngineError("unloaded") }
        let params = StableAudio3Params(seconds: 1.5, steps: 2, seed: 20_260_927)
        guard let stock = try model.generate(prompt: testPrompt, params: params) else { throw EngineError("cancelled") }
        dit.promoteWeights()
        defer { dit.restoreWeights() }
        guard let fast = try model.generate(prompt: testPrompt, params: params) else { throw EngineError("cancelled") }
        guard fast.left.allSatisfy(\.isFinite), fast.right.allSatisfy(\.isFinite) else { throw EngineError("non-finite output") }
        guard fast.left == stock.left, fast.right == stock.right else {
            let differing = zip(fast.left + fast.right, stock.left + stock.right).filter { $0 != $1 }.count
            throw EngineError("\(differing) of \(stock.left.count * 2) samples differ from stock")
        }
        return "\(stock.left.count * 2) samples bit-identical"
    }

    /// Inexact change: a 3 s, 4-step render with a fixed seed on both paths; the fp16 stream must be finite, not
    /// silent, and within `halfStreamMaxMelDB` log-mel distance of stock.
    static func halfStreamSelfTest(_ model: StableAudio3) throws -> String {
        guard let dit = model.dit else { throw EngineError("unloaded") }
        let params = StableAudio3Params(seconds: 3, steps: 4, seed: 20_260_927)
        guard let stock = try model.generate(prompt: testPrompt, params: params) else { throw EngineError("cancelled") }
        setHalfStream(dit, true)
        defer { setHalfStream(dit, false) }
        guard let fast = try model.generate(prompt: testPrompt, params: params) else { throw EngineError("cancelled") }
        let samples = fast.interleaved
        if let problem = AudioMetrics.signalProblem(samples) { throw EngineError("fp16 stream output: \(problem)") }
        let a = AudioMetrics.logMel(stock.interleaved, channels: 2, sampleRate: SA3Audio.sampleRate)
        let b = AudioMetrics.logMel(samples, channels: 2, sampleRate: SA3Audio.sampleRate)
        guard let distance = AudioMetrics.melDistance(a, b) else { throw EngineError("log-mel distance undefined") }
        guard distance <= halfStreamMaxMelDB else {
            throw EngineError(String(format: "log-mel distance %.2f dB to stock (> %.1f)", distance, halfStreamMaxMelDB))
        }
        return String(format: "log-mel distance %.2f dB to stock", distance)
    }

    /// A decoder-only change: 16 s, 2 steps with a fixed seed sampled once on the DiT path already chosen (173 latents:
    /// a chunked SAME-L decode of two 144-latent windows), plus its first 34 latents (one whole window) and its first 43
    /// (odd: the (2, 2) small-window dispatch). Each decoded by stock and by the change (`apply(true)`); same length,
    /// finite, not silent, and bit-identical ("exact") or within `decoderMaxMelDB` log-mel. The model is left on stock.
    static func decoderSelfTest(_ model: StableAudio3, label: String, exact: Bool = false, apply: (Bool) -> Void) throws -> String {
        let params = StableAudio3Params(seconds: 16, steps: 2, seed: 20_260_928)
        guard let latents = try model.sampleLatents(prompt: testPrompt, params: params)?.latents else { throw EngineError("cancelled") }
        let perLatent = Double(SA3Audio.samplesPerLatent) / Double(SA3Audio.sampleRate)
        let cases: [(MLXArray, Double)] = [(latents, params.seconds), (latents[0..., 0..., 0..<34], 34 * perLatent),
                                           (latents[0..., 0..., 0..<43], 43 * perLatent)]
        apply(false)
        let stock = try cases.map { try model.decodeAudio($0.0, seconds: $0.1) }
        apply(true)
        defer { apply(false) }
        var identical = 0, distances = [String]()
        for ((lat, seconds), reference) in zip(cases, stock) {
            let fast = try model.decodeAudio(lat, seconds: seconds)
            let samples = fast.interleaved
            if let problem = AudioMetrics.signalProblem(samples) { throw EngineError("\(label) output: \(problem)") }
            guard fast.left.count == reference.left.count else { throw EngineError("\(label) length \(fast.left.count) != \(reference.left.count)") }
            if fast.left == reference.left, fast.right == reference.right { identical += 1; continue }
            // Exact configuration: a decoder change must be bit-identical, not merely close.
            if exact { throw EngineError("\(label) differs from stock at \(lat.dim(2)) latents (Exact requires identical output)") }
            let a = AudioMetrics.logMel(reference.interleaved, channels: 2, sampleRate: SA3Audio.sampleRate)
            let b = AudioMetrics.logMel(samples, channels: 2, sampleRate: SA3Audio.sampleRate)
            guard let distance = AudioMetrics.melDistance(a, b) else { throw EngineError("log-mel distance undefined") }
            guard distance <= decoderMaxMelDB else {
                throw EngineError(String(format: "log-mel distance %.2f dB to stock at %d latents (> %.1f)", distance, lat.dim(2), decoderMaxMelDB))
            }
            distances.append(String(format: "%.3f dB at %d latents", distance, lat.dim(2)))
        }
        return identical == cases.count ? "\(cases.count) decodes bit-identical to stock"
            : "\(identical) of \(cases.count) decodes bit-identical, log-mel \(distances.joined(separator: ", "))"
    }

    /// The optimized path produced non-finite audio: stock from now on (sticky for these keys).
    private func fallBack(_ detail: String) {
        model?.dit?.restoreWeights()
        if let dit = model?.dit { Self.setHalfStream(dit, false) }
        (model?.decoder as? SAMELDecoder)?.viewWindows = false
        (engineLabel, engineReason) = ("mlx", "stock MLX (fallback: \(detail))")
        for key in gateKeys {
            KernelGate.recordRuntimeFailure(engine: id, kernelVersion: key.version, weights: key.weights, configuration: configuration, detail: detail)
        }
        gateKeys = []
    }

    public func generate(_ request: SoundRequest, control: GenerationControl) async throws -> AudioClip {
        guard let model else { throw EngineError("Stable Audio 3: model is unloaded") }
        guard kinds.contains(request.kind) else {
            throw EngineError(model.variant == .medium ? "Stable Audio 3 renders sound effects and instrumental music, not \(request.kind.rawValue)"
                                                       : "Stable Audio 3 Small SFX renders sound effects, not \(request.kind.rawValue)")
        }
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw EngineError("Stable Audio 3: empty prompt")
        }
        guard request.seconds > 0, request.seconds <= maxSeconds else {
            throw EngineError("Stable Audio 3: seconds must be between 0 and \(Int(maxSeconds))")
        }
        let params = StableAudio3Params(
            seconds: request.seconds,
            steps: request.steps ?? StableAudio3Engine.defaultSteps,
            cfgScale: Float(request.guidance ?? 1.0),
            seed: request.seed ?? UInt64.random(in: 0..<UInt64(UInt32.max)),
            negativePrompt: request.negativePrompt)
        control.report(progress: 0.02)
        func render() throws -> StableAudio3Output {
            let out = try model.generate(prompt: request.prompt, params: params) { step, total in
                control.report(progress: 0.05 + 0.85 * Double(step) / Double(total), step: step, total: total)
                return !control.isCancelled
            }
            guard let out else { throw CancellationError() }
            return out
        }
        var out = try render()
        // Lab: VIREO_SA3_INJECT_NAN=1 poisons the optimized path's output to exercise the fallback (serving path only,
        // never in the self-test; part of the verdict key like every VIREO_SA3_ switch).
        if !gateKeys.isEmpty, configuration.value("VIREO_SA3_INJECT_NAN") == "1", !out.left.isEmpty { out.left[out.left.count / 2] = .nan }
        let finite = { (o: StableAudio3Output) in !o.left.contains(where: { !$0.isFinite }) && !o.right.contains(where: { !$0.isFinite }) }
        if !finite(out), !gateKeys.isEmpty {
            fallBack("non-finite samples")
            out = try render()   // same seed, stock path
        }
        if !finite(out) {
            throw EngineError("Stable Audio 3 produced non-finite samples")
        }
        control.report(progress: 1)
        return AudioClip(samples: out.interleaved, sampleRate: sampleRate, channels: 2)
    }

    public func unload() async {
        model?.unload()
        model = nil
    }
}

extension StableAudio3Engine {
    /// All large eligible matrices in DiT, T5Gemma and SAME-L, streamed once at Get.
    static func convertIntake(_ source: ModelSource, dtype: String, into directory: URL) throws {
        guard ["fp16", "int8", "int4"].contains(dtype) else { throw EngineError("Unsupported intake precision") }
        let bits = dtype == "int8" ? 8 : dtype == "int4" ? 4 : nil
        let sourceRoot = source.directory.resolvingSymlinksInPath()
        guard let walker = FileManager.default.enumerator(atPath: sourceRoot.path) else { throw EngineError("Missing intake source") }
        for case let relative as String in walker {
            let file = sourceRoot.appendingPathComponent(relative)
            guard try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else { continue }
            let storeFiles = ["arsenal-snapshot.json", "arsenal-derived.json", "mobius-snapshot.json", "mobius-derived.json"] // legacy-compat: older stores
            if storeFiles.contains(relative) || relative == "conversion.json" { continue }
            // Rebuild coverage for this tier; every source receipt is immutable and belongs to the prior tier.
            if relative.hasSuffix(".coverage.json") { continue }
            let output = directory.appendingPathComponent(relative)
            try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
            if file.pathExtension == "npz" || file.pathExtension == "safetensors" {
                let target = output.deletingPathExtension().appendingPathExtension("safetensors")
                var keys: [String]; let sourceRead: (String) throws -> MLXArray
                if file.pathExtension == "npz" { let archive = try NPZArchive(url: file); keys = archive.keys; sourceRead = archive.array }
                else { let archive = try TensorArchive(file); keys = archive.entries.keys.sorted(); sourceRead = archive.array }
                let remapSeconds = source.id == "stable-audio-3-small-sfx" && keys.contains("cond.seconds_total_weight")
                if remapSeconds { keys = keys.map { $0 == "cond.seconds_total_weight" ? "cond.seconds_total.weight" : $0 } }
                let read: (String) throws -> MLXArray = { key in
                    try sourceRead(remapSeconds && key == "cond.seconds_total.weight" ? "cond.seconds_total_weight" : key)
                }
                let coverage = try StreamedPrecision.convert(keys: keys, read: read, to: target, half: .float16, bits: bits) { key, shape in
                    key.hasSuffix(".weight") && shape.reduce(1, *) >= 65_536
                }
                let written = try TensorArchive(target)
                let skipped = keys.filter { key in
                    guard let entry = written.entries[key] else { return false }
                    return entry.dtype != .uint32 && !key.hasSuffix(".scales") && !key.hasSuffix(".biases")
                }
                var receipt: [String: Any] = ["tensors": coverage.tensors, "quantized_tensors": coverage.quantizedTensors,
                    "source_bytes": coverage.sourceBytes, "output_bytes": coverage.outputBytes,
                    "unpacked_keys": skipped, "skip_policy": "Non-matrix, under 65536 elements, non-floating, or input width not divisible by 64."]
                if source.id == "stable-audio-3-small-sfx" {
                    receipt["output_dtypes"] = Dictionary(grouping: written.entries.values, by: { String(describing: $0.dtype) }).mapValues { $0.count }
                    receipt["skip_reasons"] = Dictionary(uniqueKeysWithValues: skipped.map { key in
                        let entry = written.entries[key]!
                        let reason = bits == nil ? "16-bit tier" : key == "transformer.memory_tokens" ? "Learned tokens consumed by concatenation, not matrix multiplication or embedding lookup; retained half" : entry.shape.count != 2 ? "Non-matrix (norm, bias, constant or convolution)"
                            : entry.shape.reduce(1, *) < 65_536 ? "Below 65536 elements"
                            : entry.shape[1] % 64 != 0 ? "Input width not divisible by 64" : "Non-floating or not a module weight"
                        return (key, reason)
                    })
                }
                try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys]).write(to: target.appendingPathExtension("coverage.json"))
            } else { try FileManager.default.copyItem(at: file, to: output) }
        }
    }
}

public extension StableAudio3SoundModel {
    func intakeComponents(fixture: URL, output: URL, capture: Bool) throws {
        guard let model, let text = model.t5 as? T5GemmaText, let dit = model.dit, let decoder = model.decoder,
            let pad = model.padding, let seconds = model.secondsEmbedder else { throw EngineError("Component capture needs loaded Stable Audio model") }
        let inputs = try ComponentCapture.inputs(at: fixture, create: capture) {
            let (ids, mask) = text.tokenize("Gentle rain on a window.")
            let embeds = text.encoder(ids, attentionMask: mask)
            let sec = seconds(5).asType(model.computeDType)
            let cross = MLX.concatenated([applyPromptPadding(embeds: embeds, mask: mask, paddingEmbedding: pad.asType(model.computeDType)), sec], axis: 1)
            let global = sec[0...,0,0...]
            // Capture a reference trajectory once; each candidate step is independently teacher-forced.
            var saved: [String: MLXArray] = ["ids": ids, "mask": mask, "conditioning": cross, "global": global]
            if model.variant == .smallSFX { saved["seconds-value"] = MLXArray([Float(5)]) }
            let noise = MLXRandom.normal([1,256,StableAudio3.latentLength(seconds: 5)], dtype: model.computeDType, key: MLXRandom.key(42))
            let sigmas = buildPingPongSchedule(steps: 8, sigmaMax: 1, useLogSNRShift: true)
            var index = 0
            let fn: (MLXArray, MLXArray) -> MLXArray = { x, t in
                if [0,3,7].contains(index) { saved["latent-\(index)"] = x; saved["t-\(index)"] = t }
                index += 1
                return dit(x, t: t, crossAttnCondRaw: cross, globalCondRaw: global)
            }
            guard let latents = sampleFlowPingPong(modelFn: fn, initial: noise, sigmas: sigmas, seed: 43) else { throw EngineError("Cancelled capture") }
            saved["decoder-input"] = latents.asType(.float32); return saved
        }
        var shapes = ["ids": [1,256], "mask": [1,256], "conditioning": [1,257,768], "global": [1,768], "decoder-input": [1,256,54]]
        if model.variant == .smallSFX { shapes["seconds-value"] = [1] }
        for i in [0,3,7] { shapes["latent-\(i)"] = [1,256,54]; shapes["t-\(i)"] = [1] }
        try ComponentCapture.validate(inputs, shapes: shapes)
        let recorder = ComponentCapture()
        func traceDecoder(_ trace: ((String, MLXArray) -> Void)?) {
            if let medium = decoder as? SAMELDecoder { medium.activationTrace = trace }
            if let small = decoder as? SAMESDecoder { small.activationTrace = trace }
        }
        defer { text.encoder.activationTrace = nil; dit.transformer.activationTrace = nil; traceDecoder(nil) }
        text.encoder.activationTrace = { recorder.observe("prompt/" + $0, $1) }
        var outputs: [String: MLXArray] = ["prompt": text.encoder(inputs["ids"]!, attentionMask: inputs["mask"]!)]
        eval(outputs["prompt"]!)
        if model.variant == .smallSFX { outputs["seconds"] = seconds(inputs["seconds-value"]!.item(Float.self)) }
        for i in [0,3,7] {
            dit.transformer.activationTrace = { recorder.observe("denoiser-\(i)/" + $0, $1) }
            outputs["denoiser-\(i)"] = dit(inputs["latent-\(i)"]!, t: inputs["t-\(i)"]!, crossAttnCondRaw: inputs["conditioning"]!, globalCondRaw: inputs["global"]!)
            eval(outputs["denoiser-\(i)"]!)
        }
        var decoderTraceIndex = 0
        traceDecoder { name, value in
            // SAME-S serves overlapping windows; retain every window rather than overwrite a block's prior extrema.
            let key = model.variant == .medium ? "decoder/" + name : "decoder/\(decoderTraceIndex)/" + name
            recorder.observe(key, value); decoderTraceIndex += 1
        }
        outputs["decoder"] = model.variant == .medium
            ? decoder.decodeWindow(inputs["decoder-input"]!)
            : sa3DecodeLatents(decoder, inputs["decoder-input"]!)
        try recorder.write(outputs, to: output)
    }
}
