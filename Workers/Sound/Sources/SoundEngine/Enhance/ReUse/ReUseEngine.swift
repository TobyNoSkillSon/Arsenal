// SPDX-License-Identifier: MIT
// Vireo engine adapter for NVIDIA RE-USE (universal speech enhancement,
// 9.6M parameters, half execution from an FP32-only source). Weights: NVIDIA's own
// nvidia/NVIDIA-NemotronLabs-RE-USE checkpoint, OpenMDW-1.1 at the pinned revision
// (earlier revisions were NVIDIA NSCLv1, non-commercial). Model code: SpeechLens (Apache-2.0).

import Foundation
import MLX
import MLXNN
import VireoCore

public enum ReUseEngine {
    /// Catalog `engine` key.
    public static let key = "re-use"
    public static let weightsFile = "model_mlx.safetensors"
    /// The maker's PyTorch checkpoint (nvidia/NVIDIA-NemotronLabs-RE-USE); Get casts it into `weightsFile`.
    public static let makerWeightsFile = "model.safetensors"
    /// Rates the model was trained on; anything else is resampled to the nearest supported rate.
    public static let supportedRates = [8_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000]

    /// `options["kernels"]`: "auto" (default: Metal kernels after `KernelGate` passed on this machine, else
    /// stock) or "stock" (plain MLX). `VIREO_KERNELS=off` forces stock. Nothing serves the kernels ungated.
    /// `options["chunk_seconds"]` overrides the 5 s window.
    public static let load: ModelLoader = { source in
        // 8/4-bit: the file converted at Get (QuantRules.reUse); the kernel gate keys its verdict on this file too.
        let url = SoundEngines.quantizedDirectory(source)?.appendingPathComponent(weightsFile)
            ?? source.directory.appendingPathComponent(source.options["weights"] ?? weightsFile)
        if let expected = source.options["quantize_bits"].flatMap(Int.init) {
            let tensors = try MLX.loadArrays(url: url, stream: .cpu)
            guard QuantizedWeights.bits(of: tensors) == expected else { throw EngineError("RE-USE checkpoint does not match the requested dtype") }
        }
        let requested = (source.options["kernels"] ?? "auto").lowercased()
        // This load's configuration: Standard and Exact cells turn the (inexact) kernels off.
        let configuration = FastPath(source: source)
        let kernels: ReUseKernels
        let reason: String
        if ReUseKernels.resolved(preferred: .metal, configuration: configuration) == .stock {
            (kernels, reason) = (.stock, configuration.forceStock ? "stock MLX (VIREO_FORCE_STOCK)" : "stock MLX (Metal kernels switched off)")
        } else if requested == "stock" {
            (kernels, reason) = (.stock, "stock MLX (catalog kernels: stock)")
        } else {
            let verdict = KernelGate.verdict(engine: key, kernelVersion: ReUseKernelGate.kernelVersion, weights: url, configuration: configuration)
            (kernels, reason) = verdict.passed
                ? (.metal, "Metal selective scan + causal conv (self-test passed: \(verdict.detail))")
                : (.stock, "stock MLX (kernel self-test failed: \(verdict.detail))")
        }
        var settings = ReUseSettings.standard
        if let raw = source.options["chunk_seconds"], let seconds = Double(raw) {
            settings = try settings.with(chunkSeconds: seconds)
        }
        return try ReUseModel(id: source.id, weights: url, kernels: kernels, settings: settings, reason: reason, configuration: configuration)
    }

    /// Converts the FP32 weights with `QuantRules.reUse`; the check loads the result through the real loader.
    static func convert(_ source: ModelSource, bits: Int, into directory: URL, report: inout QuantReport) throws {
        let output = directory.appendingPathComponent(weightsFile)
        let rule = report.rule
        try QuantizedWeights.convertFile(source: source.directory.appendingPathComponent(source.options["weights"] ?? weightsFile), output: output,
                                         skeleton: SEMamba(numBlocks: 30, kernels: .stock), rule: rule, bits: bits, report: &report) { _, expected in
            let model = try MambaEnhancer.loadModel(url: output, mode: .standard, kernels: .stock)
            let installed = model.leafModules().flattened().filter { $0.1 is QuantizedLinear }.count
            guard installed == expected else { throw EngineError("Converted RE-USE installs \(installed) quantized layers, expected \(expected)") }
        }
    }

    /// Child-process self-test (`vireo-helper --kernel-self-test re-use <weights>`): the kernel unit test on
    /// fixed-seed inputs, then the whole model on a fixed 1 s signal at 48 kHz and 0.5 s at 16 kHz, Metal vs
    /// stock. Pass: kernels within 1e-3 (relative), model output finite and ≥ 40 dB SNR against stock
    /// (M5 Max: ~80 dB, the 24-bit floor; the port notes ask ≥ 30 dB).
    static func childSelfTest(_ weights: URL) throws -> String {
        // Lab: VIREO_REUSE_GATE_FAIL=1 exercises the failed-verdict path (part of the verdict key).
        if FastPath.current.value("VIREO_REUSE_GATE_FAIL") == "1" { throw EngineError("forced failure (VIREO_REUSE_GATE_FAIL=1)") }
        let kernels = ReUseKernelGate.selfTest()
        guard kernels.passed else { throw EngineError("kernels \(kernels)") }
        let stock = try MambaEnhancer(modelWeightsURL: weights, kernels: .stock)
        let metal = try MambaEnhancer(modelWeightsURL: weights, kernels: .metal)
        var parts = [String(format: "kernels scan %.1e conv %.1e", kernels.scanMaxAbsError, kernels.convMaxAbsError)]
        for (rate, seconds) in [(48_000, 1.0), (16_000, 0.5)] {
            let x = testSignal(rate: rate, seconds: seconds)
            let (a, b) = try blocking {
                (try await ReUseModel.run(stock, x, rate: rate, settings: .standard, control: nil),
                 try await ReUseModel.run(metal, x, rate: rate, settings: .standard, control: nil))
            }
            guard b.count == a.count else { throw EngineError("Metal output length \(b.count) ≠ stock \(a.count) at \(rate) Hz") }
            guard b.allSatisfy(\.isFinite) else { throw EngineError("non-finite Metal output at \(rate) Hz") }
            let signal = a.reduce(0.0) { $0 + Double($1) * Double($1) }
            let noise = zip(a, b).reduce(0.0) { $0 + Double($1.0 - $1.1) * Double($1.0 - $1.1) }
            let snr = noise == 0 ? 999 : 10 * Foundation.log10(signal / Swift.max(noise, 1e-30))
            guard snr >= 40 else { throw EngineError(String(format: "model SNR %.1f dB vs stock at %d Hz (< 40)", snr, rate)) }
            parts.append(String(format: "model %d kHz %.0f dB", rate / 1000, snr))
        }
        return parts.joined(separator: ", ")
    }

    /// Deterministic speech-like test input: a gliding harmonic tone with an envelope plus LCG noise.
    static func testSignal(rate: Int, seconds: Double) -> [Float] {
        var state: UInt32 = 20_260_926
        return (0..<Int(Double(rate) * seconds)).map { i in
            state = state &* 1_664_525 &+ 1_013_904_223
            let t = Double(i) / Double(rate)
            let f0 = 140 + 40 * sin(2 * .pi * 1.3 * t)
            let voiced = (1...6).reduce(0.0) { $0 + sin(2 * .pi * f0 * Double($1) * t) / Double($1) }
            let envelope = 0.5 + 0.5 * sin(2 * .pi * 3 * t)
            let noise = Double(state >> 8) / Double(1 << 24) - 0.5
            return Float(0.2 * envelope * voiced + 0.05 * noise)
        }
    }

    /// Runs async work from the synchronous child entry point.
    static func blocking<T>(_ body: @escaping @Sendable () async throws -> T) throws -> T {
        let done = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var result: Result<T, Error>!
        Task { do { result = .success(try await body()) } catch { result = .failure(error) }; done.signal() }
        done.wait()
        return try result.get()
    }

    static func nearestSupportedRate(_ rate: Int) -> Int {
        supportedRates.min { abs($0 - rate) < abs($1 - rate) } ?? 48_000
    }
}

public final class ReUseModel: EnhanceModel {
    public let id: String
    /// RE-USE runs at the input's own rate; 48 kHz is its top (and upscale) rate.
    public let sampleRate = 48_000
    public private(set) var kernels: ReUseKernels
    /// /status `engine`: "optimized" while the gated Metal kernels serve, else "mlx".
    public var engineLabel: String { kernels == .metal ? "optimized" : "mlx" }
    public private(set) var engineReason: String
    private let settings: ReUseSettings
    private let weights: URL
    var enhancer: MambaEnhancer?
    /// The configuration the kernel gate ran under (keys a runtime failure).
    private let configuration: FastPath

    public init(id: String, weights: URL, kernels: ReUseKernels, settings: ReUseSettings = .standard, reason: String = "",
                configuration: FastPath = .current) throws {
        self.id = id
        self.configuration = configuration
        self.kernels = kernels
        self.settings = settings
        self.weights = weights
        self.engineReason = reason.isEmpty ? (kernels == .metal ? "Metal selective scan + causal conv" : "stock MLX") : reason
        self.enhancer = try MambaEnhancer(modelWeightsURL: weights, kernels: kernels)
    }

    /// Modes: "denoise" and "restore" (one universal model: noise, reverb, clipping, codec,
    /// bandwidth, packet loss), "upscale" (bandwidth extension: resample to
    /// `options["rate"]`, default 48000, then enhance — NVIDIA's BWE recipe).
    public func enhance(_ input: AudioClip, request: EnhanceRequest, control: GenerationControl) async throws -> AudioClip {
        guard enhancer != nil else { throw EngineError("\(id) is unloaded") }
        guard input.sampleRate > 0, input.frames > 0 else { throw invalidInput("empty audio") }
        let targetRate: Int
        switch request.mode {
        case "denoise", "restore":
            targetRate = input.sampleRate
        case "upscale":
            targetRate = Int(request.options["rate"] ?? "48000") ?? 48_000
            guard ReUseEngine.supportedRates.contains(targetRate), targetRate >= input.sampleRate else {
                throw invalidInput("upscale rate must be one of \(ReUseEngine.supportedRates.map(String.init).joined(separator: ", ")) and ≥ the input rate")
            }
        default:
            throw invalidInput("\(id) supports denoise, restore, upscale; not \(request.mode)")
        }
        let modelRate = ReUseEngine.supportedRates.contains(targetRate) ? targetRate : ReUseEngine.nearestSupportedRate(targetRate)

        let channels = EnhanceAudio.channels(of: input)
        var outputs = [[Float]]()
        let total = Double(channels.count)
        for (index, channel) in channels.enumerated() {
            let x = try EnhanceAudio.resample(channel, from: input.sampleRate, to: modelRate)
            let progress = Progress(control: control, base: Double(index) / total, span: 1 / total)
            var y = try await Self.run(enhancer!, x, rate: modelRate, settings: settings, control: progress)
            // Lab: VIREO_REUSE_INJECT_NAN=1 poisons Metal output to exercise the fallback (never in the gate child).
            if kernels == .metal, configuration.value("VIREO_REUSE_INJECT_NAN") == "1", !y.isEmpty { y[y.count / 2] = .nan }
            // Runtime fallback: the Metal path passed its self-test, but bad output here switches this model
            // to stock for good (sticky verdict) and redoes the channel.
            if kernels == .metal, let bad = y.firstIndex(where: { !$0.isFinite }) {
                let detail = "non-finite output at frame \(bad)"
                ReUseLog.inference.error("RE-USE Metal kernels: \(detail, privacy: .public); falling back to stock")
                KernelGate.recordRuntimeFailure(engine: ReUseEngine.key, kernelVersion: ReUseKernelGate.kernelVersion, weights: weights,
                                                configuration: configuration, detail: detail)
                enhancer = try MambaEnhancer(modelWeightsURL: weights, kernels: .stock)
                kernels = .stock
                engineReason = "stock MLX (fallback: Metal kernels gave \(detail))"
                y = try await Self.run(enhancer!, x, rate: modelRate, settings: settings, control: progress)
            }
            guard y.count == x.count else { throw EngineError("\(id) changed the frame count (\(x.count) → \(y.count))") }
            let back = modelRate == targetRate ? y : try EnhanceAudio.resample(y, from: modelRate, to: targetRate)
            try EnhanceAudio.requireFinite(back, model: id)
            outputs.append(back)
        }
        control.report(progress: 1)
        return AudioClip(samples: EnhanceAudio.interleave(outputs), sampleRate: targetRate, channels: channels.count)
    }

    struct Progress { let control: GenerationControl; let base: Double; let span: Double }

    /// One mono signal through a fresh session, block by block.
    static func run(_ enhancer: MambaEnhancer, _ x: [Float], rate: Int, settings: ReUseSettings, control: Progress?) async throws -> [Float] {
        let session = try await enhancer.makeSession(sampleRate: rate, settings: settings)
        let block = max(1, session.recommendedInputFrameCount)
        var y = [Float]()
        y.reserveCapacity(x.count)
        var start = 0
        while start < x.count {
            if control?.control.isCancelled == true { throw CancellationError() }
            let end = min(x.count, start + block)
            y.append(contentsOf: try await session.append(Array(x[start..<end])))
            if let control { control.control.report(progress: 0.98 * (control.base + control.span * Double(end) / Double(x.count))) }
            start = end
        }
        y.append(contentsOf: try await session.finish())
        return y
    }

    /// RE-USE uses no `compile()` closures; the Metal kernels are weight-free static
    /// programs. Dropping the enhancer releases every weight array.
    public func unload() async {
        enhancer = nil
    }
}
