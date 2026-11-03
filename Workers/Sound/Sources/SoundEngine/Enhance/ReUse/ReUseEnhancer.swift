// SPDX-License-Identifier: Apache-2.0
// Vendored from faraday/SpeechLens Sources/Inference (Apache-2.0)
// @ d20e11bcafbc32435ef18a17ed56caf22fca6833; Vireo changes listed in CHANGES.md.

import Foundation
import MLX
import MLXNN

/// The base sample rate used by the SE-Mamba model for internal scaling.
/// All STFT parameters are derived from (noisy_sr / baseSampleRate).
let SE_MAMBA_BASE_SAMPLE_RATE: Int = 8000

/// SpeechLens is a Metal-only runtime. Keep MLX graph construction and
/// evaluation explicitly on the GPU instead of relying on the package's
/// process- or task-default device semantics.
private enum ReUseExecutionContext {
    static func withGPU<Result>(_ operation: () throws -> Result) rethrows -> Result {
        try Device.withDefaultDevice(.gpu, operation)
    }
}

/// Matches the post-processing in the pinned NVIDIA RE-USE inference path.
/// A time frame is suppressed when more than half of its frequency bins were
/// clamped to zero by the non-negative log-magnitude reconstruction.
func reUseSuppressSweepArtifactFrames(_ magnitude: MLXArray) -> MLXArray {
    precondition(magnitude.ndim == 3, "Expected magnitude shaped [batch, frequency, time]")
    let zeroCounts = (magnitude .== 0).sum(axis: 1, keepDims: true)
    let suppressFrame = zeroCounts .> (magnitude.shape[1] / 2)
    return which(suppressFrame, MLXArray(0.0), magnitude)
}

/// A single-use, mono enhancement stream. Implementations may retain a bounded
/// amount of look-ahead and overlap-add state between calls.
public protocol ReUseSession: Sendable {
    var recommendedInputFrameCount: Int { get }

    func append(_ monoFrames: [Float]) async throws -> [Float]
    func finish() async throws -> [Float]
}

/// Creates independent mono sessions backed by one reusable enhancement model.
public protocol ReUseStreamingEnhancer: Sendable {
    func makeSession(
        sampleRate: Int,
        settings: ReUseSettings
    ) async throws -> any ReUseSession
}

/// MLX-native implementation of the SE-Mamba speech enhancement model.
public actor MambaEnhancer: ReUseStreamingEnhancer {
    let model: SEMamba
    public let mode: ReUseScanMode
    public let kernels: ReUseKernels

    /// Vireo: no metallib probe (the helper owns metallib placement) and no
    /// process-wide cache-limit side effect (the helper owns `Memory.cacheLimit`).
    /// Weights are mapped with `verify: .all`: every file key must land on a
    /// module parameter, every parameter must be set, shapes must match.
    public init(modelWeightsURL: URL, mode: ReUseScanMode = .standard, kernels: ReUseKernels = .metal) throws {
        self.mode = mode
        self.kernels = kernels
        self.model = try Self.loadModel(url: modelWeightsURL, mode: mode, kernels: kernels)
        ReUseLog.inference.debug("Loaded weights path: \(modelWeightsURL.path, privacy: .private)")
    }

    static func loadModel(url: URL, mode: ReUseScanMode, kernels: ReUseKernels) throws -> SEMamba {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ReUseError.modelNotFound(url)
        }
        var weights: [String: MLXArray]
        do {
            weights = try MLX.loadArrays(url: url)  // CPU stream; no GPU work
        } catch {
            throw ReUseError.weightMappingFailed("cannot read \(url.lastPathComponent): \(error)")
        }
        try validateModelWeights(weights)
        // Vireo: a converted 8/4-bit file (QuantRules.reUse) holds packed uint32 weights beside scales/biases.
        let packed = Set(weights.keys.filter { $0.hasSuffix(".scales") }.map { String($0.dropLast(".scales".count)) + ".weight" })
        guard weights.values.allSatisfy({ [.float16, .bfloat16, .uint32].contains($0.dtype) }),
            Set(weights.values.filter { $0.dtype != .uint32 }.map(\.dtype)).count == 1 else {
            throw ReUseError.unsupportedModelArchitecture("RE-USE requires uniform half learned weights and affine metadata")
        }
        let model = SEMamba(numBlocks: 30, mode: mode, kernels: kernels)
        do {
            if !packed.isEmpty { try QuantizedWeights.prepare(model, weights: weights) }
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        } catch {
            throw ReUseError.weightMappingFailed("\(error)")
        }
        return model
    }

    /// Weight/key-mapping check without inference: loads the file on the CPU,
    /// maps it strictly onto the module tree and reports counts.
    public static func verifyWeights(url: URL) throws -> (tensors: Int, parameters: Int) {
        let model = try loadModel(url: url, mode: .standard, kernels: .stock)
        let params = model.parameters().flattened()
        let count = params.reduce(0) { $0 + $1.1.size }
        return (params.count, count)
    }

    static func validateModelWeights(_ weights: [String: MLXArray]) throws {
        let branches = ["timeMamba.forward", "timeMamba.backward", "freqMamba.forward", "freqMamba.backward"]
        let blockIndices = Set(weights.keys.compactMap { key -> Int? in
            let components = key.split(separator: ".", maxSplits: 2)
            guard components.count == 3, components[0] == "tfMamba" else { return nil }
            return Int(components[1])
        })

        guard blockIndices == Set(0..<30) else {
            throw ReUseError.unsupportedModelArchitecture(
                "expected exactly 30 TFMamba blocks indexed 0 through 29"
            )
        }

        for block in 0..<30 {
            for branch in branches {
                let prefix = "tfMamba.\(block).\(branch)"
                let aLog = try requiredWeight(weights, key: "\(prefix).A_log")
                let d = try requiredWeight(weights, key: "\(prefix).D")
                let xProj = try requiredWeight(weights, key: "\(prefix).xProj.weight")
                let dtProj = try requiredWeight(weights, key: "\(prefix).dtProj.weight")
                do {
                    try ReUseMetalSelectiveScan.validateModelLayout(
                        aLogShape: aLog.shape,
                        dShape: d.shape,
                        xProjWeightShape: weights["\(prefix).xProj.scales"].map { [xProj.shape[0], $0.shape[1] * 64] } ?? xProj.shape,
                        dtProjWeightShape: dtProj.shape
                    )
                } catch let error as ReUseSelectiveScanError {
                    throw ReUseError.metalInferenceIncompatible(error.localizedDescription)
                }
            }
        }
    }

    private static func requiredWeight(
        _ weights: [String: MLXArray],
        key: String
    ) throws -> MLXArray {
        guard let weight = weights[key] else {
            throw ReUseError.unsupportedModelArchitecture("missing required tensor \(key)")
        }
        return weight
    }

    public func makeSession(
        sampleRate: Int,
        settings: ReUseSettings
    ) throws -> any ReUseSession {
        try MambaEnhancementSession(
            enhancer: self,
            sampleRate: sampleRate,
            settings: settings
        )
    }

    func enhanceWindow(_ samples: [Float], sampleRate: Int) throws -> [Float] {
        try Task.checkCancellation()
        guard sampleRate > 0 else {
            throw ReUseError.inferenceFailed("Sample rate must be greater than zero")
        }
        guard !samples.isEmpty else { return [] }

        let params = Self.scaledSTFTParams(sampleRate: sampleRate)
        ReUseLog.inference.debug(
            "Running enhancement at \(sampleRate, privacy: .public) Hz with nFFT=\(params.nFFT, privacy: .public), hop=\(params.hop, privacy: .public), win=\(params.win, privacy: .public)"
        )
        let output = try ReUseExecutionContext.withGPU {
            try enhanceChunkSamples(samples, params: params)
        }
        try Task.checkCancellation()
        guard output.count == samples.count else {
            throw ReUseError.inferenceFailed("Model window changed frame count")
        }
        return output
    }

    /// Runs STFT → model → iSTFT on one chunk and returns exactly
    /// `samples.count` samples.
    ///
    /// `MLX.eval` is called explicitly on the model output before `asArray` so
    /// intermediate graph state is released before the next chunk builds its
    /// graph.
    private func enhanceChunkSamples(
        _ samples: [Float],
        params: (nFFT: Int, hop: Int, win: Int)
    ) throws -> [Float] {
        let (magIn, phaIn, frameCount) = try stftInputs(
            samples: samples,
            nFFT: params.nFFT,
            hop: params.hop,
            win: params.win
        )
        let fBins = params.nFFT / 2 + 1

        let half = model.denseEncoder.conv1.parameters().flattened().first!.1.dtype
        let (modelOutMag, modelOutPha) = model(magIn.asType(half), phaIn.asType(half))
        if modelOutMag.shape != [1, fBins, frameCount]
            || modelOutPha.shape != [1, fBins, frameCount]
        {
            throw ReUseError.inferenceFailed("Unexpected model output shape")
        }
        let nonnegativeMagnitude = expm1(maximum(modelOutMag.asType(.float32), MLXArray(0.0)))
        let enhMag = reUseSuppressSweepArtifactFrames(nonnegativeMagnitude)

        // Force MLX to evaluate and release the forward-pass graph before we
        // move on — without this barrier, intermediates from every chunk stack
        // up and RSS grows linearly with audio length.
        MLX.eval(enhMag, modelOutPha)

        let padLen = params.nFFT / 2
        let paddedLen = samples.count + 2 * padLen
        let reconstructed = istft(
            mag: enhMag,
            pha: modelOutPha.asType(.float32),
            originalCount: paddedLen,
            nFFT: params.nFFT,
            hop: params.hop,
            win: params.win
        )
        let trimStart = padLen
        let trimEnd = min(trimStart + samples.count, reconstructed.count)
        if trimStart < trimEnd {
            return Array(reconstructed[trimStart..<trimEnd])
        }
        return [Float](repeating: 0, count: samples.count)
    }
    /// Mirrors NVIDIA RE-USE's `make_even(base * sampleRate // baseRate)`.
    /// The integer division is intentional: at 44.1 kHz, the trained hop
    /// scales to 220 rather than rounding the half sample up to 222.
    static func nvidiaScaledEvenParameter(
        base: Int,
        sampleRate: Int,
        baseSampleRate: Int = SE_MAMBA_BASE_SAMPLE_RATE
    ) -> Int {
        precondition(base > 0 && sampleRate > 0 && baseSampleRate > 0)
        let (product, overflow) = base.multipliedReportingOverflow(by: sampleRate)
        precondition(!overflow, "STFT parameter scaling overflowed Int")
        let scaled = product / baseSampleRate
        return (scaled % 2 == 0) ? max(2, scaled) : max(2, scaled + 1)
    }

    static func scaledSTFTParams(sampleRate: Int) -> (nFFT: Int, hop: Int, win: Int) {
        return (
            nFFT: nvidiaScaledEvenParameter(base: 320, sampleRate: sampleRate),
            hop: nvidiaScaledEvenParameter(base: 40, sampleRate: sampleRate),
            win: nvidiaScaledEvenParameter(base: 320, sampleRate: sampleRate)
        )
    }

    /// Returns the periodic Hann window used by PyTorch's
    /// `torch.hann_window(length)` default (`periodic == true`). Keeping the
    /// denominator at `length` is part of the NVIDIA RE-USE STFT contract;
    /// the symmetric `length - 1` form would produce a different model input
    /// and a different overlap-add synthesis window.
    static func nvidiaPeriodicHannWindow(_ length: Int) -> [Float] {
        guard length > 1 else { return [Float](repeating: 1.0, count: max(1, length)) }
        let denom = Float(length)
        return (0..<length).map { i in
            0.5 - 0.5 * Foundation.cos(2.0 * Float.pi * Float(i) / denom)
        }
    }

    private func stftInputs(
        samples: [Float],
        nFFT: Int,
        hop: Int,
        win: Int
    ) throws -> (mag: MLXArray, pha: MLXArray, frameCount: Int) {
        // Center padding (reflect) — matches PyTorch center=True, pad_mode='reflect'
        let paddedArray = centeredReflectPaddedArray(samples, nFFT: nFFT)

        // STFT framing retains only complete windows, matching PyTorch's
        // floor-based torch.stft enumeration. Any incomplete final hop is
        // discarded from the model input; iSTFT still reconstructs to the
        // requested padded length and the outer trim preserves frame count.
        let framing: ReUseFrameWindowPlan
        let frames: MLXArray
        do {
            (frames, framing) = try ReUseFrameFramer.frames(paddedArray, window: win, hop: hop)
        } catch {
            throw ReUseError.inferenceFailed(error.localizedDescription)
        }
        let window = MLXArray(Self.nvidiaPeriodicHannWindow(win), [1, win])

        var windowedFrames = frames * window
        if nFFT > win {
            let zeroTail = MLX.zeros([framing.frameCount, nFFT - win])
            windowedFrames = MLX.concatenated([windowedFrames, zeroTail], axis: 1)
        }

        let spec = rfft(windowedFrames, n: nFFT, axis: -1)
        let real = spec.realPart()
        let imag = spec.imaginaryPart()
        let mag = (real * real + imag * imag).sqrt().log1p().transposed(1, 0).expandedDimensions(axis: 0)
        let pha = MLX.atan2(imag, real).transposed(1, 0).expandedDimensions(axis: 0)
        return (mag, pha, framing.frameCount)
    }

    private func centeredReflectPaddedArray(_ samples: [Float], nFFT: Int) -> MLXArray {
        let padLen = nFFT / 2
        let n = samples.count
        if n > padLen + 1 {
            let samplesArr = MLXArray(samples)
            let prefix = samplesArr[.stride(from: padLen, to: 0, by: -1)]
            let suffix = samplesArr[.stride(from: n - 2, to: n - 2 - padLen, by: -1)]
            return MLX.concatenated([prefix, samplesArr, suffix])
        }
        var padded = [Float]()
        padded.reserveCapacity(n + 2 * padLen)
        for i in stride(from: padLen, through: 1, by: -1) {
            padded.append(samples[min(i, n - 1)])
        }
        padded.append(contentsOf: samples)
        for i in 0..<padLen {
            padded.append(samples[max(0, n - 2 - i)])
        }
        return MLXArray(padded)
    }

    private func istft(
        mag: MLXArray,
        pha: MLXArray,
        originalCount: Int,
        nFFT: Int,
        hop: Int,
        win: Int
    ) -> [Float] {
        let squeezedMag = mag.squeezed(axis: 0)
        let squeezedPha = pha.squeezed(axis: 0)
        let frameCount = squeezedMag.shape.count == 2 ? squeezedMag.shape[1] : 0
        guard frameCount > 0 else { return [Float](repeating: 0, count: originalCount) }
        let outLen = max(originalCount, (frameCount - 1) * hop + win)
        let window = Self.nvidiaPeriodicHannWindow(win)

        let magTF = squeezedMag.transposed(1, 0)
        let phaTF = squeezedPha.transposed(1, 0)
        let spectrum = magTF * phaTF.cos() + (magTF * phaTF.sin()).asImaginary()
        let timeFrames = irfft(spectrum, n: nFFT, axis: -1)
        MLX.eval(timeFrames)
        let frameTimeFlat = timeFrames.asArray(Float.self)

        var out = [Float](repeating: 0, count: outLen)
        var norm = [Float](repeating: 0, count: outLen)

        for t in 0..<frameCount {
            let start = t * hop
            let frameOffset = t * nFFT
            for i in 0..<win {
                let dst = start + i
                if dst < outLen {
                    let w = window[i]
                    out[dst] += frameTimeFlat[frameOffset + i] * w
                    norm[dst] += w * w
                }
            }
        }

        for i in 0..<outLen where norm[i] > 1e-8 {
            out[i] /= norm[i]
        }

        return Array(out.prefix(originalCount))
    }
}

/// Errors occurring during model loading or inference.
public enum ReUseError: Error, LocalizedError, Sendable, Equatable {
    case modelNotFound(URL)
    case unsupportedModelArchitecture(String)
    case metalInferenceIncompatible(String)
    case weightMappingFailed(String)
    case inferenceFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let url): return "Model weights not found at \(url.path)"
        case .unsupportedModelArchitecture(let message): return "Unsupported model architecture: \(message)"
        case .metalInferenceIncompatible(let message): return "Metal inference incompatible: \(message)"
        case .weightMappingFailed(let msg): return "Weight mapping failed: \(msg)"
        case .inferenceFailed(let msg): return "Inference failed: \(msg)"
        }
    }
}

/// Vireo: FP16 tensors widened to FP32 (exact); other dtypes unchanged.
func widenedToFloat32(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    weights.mapValues { $0.dtype == .float16 ? $0.asType(.float32) : $0 }
}
