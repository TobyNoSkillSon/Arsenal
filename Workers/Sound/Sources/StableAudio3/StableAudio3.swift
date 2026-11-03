import Foundation
import MLX
import MLXNN
import MLXRandom

// Adapted from soniqo/speech-swift `StableAudio3MusicGen.swift` (Apache-2.0,
// commit 2af9c6c) and checked line by line against Stability's MLX runtime
// (`stable-audio-3/optimized/mlx/scripts/sa3_mlx.py`, MIT). Vireo changes:
// loads the official F16 export, noise drawn from `key(seed)` like the reference,
// fp32 timestep, cancellation, and the decode dispatch mirrors the reference. Runs the
// medium (SAME-L) and small SFX (SAME-S) checkpoints (`SA3Variant`).

public struct StableAudio3Params: Sendable, Equatable {
    public var seconds: Double
    public var steps: Int
    public var cfgScale: Float
    public var apg: Float
    public var sigmaMax: Float
    public var seed: UInt64
    public var negativePrompt: String?

    /// Defaults are the reference CLI's: 8 ping-pong steps, cfg 1 (the ARC
    /// post-trained model needs no guidance), APG 1, σmax 1.
    public init(seconds: Double = 30, steps: Int = 8, cfgScale: Float = 1.0, apg: Float = 1.0,
                sigmaMax: Float = 1.0, seed: UInt64 = 0, negativePrompt: String? = nil) {
        self.seconds = seconds; self.steps = steps; self.cfgScale = cfgScale; self.apg = apg
        self.sigmaMax = sigmaMax; self.seed = seed; self.negativePrompt = negativePrompt
    }
}

/// Planar stereo Float32 at 44.1 kHz.
public struct StableAudio3Output: Sendable {
    public var left: [Float]
    public var right: [Float]
    public var sampleRate: Int { SA3Audio.sampleRate }
    public var interleaved: [Float] {
        var out = [Float](repeating: 0, count: left.count * 2)
        for i in 0..<left.count { out[2 * i] = left[i]; out[2 * i + 1] = right[i] }
        return out
    }
}

/// Prompt → (embeddings [1, S, 768], mask [1, S]); T5Gemma in production, a stand-in in CPU tests.
public protocol SA3PromptEncoder: AnyObject {
    func encodePrompt(_ prompt: String) -> (embeds: MLXArray, mask: MLXArray)
}

extension T5GemmaText: SA3PromptEncoder {
    public func encodePrompt(_ prompt: String) -> (embeds: MLXArray, mask: MLXArray) { encode(prompt) }
}

public final class StableAudio3 {
    public private(set) var t5: (any SA3PromptEncoder)?
    public private(set) var dit: DiTMedium?
    public internal(set) var decoder: (any SA3LatentDecoder)?
    public private(set) var padding: MLXArray?
    public private(set) var secondsEmbedder: SecondsTotalEmbedder?
    public let reports: [SA3KeyReport]
    public let ditWeightDType: DType
    public let ditQuantBits: Int?
    public let variant: SA3Variant

    /// DiT compute dtype for latents/conditioning (reference default fp16; weights
    /// are whatever the checkpoint stores).
    public var computeDType: DType = .float16
    /// VIREO_SA3_PROFILE=1 (set by the adapter): one stderr line per render.
    public var profile = false

    /// Load all three components strictly (every tensor matched). `ditPrecision`
    /// nil keeps the stored dtype (F16 for the official export).
    public init(checkpoint: SA3Checkpoint, variant: SA3Variant = .medium, ditPrecision: String? = nil) throws {
        let d = try sa3LoadDiTMedium(checkpoint.dit, config: variant.dit, precision: ditPrecision, component: variant.ditComponent)
        let (dec, decReport): (any SA3LatentDecoder, SA3KeyReport)
        switch variant {
        case .medium: (dec, decReport) = try sa3LoadSAMELDecoder(checkpoint.decoder)
        case .smallSFX: (dec, decReport) = try sa3LoadSAMESDecoder(checkpoint.decoder)
        }
        let (t5, t5Report) = try sa3LoadT5Gemma(checkpoint.t5gemma)
        self.variant = variant
        self.dit = d.model
        self.padding = d.padding
        self.secondsEmbedder = d.secondsEmbedder
        self.decoder = dec
        self.t5 = t5
        self.reports = [d.report, decReport, t5Report]
        self.ditWeightDType = d.weightDType
        self.ditQuantBits = d.quantBits
        if d.weightDType == .bfloat16 { computeDType = .bfloat16 }
    }

    /// Assemble from loaded components (CPU tests with tiny modules).
    public init(dit: DiTMedium, decoder: any SA3LatentDecoder, text: any SA3PromptEncoder, padding: MLXArray,
                secondsEmbedder: SecondsTotalEmbedder, variant: SA3Variant, weightDType: DType = .float16) {
        self.dit = dit; self.decoder = decoder; self.t5 = text; self.padding = padding
        self.secondsEmbedder = secondsEmbedder; self.variant = variant
        self.reports = []; self.ditWeightDType = weightDType; self.ditQuantBits = nil
        if weightDType == .bfloat16 { computeDType = .bfloat16 }
    }

    /// Drop every weight reference. No compiled closures are created by this port,
    /// so nothing else captures the weights.
    public func unload() {
        t5 = nil; dit = nil; decoder = nil; padding = nil; secondsEmbedder = nil
    }

    /// `ceil(seconds * 44100 / 4096)`, decoder-independent like the reference.
    public static func latentLength(seconds: Double) -> Int {
        max(1, Int((seconds * Double(SA3Audio.sampleRate) / Double(SA3Audio.samplesPerLatent)).rounded(.up)))
    }

    /// Render `params.seconds` of stereo audio. `onStep(step, total)` returns false to
    /// cancel (the call then returns nil).
    public func generate(prompt: String, params: StableAudio3Params,
                         onStep: ((Int, Int) -> Bool)? = nil) throws -> StableAudio3Output? {
        let started = Date()
        guard let sampled = try sampleLatents(prompt: prompt, params: params, onStep: onStep) else { return nil }
        let decodeStart = Date()
        let out = try decodeAudio(sampled.latents, seconds: params.seconds)
        if profile, let dit {
            let end = Date()
            FileHandle.standardError.write(Data(String(format: "[sa3 profile] %.1f s · %d latents · %d steps · text %.0f ms · sampling %.0f ms (%.1f ms/step) · decode %.0f ms · total %.0f ms · dit %@%@%@%@\n",
                params.seconds, sampled.latents.dim(2), params.steps, sampled.textSeconds * 1000,
                sampled.samplingSeconds * 1000, sampled.samplingSeconds * 1000 / Double(params.steps),
                end.timeIntervalSince(decodeStart) * 1000, end.timeIntervalSince(started) * 1000,
                dit.localCondDType == .float32 ? "fp32 stream" : "fp16 stream", dit.promotedWeights ? " · fp32 weights" : "",
                params.cfgScale != 1 ? " · cfg" : "", (decoder as? SAMELDecoder)?.viewWindows == true ? " · SWA views" : "").utf8))
        }
        return out
    }

    /// Text encoding and flow sampling: the latents [1, 64 or 256, T] in the compute dtype, evaluated, with the text and
    /// sampling times. nil when `onStep` cancels.
    public func sampleLatents(prompt: String, params: StableAudio3Params, onStep: ((Int, Int) -> Bool)? = nil) throws
        -> (latents: MLXArray, textSeconds: Double, samplingSeconds: Double)? {
        guard let t5, let dit, let padding, let secondsEmbedder else {
            throw StableAudio3Error.weightLoadFailed("model is unloaded")
        }
        guard params.seconds > 0, params.seconds <= variant.maxSeconds else {
            throw StableAudio3Error.precision("seconds must be in (0, \(Int(variant.maxSeconds))]")
        }
        guard params.steps >= 1 else { throw StableAudio3Error.precision("steps must be ≥ 1") }
        let dtype = computeDType
        let started = Date()

        // 1) Text → cross-attention and global conditioning.
        let (embeds, mask) = t5.encodePrompt(prompt)
        let pad = padding.asType(dtype)
        let secondsEmbed = secondsEmbedder(Float(params.seconds)).asType(dtype)          // [1,1,768]
        let crossAttn = MLX.concatenated([applyPromptPadding(embeds: embeds.asType(dtype), mask: mask, paddingEmbedding: pad),
                                          secondsEmbed], axis: 1)                        // [1,257,768]
        let globalCond = secondsEmbed[0..., 0, 0...]                                     // [1,768]
        var nullCross: MLXArray? = nil
        if params.cfgScale != 1.0 {
            if let neg = params.negativePrompt {
                let (ne, nm) = t5.encodePrompt(neg)
                nullCross = MLX.concatenated([applyPromptPadding(embeds: ne.asType(dtype), mask: nm, paddingEmbedding: pad),
                                              secondsEmbed], axis: 1)
            } else {
                nullCross = MLXArray.zeros(crossAttn.shape, dtype: dtype)
            }
        }
        eval(crossAttn, globalCond)
        let textDone = Date()

        // 2) Noise from key(seed), as `mx.random.normal(..., key=mx.random.key(seed))`.
        let tLat = Self.latentLength(seconds: params.seconds)
        let noise = MLXRandom.normal([1, dit.config.ioChannels, tLat], dtype: dtype, key: MLXRandom.key(params.seed))

        // 3) Ping-pong rectified-flow sampling.
        let sigmas = buildPingPongSchedule(steps: params.steps, sigmaMax: params.sigmaMax, useLogSNRShift: true)
        let modelFn: (MLXArray, MLXArray) -> MLXArray = { x, t in
            sa3Velocity(dit: dit, x: x, t: t, cross: crossAttn, nullCross: nullCross, global: globalCond,
                        cfg: params.cfgScale, apg: params.apg)
        }
        guard let latents = sampleFlowPingPong(modelFn: modelFn, initial: noise, sigmas: sigmas,
                                               seed: params.seed &+ 1, onStep: onStep) else { return nil }
        eval(latents)
        return (latents, textDone.timeIntervalSince(started), Date().timeIntervalSince(textDone))
    }

    /// SAME decode and unpatch of sampled latents to `seconds` of stereo audio.
    public func decodeAudio(_ latents: MLXArray, seconds: Double) throws -> StableAudio3Output {
        guard let decoder else { throw StableAudio3Error.weightLoadFailed("model is unloaded") }
        // 4) SAME dispatch (SAME-L chunk 128 / overlap 8, SAME-S 8 / 2). SAME-L anchors its stream to stored half weights.
        // Float32 here is the transport input/output type, not SAME-L component compute.
        let patches = sa3DecodeLatents(decoder, latents.asType(.float32))
        eval(patches)

        // 5) Unpatch → [1, 2, T*4096], trim to round(seconds * 44100) like the reference.
        let audio = patchedDecode(patches, patchSize: SA3Audio.patchSize, channels: SA3Audio.channels).asType(.float32)
        let n = min(audio.dim(2), Int((seconds * Double(SA3Audio.sampleRate)).rounded()))
        let trimmed = audio[0..., 0..., 0..<n]
        eval(trimmed)
        return StableAudio3Output(left: trimmed[0, 0, 0...].asArray(Float.self),
                                  right: trimmed[0, 1, 0...].asArray(Float.self))
    }
}

/// Velocity for one sampler step, with batched CFG + APG exactly as `model_fn` in
/// Stability's `sa3_mlx.py` (fp32 guidance maths, result cast back to `x.dtype`).
public func sa3Velocity(dit: DiTMedium, x: MLXArray, t: MLXArray, cross: MLXArray, nullCross: MLXArray?,
                        global: MLXArray, cfg: Float, apg: Float) -> MLXArray {
    guard cfg != 1.0, let nullCross else {
        return dit(x, t: t, crossAttnCondRaw: cross, globalCondRaw: global)
    }
    let v2 = dit(MLX.concatenated([x, x], axis: 0), t: MLX.concatenated([t, t], axis: 0),
                 crossAttnCondRaw: MLX.concatenated([cross, nullCross], axis: 0),
                 globalCondRaw: MLX.concatenated([global, global], axis: 0))
    let halves = MLX.split(v2, parts: 2, axis: 0)
    let sigma = t.reshaped([-1, 1, 1]).asType(.float32)
    let x32 = x.asType(.float32)
    let condD = x32 - halves[0].asType(.float32) * sigma
    let uncondD = x32 - halves[1].asType(.float32) * sigma
    let diff = condD - uncondD
    var cfgDiff = diff
    if apg > 0 {
        let norm = MLX.sqrt((condD * condD).sum(axes: [-2, -1], keepDims: true))
        let unit = condD / MLX.maximum(norm, MLXArray(Float(1e-8)))
        let orth = diff - (diff * unit).sum(axes: [-2, -1], keepDims: true) * unit
        cfgDiff = apg >= 1 ? orth : MLXArray(apg) * orth + MLXArray(1 - apg) * diff
    }
    let cfgD = condD + MLXArray(cfg - 1) * cfgDiff
    return ((x32 - cfgD) / sigma).asType(x.dtype)
}

/// A SAME decoder ([B, 256, T] latents → [B, 512, T·16] patches) with its chunked-decode settings
/// (`DECODER_CHOICES` in `sa3_mlx.py`).
public protocol SA3LatentDecoder: AnyObject {
    func decodeWindow(_ latents: MLXArray) -> MLXArray
    var decodeChunk: Int { get }
    var decodeOverlap: Int { get }
}

extension SAMELDecoder: SA3LatentDecoder {
    public func decodeWindow(_ latents: MLXArray) -> MLXArray { self(latents) }
    public var decodeChunk: Int { 128 }
    public var decodeOverlap: Int { 8 }
}

extension SAMESDecoder: SA3LatentDecoder {
    public func decodeWindow(_ latents: MLXArray) -> MLXArray { self(latents) }
    public var decodeChunk: Int { 8 }
    public var decodeOverlap: Int { 2 }
}

/// Decoder dispatch from `sa3_mlx.py` step 4 (decoder-independent T_lat): chunked
/// beyond one kernel; even T whole; odd T > 6 chunked (2, 2); tiny odd T reflect-padded.
/// Every SAME-S window is even, as that decoder requires.
public func sa3DecodeLatents(_ decoder: any SA3LatentDecoder, _ lat: MLXArray) -> MLXArray {
    let T = lat.dim(2), chunk = decoder.decodeChunk, overlap = decoder.decodeOverlap
    let decode = { (x: MLXArray) in decoder.decodeWindow(x) }
    if T > chunk + 2 * overlap { return sa3DecodeChunked(decode, latents: lat, chunkSize: chunk, overlap: overlap) }
    if T % 2 == 0 { return decode(lat) }
    if T > 6 { return sa3DecodeChunked(decode, latents: lat, chunkSize: 2, overlap: 2) }
    let even = MLX.concatenated([lat, lat[0..., 0..., (T - 1)...]], axis: -1)
    return decode(even)[0..., 0..., 0..<(T * SAMELDims.sinPerPos)]
}
