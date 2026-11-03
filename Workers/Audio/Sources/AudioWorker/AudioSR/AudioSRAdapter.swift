// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN
import MLXRandom
import ModelPrecision
import ArsenalWire

/// Get-time conversion of `haoheliu/audiosr_basic` (one PyTorch state dict with the VAE, UNet, HiFi-GAN, an unused CLAP,
/// EMA copies and a duplicate condition VAE): keeps the three networks the pipeline runs, transposes kernels to MLX
/// layout and writes FP16 safetensors plus the latent scale factor.
enum AudioSRConversion {
    static let recipe = "cast:audio-audiosr-v1"
    static let marker = "arsenal-audiosr-layout.json"
    static let legacyMarker = "mobius-audiosr-layout.json" // legacy-compat: read in snapshots written before the rename
    static let base = NativeConversion(recipe: recipe, convert: prepare)

    /// Checkpoint name → snapshot name and the axes that move it into MLX layout; nil for tensors the pipeline never runs.
    static func name(_ source: String, rank: Int) -> (String, [Int]?)? {
        let target: String
        if source.hasPrefix("model.diffusion_model.") {
            var parts = String(source.dropFirst("model.diffusion_model.".count)).split(separator: ".").map(String.init)
            if parts.first == "input_blocks" || parts.first == "output_blocks", parts.count > 2 { parts.insert("layers", at: 2) }
            if parts.first == "middle_block", parts.count > 1 { parts.insert("layers", at: 1) }
            target = "unet." + parts.joined(separator: ".")
        } else if source.hasPrefix("first_stage_model.vocoder.") {
            target = "vocoder." + source.dropFirst("first_stage_model.vocoder.".count)
        } else if source.hasPrefix("first_stage_model.") {
            target = "vae." + source.dropFirst("first_stage_model.".count)
        } else {
            return nil
        }
        guard target.hasSuffix(".weight") else { return (target, nil) }
        switch rank {
        case 4: return (target, [0, 2, 3, 1]) // OIHW → OHWI
        case 3: return (target, target.hasPrefix("vocoder.ups.") ? [1, 2, 0] : [0, 2, 1]) // transposed (I,O,K) → (O,K,I); conv (O,I,K) → (O,K,I)
        default: return (target, nil)
        }
    }

    /// The DDPM's cosine schedule (s = 0.008, 1000 steps) in Float64, as `make_beta_schedule("cosine")` builds it.
    static func alphasCumprod(steps: Int = 1000, s: Double = 8e-3) -> [Double] {
        let bar = (0...steps).map { pow(cos((Double($0) / Double(steps) + s) / (1 + s) * Double.pi / 2), 2) }
        var product = 1.0
        return (0..<steps).map { i in
            product *= bar[i + 1] / bar[i]; return product
        }
    }

    static func prepare(_ context: NativeConversionContext, _ cancellation: NativeCancellation) throws -> JSONObject {
        guard context.bits == nil else { throw PrecisionError("AudioSR ships at 16-bit only") }
        let checkpoint = try TorchCheckpoint(context.source.appendingPathComponent("pytorch_model.bin"))
        let tensors = try checkpoint.tensors(at: ["state_dict"])
        guard let scaleTensor = tensors["scale_factor"], let alphasTensor = tensors["alphas_cumprod"] else {
            throw PrecisionError("AudioSR checkpoint lacks its diffusion buffers")
        }
        let scale = try checkpoint.array(scaleTensor).item(Float.self)
        let stored = try checkpoint.array(alphasTensor).asArray(Float.self)
        let computed = alphasCumprod()
        guard stored.count == computed.count, zip(stored, computed).allSatisfy({ abs(Double($0) - $1) <= 1e-6 }) else {
            throw PrecisionError("AudioSR checkpoint schedule is not the cosine schedule")
        }
        var mapping: [String: (TorchCheckpoint.Tensor, [Int]?)] = [:]
        for (source, tensor) in tensors {
            guard let (target, axes) = name(source, rank: tensor.shape.count) else { continue }
            guard mapping.updateValue((tensor, axes), forKey: target) == nil else { throw PrecisionError("Duplicate AudioSR tensor " + target) }
        }
        let expected = Dictionary(uniqueKeysWithValues: AudioSRModel().parameters().flattened().map { ($0.0, $0.1.shape) })
        for (key, (tensor, axes)) in mapping {
            let shape = axes.map { $0.map { tensor.shape[$0] } } ?? tensor.shape
            guard expected[key] == shape else { throw PrecisionError("AudioSR tensor does not fit the graph: " + key) }
        }
        guard Set(expected.keys) == Set(mapping.keys) else { throw PrecisionError("AudioSR checkpoint does not cover the graph") }
        _ = try StreamedPrecision.convert(
            keys: mapping.keys.sorted(),
            read: { key in
                try cancellation.check()
                let (tensor, axes) = mapping[key]!
                let array = try checkpoint.array(tensor)
                // Materialize transposed kernels row-major on the CPU: the shared writer copies a strided view element by
                // element (about 8 minutes for this checkpoint), a contiguous one as a single block.
                return axes.map { contiguous(array.transposed(axes: $0), stream: .cpu) } ?? array
            }, to: context.destination.appendingPathComponent("model.safetensors"), half: context.half, eligible: { _, _ in false })
        try JSONSerialization.data(withJSONObject: ["architecture": "audiosr", "layout_version": 1, "scale_factor": Double(scale)], options: [.sortedKeys])
            .write(to: context.destination.appendingPathComponent(marker))
        return try context.receipt(cancellation: cancellation)
    }
}

/// The AudioSR inference pipeline (audiosr 0.0.7 `super_resolution`): 48 kHz mono, padded to a multiple of 5.12 s,
/// low-passed at its detected cutoff, VAE-encoded as the condition, DDIM (v-prediction, η = 1, classifier-free
/// guidance), VAE-decoded, low mel bins restored from the input, HiFi-GAN, low STFT bins restored from the input.
struct AudioSRPipeline {
    /// 5.12 s at 48 kHz: 512 mel frames, 64 latent frames.
    static let block = 245_760
    let model: AudioSRModel
    let scale: Float
    let basis = AudioSRDSP.melBasis()
    /// Supplies every Gaussian draw in the reference's order (condition sample, initial latent, one per step).
    var noise: (_ shape: [Int]) -> MLXArray

    struct Prepared {
        var frames: Int, length: Int, targetFrames: Int
        var wave: [Float], lowpassed: [Float], lowpassMel: MLXArray
        var highcut: Int, filter: AudioSRDSP.Filter
    }

    /// (mel frames, samples) after padding to whole 5.12 s blocks.
    static func padding(frames: Int) -> (Int, Int) {
        let blocks = max(1, (frames + block - 1) / block)
        return (blocks * 512, blocks * block)
    }

    static func padSpec(_ x: MLXArray, frames: Int) -> MLXArray {
        let n = x.dim(0)
        if n == frames { return x }
        if n > frames { return x[..<frames] }
        return concatenated([x, MLXArray.zeros([frames - n, x.dim(1)], type: Float.self)], axis: 0)
    }

    /// `read_wav_file` + `lowpass_filtering_prepare_inference` + `wav_feature_extraction` on decoded 48 kHz mono samples.
    func prepare(_ samples: [Float], seed: UInt32) -> Prepared {
        let frames = samples.count
        // Padded up to whole 5.12 s blocks, in integers: the reference's float arithmetic lands one frame short for
        // some lengths (e.g. 1186 samples), which would leave its mel and latent shapes out of step.
        let (targetFrames, length) = Self.padding(frames: frames)
        let mean = samples.reduce(Float(0), +) / Float(frames)
        var wave = samples.map { $0 - mean }
        let peak = wave.reduce(Float(0)) { max($0, abs($1)) }
        wave = wave.map { $0 / (peak + 1e-8) * 0.5 }
        wave = Array((wave + [Float](repeating: 0, count: max(0, length - frames))).prefix(length))
        let (_, magnitude) = AudioSRDSP.melSpectrogram(wave, basis: basis)
        let energy = Self.padSpec(magnitude, frames: targetFrames).sum(axis: 0).asArray(Float.self)
        var cumulative: [Double] = [], total = 0.0
        for value in energy { total += Double(value); cumulative.append(total) }
        var cutoff = Double(AudioSRDSP.cutoff(cumulative: cumulative, percentile: 0.985)) / 1024 * 24_000
        if cutoff < 1000 { cutoff = 24_000 }
        let filter = AudioSRDSP.filter(seed: seed)
        let lowpassed = AudioSRDSP.lowpass(wave.map(Double.init), highcut: Int(cutoff), filter: filter).map(Float.init)
        let lowpassMel = Self.padSpec(AudioSRDSP.melSpectrogram(lowpassed, basis: basis).mel, frames: targetFrames)
        return Prepared(
            frames: frames, length: length, targetFrames: targetFrames, wave: wave, lowpassed: lowpassed, lowpassMel: lowpassMel, highcut: Int(cutoff),
            filter: filter)
    }

    /// DDIM timesteps (uniform, +1) with their ᾱ, ᾱ_prev and σ (η = 1), last step first.
    static func schedule(steps: Int) -> [(t: Int, a: Double, prev: Double, sigma: Double)] {
        let alphas = AudioSRConversion.alphasCumprod().map { Double(Float($0)) }
        let c = 1000 / steps
        let ts = stride(from: 0, to: 1000, by: c).map { $0 + 1 }
        // Step counts whose grid reaches t = 1000 (c divides 999, e.g. 300) index past the 1000-entry schedule; the
        // reference raises there. Arsenal reads the last entry instead.
        return ts.indices.reversed().map { i in
            let a = alphas[min(ts[i], 999)], prev = i == 0 ? alphas[0] : alphas[min(ts[i - 1], 999)]
            return (ts[i], a, prev, ((1 - prev) / (1 - a) * (1 - a / prev)).squareRoot())
        }
    }

    /// Condition latent → denoised latent [1, t, f, 16] (float32).
    func sample(condition: MLXArray, steps: Int, guidance: Float, cancellation: NativeCancellation, progress: (Int) -> Void = { _ in }) throws -> MLXArray {
        let shape = condition.shape
        let unconditional = (MLXArray.zeros(shape, type: Float.self) + Float(-11.4981) * scale).asType(.float16)
        let conditions = concatenated([(condition * scale).asType(.float16), unconditional], axis: 0)
        var x = noise(shape)
        for (index, step) in Self.schedule(steps: steps).enumerated() {
            try cancellation.check()
            let input = concatenated([x, x], axis: 0).asType(.float16)
            let v2 = model.unet(input, condition: conditions, timesteps: MLXArray([Int32(step.t), Int32(step.t)])).asType(.float32)
            let v = v2[1..<2] + guidance * (v2[0..<1] - v2[1..<2])
            let (sa, sb) = (Float(step.a.squareRoot()), Float((1 - step.a).squareRoot()))
            let eps = sa * v + sb * x, x0 = sa * x - sb * v
            let direction = Float((1 - step.prev - step.sigma * step.sigma).squareRoot()) * eps
            x = Float(step.prev.squareRoot()) * x0 + direction + Float(step.sigma) * noise(shape)
            eval(x)
            progress(index)
        }
        return x
    }

    /// `mel_replace_ops`: the input's low mel bins replace the generated ones.
    static func replaceLowBins(_ mel: MLXArray, lowpassMel: MLXArray) -> MLXArray {
        let energy = exp(lowpassMel).sum(axis: 0).asArray(Float.self)
        var cumulative: [Double] = [], total = 0.0
        for value in energy { total += Double(value); cumulative.append(total) }
        let bin = AudioSRDSP.cutoff(cumulative: cumulative, percentile: 0.985)
        guard bin > 0 else { return mel }
        return concatenated([lowpassMel[0..., ..<bin], mel[0..., bin...]], axis: 1)
    }

    /// `postprocessing`: below the input's cutoff STFT bin, the low-passed input's spectrum (energy-matched) replaces the
    /// generated one.
    static func replaceLowBand(_ out: [Float], lowpassed: [Float]) -> [Float] {
        let gt = AudioSRDSP.stft(lowpassed), generated = AudioSRDSP.stft(out)
        let magnitude = sqrt(gt.real * gt.real + gt.imag * gt.imag)
        var cumulative: [Double] = [], total = 0.0
        for value in magnitude.sum(axis: 0).asArray(Float.self) { total += Double(value); cumulative.append(total) }
        let bin = AudioSRDSP.cutoff(cumulative: cumulative, percentile: 0.985)
        let generatedMagnitude = sqrt(generated.real * generated.real + generated.imag * generated.imag)
        let ratioRaw = magnitude[0..., bin].sum().item(Float.self) / generatedMagnitude[0..., bin].sum().item(Float.self)
        let ratio = min(max(ratioRaw.isFinite ? ratioRaw : 1, 0.8), 1.2)
        let real = concatenated([gt.real[0..., ..<bin] / ratio, generated.real[0..., bin...]], axis: 1)
        let imag = concatenated([gt.imag[0..., ..<bin] / ratio, generated.imag[0..., bin...]], axis: 1)
        return AudioSRDSP.istft(real: real, imag: imag, length: out.count).asArray(Float.self)
    }

    func run(_ samples: [Float], seed: UInt32, steps: Int, guidance: Float, cancellation: NativeCancellation) throws -> [Float] {
        let prepared = prepare(samples, seed: seed)
        try cancellation.check()
        let posterior = model.vae.encode(prepared.lowpassMel.reshaped(1, prepared.targetFrames, 256, 1).asType(.float16))
        let condition = posterior.mean.asType(.float32) + posterior.std.asType(.float32) * noise(posterior.mean.shape)
        let latent = try sample(condition: condition, steps: steps, guidance: guidance, cancellation: cancellation)
        var mel = model.vae.decode((latent / scale).asType(.float16)).asType(.float32)[0, 0..., 0..., 0]
        mel = Self.replaceLowBins(mel, lowpassMel: prepared.lowpassMel)
        try cancellation.check()
        let wave = model.vocoder(mel.expandedDimensions(axis: 0).asType(.float16)).asType(.float32)[0].asArray(Float.self)
        var out = Self.replaceLowBand(wave, lowpassed: prepared.lowpassed)
        let peak = out.reduce(Float(0)) { max($0, abs($1)) }
        if peak > 0 { out = out.map { 0.5 * $0 / peak } }
        let mean = out.reduce(Float(0), +) / Float(out.count)
        return Array(out.map { $0 - mean }.prefix(prepared.frames))
    }
}

final class AudioSRAdapter: AudioAdapter {
    static let maximumSeconds = 60
    /// The authors' default is 50. A 60 s clip took 36.7 s at 50 steps on an M5 Max (quick check, 22 Oct), so 100 steps is
    /// about 75 s there, leaving roughly 8x headroom for slower Macs inside the op's 590 s timeout; 200 would leave 4x.
    static let stepRange = 10...100
    var scale: Float = 0.3342
    override var architecture: String { "audiosr" }
    override var operations: Set<String> { ["enhance"] }

    override func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        let marker = StoredNames.current(AudioSRConversion.marker, legacy: AudioSRConversion.legacyMarker, in: context.weights)
        if let data = try? Data(contentsOf: marker), let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let value = object["scale_factor"] as? Double
        {
            scale = Float(value)
        } else if !context.randomWeights {
            throw audioFailure("AudioSR snapshot lacks its layout marker; Get it again.")
        }
        try prepare(AudioSRModel(), context, cancellation: cancellation, eligible: { _, _ in false })
    }

    override func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        // `stream` is part of the enhancement kind's request (the benchmark workload sends it); the result is one file either way.
        try audioKeys(input, allowed: ["audio", "steps", "guidance_scale", "seed", "stream"])
        if let stream = input["stream"], stream.bool == nil { throw audioFailure("stream must be a boolean.") }
        let steps = try audioInteger(input, key: "steps", fallback: 50, range: Self.stepRange)
        let seed = try audioInteger(input, key: "seed", fallback: 42, range: 0...Int(UInt32.max))
        let guidance = input["guidance_scale"]?.number ?? 3.5
        guard guidance.isFinite, (1...10).contains(guidance) else { throw audioFailure("guidance_scale must be between 1 and 10.") }
        guard operation == "enhance", let model = graph as? AudioSRModel else { throw audioFailure("AudioSR is not loaded.") }
        let pcm = try audioInput(input, rate: AudioSRDSP.rate, seconds: Self.maximumSeconds)
        guard pcm.samples.count > 100 else { throw audioFailure("Audio is too short.") }
        // One key per Gaussian draw (condition sample, initial latent, one per step), split from the seed, so a seed always
        // gives the same result whatever else used MLX's global generator. Seeds do not reproduce PyTorch's samples.
        let keys = MLXRandom.split(key: MLXRandom.key(UInt64(seed)), into: AudioSRPipeline.schedule(steps: steps).count + 2)
        var draw = 0
        let pipeline = AudioSRPipeline(model: model, scale: scale) { shape in
            defer { draw += 1 }
            return MLXRandom.normal(shape, key: keys[draw])
        }
        let out = try pipeline.run(pcm.samples, seed: UInt32(seed), steps: steps, guidance: Float(guidance), cancellation: cancellation)
        let file = try audioWave(out, rate: AudioSRDSP.rate, channels: 1, id: "audio", input: input, cancellation: cancellation)
        return ["files": .array([file])]
    }
}
