// SPDX-License-Identifier: MIT
// Fish S1 DAC (44.1 kHz, 10 codebooks, 2048 samples per frame), the codec of Fish Audio Drama 3. Derived from mlx-audio-swift
// MLXAudioCodecs/FishS1DAC/FishS1DAC.swift at 01dec7c9bdce3088a6b6b7ab9f2e403458195efb (MIT), restructured to follow the
// reference mlx-audio Python mlx_audio/codec/models/fish_s1_dac at dad2087f: its parameter tree, sanitize, weight-norm
// folding, and causal streaming decode. The Hugging Face download path and the AudioCodecModel conformance are dropped.
// See lab/drama3-port/codec-notes.md.
import Foundation
import MLX
import MLXNN

/// The architecture constants. `drama3` is the reference `build_ae()`; tests use small variants.
struct FishS1CodecConfig {
    var encoderDim = 64
    var encoderRates = [2, 4, 8, 8]
    var latentDim = 1024
    var decoderDim = 1536
    var decoderRates = [8, 8, 4, 2]
    var nCodebooks = 9
    var codebookSize = 1024
    var codebookDim = 8
    var semanticCodebookSize = 4096
    var downsampleFactor = [2, 2]
    /// Transformer layers after each encoder block; only the last (dim 1024) has any.
    var encoderTransformerLayers = [0, 0, 0, 4]
    var encoderWindow = 512
    var encoderBlockSize = 16_384
    /// Encoder transformers use n_head = dim / encoderHeadDim heads of encoderHeadDim (the reference hardcodes dim // 64, head 64).
    var encoderHeadDim = 64
    var quantizerLayers = 8
    var quantizerHeads = 16
    var quantizerHeadDim = 64
    var quantizerIntermediate = 3072
    var quantizerWindow = 128
    var quantizerBlockSize = 4096
    var ropeBase = 10_000
    var normEps: Float = 1e-5
    var sampleRate = 44_100

    static let drama3 = FishS1CodecConfig()

    var hopLength: Int { encoderRates.reduce(1, *) }
    /// The reference hardcodes hop * 4; 4 is the product of the quantizer's downsample factors.
    var frameLength: Int { hopLength * downsampleFactor.reduce(1, *) }
}

final class FishS1EncoderBlock: Module, UnaryLayer {
    @ModuleInfo(key: "block") var block: [Module]

    init(dim: Int, stride: Int, transformerLayers: Int, config: FishS1CodecConfig) {
        let transformer: Module
        if transformerLayers == 0 {
            transformer = FishS1Identity()
        } else {
            let args = FishS1TransformerArgs(
                blockSize: config.encoderBlockSize, nLayer: transformerLayers, nHead: dim / config.encoderHeadDim, dim: dim,
                intermediateSize: dim * 3, nLocalHeads: dim / config.encoderHeadDim, headDim: config.encoderHeadDim,
                ropeBase: config.ropeBase, normEps: config.normEps)
            transformer = FishS1WindowLimitedTransformer(args, windowSize: config.encoderWindow)
        }
        _block.wrappedValue = [
            FishS1ResidualUnit(dim: dim / 2, dilation: 1),
            FishS1ResidualUnit(dim: dim / 2, dilation: 3),
            FishS1ResidualUnit(dim: dim / 2, dilation: 9),
            FishS1Snake1d(channels: dim / 2),
            FishS1CausalWNConv1d(inChannels: dim / 2, outChannels: dim, kernelSize: 2 * stride, stride: stride),
            transformer,
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in block { h = fishS1Call(layer, h) }
        return h
    }
}

final class FishS1Encoder: Module, UnaryLayer {
    @ModuleInfo(key: "block") var block: [Module]

    init(_ config: FishS1CodecConfig) {
        var dim = config.encoderDim
        var layers: [Module] = [FishS1CausalWNConv1d(inChannels: 1, outChannels: dim, kernelSize: 7)]
        for (stride, transformerLayers) in zip(config.encoderRates, config.encoderTransformerLayers) {
            dim *= 2
            layers.append(FishS1EncoderBlock(dim: dim, stride: stride, transformerLayers: transformerLayers, config: config))
        }
        layers += [FishS1Snake1d(channels: dim), FishS1CausalWNConv1d(inChannels: dim, outChannels: config.latentDim, kernelSize: 3)]
        _block.wrappedValue = layers
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in block { h = fishS1Call(layer, h) }
        return h
    }
}

final class FishS1DecoderBlock: Module, UnaryLayer, FishS1Steppable {
    @ModuleInfo(key: "block") var block: [Module]

    init(inputDim: Int, outputDim: Int, stride: Int) {
        _block.wrappedValue = [
            FishS1Snake1d(channels: inputDim),
            FishS1CausalWNConvTranspose1d(inChannels: inputDim, outChannels: outputDim, kernelSize: 2 * stride, stride: stride),
            FishS1ResidualUnit(dim: outputDim, dilation: 1),
            FishS1ResidualUnit(dim: outputDim, dilation: 3),
            FishS1ResidualUnit(dim: outputDim, dilation: 9),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in block { h = fishS1Call(layer, h) }
        return h
    }

    func step(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in block { h = fishS1Step(layer, h) }
        return h
    }
}

final class FishS1Decoder: Module, UnaryLayer, FishS1Steppable {
    @ModuleInfo(key: "model") var model: [Module]

    init(_ config: FishS1CodecConfig) {
        let channels = config.decoderDim
        var layers: [Module] = [FishS1CausalWNConv1d(inChannels: config.latentDim, outChannels: channels, kernelSize: 7)]
        var outputDim = channels
        for (i, stride) in config.decoderRates.enumerated() {
            let inputDim = channels >> i
            outputDim = channels >> (i + 1)
            layers.append(FishS1DecoderBlock(inputDim: inputDim, outputDim: outputDim, stride: stride))
        }
        layers += [FishS1Snake1d(channels: outputDim), FishS1CausalWNConv1d(inChannels: outputDim, outChannels: 1, kernelSize: 7), Tanh()]
        _model.wrappedValue = layers
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in model { h = fishS1Call(layer, h) }
        return h
    }

    func step(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in model { h = fishS1Step(layer, h) }
        return h
    }
}

/// The Fish S1 DAC codec. Encode a reference clip to codes; decode codes to 44.1 kHz mono, whole or streamed.
public final class FishS1Codec: Module {
    @ModuleInfo(key: "encoder") var encoder: FishS1Encoder
    @ModuleInfo(key: "quantizer") var quantizer: FishS1DownsampleResidualVectorQuantize
    @ModuleInfo(key: "decoder") var decoder: FishS1Decoder
    let config: FishS1CodecConfig
    /// Compute dtype: inputs are cast to it. Float32 until `load` casts the weights.
    public private(set) var dtype: DType = .float32

    public var sampleRate: Int { config.sampleRate }
    /// Samples per codec frame (2048 at 44.1 kHz: 21.5 frames per second).
    public var frameLength: Int { config.frameLength }
    /// Codebooks per frame: one semantic plus the residual ones (10).
    public var numCodebooks: Int { 1 + config.nCodebooks }
    public var semanticCodebookSize: Int { config.semanticCodebookSize }
    public var codebookSize: Int { config.codebookSize }

    /// The Drama 3 codec with initial (random) weights; `load` fills them.
    public convenience override init() { self.init(config: .drama3) }

    init(config: FishS1CodecConfig) {
        self.config = config
        let quantizerArgs = FishS1TransformerArgs(
            blockSize: config.quantizerBlockSize, nLayer: config.quantizerLayers, nHead: config.quantizerHeads, dim: config.latentDim,
            intermediateSize: config.quantizerIntermediate, nLocalHeads: config.quantizerHeads, headDim: config.quantizerHeadDim,
            ropeBase: config.ropeBase, normEps: config.normEps)
        _encoder.wrappedValue = FishS1Encoder(config)
        _quantizer.wrappedValue = FishS1DownsampleResidualVectorQuantize(
            inputDim: config.latentDim, nCodebooks: config.nCodebooks, codebookDim: config.codebookDim, codebookSize: config.codebookSize,
            semanticCodebookSize: config.semanticCodebookSize, downsampleFactor: config.downsampleFactor,
            preModule: FishS1WindowLimitedTransformer(quantizerArgs, windowSize: config.quantizerWindow),
            postModule: FishS1WindowLimitedTransformer(quantizerArgs, windowSize: config.quantizerWindow))
        _decoder.wrappedValue = FishS1Decoder(config)
    }

    // MARK: Loading

    /// Loads `codec.safetensors` (float32 PyTorch-layout weights) and casts it to `dtype` in memory. Weight-norm kernels are
    /// folded from the checkpoint's own precision before the cast; the RoPE tables stay float32. Nothing is written to disk.
    public static func load(url: URL, dtype: DType = .float16) throws -> FishS1Codec {
        let codec = FishS1Codec(config: .drama3)
        try codec.loadWeights(try loadArrays(url: url), dtype: dtype)
        return codec
    }

    func loadWeights(_ raw: [String: MLXArray], dtype: DType) throws {
        let weights = Self.sanitize(raw)
        try update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        foldWeightNorm(dtype: dtype)
        var cast: [String: MLXArray] = [:]
        for (key, value) in weights { cast[key] = key.hasSuffix("freqs_cis") ? value : value.asType(dtype) }
        try update(parameters: ModuleParameters.unflattened(cast), verify: .all)
        eval(parameters())
        self.dtype = dtype
    }

    /// Computes every weight-norm kernel once (Python `fold_weight_norm`) and keeps it in `dtype`.
    func foldWeightNorm(dtype: DType) {
        let kernels = modules().compactMap { ($0 as? FishS1WeightNormed)?.fold(dtype: dtype) }
        eval(kernels)
    }

    /// Checkpoint key -> module key, exactly as the reference's `DAC.sanitize`; nil drops the tensor.
    static func sanitizedKey(_ key: String, weightNormPrefixes: Set<String>) -> String? {
        if key.hasSuffix(".causal_mask") { return nil }
        if key.contains(".conv.parametrizations.weight.original0") {
            return key.replacingOccurrences(of: ".conv.parametrizations.weight.original0", with: ".weight_g")
        }
        if key.contains(".conv.parametrizations.weight.original1") {
            return key.replacingOccurrences(of: ".conv.parametrizations.weight.original1", with: ".weight_v")
        }
        if key.hasSuffix(".conv.bias") {
            let prefix = String(key.dropLast(".conv.bias".count))
            return weightNormPrefixes.contains(prefix) ? prefix + ".bias" : key
        }
        if key.contains(".parametrizations.weight.original0") {
            return key.replacingOccurrences(of: ".parametrizations.weight.original0", with: ".weight_g")
        }
        if key.contains(".parametrizations.weight.original1") {
            return key.replacingOccurrences(of: ".parametrizations.weight.original1", with: ".weight_v")
        }
        return key
    }

    static func weightNormPrefixes(_ keys: some Sequence<String>) -> Set<String> {
        let marker = ".conv.parametrizations.weight.original0"
        return Set(keys.compactMap { key in key.range(of: marker).map { String(key[..<$0.lowerBound]) } })
    }

    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let prefixes = weightNormPrefixes(weights.keys)
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            if let mapped = sanitizedKey(key, weightNormPrefixes: prefixes) { out[mapped] = value }
        }
        return out
    }

    // MARK: Encode

    /// Mono samples at 44.1 kHz -> codes [numCodebooks, frames] (int32), frames = ceil(count / frameLength).
    public func encode(_ samples: [Float]) -> MLXArray {
        encode(MLXArray(samples))
    }

    /// Audio [L], [1, L] or [1, 1, L] at 44.1 kHz -> codes [numCodebooks, frames] (int32).
    public func encode(_ audio: MLXArray) -> MLXArray {
        var x = audio.reshaped(1, 1, -1).asType(dtype)
        let length = x.dim(2)
        let rightPad = (length + frameLength - 1) / frameLength * frameLength - length
        if rightPad > 0 { x = MLX.padded(x, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((0, rightPad))]) }
        let codes = quantizer.encodeCodes(encoder(x))
        return codes[0].asType(.int32)
    }

    // MARK: Decode

    private func batched(_ codes: MLXArray) -> MLXArray {
        let c = codes.ndim == 2 ? codes[.newAxis] : codes
        precondition(c.ndim == 3 && c.dim(0) == 1 && c.dim(1) == numCodebooks, "codes must be [\(numCodebooks), n] or [1, \(numCodebooks), n]")
        return c
    }

    /// Codes [numCodebooks, n] or [1, numCodebooks, n] -> float32 samples [n * frameLength], decoding all frames at once.
    public func decode(_ codes: MLXArray) -> MLXArray {
        decoder(quantizer.decode(batched(codes)))[0, 0].asType(.float32)
    }

    /// Clears decoder conv buffers, overlap tails and transformer KV caches. Call before each new stream.
    public func resetStreamingState() {
        for module in modules() { (module as? FishS1Resettable)?.resetState() }
    }

    /// Decodes only the new frames of a stream: codes [1, numCodebooks, n] or [numCodebooks, n] -> float32 samples
    /// [n * frameLength]. Concatenating successive calls matches `decode` of all frames at once.
    public func streamingStep(_ codes: MLXArray) -> MLXArray {
        decoder.step(quantizer.decodeStep(batched(codes)))[0, 0].asType(.float32)
    }
}
