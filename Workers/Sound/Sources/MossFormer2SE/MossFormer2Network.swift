// SPDX-License-Identifier: MIT
// MossFormer2 SE mask network. Vendored from Blaizzy/mlx-audio-swift @ 0dda61b29c308175e302dceef77c3bd84551e92b
// (Sources/MLXAudioSTS/Models/MossFormer2SE/MossFormer2Model.swift, MIT, licenses/mlx-audio-swift-MIT.txt).
// Arsenal changes: no Hugging Face download, STS protocol or safetensors directory scan (Arsenal's Get supplies the
// weights); the positional embedding follows the activation dtype; a named activation trace for component parity.
import Foundation
import MLX
import MLXNN

/// Called with (name, activation) at the component boundaries the parity gate compares.
public typealias MossFormer2Trace = (String, MLXArray) -> Void

final class MossFormerM: Module {
    @ModuleInfo(key: "mossformerM") var mossformerM: MossFormerBlock_GFSMN
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(numBlocks: Int, dModel: Int) {
        self._mossformerM.wrappedValue = MossFormerBlock_GFSMN(dim: dModel, depth: numBlocks)
        // The maker: nn.LayerNorm(d_model, eps=1e-6).
        self._norm.wrappedValue = LayerNorm(dimensions: dModel, eps: 1e-6)
    }

    func callAsFunction(_ src: MLXArray, trace: MossFormer2Trace?) -> MLXArray {
        norm(mossformerM(src, trace: trace))
    }
}

final class ComputationBlock: Module {
    @ModuleInfo(key: "intra_mdl") var intraMdl: MossFormerM
    /// The maker's select_norm("ln", dim, 3) is nn.GroupNorm(1, dim, eps=1e-8): statistics over channels and time of
    /// [B, N, S], per-channel affine. (The mlx-audio-swift port used a per-frame LayerNorm here.)
    @ModuleInfo(key: "intra_norm") var intraNorm: GlobalLayerNorm

    init(numBlocks: Int, outChannels: Int) {
        self._intraMdl.wrappedValue = MossFormerM(numBlocks: numBlocks, dModel: outChannels)
        self._intraNorm.wrappedValue = GlobalLayerNorm(dim: outChannels, shape: 2, eps: 1e-8)
    }

    func callAsFunction(_ x: MLXArray, trace: MossFormer2Trace?) -> MLXArray {
        let intra = intraMdl(x.transposed(0, 2, 1), trace: trace)
        trace?("intra_mdl", intra)
        let normed = intraNorm(intra.transposed(0, 2, 1))
        trace?("intra_norm", normed)
        return normed + x
    }
}

final class MossFormerMaskNet: Module {
    let numSpks = 2

    @ModuleInfo(key: "norm") var norm: GlobalLayerNorm
    @ModuleInfo(key: "conv1d_encoder") var conv1dEncoder: Conv1d
    @ModuleInfo(key: "pos_enc") var posEnc: ScaledSinuEmbedding
    @ModuleInfo(key: "mdl") var mdl: ComputationBlock
    @ModuleInfo(key: "conv1d_out") var conv1dOut: Conv1d
    @ModuleInfo(key: "conv1_decoder") var conv1Decoder: Conv1d
    @ModuleInfo(key: "prelu") var prelu: PReLU
    @ModuleInfo(key: "output") var output: Conv1d
    @ModuleInfo(key: "output_gate") var outputGate: Conv1d

    init(config: MossFormer2Config) {
        self._norm.wrappedValue = GlobalLayerNorm(dim: config.inChannels, shape: 3)
        self._conv1dEncoder.wrappedValue = Conv1d(inputChannels: config.inChannels, outputChannels: config.outChannels, kernelSize: 1, bias: false)
        self._posEnc.wrappedValue = ScaledSinuEmbedding(dim: config.outChannels)
        self._mdl.wrappedValue = ComputationBlock(numBlocks: config.numBlocks, outChannels: config.outChannels)
        self._conv1dOut.wrappedValue = Conv1d(inputChannels: config.outChannels, outputChannels: config.outChannels * numSpks, kernelSize: 1, bias: true)
        self._conv1Decoder.wrappedValue = Conv1d(inputChannels: config.outChannels, outputChannels: config.outChannelsFinal, kernelSize: 1, bias: false)
        self._prelu.wrappedValue = PReLU()
        self._output.wrappedValue = Conv1d(inputChannels: config.outChannels, outputChannels: config.outChannels, kernelSize: 1, bias: true)
        self._outputGate.wrappedValue = Conv1d(inputChannels: config.outChannels, outputChannels: config.outChannels, kernelSize: 1, bias: true)
    }

    /// `input` [B, 180, T] → mask [B, T, 961].
    func callAsFunction(_ input: MLXArray, trace: MossFormer2Trace?) -> MLXArray {
        var x = norm(input)
        x = conv1dEncoder(x.transposed(0, 2, 1)).transposed(0, 2, 1)
        trace?("encoder", x)
        // Positions stay float32 (exact up to 2^24 frames); the embedding joins the stream in its dtype.
        let emb = posEnc(x.transposed(0, 2, 1)).asType(x.dtype)
        x += emb.transposed(1, 0).expandedDimensions(axis: 0)
        x = mdl(x, trace: trace)
        trace?("computation_block", x)
        x = prelu(x)
        x = conv1dOut(x.transposed(0, 2, 1)).transposed(0, 2, 1)
        let batch = x.shape[0], seqLen = x.shape[2]
        x = x.reshaped(batch * numSpks, -1, seqLen).transposed(0, 2, 1)
        x = tanh(output(x)) * sigmoid(outputGate(x))
        x = conv1Decoder(x).transposed(0, 2, 1)
        x = relu(x.reshaped(batch, numSpks, x.shape[1], x.shape[2])).transposed(1, 0, 2, 3)
        return x[0].transposed(0, 2, 1)
    }
}

final class MossFormer2TestNet: Module {
    @ModuleInfo(key: "mossformer") var mossformer: MossFormerMaskNet
    init(config: MossFormer2Config) { self._mossformer.wrappedValue = MossFormerMaskNet(config: config) }
}

public struct MossFormer2Config: Sendable, Equatable {
    public var sampleRate = 48_000
    public var winLen = 1920
    public var winInc = 384
    public var fftLen = 1920
    public var numMels = 60
    public var preemphasis: Float = 0.97
    public var inChannels = 180
    public var outChannels = 512
    public var outChannelsFinal = 961
    public var numBlocks = 24
    /// Maker decode: one pass up to this many seconds, else `decodeWindow`-second windows at 3/4 stride.
    public var oneTimeDecodeLength = 20
    public var decodeWindow = 4
    public init() {}
}

/// The MossFormer2_SE_48K network: log-mel features (+ deltas) → magnitude mask.
public final class MossFormer2SE: Module {
    @ModuleInfo(key: "model") var model: MossFormer2TestNet
    public let config: MossFormer2Config

    public init(config: MossFormer2Config = .init()) {
        self.config = config
        self._model.wrappedValue = MossFormer2TestNet(config: config)
    }

    /// `features` [B, T, 180] (fbank, delta, delta-delta) → mask [B, T, 961], in the weights' dtype.
    public func mask(_ features: MLXArray, trace: MossFormer2Trace? = nil) -> MLXArray {
        model.mossformer(features.transposed(0, 2, 1), trace: trace)
    }
}
