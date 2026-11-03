import Foundation
import MLX
import MLXNN

private final class Nemotron3RoPE {
    let cosine: MLXArray
    let sine: MLXArray

    init(headDimension: Int = 64, maximumLength: Int = 684) {
        let half = headDimension / 2
        let exponents = MLXArray(
            (0..<half).map { Float(2 * $0) / Float(headDimension) })
        let inverse =
            MLXArray(Float(1))
            / MLX.pow(MLXArray(Float(10_000)), exponents)
        let positions = MLXArray((0..<maximumLength).map(Float.init))
        let frequencies =
            positions.reshaped(maximumLength, 1)
            * inverse.reshaped(1, half)
        let halfCosine = cos(frequencies)
        let halfSine = sin(frequencies)
        cosine = concatenated([halfCosine, halfCosine], axis: -1)
        sine = concatenated([halfSine, halfSine], axis: -1)
    }

    func apply(_ value: MLXArray) -> MLXArray {
        // value: [B, H, T, D], NeMo/GPT-NeoX split-half rotation.
        let time = value.dim(2)
        let dimensions = value.dim(3)
        let half = dimensions / 2
        let cosine = cosine[0..<time, 0...]
            .reshaped(1, 1, time, dimensions).asType(value.dtype)
        let sine = sine[0..<time, 0...]
            .reshaped(1, 1, time, dimensions).asType(value.dtype)
        let first = value[0..., 0..., 0..., 0..<half]
        let second = value[0..., 0..., 0..., half..<dimensions]
        return value * cosine + concatenated([-second, first], axis: -1) * sine
    }
}

private final class Nemotron3MLXAttention: Module {
    private let heads = 8
    private let headDimension = 64
    private let scale: Float = 1 / 8
    private let rope: Nemotron3RoPE

    @ModuleInfo(key: "w_qkv") var qkvProjection: Linear
    @ModuleInfo(key: "out_proj") var outputProjection: Linear

    init(rope: Nemotron3RoPE) {
        self.rope = rope
        _qkvProjection.wrappedValue = Linear(512, 1_536, bias: false)
        _outputProjection.wrappedValue = Linear(512, 512, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let time = input.dim(1)
        let qkv = qkvProjection(input)
            .reshaped(batch, time, 3, heads, headDimension)
            .transposed(2, 0, 3, 1, 4)
        var query = qkv[0]
        var key = qkv[1]
        let value = qkv[2]
        query = rope.apply(query)
        key = rope.apply(key)
        let attended = nemotronAttend(
            qHeads: query,
            kHeads: key,
            vHeads: value,
            scale: scale,
            mask: mask)
        return outputProjection(attended)
    }
}

private final class Nemotron3MLXFeedForward: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    override init() {
        _linear1.wrappedValue = Linear(512, 2_048, bias: true)
        _linear2.wrappedValue = Linear(2_048, 512, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        linear2(gelu(linear1(input)))
    }
}

private final class Nemotron3MLXLayer: Module {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attention: Nemotron3MLXAttention
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo(key: "ffn") var feedForward: Nemotron3MLXFeedForward

    init(rope: Nemotron3RoPE) {
        _norm1.wrappedValue = LayerNorm(dimensions: 512, eps: 1.0e-5)
        _attention.wrappedValue = Nemotron3MLXAttention(rope: rope)
        _norm2.wrappedValue = LayerNorm(dimensions: 512, eps: 1.0e-5)
        _feedForward.wrappedValue = Nemotron3MLXFeedForward()
        super.init()
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray) -> MLXArray {
        let attended = input + attention(norm1(input), mask: mask)
        return attended + feedForward(norm2(attended))
    }
}

private final class Nemotron3MLXPreEncoder: Module {
    @ModuleInfo var proj: Linear

    override init() {
        _proj.wrappedValue = Linear(1_024, 512, bias: false)
        super.init()
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        let time = features.dim(1)
        precondition(time % 8 == 0)
        return proj(features.reshaped(-1, time / 8, 1_024))
    }
}

private final class Nemotron3MLXEncoder: Module {
    @ModuleInfo(key: "pre_encode") var preEncode: Nemotron3MLXPreEncoder
    @ModuleInfo(key: "embed_norm") var embedNorm: LayerNorm
    @ModuleInfo var layers: [Nemotron3MLXLayer]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

    init(layerCount: Int = 31) {
        let rope = Nemotron3RoPE()
        _preEncode.wrappedValue = Nemotron3MLXPreEncoder()
        _embedNorm.wrappedValue = LayerNorm(dimensions: 512, eps: 1.0e-5)
        _layers.wrappedValue = (0..<layerCount).map { _ in
            Nemotron3MLXLayer(rope: rope)
        }
        _finalNorm.wrappedValue = LayerNorm(dimensions: 512, eps: 1.0e-5)
        super.init()
    }

    func encode(_ embeddings: MLXArray, mask: MLXArray) -> MLXArray {
        var hidden = embedNorm(embeddings)
        for layer in layers {
            hidden = layer(hidden, mask: mask)
        }
        return finalNorm(hidden)
    }
}

private final class Nemotron3MLXHead: Module {
    @ParameterInfo(key: "learnable_sil_emb") var silence = MLXArray.zeros([512])
    @ModuleInfo(key: "hidden_to_spks") var pairSpeaker = Linear(384, 8)
    @ModuleInfo(key: "activity_head") var activity = NemotronActivity()
    @ModuleInfo(key: "encoder_proj") var encoderProjection: Linear
    @ModuleInfo(key: "subpixel_upsample") var subpixelUpsample: Conv1d
    @ModuleInfo(key: "first_hidden_to_hidden") var hiddenProjection: Linear
    @ModuleInfo(key: "single_hidden_to_spks") var speakerProjection: Linear

    override init() {
        _encoderProjection.wrappedValue = Linear(512, 192, bias: true)
        _subpixelUpsample.wrappedValue = Conv1d(
            inputChannels: 192,
            outputChannels: 1_536,
            kernelSize: 3,
            padding: 1,
            bias: true)
        _hiddenProjection.wrappedValue = Linear(192, 192, bias: true)
        _speakerProjection.wrappedValue = Linear(192, 8, bias: true)
        super.init()
    }

    func callAsFunction(
        _ encoded: MLXArray, validLength: Int
    ) -> (high: MLXArray, low: MLXArray) {
        let batch = encoded.dim(0)
        let frames = encoded.dim(1)
        var hidden = subpixelUpsample(encoderProjection(encoded))
        hidden = hidden.reshaped(batch, frames, 8, 192)
            .reshaped(batch, frames * 8, 192)
        hidden = maximum(hidden, MLXArray(Float(0)))
        hidden = maximum(hiddenProjection(hidden), MLXArray(Float(0)))
        var high = sigmoid(speakerProjection(hidden))
        let valid =
            MLXArray(0..<Int32(frames * 8))
            .< MLXArray(Int32(validLength * 8))
        // swiftlint:disable:next shorthand_operator
        high = high * valid.reshaped(1, frames * 8, 1).asType(high.dtype)
        let low = high.reshaped(batch, frames, 8, 8).mean(axis: 2)
        return (high, low)
    }
}

final class Nemotron3MLXNetwork: Module {
    @ModuleInfo private var encoder: Nemotron3MLXEncoder
    @ModuleInfo(key: "sortformer_modules") private var head: Nemotron3MLXHead

    init(layerCount: Int = 31) {
        _encoder.wrappedValue = Nemotron3MLXEncoder(layerCount: layerCount)
        _head.wrappedValue = Nemotron3MLXHead()
        super.init()
    }

    func preencode(_ features: MLXArray) -> MLXArray {
        encoder.preEncode(features)
    }

    func inferHead(
        _ embeddings: MLXArray, validLength: Int
    ) -> (high: MLXArray, low: MLXArray) {
        let frames = embeddings.dim(1)
        let valid =
            MLXArray(0..<Int32(frames))
            .< MLXArray(Int32(validLength))
        let zero = MLXArray(Float(0))
        let negative = MLXArray(Float(-10_000))
        let mask = MLX.where(valid, zero, negative)
            .reshaped(1, 1, 1, frames)
        return head(encoder.encode(embeddings, mask: mask), validLength: validLength)
    }
}

private final class NemotronActivity: Module {
    @ModuleInfo var layers: [Module] = [LayerNorm(dimensions: 192), Linear(192, 3)]
}
private func nemotronAttend(qHeads: MLXArray, kHeads: MLXArray, vHeads: MLXArray, scale: Float, mask: MLXArray) -> MLXArray {
    let output = MLXFast.scaledDotProductAttention(queries: qHeads, keys: kHeads, values: vHeads, scale: scale, mask: mask.asType(qHeads.dtype))
    return output.transposed(0, 2, 1, 3).reshaped(qHeads.dim(0), qHeads.dim(2), -1)
}
