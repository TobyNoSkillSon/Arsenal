// swiftlint:disable identifier_name shorthand_operator force_cast
// Copyright © 2026 Apple Inc.
import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM
import IntegrationKit
public struct Qwen35Configuration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public var modelType: String = ""
        public var hiddenSize: Int = 4096
        public var hiddenLayers: Int = 32
        public var intermediateSize: Int = 14_336
        public var attentionHeads: Int = 32
        public var kvHeads: Int = 8
        public var linearNumValueHeads: Int = 64
        public var linearNumKeyHeads: Int = 16
        public var linearKeyHeadDim: Int = 192
        public var linearValueHeadDim: Int = 128
        public var linearConvKernelDim: Int = 4
        public var rmsNormEps: Float = 1e-6
        public var vocabularySize: Int = 248_320
        public var ropeTheta: Float = 100_000.0
        public var partialRotaryFactor: Float = 0.25
        public var maxPositionEmbeddings: Int = 131_072
        public var tieWordEmbeddings: Bool = false
        public var attentionBias: Bool = false
        public var headDim: Int?
        public var ropeParameters: [String: StringOrNumber]?
        public var fullAttentionInterval: Int = 4

        // MoE fields
        public var numExperts: Int = 0
        public var numExpertsPerTok: Int = 0
        public var decoderSparseStep: Int = 1
        public var sharedExpertIntermediateSize: Int = 0
        public var moeIntermediateSize: Int = 0
        public var normTopkProb: Bool = true

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case linearNumValueHeads = "linear_num_value_heads"
            case linearNumKeyHeads = "linear_num_key_heads"
            case linearKeyHeadDim = "linear_key_head_dim"
            case linearValueHeadDim = "linear_value_head_dim"
            case linearConvKernelDim = "linear_conv_kernel_dim"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
            case maxPositionEmbeddings = "max_position_embeddings"
            case tieWordEmbeddings = "tie_word_embeddings"
            case attentionBias = "attention_bias"
            case headDim = "head_dim"
            case ropeParameters = "rope_parameters"
            case fullAttentionInterval = "full_attention_interval"
            case numExperts = "num_experts"
            case numExpertsPerTok = "num_experts_per_tok"
            case decoderSparseStep = "decoder_sparse_step"
            case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
            case moeIntermediateSize = "moe_intermediate_size"
            case normTopkProb = "norm_topk_prob"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)

            self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
            self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
            self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
            self.intermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14_336
            self.attentionHeads =
                try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
            self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
            self.linearNumValueHeads =
                try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
            self.linearNumKeyHeads =
                try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
            self.linearKeyHeadDim =
                try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 192
            self.linearValueHeadDim =
                try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
            self.linearConvKernelDim =
                try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
            self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            self.vocabularySize =
                try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 248_320
            self.maxPositionEmbeddings =
                try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
            self.tieWordEmbeddings =
                try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
            self.attentionBias =
                try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
            self.fullAttentionInterval =
                try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4

            self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
            self.numExpertsPerTok =
                try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
            self.decoderSparseStep =
                try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
            self.sharedExpertIntermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
            self.moeIntermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
            self.normTopkProb =
                try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true

            let defaultRopeParameters: [String: StringOrNumber] = [
                "type": .string("default"),
                "mrope_section": .ints([11, 11, 10]),
                "rope_theta": .float(100_000.0),
                "partial_rotary_factor": .float(0.25)
            ]

            var decodedRope = try container.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeParameters)

            if decodedRope == nil {
                let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                let partial = try container.decodeIfPresent(
                    Float.self, forKey: .partialRotaryFactor)
                if ropeTheta != nil || partial != nil {
                    decodedRope = defaultRopeParameters
                    if let ropeTheta {
                        decodedRope?["rope_theta"] = .float(ropeTheta)
                    }
                    if let partial {
                        decodedRope?["partial_rotary_factor"] = .float(partial)
                    }
                }
            }

            if var decodedRope {
                if decodedRope["type"] == nil, let ropeType = decodedRope["rope_type"] {
                    decodedRope["type"] = ropeType
                }
                self.ropeParameters = decodedRope
                self.ropeTheta = decodedRope["rope_theta"]?.asFloat() ?? 100_000.0
                self.partialRotaryFactor = decodedRope["partial_rotary_factor"]?.asFloat() ?? 0.25
            } else {
                self.ropeParameters = defaultRopeParameters
                self.ropeTheta = 100_000.0
                self.partialRotaryFactor = 0.25
            }

            if self.headDim == nil {
                self.headDim = self.hiddenSize / self.attentionHeads
            }
        }
    }

    public typealias VisionConfiguration = MLXVLM.Qwen3VLConfiguration.VisionConfiguration

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _ignoreIndex: Int?
    public var ignoreIndex: Int { _ignoreIndex ?? -100 }
    private let _imageTokenId: Int?
    public var imageTokenId: Int { _imageTokenId ?? 248_056 }
    private let _videoTokenId: Int?
    public var videoTokenId: Int { _videoTokenId ?? 248_057 }
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int { _imageTokenIndex ?? imageTokenId }
    private let _videoTokenIndex: Int?
    public var videoTokenIndex: Int { _videoTokenIndex ?? videoTokenId }
    private let _visionStartTokenId: Int?
    public var visionStartTokenId: Int { _visionStartTokenId ?? 248_045 }
    private let _visionEndTokenId: Int?
    public var visionEndTokenId: Int { _visionEndTokenId ?? 248_046 }
    private let _vocabSize: Int?
    public var vocabSize: Int { _vocabSize ?? textConfiguration.vocabularySize }
    private let _eosTokenId: IntOrIntArray?
    public var eosTokenId: [Int]? { _eosTokenId?.values }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _ignoreIndex = "ignore_index"
        case _imageTokenId = "image_token_id"
        case _videoTokenId = "video_token_id"
        case _imageTokenIndex = "image_token_index"
        case _videoTokenIndex = "video_token_index"
        case _visionStartTokenId = "vision_start_token_id"
        case _visionEndTokenId = "vision_end_token_id"
        case _vocabSize = "vocab_size"
        case _eosTokenId = "eos_token_id"
    }
}

// MARK: - Language

enum Qwen35Language {

    final class RotaryEmbedding {
        private let invFreq: MLXArray
        private let mropeSection: [Int]

        init(dim: Int, base: Float, mropeSection: [Int]) {
            let safeDim = max(1, dim)
            var freq = MLXArray(stride(from: 0, to: safeDim, by: 2)).asType(.float32)
            freq = freq / Float(safeDim)
            self.invFreq = 1.0 / pow(MLXArray(base), freq)
            self.mropeSection =
                mropeSection.count >= 3 ? mropeSection : [11, 11, 10]
        }

        private func applyInterleavedMRope(_ freqs: MLXArray) -> MLXArray {
            let freqsT = freqs[0, 0..., 0..., 0...]
            let dims = freqsT.dim(-1)
            var slices: [MLXArray] = []
            slices.reserveCapacity(dims)

            for idx in 0..<dims {
                var slice = freqsT[0..., 0..., idx]
                for (dim, offset) in [(1, 1), (2, 2)] {
                    let length = min(mropeSection[dim] * 3, dims)
                    if idx >= offset && idx < length && ((idx - offset) % 3 == 0) {
                        slice = freqs[dim, 0..., 0..., idx]
                        break
                    }
                }
                slices.append(slice)
            }

            return stacked(slices, axis: -1)
        }

        func callAsFunction(x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
            var positionIds = positionIds
            if positionIds.ndim == 2 {
                positionIds = broadcast(
                    positionIds[.newAxis, 0..., 0...],
                    to: [3, positionIds.dim(0), positionIds.dim(1)])
            }

            let pos = positionIds.asType(.float32)
            var inv = invFreq.asType(.float32)
            inv = inv[.newAxis, .newAxis, .newAxis, 0...]
            var freqs = pos[0..., 0..., 0..., .newAxis] * inv
            freqs = applyInterleavedMRope(freqs)

            let emb = concatenated([freqs, freqs], axis: -1)
            return (cos(emb).asType(x.dtype), sin(emb).asType(x.dtype))
        }
    }

    static func applyMultimodalRotaryPosEmb(
        q: MLXArray,
        k: MLXArray,
        cos: MLXArray,
        sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        let cos = expandedDimensions(cos, axis: 1)
        let sin = expandedDimensions(sin, axis: 1)

        let rotaryDim = cos.dim(-1)
        let qDim = q.dim(-1)
        let kDim = k.dim(-1)

        let qRot = q[.ellipsis, ..<rotaryDim]
        let kRot = k[.ellipsis, ..<rotaryDim]

        let qEmbedded = (qRot * cos) + (OvisVision.rotateHalf(qRot) * sin)
        let kEmbedded = (kRot * cos) + (OvisVision.rotateHalf(kRot) * sin)

        let qOut: MLXArray
        if rotaryDim < qDim {
            qOut = concatenated([qEmbedded, q[.ellipsis, rotaryDim...]], axis: -1)
        } else {
            qOut = qEmbedded
        }

        let kOut: MLXArray
        if rotaryDim < kDim {
            kOut = concatenated([kEmbedded, k[.ellipsis, rotaryDim...]], axis: -1)
        } else {
            kOut = kEmbedded
        }

        return (qOut, kOut)
    }

    final class RMSNormGated: Module {
        @ParameterInfo(key: "weight") var weight: MLXArray
        let eps: Float

        init(dimensions: Int, eps: Float = 1e-6) {
            self.eps = eps
            _weight.wrappedValue = MLXArray.ones([dimensions])
            super.init()
        }

        func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray? = nil) -> MLXArray {
            var x = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: eps)
            if let gate {
                x = x * silu(gate)
            }
            return x
        }
    }

    final class Attention: Module {
        let numKeyValueHeads: Int
        let numAttentionHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        let rotaryEmbedding: RotaryEmbedding

        init(_ args: Qwen35Configuration.TextConfiguration) {
            self.numKeyValueHeads = args.kvHeads
            self.numAttentionHeads = args.attentionHeads
            self.headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
            self.scale = pow(Float(headDim), -0.5)

            _qProj.wrappedValue = Linear(
                args.hiddenSize, numAttentionHeads * headDim * 2, bias: args.attentionBias)
            _kProj.wrappedValue = Linear(
                args.hiddenSize, numKeyValueHeads * headDim, bias: args.attentionBias)
            _vProj.wrappedValue = Linear(
                args.hiddenSize, numKeyValueHeads * headDim, bias: args.attentionBias)
            _oProj.wrappedValue = Linear(
                numAttentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

            let mrope = args.ropeParameters?["mrope_section"]?.asInts() ?? [11, 11, 10]
            let rotaryDim = Int(Float(headDim) * args.partialRotaryFactor)
            self.rotaryEmbedding = RotaryEmbedding(
                dim: rotaryDim, base: args.ropeTheta, mropeSection: mrope)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let B = x.dim(0)
            let L = x.dim(1)

            let qProjOutput = qProj(x)
            let qSplit = qProjOutput.reshaped(B, L, numAttentionHeads, -1).split(parts: 2, axis: -1)
            var queries = qSplit[0]
            let gate = qSplit[1].reshaped(B, L, -1)

            var keys = kProj(x)
            var values = vProj(x)

            queries = qNorm(queries).transposed(0, 2, 1, 3)
            keys = kNorm(keys.reshaped(B, L, numKeyValueHeads, -1)).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)

            var kvSeqLen = keys.dim(-2)
            var positionIds = positionIds

            if positionIds == nil {
                let offset = cache?.offset ?? 0
                kvSeqLen += offset + 1
                var base = MLXArray(stride(from: offset, to: offset + L, by: 1)).asType(.int32)
                base = tiled(base[.newAxis, 0...], repetitions: [B, 1])
                positionIds = base[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds!, repetitions: [3, 1, 1])
            } else if let cache {
                kvSeqLen += cache.offset + 1
            }

            let (cosValues, sinValues) = rotaryEmbedding(x: values, positionIds: positionIds!)
            (queries, keys) = applyMultimodalRotaryPosEmb(
                q: queries, k: keys, cos: cosValues, sin: sinValues)

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
            if case .array(let array) = mask {
                attentionMask = .array(array[.ellipsis, 0..<kvSeqLen])
            } else {
                attentionMask = mask
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: attentionMask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return oProj(output * sigmoid(gate))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    final class GatedDeltaNet: Module {
        let hiddenSize: Int
        let numVHeads: Int
        let numKHeads: Int
        let headKDim: Int
        let headVDim: Int
        let keyDim: Int
        let valueDim: Int
        let convKernelSize: Int
        let convDim: Int

        @ModuleInfo(key: "conv1d") var conv1d: Conv1d
        @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
        @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
        @ModuleInfo(key: "in_proj_b") var inProjB: Linear
        @ModuleInfo(key: "in_proj_a") var inProjA: Linear

        @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
        @ParameterInfo(key: "A_log") var aLog: MLXArray

        @ModuleInfo(key: "norm") var norm: RMSNormGated
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ args: Qwen35Configuration.TextConfiguration) {
            self.hiddenSize = args.hiddenSize
            self.numVHeads = args.linearNumValueHeads
            self.numKHeads = args.linearNumKeyHeads
            self.headKDim = args.linearKeyHeadDim
            self.headVDim = args.linearValueHeadDim
            self.keyDim = headKDim * numKHeads
            self.valueDim = headVDim * numVHeads
            self.convKernelSize = args.linearConvKernelDim
            self.convDim = keyDim * 2 + valueDim

            precondition(
                numVHeads % numKHeads == 0,
                "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))"
            )

            _conv1d.wrappedValue = Conv1d(
                inputChannels: convDim,
                outputChannels: convDim,
                kernelSize: convKernelSize,
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: convDim,
                bias: false
            )

            _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
            _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
            _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
            _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

            _dtBias.wrappedValue = MLXArray.ones([numVHeads])
            let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
            _aLog.wrappedValue = log(a)

            _norm.wrappedValue = RMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
            _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            mask: MLXArray? = nil,
            cache: MambaCache? = nil
        ) -> MLXArray {
            let B = inputs.dim(0)
            let S = inputs.dim(1)

            var mixedQKV = inProjQKV(inputs)
            let z = inProjZ(inputs).reshaped(B, S, numVHeads, headVDim)
            let b = inProjB(inputs)
            let a = inProjA(inputs)

            let convState: MLXArray
            if let cacheState = cache?[0] {
                convState = cacheState
            } else {
                convState = MLXArray.zeros(
                    [B, max(0, convKernelSize - 1), convDim], dtype: inputs.dtype)
            }

            if let mask {
                mixedQKV = MLX.where(mask[.ellipsis, .newAxis], mixedQKV, 0)
            }

            let convInput = concatenated([convState, mixedQKV], axis: 1)
            if let cache, convKernelSize > 1 {
                cache[0] = contiguous(convInput[0..., (-(convKernelSize - 1))..., 0...])
            }

            let convOut = silu(conv1d(convInput))
            let split = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
            let q = split[0].reshaped(B, S, numKHeads, headKDim)
            let k = split[1].reshaped(B, S, numKHeads, headKDim)
            let v = split[2].reshaped(B, S, numVHeads, headVDim)

            var state = cache?[1]
            let dtype = q.dtype
            let invScale = pow(Float(headKDim), -0.5)
            let qNormed =
                MLXArray(pow(invScale, 2)).asType(dtype)
                * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
            let kNormed =
                MLXArray(invScale).asType(dtype)
                * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

            var out: MLXArray
            (out, state) = gatedDeltaUpdate(
                q: qNormed,
                k: kNormed,
                v: v,
                a: a,
                b: b,
                aLog: aLog,
                dtBias: dtBias,
                state: state,
                mask: mask
            )

            if let cache {
                cache[1] = state
                cache.advance(S)
            }

            out = norm(out, gate: z)
            return outProj(out.reshaped(B, S, -1))
        }
    }

    final class SparseMoeBlock: Module, UnaryLayer {
        let normTopkProb: Bool
        let numExperts: Int
        let topK: Int

        @ModuleInfo(key: "gate") var gate: Linear
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

        @ModuleInfo(key: "shared_expert") var sharedExpert: MLP
        @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

        init(_ args: Qwen35Configuration.TextConfiguration) {
            self.normTopkProb = args.normTopkProb
            self.numExperts = args.numExperts
            self.topK = args.numExpertsPerTok

            _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: args.hiddenSize,
                hiddenDims: args.moeIntermediateSize,
                numExperts: args.numExperts
            )

            _sharedExpert.wrappedValue = MLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.sharedExpertIntermediateSize
            )
            _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var gates = gate(x)
            gates = MLX.softmax(gates, axis: -1, precise: true)

            let kth = gates.dim(-1) - topK
            let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, kth...]
            var scores = MLX.takeAlong(gates, inds, axis: -1)
            if normTopkProb {
                scores = scores / scores.sum(axis: -1, keepDims: true)
            }

            let y = switchMLP(x, inds)
            let combined = weightedExpertSum(y, scores)

            var sharedY = sharedExpert(x)
            sharedY = sigmoid(sharedExpertGate(x)) * sharedY

            return combined + sharedY
        }
    }

    final class DecoderLayer: Module {
        let isLinear: Bool

        @ModuleInfo(key: "self_attn") var selfAttn: Attention?
        @ModuleInfo(key: "linear_attn") var linearAttn: GatedDeltaNet?

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        @ModuleInfo(key: "mlp") var mlp: Module

        init(_ args: Qwen35Configuration.TextConfiguration, layerIdx: Int) {
            self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

            if isLinear {
                _linearAttn.wrappedValue = GatedDeltaNet(args)
            } else {
                _selfAttn.wrappedValue = Attention(args)
            }

            if args.numExperts > 0 {
                _mlp.wrappedValue = SparseMoeBlock(args)
            } else {
                _mlp.wrappedValue = MLP(
                    dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            }

            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)

            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
            ssmMask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let r: MLXArray
            if isLinear {
                r = linearAttn!(inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache)
            } else {
                r = selfAttn!(
                    inputLayerNorm(x), mask: attentionMask, cache: cache, positionIds: positionIds)
            }

            let h = x + r
            return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
        }
    }

    final class Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") fileprivate var layers: [DecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm

        let ssmIdx: Int
        let faIdx: Int

        init(_ args: Qwen35Configuration.TextConfiguration) {
            precondition(args.vocabularySize > 0)
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
            _layers.wrappedValue = (0..<args.hiddenLayers).map {
                DecoderLayer(args, layerIdx: $0)
            }
            _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

            self.ssmIdx = 0
            self.faIdx = args.fullAttentionInterval - 1
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            positionIds: MLXArray? = nil,
            cancellation: NativeCancellation,
            didEvaluateLayer: ((Int) -> Void)? = nil
        ) throws -> MLXArray {
            var hiddenStates: MLXArray
            if let inputsEmbeds {
                hiddenStates = inputsEmbeds
            } else {
                hiddenStates = embedTokens(inputs)
            }

            var cacheArray = cache
            if cacheArray == nil {
                cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
            }

            // Arsenal: the symbolic causal mode, not a materialized L×L mask. An explicit mask forces MLX's unfused
            // attention for these 256-wide heads, whose scores grow with the square of the input (a 32,768-token page
            // set held ~17 GB per score tensor); the symbolic mode lets MLX use its fused causal kernel where the GPU
            // has one. A rotating cache still gets the explicit windowed mask.
            let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[faIdx], returnArray: false)
            let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

            for (index, layer) in layers.enumerated() {
                try cancellation.check()
                let layerSSMMask = layer.isLinear ? ssmMask : nil
                hiddenStates = layer(
                    hiddenStates,
                    attentionMask: faMask,
                    ssmMask: layerSSMMask,
                    cache: cacheArray?[index],
                    positionIds: positionIds
                )
                eval(hiddenStates)
                didEvaluateLayer?(index)
                try cancellation.check()
            }

            return norm(hiddenStates)
        }
    }

}

// swiftlint:enable identifier_name shorthand_operator force_cast
