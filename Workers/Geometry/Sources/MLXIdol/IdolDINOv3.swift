// SPDX-License-Identifier: MIT
// DINOv3 ViT-H+/16 (facebook/dinov3-vith16plus-pretrain-lvd1689m, transformers DINOv3ViTModel) as IDOL uses it:
// last_hidden_state (after the model's final norm), then an affine-less LayerNorm. Patch conv 16×16 → [cls, 4 registers,
// patches] → 32 layers of (LayerNorm → attention with axial 2D RoPE on the patch tokens (rotate-half), no key bias →
// LayerScale) and (LayerNorm → SwiGLU MLP → LayerScale). Runs in BF16, the dtype IDOL itself runs this encoder in.
import Foundation
import MLX
import MLXFast
import MLXNN

public struct IdolDINOv3Config: Sendable {
    public var hidden = 1280, layers = 32, heads = 20, intermediate = 5120, registers = 4, patch = 16
    public var ropeBase: Float = 100, eps: Float = 1e-5
    public init() {}
}

final class DinoEmbeddings: Module {
    @ParameterInfo(key: "cls_token") var cls: MLXArray
    @ParameterInfo(key: "register_tokens") var registers: MLXArray
    @ModuleInfo(key: "patch_embeddings") var patch: Conv2d
    init(_ c: IdolDINOv3Config) {
        _cls.wrappedValue = MLXArray.zeros([1, 1, c.hidden]); _registers.wrappedValue = MLXArray.zeros([1, c.registers, c.hidden])
        _patch.wrappedValue = Conv2d(inputChannels: 3, outputChannels: c.hidden, kernelSize: IntOrPair(c.patch), stride: IntOrPair(c.patch))
    }
}

final class DinoAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var o: Linear
    init(_ c: Int) {
        _q.wrappedValue = Linear(c, c); _k.wrappedValue = Linear(c, c, bias: false)
        _v.wrappedValue = Linear(c, c); _o.wrappedValue = Linear(c, c)
    }
}

final class DinoGatedMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ c: Int, _ inner: Int) { _gate.wrappedValue = Linear(c, inner); _up.wrappedValue = Linear(c, inner); _down.wrappedValue = Linear(inner, c) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class DinoLayerScale: Module {
    let lambda1: MLXArray
    init(_ c: Int) { lambda1 = MLXArray.ones([c]) }
}

final class DinoLayer: Module {
    @ModuleInfo var norm1: IdolLayerNorm
    @ModuleInfo var attention: DinoAttention
    @ModuleInfo(key: "layer_scale1") var scale1: DinoLayerScale
    @ModuleInfo var norm2: IdolLayerNorm
    @ModuleInfo var mlp: DinoGatedMLP
    @ModuleInfo(key: "layer_scale2") var scale2: DinoLayerScale
    init(_ c: IdolDINOv3Config) {
        _norm1.wrappedValue = IdolLayerNorm(c.hidden, eps: c.eps); _attention.wrappedValue = DinoAttention(c.hidden); _scale1.wrappedValue = DinoLayerScale(c.hidden)
        _norm2.wrappedValue = IdolLayerNorm(c.hidden, eps: c.eps); _mlp.wrappedValue = DinoGatedMLP(c.hidden, c.intermediate); _scale2.wrappedValue = DinoLayerScale(c.hidden)
    }
}

public final class IdolDINOv3: Module {
    @ModuleInfo var embeddings: DinoEmbeddings
    @ModuleInfo var layer: [DinoLayer]
    @ModuleInfo var norm: IdolLayerNorm
    public let config: IdolDINOv3Config

    public init(_ c: IdolDINOv3Config = IdolDINOv3Config()) {
        config = c
        _embeddings.wrappedValue = DinoEmbeddings(c)
        _layer.wrappedValue = (0..<c.layers).map { _ in DinoLayer(c) }
        _norm.wrappedValue = IdolLayerNorm(c.hidden, eps: c.eps)
    }

    /// Checkpoint name → module name: transformers 5 writes `model.layer.N`, the Hub checkpoints `layer.N`; the
    /// pretraining-only mask token is not used.
    public static func moduleName(_ key: String) -> String? {
        if key == "embeddings.mask_token" { return nil }
        return key.hasPrefix("model.") ? String(key.dropFirst(6)) : key
    }

    /// Axial 2D RoPE angles for an nH × nW patch grid (DINOv3ViTRopePositionEmbedding): (cos, sin) [patches, headDim].
    func rope(_ nH: Int, _ nW: Int) -> (MLXArray, MLXArray) {
        let headDim = config.hidden / config.heads, nf = headDim / 4
        let invFreq = (0..<nf).map { 1 / powf(config.ropeBase, Float($0) * 4 / Float(headDim)) }
        var angles = [Float](repeating: 0, count: nH * nW * headDim)
        var index = 0
        for h in 0..<nH {
            let cy = 2 * ((Float(h) + 0.5) / Float(nH)) - 1
            for w in 0..<nW {
                let cx = 2 * ((Float(w) + 0.5) / Float(nW)) - 1
                for t in 0..<2 {
                    for f in 0..<nf {
                        angles[index + t * 2 * nf + f] = 2 * Float.pi * cy * invFreq[f]
                        angles[index + t * 2 * nf + nf + f] = 2 * Float.pi * cx * invFreq[f]
                    }
                }
                index += headDim
            }
        }
        let a = MLXArray(angles, [nH * nW, headDim])
        return (MLX.cos(a), MLX.sin(a))
    }

    private func applyRope(_ x: MLXArray, _ cos: MLXArray, _ sin: MLXArray) -> MLXArray {
        let prefix = 1 + config.registers, l = x.dim(2), d = x.dim(3)
        let patches = x[0..., 0..., prefix..<l]
        let rotated = MLX.concatenated([-patches[.ellipsis, (d / 2)...], patches[.ellipsis, ..<(d / 2)]], axis: -1)
        let c = cos.asType(x.dtype).reshaped([1, 1, -1, d]), s = sin.asType(x.dtype).reshaped([1, 1, -1, d])
        return MLX.concatenated([x[0..., 0..., ..<prefix], patches * c + rotated * s], axis: 2)
    }

    /// pixels [1, 3, H, W] ImageNet-normalised → IDOL's feature1, [1, 1 + 4 + (H/16)(W/16), 1280] float32.
    public func callAsFunction(_ pixels: MLXArray, checkpoint: () throws -> Void = {}) throws -> MLXArray {
        let dtype = norm.weight.dtype
        let nH = pixels.dim(2) / config.patch, nW = pixels.dim(3) / config.patch
        let patches = embeddings.patch(pixels.transposed(0, 2, 3, 1).asType(dtype)).reshaped([1, nH * nW, config.hidden])
        var h = MLX.concatenated([embeddings.cls.asType(dtype), embeddings.registers.asType(dtype), patches], axis: 1)
        let (cos, sin) = rope(nH, nW)
        let heads = config.heads, headDim = config.hidden / heads, l = h.dim(1)
        for block in layer {
            try checkpoint()
            let x = block.norm1(h)
            func split(_ t: MLXArray) -> MLXArray { t.reshaped([1, l, heads, headDim]).transposed(0, 2, 1, 3) }
            let q = applyRope(split(block.attention.q(x)), cos, sin), k = applyRope(split(block.attention.k(x)), cos, sin)
            let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: split(block.attention.v(x)), scale: 1 / Float(headDim).squareRoot(), mask: nil)
            h += block.attention.o(o.transposed(0, 2, 1, 3).reshaped([1, l, config.hidden])) * block.scale1.lambda1
            h += block.mlp(block.norm2(h)) * block.scale2.lambda1
            eval(h)
        }
        return layerNorm32(norm(h).asType(.float32), eps: 1e-5)
    }
}
