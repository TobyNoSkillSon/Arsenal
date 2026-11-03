// SPDX-License-Identifier: MIT
// IDOL's denoiser (idol/models/dit.py LatentSeqMMFlowModel): a rectified-flow transformer over the object latent
// (8,192 tokens of 16) and the input camera (one token of 5), conditioned on the image tokens by joint attention over
// [latent, image, camera] with learned rotary positions (RePo).
import Foundation
import MLX
import MLXNN

public struct IdolDenoiserConfig: Sendable {
    public var tokens = 8192, inChannels = 16, camChannels = 5, channels = 1024
    public var condChannels = 1280, cond2Channels = 128, blocks = 24, refinerBlocks = 2, heads = 16, headChannels = 64
    public init() {}
}

/// The velocity for the latent and the camera.
public struct IdolVelocity {
    public let latent: MLXArray
    public let camera: MLXArray
}

public final class IdolDenoiser: Module {
    @ModuleInfo(key: "t_embedder") var tEmbedder: SinusoidalEmbedder
    @ModuleInfo(key: "adaLN_modulation") var adaLN: SiLULinear
    @ModuleInfo(key: "input_layer") var inputLayer: Linear
    @ModuleInfo(key: "cond_embedder") var condEmbedder: Linear
    @ModuleInfo(key: "cond_embedder2") var condEmbedder2: Linear
    @ModuleInfo(key: "noise_repo_layers") var noiseRepo: [RePo3DRotaryEmbedding]
    @ModuleInfo(key: "context_repo_layers") var contextRepo: [RePo3DRotaryEmbedding]
    @ModuleInfo(key: "repo_layers") var repo: [RePo3DRotaryEmbedding]
    @ModuleInfo(key: "noise_refiner") var noiseRefiner: [UnifiedTransformerBlock]
    @ModuleInfo(key: "context_refiner") var contextRefiner: [UnifiedTransformerBlock]
    @ModuleInfo(key: "cam_refiner") var camRefiner: CamMLP
    @ModuleInfo var blocks: [UnifiedTransformerBlock]
    @ParameterInfo(key: "shift_table") var shiftTable: MLXArray
    @ModuleInfo(key: "out_layer") var outLayer: Linear
    @ModuleInfo(key: "cam_out_layer") var camOutLayer: Linear
    /// The fixed latent token positions [1, tokens, 3] (a scrambled Sobol sequence stored with the weights). Kept in
    /// float32: 98 KB, and their embedding resolves 2^16 cycles per unit.
    @ParameterInfo(key: "pos_pe") var positions: MLXArray

    public let config: IdolDenoiserConfig
    /// Kept outside the module's parameters: a derived cache, not a weight.
    private let positionCache = PositionCache()

    public init(_ c: IdolDenoiserConfig = IdolDenoiserConfig()) {
        config = c
        _tEmbedder.wrappedValue = SinusoidalEmbedder(c.channels)
        _adaLN.wrappedValue = SiLULinear(c.channels, 6 * c.channels)
        _inputLayer.wrappedValue = Linear(c.inChannels, c.channels)
        _condEmbedder.wrappedValue = Linear(c.condChannels, c.channels)
        _condEmbedder2.wrappedValue = Linear(c.cond2Channels, c.channels)
        func rope(_ n: Int) -> [RePo3DRotaryEmbedding] { (0..<n).map { _ in RePo3DRotaryEmbedding(channels: c.channels, heads: c.heads, headDim: c.headChannels) } }
        _noiseRepo.wrappedValue = rope(c.refinerBlocks); _contextRepo.wrappedValue = rope(c.refinerBlocks); _repo.wrappedValue = rope(c.blocks)
        _noiseRefiner.wrappedValue = (0..<c.refinerBlocks).map { _ in UnifiedTransformerBlock(c.channels, heads: c.heads, modulation: true) }
        _contextRefiner.wrappedValue = (0..<c.refinerBlocks).map { _ in UnifiedTransformerBlock(c.channels, heads: c.heads, modulation: false) }
        _camRefiner.wrappedValue = CamMLP(c.camChannels, c.channels)
        _blocks.wrappedValue = (0..<c.blocks).map { _ in UnifiedTransformerBlock(c.channels, heads: c.heads, modulation: true) }
        _shiftTable.wrappedValue = MLXArray.zeros([1, 2, c.channels])
        _outLayer.wrappedValue = Linear(c.channels, c.inChannels)
        _camOutLayer.wrappedValue = Linear(c.channels, c.camChannels)
        _positions.wrappedValue = MLXArray.zeros([1, c.tokens, 3])
    }

    /// The latent tokens' position embedding [1, tokens, channels] in half, computed once on the host (see
    /// `PositionEmbedding.latentTable`). Uses the first `count` positions (all of them in the product).
    func positionEmbedding(count: Int) -> MLXArray {
        if let table = positionCache.table, table.dim(1) == count { return table }
        let flat = positions.asType(.float32)[0, 0..<count].reshaped([-1]).asArray(Float.self)
        let table = MLXArray(PositionEmbedding.latentTable(flat, channels: config.channels), [1, count, config.channels]).asType(.float16)
        eval(table)
        positionCache.table = table
        return table
    }

    /// latent [B, N, 16] and camera [B, 1, 5] float32; t [B] in [0, 1000]; feature1 [B, Lc, 1280] (DINOv3) and
    /// feature2 [B, Lc, 128] (FLUX.2 VAE) float32. Returns the velocity in float32.
    public func callAsFunction(latent: MLXArray, camera: MLXArray, t: MLXArray, feature1: MLXArray, feature2: MLXArray) -> IdolVelocity {
        let half = DType.float16
        var hx = inputLayer(latent).asType(half)
        var hc = (condEmbedder(feature1) + condEmbedder2(feature2)).asType(half)
        let tEmb = tEmbedder(t)
        let tMod = adaLN(tEmb).asType(half)
        hx += positionEmbedding(count: latent.dim(1))
        for (r, block) in zip(noiseRepo, noiseRefiner) { hx = block(hx, mod: tMod, rope: r(hx)) }
        for (r, block) in zip(contextRepo, contextRefiner) { hc = block(hc, mod: nil, rope: r(hc)) }
        let hcam = camRefiner(camera).asType(half)
        var h = MLX.concatenated([hx, hc, hcam], axis: 1)
        for (r, block) in zip(repo, blocks) { h = block(h, mod: tMod, rope: r(h)) }
        let n = latent.dim(1), total = h.dim(1)
        let table = shiftTable.asType(.float32) + tEmb.expandedDimensions(axis: 1) // [B, 2, C]
        let shift = table[0..., 0..<1], scale = table[0..., 1..<2]
        let outX = layerNorm32(h[0..., 0..<n].asType(.float32), eps: 1e-5) * (1 + scale) + shift
        let outCam = layerNorm32(h[0..., (total - camera.dim(1))...].asType(.float32), eps: 1e-5) * (1 + scale) + shift
        return IdolVelocity(latent: outLayer(outX).asType(.float32), camera: camOutLayer(outCam).asType(.float32))
    }
}

/// The camera refiner, MLP(5, 1024, 1024) with two layers: `mlp.0`, `mlp.2`.
final class CamMLP: Module {
    @ModuleInfo var mlp: GeluMLP
    init(_ input: Int, _ channels: Int) { _mlp.wrappedValue = GeluMLP(input, channels, channels) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { mlp(x) }
}

final class PositionCache {
    var table: MLXArray?
}
