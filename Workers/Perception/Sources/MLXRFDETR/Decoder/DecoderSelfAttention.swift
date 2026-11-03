// Vendored model implementation; source and adaptations: Workers/Perception/licenses/MLXRFDETR/PORTED-FROM.md
// Multi-head self-attention for decoder queries.
//
// Position embedding is added to q and k only (not v).
//
// PORT FROM: ../../python/rf-detr/src/rfdetr/models/transformer.py (decoder self-attn)

import Foundation
import MLX
import MLXFast
import MLXNN

/// Multi-head self-attention with position added to q,k only.
public final class DecoderSelfAttention: Module {
    @ModuleInfo(key: "in_proj") public var inProj: Linear
    @ModuleInfo(key: "out_proj") public var outProj: Linear

    public let numHeads: Int
    public let headDim: Int
    public let scale: Float

    public init(dModel: Int, nHeads: Int) {
        self.numHeads = nHeads
        self.headDim = dModel / nHeads
        self.scale = 1.0 / sqrt(Float(headDim))
        self._inProj = ModuleInfo(wrappedValue: Linear(dModel, 3 * dModel), key: "in_proj")
        self._outProj = ModuleInfo(wrappedValue: Linear(dModel, dModel), key: "out_proj")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, queryPos: MLXArray) -> MLXArray {
        // swiftlint:disable:next identifier_name
        let B = x.dim(0); let N = x.dim(1); let D = x.dim(2)
        // swiftlint:disable:next identifier_name
        let H = numHeads; let d = headDim

        let qkInput = x + queryPos
        let fused = inProj(qkInput)
        let q = fused[.ellipsis, ..<D].reshaped([B, N, H, d]).transposed(0, 2, 1, 3)
        let k = fused[.ellipsis, D..<(2 * D)].reshaped([B, N, H, d]).transposed(0, 2, 1, 3)
        let v = inProj(x)[.ellipsis, (2 * D)...].reshaped([B, N, H, d]).transposed(0, 2, 1, 3)

        var y = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil
        )
        y = y.transposed(0, 2, 1, 3).reshaped([B, N, D])
        return outProj(y)
    }
}
