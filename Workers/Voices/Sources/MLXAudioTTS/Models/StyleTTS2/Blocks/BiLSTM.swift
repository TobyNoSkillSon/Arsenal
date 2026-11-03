import Foundation
@preconcurrency import MLX
import MLXNN

public class BiLSTM: Module {
    public let inputSize: Int
    public let hiddenSize: Int

    public var Wx_forward: MLXArray
    public var Wh_forward: MLXArray
    public var bias_ih_forward: MLXArray
    public var bias_hh_forward: MLXArray

    public var Wx_backward: MLXArray
    public var Wh_backward: MLXArray
    public var bias_ih_backward: MLXArray
    public var bias_hh_backward: MLXArray

    public init(inputSize: Int, hiddenSize: Int) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        let scale = 1.0 / Float(hiddenSize).squareRoot()

        Wx_forward = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize, inputSize])
        Wh_forward = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize, hiddenSize])
        bias_ih_forward = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize])
        bias_hh_forward = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize])

        Wx_backward = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize, inputSize])
        Wh_backward = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize, hiddenSize])
        bias_ih_backward = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize])
        bias_hh_backward = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize])
    }

    private func forwardDirection(_ x: MLXArray) -> MLXArray {
        let xProj = MLX.addmm(bias_ih_forward + bias_hh_forward, x, Wx_forward.transposed())

        let seqLen = x.shape[x.ndim - 2]
        let batchSize = x.ndim == 3 ? x.shape[0] : 1
        var hidden = MLXArray.zeros([batchSize, hiddenSize], dtype: x.dtype)
        var cell = MLXArray.zeros([batchSize, hiddenSize], dtype: x.dtype)
        var allHidden = [MLXArray]()
        allHidden.reserveCapacity(seqLen)

        for idx in 0..<seqLen {
            var ifgo = xProj[.ellipsis, idx, 0...]
            ifgo = ifgo + MLX.matmul(hidden, Wh_forward.transposed())

            let gates = MLX.split(ifgo, parts: 4, axis: -1)
            let i = MLX.sigmoid(gates[0])
            let f = MLX.sigmoid(gates[1])
            let g = MLX.tanh(gates[2])
            let o = MLX.sigmoid(gates[3])

            cell = f * cell + i * g
            hidden = o * MLX.tanh(cell)
            allHidden.append(hidden)
        }

        return MLX.stacked(allHidden, axis: -2)
    }

    private func backwardDirection(_ x: MLXArray) -> MLXArray {
        let xProj = MLX.addmm(bias_ih_backward + bias_hh_backward, x, Wx_backward.transposed())

        let seqLen = x.shape[x.ndim - 2]
        let batchSize = x.ndim == 3 ? x.shape[0] : 1
        var hidden = MLXArray.zeros([batchSize, hiddenSize], dtype: x.dtype)
        var cell = MLXArray.zeros([batchSize, hiddenSize], dtype: x.dtype)
        var allHidden = [MLXArray]()
        allHidden.reserveCapacity(seqLen)

        for idx in stride(from: seqLen - 1, through: 0, by: -1) {
            var ifgo = xProj[.ellipsis, idx, 0...]
            ifgo = ifgo + MLX.matmul(hidden, Wh_backward.transposed())

            let gates = MLX.split(ifgo, parts: 4, axis: -1)
            let i = MLX.sigmoid(gates[0])
            let f = MLX.sigmoid(gates[1])
            let g = MLX.tanh(gates[2])
            let o = MLX.sigmoid(gates[3])

            cell = f * cell + i * g
            hidden = o * MLX.tanh(cell)
            allHidden.append(hidden)
        }

        return MLX.stacked(allHidden.reversed(), axis: -2)
    }

    /// Vireo fast path (set by `KokoroModel.setFastOptions`): both directions in one pass.
    public var fused = false

    /// LSTM cell update of both directions at once: gates i, f, g, o [2, B, H] and cell [2, B, H] → (hidden, cell).
    /// Weight-free and shapeless, so the compiled closure captures nothing (no weights kept alive after unload). The
    /// gate split stays outside: a shapeless compile cannot infer `split` output shapes (MLX aborts the process).
    private static let cell: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { inputs in
        let i = MLX.sigmoid(inputs[0])
        let f = MLX.sigmoid(inputs[1])
        let g = MLX.tanh(inputs[2])
        let o = MLX.sigmoid(inputs[3])
        let cell = f * inputs[4] + i * g
        return [o * MLX.tanh(cell), cell]
    }

    /// Both directions as a batch of two per step: the input projections of step t (forward) and T-1-t (backward)
    /// are stacked, the recurrent GEMM is one batched `addmm` over [Wh_forward, Wh_backward] in stock's transposed
    /// layout, and the cell update is one compiled kernel: T steps of ~3 dispatches instead of 2T steps of ~11.
    private func fusedDirections(_ x: MLXArray) -> MLXArray {
        let xProjF = MLX.addmm(bias_ih_forward + bias_hh_forward, x, Wx_forward.transposed())
        let xProjB = MLX.addmm(bias_ih_backward + bias_hh_backward, x, Wx_backward.transposed())
        let seqLen = x.shape[x.ndim - 2]
        let batchSize = x.ndim == 3 ? x.shape[0] : 1
        // [2, B, T, 4H]: the backward half reversed in time, so step t reads index t of both.
        let projections = MLX.stacked([xProjF, xProjB[.ellipsis, .stride(by: -1), 0...]], axis: 0)
        let recurrent = MLX.stacked([Wh_forward, Wh_backward], axis: 0).transposed(0, 2, 1)
        var hidden = MLXArray.zeros([2, batchSize, hiddenSize], dtype: x.dtype)
        var cell = MLXArray.zeros([2, batchSize, hiddenSize], dtype: x.dtype)
        var allHidden = [MLXArray]()
        allHidden.reserveCapacity(seqLen)
        for idx in 0..<seqLen {
            let gates = MLX.addmm(projections[0..., 0..., idx, 0...], hidden, recurrent)
            let next = Self.cell(MLX.split(gates, parts: 4, axis: -1) + [cell])
            hidden = next[0]; cell = next[1]
            allHidden.append(hidden)
        }
        let states = MLX.stacked(allHidden, axis: -2)   // [2, B, T, H]
        return MLX.concatenated([states[0], states[1][.ellipsis, .stride(by: -1), 0...]], axis: -1)
    }

    public func callAsFunction(_ x: MLXArray) -> (MLXArray, ()) {
        var input = x.asType(Wx_forward.dtype)
        if input.ndim == 2 {
            input = input.expandedDimensions(axis: 0)
        }
        if fused { return (fusedDirections(input), ()) }
        let fwd = forwardDirection(input)
        let bwd = backwardDirection(input)
        let output = MLX.concatenated([fwd, bwd], axis: -1)
        return (output, ())
    }
}
