// SPDX-License-Identifier: MIT
import MLX
import MLXNN

/// The pinned MLXNN LSTM equations with ordinary Linear owners so both recurrent matrices accept affine packs.
public final class AudioLSTM: Module {
    @ModuleInfo(key: "x_proj") var inputProjection: Linear
    @ModuleInfo(key: "h_proj") var recurrentProjection: Linear
    public init(inputSize: Int, hiddenSize: Int, bias: Bool = true) {
        _inputProjection.wrappedValue = Linear(inputSize, 4 * hiddenSize, bias: bias)
        _recurrentProjection.wrappedValue = Linear(hiddenSize, 4 * hiddenSize, bias: false)
    }
    public func callAsFunction(_ input: MLXArray, hidden: MLXArray? = nil, cell: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let projected = inputProjection(input)
        var hidden = hidden, cell = cell
        var states: [MLXArray] = [], cells: [MLXArray] = []
        for index in 0..<input.dim(-2) {
            let current = projected[.ellipsis, index, 0...]
            let gates = hidden.map { current + recurrentProjection($0) } ?? current
            let parts = split(gates, parts: 4, axis: -1)
            let next = cell.map { sigmoid(parts[1]) * $0 + sigmoid(parts[0]) * tanh(parts[2]) } ?? (sigmoid(parts[0]) * tanh(parts[2]))
            let state = sigmoid(parts[3]) * tanh(next)
            cells.append(next); states.append(state)
            cell = next; hidden = state
        }
        return (stacked(states, axis: -2), stacked(cells, axis: -2))
    }
}

func audioLSTMKey(_ name: String) -> String {
    if name.hasSuffix(".Wx") { return String(name.dropLast(3)) + ".x_proj.weight" }
    if name.hasSuffix(".Wh") { return String(name.dropLast(3)) + ".h_proj.weight" }
    if name.hasSuffix(".bias"), name.contains(".lstm.layers.") || name.contains(".forward_lstms.") || name.contains(".backward_lstms.") {
        return String(name.dropLast(5)) + ".x_proj.bias"
    }
    return name
}
