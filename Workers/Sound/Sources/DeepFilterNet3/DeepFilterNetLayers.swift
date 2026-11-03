// SPDX-License-Identifier: MIT
// Vendored from Blaizzy/mlx-audio-swift @ 0dda61b29c308175e302dceef77c3bd84551e92b (Sources/MLXAudioSTS/Models/DeepFilterNet,
// MIT, licenses/mlx-audio-swift-MIT.txt). Arsenal changes: version 1, CPU feature loops and the Accelerate GRU removed;
// folded batch norm and transposed-conv weights keep the half dtype. See DeepFilterNetModel.swift.
// swiftlint:disable all
import MLX
import MLXNN

// MARK: - Compound Layer Helpers

extension DeepFilterNetModel {

    func applyEncoderConv(
        _ x: MLXArray,
        prefix: String,
        main: Int,
        pointwise: Int?,
        bn: Int,
        fstride: Int
    ) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weightKey: "\(prefix).\(main).weight",
            bias: nil,
            fstride: fstride,
            lookahead: 0
        )
        if let pointwise {
            y = try conv2dLayer(
                y,
                weightKey: "\(prefix).\(pointwise).weight",
                bias: nil,
                fstride: 1,
                lookahead: 0
            )
        }
        y = try batchNorm(y, prefix: "\(prefix).\(bn)")
        return relu(y)
    }

    func applyPathwayConv(_ x: MLXArray, prefix: String) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weightKey: "\(prefix).0.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        y = try batchNorm(y, prefix: "\(prefix).1")
        return relu(y)
    }

    func applyTransposeBlock(_ x: MLXArray, prefix: String, fstride: Int) throws -> MLXArray {
        var y = try convTranspose2dLayer(
            x,
            weightKey: "\(prefix).0.weight",
            fstride: fstride,
            groups: config.convCh
        )
        y = try conv2dLayer(
            y,
            weightKey: "\(prefix).1.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        return try batchNorm(y, prefix: "\(prefix).2")
    }

    func applyRegularBlock(_ x: MLXArray, prefix: String) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weightKey: "\(prefix).0.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        y = try conv2dLayer(
            y,
            weightKey: "\(prefix).1.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        return try batchNorm(y, prefix: "\(prefix).2")
    }

    func applyOutputConv(_ x: MLXArray, prefix: String) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weightKey: "\(prefix).0.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        y = try batchNorm(y, prefix: "\(prefix).1")
        return y
    }
}

// MARK: - Conv2d

extension DeepFilterNetModel {

    func conv2dLayer(
        _ xBCHW: MLXArray,
        weightKey: String,
        bias: MLXArray?,
        fstride: Int,
        lookahead: Int
    ) throws -> MLXArray {
        if let wOHWI = conv2dWeightsOHWI[weightKey] {
            return conv2dLayer(
                xBCHW,
                weightOHWI: wOHWI,
                bias: bias,
                fstride: fstride,
                lookahead: lookahead
            )
        }
        return try conv2dLayer(
            xBCHW,
            weight: w(weightKey),
            bias: bias,
            fstride: fstride,
            lookahead: lookahead
        )
    }

    func conv2dLayer(
        _ xBCHW: MLXArray,
        weight: MLXArray,
        bias: MLXArray?,
        fstride: Int,
        lookahead: Int
    ) throws -> MLXArray {
        conv2dLayer(
            xBCHW,
            weightOHWI: weight.transposed(0, 2, 3, 1),
            bias: bias,
            fstride: fstride,
            lookahead: lookahead
        )
    }

    func conv2dLayer(
        _ xBCHW: MLXArray,
        weightOHWI: MLXArray,
        bias: MLXArray?,
        fstride: Int,
        lookahead: Int
    ) -> MLXArray {
        let kT = weightOHWI.shape[1]
        let kF = weightOHWI.shape[2]
        let inPerGroup = weightOHWI.shape[3]
        let inChannels = xBCHW.shape[1]
        let groups = max(1, inChannels / max(1, inPerGroup))

        let rawLeft = kT - 1 - lookahead
        let timeCrop = max(0, -rawLeft)
        let timePadLeft = max(0, rawLeft)
        let timePadRight = max(0, lookahead)
        let freqPad = kF / 2

        var x = xBCHW.transposed(0, 2, 3, 1) // NHWC
        if timeCrop > 0, x.shape[1] > timeCrop {
            x = x[0..., timeCrop..., 0..., 0...]
        }
        x = MLX.padded(
            x,
            widths: [
                .init(0),
                .init((timePadLeft, timePadRight)),
                .init((freqPad, freqPad)),
                .init(0)
            ],
            mode: .constant
        )

        var y = MLX.conv2d(x, weightOHWI, stride: [1, fstride], padding: [0, 0], groups: groups)
        if let bias {
            y = y + bias.reshaped([1, 1, 1, bias.shape[0]])
        }
        return y.transposed(0, 3, 1, 2)
    }
}

// MARK: - ConvTranspose2d

extension DeepFilterNetModel {

    func convTranspose2dLayer(
        _ xBCHW: MLXArray,
        weightKey: String,
        fstride: Int,
        groups: Int
    ) throws -> MLXArray {
        if groups > 1, let denseWeight = convTransposeDenseWeights[weightKey] {
            var x = xBCHW.transposed(0, 2, 3, 1) // NHWC
            let kT = denseWeight.shape[1]
            let kF = denseWeight.shape[2]
            let padding = IntOrPair((kT - 1, kF / 2))
            let outputPadding = IntOrPair((0, kF / 2))
            x = MLX.convTransposed2d(
                x,
                denseWeight,
                stride: [1, fstride],
                padding: padding,
                outputPadding: outputPadding,
                groups: 1
            )
            return x.transposed(0, 3, 1, 2)
        }
        if groups > 1, let groupedWeights = convTransposeGroupWeights[weightKey], groupedWeights.count == groups {
            return convTranspose2dLayer(
                xBCHW,
                groupedWeights: groupedWeights,
                fstride: fstride
            )
        }
        return try convTranspose2dLayer(
            xBCHW,
            weight: w(weightKey),
            fstride: fstride,
            groups: groups
        )
    }

    private func convTranspose2dLayer(
        _ xBCHW: MLXArray,
        weight: MLXArray,
        fstride: Int,
        groups: Int
    ) throws -> MLXArray {
        var x = xBCHW.transposed(0, 2, 3, 1) // NHWC
        let kT = weight.shape[2]
        let kF = weight.shape[3]
        let padding = IntOrPair((kT - 1, kF / 2))
        let outputPadding = IntOrPair((0, kF / 2))

        if groups <= 1 {
            let w = weight.transposed(1, 2, 3, 0)
            x = MLX.convTransposed2d(
                x,
                w,
                stride: [1, fstride],
                padding: padding,
                outputPadding: outputPadding,
                groups: 1
            )
            return x.transposed(0, 3, 1, 2)
        }

        let inPerGroup = max(1, x.shape[3] / groups)
        var ys = [MLXArray]()
        ys.reserveCapacity(groups)

        for g in 0..<groups {
            let inStart = g * inPerGroup
            let inEnd = inStart + inPerGroup
            let xg = x[0..., 0..., 0..., inStart..<inEnd]

            let wg = weight[inStart..<inEnd, 0..., 0..., 0...]
            let wT = wg.transposed(1, 2, 3, 0)
            let yg = MLX.convTransposed2d(
                xg,
                wT,
                stride: [1, fstride],
                padding: padding,
                outputPadding: outputPadding,
                groups: 1
            )
            ys.append(yg)
        }

        return MLX.concatenated(ys, axis: 3).transposed(0, 3, 1, 2)
    }

    private func convTranspose2dLayer(
        _ xBCHW: MLXArray,
        groupedWeights: [MLXArray],
        fstride: Int
    ) -> MLXArray {
        let groups = groupedWeights.count
        var x = xBCHW.transposed(0, 2, 3, 1) // NHWC
        let kT = groupedWeights[0].shape[1]
        let kF = groupedWeights[0].shape[2]
        let padding = IntOrPair((kT - 1, kF / 2))
        let outputPadding = IntOrPair((0, kF / 2))
        let inPerGroup = max(1, x.shape[3] / groups)

        var ys = [MLXArray]()
        ys.reserveCapacity(groups)
        for (g, wT) in groupedWeights.enumerated() {
            let inStart = g * inPerGroup
            let inEnd = inStart + inPerGroup
            let xg = x[0..., 0..., 0..., inStart..<inEnd]
            let yg = MLX.convTransposed2d(
                xg,
                wT,
                stride: [1, fstride],
                padding: padding,
                outputPadding: outputPadding,
                groups: 1
            )
            ys.append(yg)
        }
        x = MLX.concatenated(ys, axis: 3)
        return x.transposed(0, 3, 1, 2)
    }
}

// MARK: - BatchNorm, GRU, Linear

extension DeepFilterNetModel {

    func batchNorm(_ x: MLXArray, prefix: String) throws -> MLXArray {
        guard let scale = bnScale[prefix], let bias = bnBias[prefix] else { throw DeepFilterNetError.missingWeightKey(prefix + ".running_mean") }
        return x * scale + bias
    }

    func squeezedGRU(
        _ x: MLXArray,
        prefix: String,
        hiddenSize: Int,
        linearOut: Bool
    ) throws -> MLXArray {
        var y = relu(groupedLinear(x, weight: try w("\(prefix).linear_in.0.weight")))

        var layer = 0
        while weights["\(prefix).gru.weight_ih_l\(layer)"] != nil {
            y = try gruLayer(y, prefix: "\(prefix).gru", layer: layer, hiddenSize: hiddenSize)
            layer += 1
        }

        if linearOut, weights["\(prefix).linear_out.0.weight"] != nil {
            y = relu(groupedLinear(y, weight: try w("\(prefix).linear_out.0.weight")))
        }
        return y
    }

    func groupedLinear(_ x: MLXArray, weight: MLXArray) -> MLXArray {
        let groups = weight.shape[0]
        let ws = weight.shape[1]
        let hs = weight.shape[2]
        let b = x.shape[0]
        let t = x.shape[1]
        let reshaped = x.reshaped([b, t, groups, ws])
        let out = MLX.einsum("btgi,gih->btgh", reshaped, weight)
        return out.reshaped([b, t, groups * hs])
    }

    func linear(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
        let b = x.shape[0]
        let t = x.shape[1]
        let x2 = x.reshaped([b * t, x.shape[2]])
        var y = MLX.matmul(x2, weight.transposed())
        y = y + bias
        return y.reshaped([b, t, weight.shape[0]])
    }
}
// swiftlint:enable all
