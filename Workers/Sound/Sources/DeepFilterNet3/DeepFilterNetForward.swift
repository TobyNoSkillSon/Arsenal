// SPDX-License-Identifier: MIT
// Vendored from Blaizzy/mlx-audio-swift @ 0dda61b29c308175e302dceef77c3bd84551e92b (Sources/MLXAudioSTS/Models/DeepFilterNet,
// MIT, licenses/mlx-audio-swift-MIT.txt). Arsenal changes: version 1, CPU feature loops and the Accelerate GRU removed;
// folded batch norm and transposed-conv weights keep the half dtype. See DeepFilterNetModel.swift.
// swiftlint:disable all
import MLX
import MLXNN

// MARK: - Network Forward (V2/V3)

extension DeepFilterNetModel {

    func forward(spec: MLXArray, featErb: MLXArray, featSpec5D: MLXArray, trace: DeepFilterNetTrace? = nil) throws -> MLXArray {
        let featSpec = featSpec5D.squeezed(axis: 1).transposed(0, 3, 1, 2)
        let featErbShift = applyLookahead(feature: featErb, lookahead: config.convLookahead)
        let featSpecShift = applyLookahead(feature: featSpec, lookahead: config.convLookahead)
        let (e0, e1, e2, e3, emb, c0, lsnr) = try encode(featErb: featErbShift, featSpec: featSpecShift)
        for (i, value) in [e0, e1, e2, e3, emb, c0, lsnr].enumerated() { trace?("enc\(i)", value) }

        let mask = try decodeErb(emb: emb, e3: e3, e2: e2, e1: e1, e0: e0)
        trace?("erb_mask", mask)
        let specMasked = applyMask(spec: spec, mask: mask)

        let dfCoefs = try decodeDf(emb: emb, c0: c0)
        trace?("df_coefs", dfCoefs)
        let b = dfCoefs.shape[0], t = dfCoefs.shape[1]
        let dfCoefs5 = dfCoefs.reshaped([b, t, config.nbDf, config.dfOrder, 2]).transposed(0, 3, 1, 2, 4)
        // DeepFilterNet3 (enc_concat false): deep-filter the noisy low band, keep the masked high band.
        let specDf = dsp.deepFilter(spec: spec, coefs: dfCoefs5)
        return concatenated([specDf[0..., 0..., 0..., 0..<config.nbDf, 0...], specMasked[0..., 0..., 0..., config.nbDf..., 0...]], axis: 3)
    }

    private func encode(featErb: MLXArray, featSpec: MLXArray)
        throws -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray)
    {
        let e0 = try applyEncoderConv(featErb, prefix: "enc.erb_conv0", main: 1, pointwise: nil, bn: 2, fstride: 1)
        let e1 = try applyEncoderConv(e0, prefix: "enc.erb_conv1", main: 0, pointwise: 1, bn: 2, fstride: 2)
        let e2 = try applyEncoderConv(e1, prefix: "enc.erb_conv2", main: 0, pointwise: 1, bn: 2, fstride: 2)
        let e3 = try applyEncoderConv(e2, prefix: "enc.erb_conv3", main: 0, pointwise: 1, bn: 2, fstride: 1)

        let c0 = try applyEncoderConv(featSpec, prefix: "enc.df_conv0", main: 1, pointwise: 2, bn: 3, fstride: 1)
        let c1 = try applyEncoderConv(c0, prefix: "enc.df_conv1", main: 0, pointwise: 1, bn: 2, fstride: 2)

        let b = c1.shape[0]
        let t = c1.shape[2]
        var cemb = c1.transposed(0, 2, 3, 1).reshaped([b, t, -1])
        cemb = relu(groupedLinear(cemb, weight: try w("enc.df_fc_emb.0.weight")))

        var emb = e3.transposed(0, 2, 3, 1).reshaped([b, t, -1])
        emb = config.encConcat ? MLX.concatenated([emb, cemb], axis: -1) : (emb + cemb)

        emb = try squeezedGRU(
            emb,
            prefix: "enc.emb_gru",
            hiddenSize: config.embHiddenDim,
            linearOut: true
        )

        let lsnr =
            sigmoid(
                linear(
                    emb,
                    weight: try w("enc.lsnr_fc.0.weight"),
                    bias: try w("enc.lsnr_fc.0.bias")
                )) * Float(config.lsnrMax - config.lsnrMin) + Float(config.lsnrMin)

        return (e0, e1, e2, e3, emb, c0, lsnr)
    }

    private func decodeErb(
        emb: MLXArray,
        e3: MLXArray,
        e2: MLXArray,
        e1: MLXArray,
        e0: MLXArray
    ) throws -> MLXArray {
        var embDec = try squeezedGRU(
            emb,
            prefix: "erb_dec.emb_gru",
            hiddenSize: config.embHiddenDim,
            linearOut: true
        )

        let b = embDec.shape[0]
        let t = embDec.shape[1]
        let f8 = e3.shape[3]
        embDec = embDec.reshaped([b, t, f8, -1]).transposed(0, 3, 1, 2)

        var d3 = relu(try applyPathwayConv(e3, prefix: "erb_dec.conv3p")) + embDec
        d3 = relu(try applyRegularBlock(d3, prefix: "erb_dec.convt3"))
        var d2 = relu(try applyPathwayConv(e2, prefix: "erb_dec.conv2p")) + d3
        d2 = relu(try applyTransposeBlock(d2, prefix: "erb_dec.convt2", fstride: 2))
        var d1 = relu(try applyPathwayConv(e1, prefix: "erb_dec.conv1p")) + d2
        d1 = relu(try applyTransposeBlock(d1, prefix: "erb_dec.convt1", fstride: 2))
        let d0 = relu(try applyPathwayConv(e0, prefix: "erb_dec.conv0p")) + d1
        let out = try applyOutputConv(d0, prefix: "erb_dec.conv0_out")
        return sigmoid(out)
    }

    private func decodeDf(emb: MLXArray, c0: MLXArray) throws -> MLXArray {
        var c = try squeezedGRU(
            emb,
            prefix: "df_dec.df_gru",
            hiddenSize: config.dfHiddenDim,
            linearOut: false
        )

        if weights["df_dec.df_skip.weight"] != nil {
            c = c + groupedLinear(emb, weight: try w("df_dec.df_skip.weight"))
        }

        var c0p = try conv2dLayer(
            c0,
            weightKey: "df_dec.df_convp.1.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        c0p = try conv2dLayer(
            c0p,
            weightKey: "df_dec.df_convp.2.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        c0p = relu(try batchNorm(c0p, prefix: "df_dec.df_convp.3"))
        c0p = c0p.transposed(0, 2, 3, 1)

        let b = c.shape[0]
        let t = c.shape[1]
        let dfOut = tanh(groupedLinear(c, weight: try w("df_dec.df_out.0.weight")))
            .reshaped([b, t, config.nbDf, config.dfOrder * 2])

        return dfOut + c0p
    }

    func applyMask(spec: MLXArray, mask: MLXArray) -> MLXArray {
        let b = mask.shape[0]
        let t = mask.shape[2]
        let e = mask.shape[3]
        let flat = mask.reshaped([b * t, e])
        let gains = MLX.matmul(flat, erbInvFB.asType(flat.dtype)).reshaped([b, 1, t, config.freqBins, 1])
        return spec * gains.asType(spec.dtype)
    }

}
// swiftlint:enable all
