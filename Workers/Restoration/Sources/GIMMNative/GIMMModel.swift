// SPDX-License-Identifier: LicenseRef-S-Lab-1.0 (Swift port of GSeanCDAT/GIMM-VFI; see LICENSE-upstream-GIMMNative)
//
//  GIMM-VFI-R (Guo et al., NeurIPS 2024; GSeanCDAT/GIMM-VFI, gimmvfi_r.py): RAFT flows between the two frames feed a
//  generalizable implicit motion model (a SIREN coordinate network modulated by forward-splatted flow latents) that
//  predicts the flow at any timestep t, and AMT-style decoders refine it and synthesise the frame. The pair stage
//  (RAFT, splatting weights, flow latents, decoder features, correlation pyramids) runs once per frame pair; each
//  timestep then costs the splat, the coordinate network and the decoders.
//

import Foundation
import MLX
import MLXNN

public final class GIMMVFIR: @unchecked Sendable {
    public let p: GIMMParams
    /// Hyponet layer matrices with the maker's weight normalisation applied once (F.normalize over the input dimension,
    /// float32, stored FP16 for the matmul as autocast does), bias row appended.
    let hypoLayers: [MLXArray]

    public init(weights: [String: MLXArray]) {
        p = GIMMParams(weights)
        hypoLayers = (0..<5).map { i in
            let wb = weights["hyponet.params_dict.linear_wb\(i)"]!
            let rows = wb.dim(0)
            let w = wb[0..<(rows - 1)].asType(.float32), b = wb[(rows - 1)..<rows].asType(.float32)
            let norm = MLX.maximum(MLX.sqrt(MLX.sum(MLX.square(w), axis: 0, keepDims: true)), MLXArray(Float(1e-12)))
            return concatenated([w / norm, b], axis: 0).asType(.float16)
        }
        eval(hypoLayers)
    }

    // MARK: Pair stage

    /// Everything shared by all timesteps of one frame pair.
    public struct Pair {
        /// Processed frames (padded, and downscaled when `scale` < 1) [1, h, w, 3] FP16 in 0...1.
        public let frame0: MLXArray, frame1: MLXArray
        /// Full-resolution padded frames when `scale` < 1 (the maker's `full_img`), else nil.
        public let full0: MLXArray?, full1: MLXArray?
        public let raftFlow01: MLXArray, raftFlow10: MLXArray
        /// normalize_flow's per-pair scaler (max |flow|).
        public let scaler: MLXArray
        public let weights0: MLXArray, weights1: MLXArray
        public let latent0: MLXArray, latent1: MLXArray
        /// AMT features: [1/4 (128 ch), 1/8 (256 ch)] per frame, FP16.
        public let features0: [MLXArray], features1: [MLXArray]
        let corr: CorrPyramid, corrT: CorrPyramid
        public let scale: Double
    }

    /// `frame0`, `frame1`: [1, H, W, 3] FP16 in 0...1 with H and W multiples of 32 (`Padding`); `scale` the maker's
    /// ds_factor.
    public func pair(_ frame0: MLXArray, _ frame1: MLXArray, scale: Double = 1, cancellation: () throws -> Void = {}) throws -> Pair {
        try Self.checkCorrelationSize(height: frame0.dim(1), width: frame0.dim(2), scale: scale)
        var x0 = frame0.asType(.float16), x1 = frame1.asType(.float16)
        var full0: MLXArray?, full1: MLXArray?
        if scale != 1 {
            full0 = x0; full1 = x1; x0 = resize(x0, scale); x1 = resize(x1, scale)
        }
        let raft = try RAFT.run(concatenated([255 * x0, 255 * x1], axis: 0), p, cancellation: cancellation)
        try cancellation()
        let f01 = raft.flow[0..<1], f10 = raft.flow[1..<2]
        let (n01, n10, scaler) = Self.normalizeFlow(f01, f10)
        let (w0, w1) = splattingWeights(f01, f10)
        let latents = cnnEncoder(concatenated([n01, n10], axis: 0))
        func project(_ i: Int) -> [MLXArray] {
            [conv(raft.features[0][i..<(i + 1)], p, "amt_second_last_cproj"), conv(raft.features[1][i..<(i + 1)], p, "amt_last_cproj")]
        }
        let fproj = conv(raft.fmap, p, "amt_fproj")
        let volume = CorrPyramid.volume(fproj[0..<1], fproj[1..<2])
        let h8 = fproj.dim(1), w8 = fproj.dim(2)
        let pair = Pair(
            frame0: x0, frame1: x1, full0: full0, full1: full1, raftFlow01: f01, raftFlow10: f10, scaler: scaler,
            weights0: w0, weights1: w1, latent0: latents[0..<1], latent1: latents[1..<2], features0: project(0), features1: project(1),
            corr: CorrPyramid(volume), corrT: CorrPyramid(volume.reshaped([h8, w8, h8 * w8]).transposed(2, 0, 1)), scale: scale)
        eval(
            pair.raftFlow01, pair.raftFlow10, pair.scaler, pair.weights0, pair.weights1, pair.latent0, pair.latent1,
            pair.features0, pair.features1, pair.corr.levels, pair.corrT.levels)
        return pair
    }

    /// The correlation lookups index each volume with Int32: rows × cells = (h/8 · w/8)² must stay below 2³¹, which
    /// holds for every size `GIMMPadding` produces (at most about 2.2e8). Anything larger is refused rather than wrapped.
    public static func checkCorrelationSize(height: Int, width: Int, scale: Double) throws {
        let cells = (Int((Double(height) * scale).rounded(.down)) / 8) * (Int((Double(width) * scale).rounded(.down)) / 8)
        guard cells > 0, cells.multipliedReportingOverflow(by: cells).partialValue < Int(Int32.max), cells <= 46_340 else {
            throw GIMMWeightError("The frames are too large for GIMM-VFI-R's correlation volume at this flow scale (\(height)x\(width) at \(scale)).")
        }
    }

    /// normalize_flow over [flow01, -flow10]: divided by their joint max |value|, then mapped to 0...1.
    static func normalizeFlow(_ f01: MLXArray, _ f10: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let scaler = MLX.maximum(MLX.max(MLX.abs(f01)), MLX.max(MLX.abs(f10)))
        return ((f01 / scaler + 1) / 2, (-f10 / scaler + 1) / 2, scaler)
    }

    /// cal_splatting_weights: 1 / (1 + backward-consistency error * alpha_fe) + 1 / (1 + local flow deviation * alpha_v).
    /// The deviation is sqrt(E[f²] - E[f]²) under the 3x3 Gaussian (g_filter) after reflect padding, per flow channel,
    /// averaged over x and y. It is computed in float32, an accumulator: the maker's autocast runs this convolution in FP16,
    /// where the difference of two large means loses most of its digits and flow² overflows past 255 px.
    public func splattingWeights(_ f01: MLXArray, _ f10: MLXArray) -> (MLXArray, MLXArray) {
        let flows = concatenated([f01, f10], axis: 0).asType(.float32)
        let devTerm = 1 / (1 + Self.flowDeviation(flows, kernel: p["g_filter"]) * p["alpha_v"].asType(.float32))
        let warped01 = -warp(f10, f01), warped10 = -warp(f01, f10)
        let err01 = MLX.mean(MLX.abs(warped01 - f01), axis: -1, keepDims: true), err10 = MLX.mean(MLX.abs(warped10 - f10), axis: -1, keepDims: true)
        let alphaFE = p["alpha_fe"].asType(.float32)
        return (1 / (1 + err01 * alphaFE) + devTerm[0..<1], 1 / (1 + err10 * alphaFE) + devTerm[1..<2])
    }

    /// The flow deviation map [n, h, w, 1] (float32) of flows [n, h, w, 2] under the 3x3 `kernel`.
    public static func flowDeviation(_ flows: MLXArray, kernel: MLXArray) -> MLXArray {
        let f = flows.asType(.float32)
        let stack = reflectPad1(concatenated([MLX.square(f), f], axis: -1))
        let blurred = conv2d(stack, broadcast(kernel.asType(.float32).reshaped([1, 3, 3, 1]), to: [4, 3, 3, 1]), groups: 4)
        let squareMean = blurred[0..., 0..., 0..., 0..<2], mean = blurred[0..., 0..., 0..., 2..<4]
        return MLX.mean(MLX.sqrt(MLX.maximum(squareMean - MLX.square(mean), MLXArray(Float(1e-9)))), axis: -1, keepDims: true)
    }

    func lateral(_ x: MLXArray, _ key: String) -> MLXArray { x + conv(lrelu(conv(x, p, key + ".layers.0")), p, key + ".layers.2") }

    /// cnn_encoder: normalised flow [n, h, w, 2] → 16-channel latent (FP16).
    public func cnnEncoder(_ x: MLXArray) -> MLXArray {
        var h = conv(conv(x, p, "cnn_encoder.0"), p, "cnn_encoder.1")
        h = lrelu(h)
        for i in 3...5 { h = lateral(h, "cnn_encoder.\(i)") }
        return conv(lrelu(h), p, "cnn_encoder.7", reflect: true)
    }

    public func resConv(_ x: MLXArray) -> MLXArray {
        var h = lrelu(conv(conv(x, p, "res_conv.0"), p, "res_conv.1"))
        h = lateral(h, "res_conv.3")
        return conv(lrelu(h), p, "res_conv.5", reflect: true)
    }

    /// predict_flow's latent for timestep t: both latents forward-splatted to t, plus the residual refinement. float32.
    public func pixelLatent(_ pair: Pair, t: Float) -> MLXArray {
        let s0 = softsplatLinear(pair.latent0, flow: pair.raftFlow01 * t, metric: pair.weights0)
        let s1 = softsplatLinear(pair.latent1, flow: pair.raftFlow10 * (1 - t), metric: pair.weights1)
        let splatted = concatenated([s0, s1], axis: -1)
        return splatted + resConv(concatenated([pair.latent0.asType(.float32), pair.latent1.asType(.float32), splatted], axis: -1))
    }

    /// CoordSampler3D for one timestep: [1, h, w, 3] float32 holding (t, y, x), y and x at pixel centres in -1...1.
    public static func coordinates(t: Float, height h: Int, width w: Int) -> MLXArray {
        let ys = MLXArray((0..<h).map { Float(-1) + 2 * ((0.5 + Float($0)) / Float(h)) }).reshaped([1, h, 1, 1])
        let xs = MLXArray((0..<w).map { Float(-1) + 2 * ((0.5 + Float($0)) / Float(w)) }).reshaped([1, 1, w, 1])
        return concatenated(
            [MLXArray.full([1, h, w, 1], values: MLXArray(t)), broadcast(ys, to: [1, h, w, 1]), broadcast(xs, to: [1, h, w, 1])], axis: -1)
    }

    /// HypoNet: [latent, (t, y, x), 1] through five weight-normalised layers with sine activations (w0 = 1), + 0.5.
    /// Returns the normalised flow at t, [1, h, w, 2] FP16.
    public func hyponet(coordinates: MLXArray, latent: MLXArray) -> MLXArray {
        let h = latent.dim(1), w = latent.dim(2)
        var hidden = concatenated([latent.asType(.float32), coordinates], axis: -1).reshaped([h * w, 35])
        for (i, layer) in hypoLayers.enumerated() {
            hidden = matmul(concatenated([hidden.asType(.float16), MLXArray.ones([h * w, 1], dtype: .float16)], axis: -1), layer)
            if i < hypoLayers.count - 1 { hidden = MLX.sin(hidden) }
        }
        return (hidden + 0.5).reshaped([1, h, w, 2])
    }

    /// The implicit motion model's flow at t in pixels, [1, h, w, 2] float32.
    public func flow(_ pair: Pair, t: Float) -> MLXArray {
        let latent = pixelLatent(pair, t: t)
        let normalized = hyponet(coordinates: Self.coordinates(t: t, height: latent.dim(1), width: latent.dim(2)), latent: latent)
        return (normalized * 2 - 1) * pair.scaler
    }

    // MARK: Frame synthesis

    func resBlock(_ x: MLXArray, _ key: String, side: Int) -> MLXArray {
        let c = x.dim(3)
        var out = convPReLU(x, p, key + ".conv1")
        var sideFeat = convPReLU(out[0..., 0..., 0..., (c - side)..<c], p, key + ".conv2")
        out = convPReLU(concatenated([out[0..., 0..., 0..., 0..<(c - side)], sideFeat], axis: -1), p, key + ".conv3")
        sideFeat = convPReLU(out[0..., 0..., 0..., (c - side)..<c], p, key + ".conv4")
        out = conv(concatenated([out[0..., 0..., 0..., 0..<(c - side)], sideFeat], axis: -1), p, key + ".conv5")
        return prelu(x + out, p[key + ".prelu.weight"])
    }

    /// NewInitDecoder at 1/4: refined flows to t and the 129-channel state (mask logit + 128 features, FP16).
    public func initDecoder(_ f0: MLXArray, _ f1: MLXArray, flow0: MLXArray, flow1: MLXArray, img0: MLXArray, img1: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let k = "amt_init_decoder"
        func up(_ f: MLXArray) -> MLXArray {
            var h = pixelShuffle(f)
            for i in 1...5 { h = convPReLU(h, p, "\(k).upsample.\(i)") }
            return relu(batchNorm(conv(h, p, "\(k).upsample.6"), p, "\(k).upsample.7"))
        }
        let u0 = up(f0), u1 = up(f1)
        let scale = Double(u0.dim(1)) / Double(img0.dim(1))
        let i0 = resize(img0, scale), i1 = resize(img1, scale)
        var h = concatenated(
            [warp(u0, flow0), warp(u1, flow1), flow0.asType(.float32), flow1.asType(.float32), i0.asType(.float32), i1.asType(.float32), warp(i0, flow0), warp(i1, flow1)],
            axis: -1)
        h = convPReLU(h, p, "\(k).convblock.0")
        for i in 1...3 { h = resBlock(h, "\(k).convblock.\(i)", side: 64) }
        let out = conv(h, p, "\(k).convblock.4")
        return (flow0 + out[0..., 0..., 0..., 0..<2], flow1 + out[0..., 0..., 0..., 2..<4], out[0..., 0..., 0..., 4...])
    }

    /// The AMT BasicUpdateBlock (`amt_update4_low` with scale 2, `amt_update4_high` without): (delta features, delta flows).
    public func updateBlock(_ key: String, net: MLXArray, flow: MLXArray, corr: MLXArray, scale: Double?) -> (MLXArray, MLXArray) {
        let state = scale.map { resize(net, 1 / $0) } ?? net
        var cor = lrelu(conv(corr, p, key + ".convc1"))
        cor = lrelu(conv(cor, p, key + ".convc2"))
        var flo = lrelu(conv(flow, p, key + ".convf1"))
        flo = lrelu(conv(flo, p, key + ".convf2"))
        let inp = lrelu(conv(concatenated([cor, flo], axis: -1), p, key + ".conv"))
        let out = conv(lrelu(conv(concatenated([inp.asType(.float32), flow.asType(.float32), state.asType(.float32)], axis: -1), p, key + ".gru.0")), p, key + ".gru.2")
        var deltaNet = conv(lrelu(conv(out, p, key + ".feat_head.0")), p, key + ".feat_head.2")
        var deltaFlow = conv(lrelu(conv(out, p, key + ".flow_head.0")), p, key + ".flow_head.2")
        if let scale {
            deltaNet = resize(deltaNet, scale); deltaFlow = Float(scale) * resize(deltaFlow, scale)
        }
        return (deltaNet, deltaFlow)
    }

    /// NewMultiFlowDecoder at full (processed) resolution: three flow pairs (6 channels each), three masks and image residuals.
    public func finalDecoder(
        _ ft: MLXArray, _ f0: MLXArray, _ f1: MLXArray, flow0: MLXArray, flow1: MLXArray, mask: MLXArray, img0: MLXArray, img1: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let k = "amt_final_decoder"
        func up(_ f: MLXArray) -> MLXArray {
            var h = pixelShuffle(pixelShuffle(f))
            for i in 2...6 { h = convPReLU(h, p, "\(k).upsample.\(i)") }
            return relu(batchNorm(conv(h, p, "\(k).upsample.7"), p, "\(k).upsample.8"))
        }
        let u0 = up(f0), u1 = up(f1)
        let fl0 = 4 * resize(flow0, 4), fl1 = 4 * resize(flow1, 4)
        let state = resize(ft, 4), m = resize(mask, 4)
        var h = concatenated(
            [
                state.asType(.float32), warp(u0, fl0), warp(u1, fl1), fl0, fl1, m.asType(.float32), img0.asType(.float32), img1.asType(.float32),
                warp(img0, fl0), warp(img1, fl1)
            ], axis: -1)
        h = convPReLU(h, p, "\(k).convblock.0")
        for i in 1...3 { h = resBlock(h, "\(k).convblock.\(i)", side: 64) }
        let out = conv(h, p, "\(k).convblock.4")
        let tile0 = concatenated([fl0, fl0, fl0], axis: -1), tile1 = concatenated([fl1, fl1, fl1], axis: -1)
        let maskOut = MLX.sigmoid(out[0..., 0..., 0..., 12..<15] + concatenated([m, m, m], axis: -1))
        return (out[0..., 0..., 0..., 0..<6] + tile0, out[0..., 0..., 0..., 6..<12] + tile1, maskOut, out[0..., 0..., 0..., 15..<24])
    }

    /// frame_synthesize's intermediate results, for parity checks.
    public struct Synthesis {
        public let image: MLXArray
        public let flowT0Quarter: MLXArray, flowT1Quarter: MLXArray
        public let flowT0: MLXArray, flowT1: MLXArray
    }

    /// frame_synthesize: the frame at t from the motion model's flow `flowT` ([1, h, w, 2] float32). Image [1, H, W, 3]
    /// float32 in 0...1 at the padded input resolution.
    public func synthesize(_ pair: Pair, flowT: MLXArray, t: Float) -> Synthesis {
        let img0 = 2 * pair.frame0 - 1, img1 = 2 * pair.frame1 - 1
        let h8 = pair.features0[1].dim(1), w8 = pair.features0[1].dim(2)
        let lookup = coordsGrid(1, h8, w8)
        let fullT0 = flowT * (-t), fullT1 = flowT * (1 - t)
        let q0 = 0.25 * resize(fullT0, 0.25), q1 = 0.25 * resize(fullT1, 0.25)
        var (flowT0, flowT1, state) = initDecoder(pair.features0[1], pair.features1[1], flow0: q0, flow1: q1, img0: img0, img1: img1)
        let mask = state[0..., 0..., 0..., 0..<1]
        var ft = state[0..., 0..., 0..., 1...]
        // _amt_corr_scale_lookup at 1/8, linear-motion rescaling of the flows to the pair's endpoints.
        let low0 = 0.5 * resize(flowT0, 0.5), low1 = 0.5 * resize(flowT1, 0.5)
        let corr = concatenated(
            [pair.corr.lookup(lookup + low1 * (1 / (1 - t))), pair.corrT.lookup(lookup + low0 * (1 / t))], axis: -1)
        var (dNet, dFlow) = updateBlock("amt_update4_low", net: ft, flow: concatenated([low0, low1], axis: -1), corr: corr, scale: 2)
        // MLX += mutates aliases of the array; the state slices share storage with the decoder output.
        // swiftlint:disable:next shorthand_operator
        flowT0 = flowT0 + dFlow[0..., 0..., 0..., 0..<2]
        // swiftlint:disable:next shorthand_operator
        flowT1 = flowT1 + dFlow[0..., 0..., 0..., 2..<4]
        // swiftlint:disable:next shorthand_operator
        ft = ft + dNet
        (dNet, dFlow) = updateBlock("amt_update4_high", net: ft, flow: concatenated([flowT0, flowT1], axis: -1), corr: resize(corr, 2), scale: nil)
        // MLX += mutates aliases of the array; the state slices share storage with the decoder output.
        // swiftlint:disable:next shorthand_operator
        flowT0 = flowT0 + dFlow[0..., 0..., 0..., 0..<2]
        // swiftlint:disable:next shorthand_operator
        flowT1 = flowT1 + dFlow[0..., 0..., 0..., 2..<4]
        // swiftlint:disable:next shorthand_operator
        ft = ft + dNet
        var (f0, f1, masks, residual) = finalDecoder(ft, pair.features0[0], pair.features1[0], flow0: flowT0, flow1: flowT1, mask: mask, img0: img0, img1: img1)
        var a = img0, b = img1
        if let full0 = pair.full0, let full1 = pair.full1 {
            a = 2 * full0 - 1; b = 2 * full1 - 1
            let inv = Double(a.dim(1)) / Double(f0.dim(1))
            f0 = Float(inv) * resize(f0, inv); f1 = Float(inv) * resize(f1, inv); masks = resize(masks, inv); residual = resize(residual, inv)
        }
        // multi_flow_combine
        var warps: [MLXArray] = []
        for i in 0..<3 {
            let m = masks[0..., 0..., 0..., i..<(i + 1)]
            warps.append(
                m * warp(a, f0[0..., 0..., 0..., (2 * i)..<(2 * i + 2)]) + (1 - m) * warp(b, f1[0..., 0..., 0..., (2 * i)..<(2 * i + 2)])
                    + residual[0..., 0..., 0..., (3 * i)..<(3 * i + 3)])
        }
        let combined = concatenated(warps, axis: -1)
        let res = conv(prelu(conv(combined, p, "amt_comb_block.0"), p["amt_comb_block.1.weight"]), p, "amt_comb_block.2")
        let image = MLX.clip(((warps[0] + warps[1] + warps[2]) / 3 + res + 1) / 2, min: 0, max: 1)
        return Synthesis(image: image, flowT0Quarter: flowT0, flowT1Quarter: flowT1, flowT0: f0, flowT1: f1)
    }

    /// The frame at t (0 < t < 1) of a prepared pair: [1, H, W, 3] float32 in 0...1, at the padded input resolution.
    public func frame(_ pair: Pair, t: Float) -> MLXArray {
        synthesize(pair, flowT: flow(pair, t: t), t: t).image
    }
}

/// The maker's demo geometry (video_Nx.py): frames are replicate-padded, centred, to multiples of 32, and the flow
/// stages run at `scale` (ds_factor) of that size. Arsenal picks the scale from the frame size so the correlation
/// volumes stay bounded: full size up to about a megapixel (1280×736 padded), half up to four times that (1080p and
/// 1440p), a quarter above (4K), as the maker runs its 2K demo at 0.5; never so small that the short side drops under 64.
public struct GIMMPadding: Equatable {
    public let height: Int, width: Int
    public let top: Int, bottom: Int, left: Int, right: Int
    public let scale: Double
    public init(height: Int, width: Int) {
        self.height = height; self.width = width
        // The maker's InputPadder(divisor 32); a side under 64 pixels is padded to 64 so RAFT's four correlation levels
        // stay at least one cell wide.
        let ph = Swift.max((((height / 32) + 1) * 32 - height) % 32, 64 - height), pw = Swift.max((((width / 32) + 1) * 32 - width) % 32, 64 - width)
        top = ph / 2; bottom = ph - ph / 2; left = pw / 2; right = pw - pw / 2
        let pixels = (height + ph) * (width + pw), shortSide = Double(Swift.min(height + ph, width + pw))
        var s = pixels <= 1280 * 736 ? 1.0 : pixels <= 4 * 1280 * 736 ? 0.5 : 0.25
        while s < 1, shortSide * s < 64 { s *= 2 }
        scale = s
    }
    public var paddedHeight: Int { height + top + bottom }
    public var paddedWidth: Int { width + left + right }

    /// F.pad(mode="replicate") on NHWC.
    public func pad(_ x: MLXArray) -> MLXArray {
        var y = x
        if top + bottom > 0 {
            y = concatenated(
                [broadcast(y[0..., 0..<1], to: [y.dim(0), top, y.dim(2), y.dim(3)]), y, broadcast(y[0..., (height - 1)..<height], to: [y.dim(0), bottom, y.dim(2), y.dim(3)])],
                axis: 1)
        }
        if left + right > 0 {
            y = concatenated(
                [
                    broadcast(y[0..., 0..., 0..<1], to: [y.dim(0), y.dim(1), left, y.dim(3)]), y,
                    broadcast(y[0..., 0..., (width - 1)..<width], to: [y.dim(0), y.dim(1), right, y.dim(3)])
                ],
                axis: 2)
        }
        return y
    }
    public func unpad(_ x: MLXArray) -> MLXArray { x[0..., top..<(top + height), left..<(left + width)] }
}
