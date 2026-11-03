// SPDX-License-Identifier: LicenseRef-S-Lab-1.0 (Swift port of GSeanCDAT/GIMM-VFI; see LICENSE-upstream-GIMMNative)
import Foundation
import MLX
import MLXRandom

public struct GIMMWeightError: Error, LocalizedError {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

/// The exact stored inventory of GIMM-VFI-R (`gimmvfi_r_arb.pt`'s state_dict after Get): 387 tensors under the maker's
/// names, convolution kernels as OHWI. Get drops the 19 `num_batches_tracked` counters and the 8 `norm3` aliases (the
/// same BatchNorm tensors RAFT's context network also registers as `downsample.1`), so Load refuses anything missing,
/// extra or reshaped, and random-weight smokes build the real graph.
public enum GIMMWeights {
    public static func shapes() -> [String: [Int]] {
        var s: [String: [Int]] = [:]
        func conv(_ k: String, _ o: Int, _ i: Int, _ kh: Int, _ kw: Int? = nil, bias: Bool = true) {
            s[k + ".weight"] = [o, kh, kw ?? kh, i]
            if bias { s[k + ".bias"] = [o] }
        }
        func bn(_ k: String, _ n: Int) { for name in ["weight", "bias", "running_mean", "running_var"] { s[k + "." + name] = [n] } }
        func convPReLU(_ k: String, _ o: Int, _ i: Int, _ kh: Int) { conv(k + ".0", o, i, kh); s[k + ".1.weight"] = [o] }

        // RAFT (flow_estimator): feature (instance norm) and context (batch norm) encoders, update block.
        for (name, out, batch) in [("fnet", 256, false), ("cnet", 256, true)] {
            let e = "flow_estimator." + name
            conv(e + ".conv1", 64, 3, 7)
            if batch { bn(e + ".norm1", 64) }
            var inPlanes = 64
            for (layer, planes, stride) in [(1, 64, 1), (2, 96, 2), (3, 128, 2)] {
                for block in 0..<2 {
                    let b = "\(e).layer\(layer).\(block)"
                    let cin = block == 0 ? inPlanes : planes
                    conv(b + ".conv1", planes, cin, 3); conv(b + ".conv2", planes, planes, 3)
                    if batch { bn(b + ".norm1", planes); bn(b + ".norm2", planes) }
                    if block == 0 && stride != 1 {
                        conv(b + ".downsample.0", planes, cin, 1)
                        if batch { bn(b + ".downsample.1", planes) }
                    }
                }
                inPlanes = planes
            }
            conv(e + ".conv2", out, 128, 1)
        }
        let u = "flow_estimator.update_block."
        conv(u + "encoder.convc1", 256, 324, 1); conv(u + "encoder.convc2", 192, 256, 3)
        conv(u + "encoder.convf1", 128, 2, 7); conv(u + "encoder.convf2", 64, 128, 3); conv(u + "encoder.conv", 126, 256, 3)
        for gate in ["z", "r", "q"] {
            conv(u + "gru.conv\(gate)1", 128, 384, 1, 5); conv(u + "gru.conv\(gate)2", 128, 384, 5, 1)
        }
        conv(u + "flow_head.conv1", 256, 128, 3); conv(u + "flow_head.conv2", 2, 256, 3)
        conv(u + "mask.0", 256, 128, 3); conv(u + "mask.2", 576, 256, 1)

        // AMT feature projections, decoders and update blocks.
        conv("amt_last_cproj", 256, 128, 1); conv("amt_second_last_cproj", 128, 96, 1); conv("amt_fproj", 256, 256, 1)
        func resBlock(_ k: String, _ c: Int, _ side: Int) {
            convPReLU(k + ".conv1", c, c, 3); convPReLU(k + ".conv2", side, side, 3)
            convPReLU(k + ".conv3", c, c, 3); convPReLU(k + ".conv4", side, side, 3)
            conv(k + ".conv5", c, c, 3); s[k + ".prelu.weight"] = [c]
        }
        let i = "amt_init_decoder"
        convPReLU(i + ".upsample.1", 64, 64, 5)
        for j in 2...4 { convPReLU(i + ".upsample.\(j)", 64, 64, 3) }
        convPReLU(i + ".upsample.5", 128, 64, 3); conv(i + ".upsample.6", 128, 128, 1); bn(i + ".upsample.7", 128)
        convPReLU(i + ".convblock.0", 128, 272, 1)
        for j in 1...3 { resBlock(i + ".convblock.\(j)", 128, 64) }
        conv(i + ".convblock.4", 133, 128, 3)
        let f = "amt_final_decoder"
        convPReLU(f + ".upsample.2", 32, 8, 5)
        for j in 3...5 { convPReLU(f + ".upsample.\(j)", 32, 32, 3) }
        convPReLU(f + ".upsample.6", 64, 32, 3); conv(f + ".upsample.7", 64, 64, 1); bn(f + ".upsample.8", 64)
        convPReLU(f + ".convblock.0", 256, 273, 3)
        for j in 1...3 { resBlock(f + ".convblock.\(j)", 256, 64) }
        conv(f + ".convblock.4", 24, 256, 3)
        for block in ["amt_update4_low", "amt_update4_high"] {
            conv(block + ".convc1", 256, 648, 1); conv(block + ".convc2", 192, 256, 3)
            conv(block + ".convf1", 128, 4, 7); conv(block + ".convf2", 64, 128, 3); conv(block + ".conv", 188, 256, 3)
            conv(block + ".gru.0", 192, 320, 3); conv(block + ".gru.2", 192, 192, 3)
            conv(block + ".feat_head.0", 192, 192, 3); conv(block + ".feat_head.2", 128, 192, 3)
            conv(block + ".flow_head.0", 192, 192, 3); conv(block + ".flow_head.2", 4, 192, 3)
        }
        conv("amt_comb_block.0", 18, 9, 7); s["amt_comb_block.1.weight"] = [18]; conv("amt_comb_block.2", 3, 18, 7)

        // Implicit motion model.
        s["g_filter"] = [1, 1, 1, 3, 3]; s["alpha_v"] = [1]; s["alpha_fe"] = [1]
        conv("cnn_encoder.0", 16, 2, 3); conv("cnn_encoder.1", 32, 16, 3)
        for j in 3...5 { conv("cnn_encoder.\(j).layers.0", 32, 32, 3); conv("cnn_encoder.\(j).layers.2", 32, 32, 3) }
        conv("cnn_encoder.7", 16, 32, 3)
        conv("res_conv.0", 32, 64, 3); conv("res_conv.1", 64, 32, 3)
        conv("res_conv.3.layers.0", 64, 64, 3); conv("res_conv.3.layers.2", 64, 64, 3); conv("res_conv.5", 32, 64, 3)
        for (j, shape) in [[36, 128], [129, 128], [129, 128], [129, 128], [129, 2]].enumerated() { s["hyponet.params_dict.linear_wb\(j)"] = shape }
        return s
    }

    /// Load's check: exactly the inventory, every tensor in `half`.
    public static func validate(_ weights: [String: MLXArray], half: DType) throws {
        let expected = shapes()
        for key in Set(expected.keys).subtracting(weights.keys).sorted().prefix(1) { throw GIMMWeightError("GIMM-VFI-R checkpoint is missing \(key)") }
        for key in Set(weights.keys).subtracting(expected.keys).sorted().prefix(1) { throw GIMMWeightError("GIMM-VFI-R checkpoint has an unexpected tensor \(key)") }
        for (key, value) in weights {
            guard value.shape == expected[key] else { throw GIMMWeightError("GIMM-VFI-R tensor \(key) is \(value.shape), expected \(expected[key]!)") }
            guard value.dtype == half else { throw GIMMWeightError("Prepare the 16-bit snapshot before Load: \(key) is \(value.dtype)") }
        }
    }

    /// Random weights of the right shapes for no-weight smokes (BatchNorm variances positive; g_filter and the two splat
    /// weights at the maker's constant and initial values).
    public static func random(dtype: DType) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, shape) in shapes() {
            if key.hasSuffix("running_var") {
                out[key] = MLXArray.ones(shape, dtype: dtype)
            } else if key == "alpha_v" || key == "alpha_fe" {
                // The maker's initial value: a random sign could make the splat's normaliser cross zero.
                out[key] = MLXArray.ones(shape, dtype: dtype)
            } else if key == "g_filter" {
                out[key] = (MLXArray([1, 2, 1, 2, 4, 2, 1, 2, 1] as [Float]) / 16).reshaped(shape).asType(dtype)
            } else {
                // Fan-in scaled, so activations keep their size through the deep FP16 convolution stacks.
                let fanIn = shape.count >= 2 ? shape.dropFirst().reduce(1, *) : 400
                out[key] = (MLXRandom.normal(shape) * (1 / Float(fanIn).squareRoot())).asType(dtype)
            }
        }
        return out
    }
}
