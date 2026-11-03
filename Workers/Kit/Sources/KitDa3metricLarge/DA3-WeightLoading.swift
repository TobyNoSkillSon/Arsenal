// swiftlint:disable all
// Vendored inference source; see PORTED-FROM.md.
import Foundation
import MLX
import MLXNN

public enum DA3ModelLoadingError: Error, CustomStringConvertible {
    case missingWeights(String)

    public var description: String {
        switch self {
        case let .missingWeights(path):
            return "Could not find DA3 weights at '\(path)'"
        }
    }
}

// MARK: - Key Remapping

/// Split a string at the first occurrence of a separator.
private func splitFirst(_ s: String, separator: Character = ".") -> (String, String)? {
    guard let idx = s.firstIndex(of: separator) else { return nil }
    return (String(s[s.startIndex..<idx]), String(s[s.index(after: idx)...]))
}

/// Map a PyTorch weight key to the corresponding MLX model key.
/// Returns nil for keys that should be skipped.
public func remapKey(_ key: String) -> String? {
    // Strip "model." prefix
    guard key.hasPrefix("model.") else { return nil }
    var rest = String(key.dropFirst("model.".count))

    // Nested checkpoints (e.g. da3nested-giant-large) wrap weights as
    // `model.da3.<...>` (anyview / multi-view branch) and `model.da3_metric.<...>`
    // (mono branch). For Phase 1 streaming we only load the multi-view branch.
    if rest.hasPrefix("da3.") {
        rest = String(rest.dropFirst("da3.".count))
    } else if rest.hasPrefix("da3_metric.") {
        // Skip the metric branch entirely.
        return nil
    }

    // Skip camera encoder/decoder + GS head (not ported)
    if rest.hasPrefix("cam_enc.") || rest.hasPrefix("cam_dec.") {
        return nil
    }
    if rest.hasPrefix("gs_head.") || rest.hasPrefix("gs_adapter.") {
        return nil
    }

    // ---- Backbone keys ----
    if rest.hasPrefix("backbone.pretrained.") {
        rest = String(rest.dropFirst("backbone.pretrained.".count))

        if rest.hasPrefix("patch_embed.") { return "backbone.embeddings.\(rest)" }
        if rest == "cls_token" || rest == "pos_embed" { return "backbone.embeddings.\(rest)" }
        if rest == "camera_token" { return "backbone.\(rest)" }
        if rest.hasPrefix("blocks.") { return "backbone.\(rest)" }
        if rest.hasPrefix("norm.") { return "backbone.\(rest)" }
        return nil
    }

    // ---- Head keys ----
    if rest.hasPrefix("head.") {
        let headRest = String(rest.dropFirst("head.".count))

        if headRest.hasPrefix("norm.") { return rest }
        if headRest.hasPrefix("projects.") { return rest }

        // resize_layers.N.param -> resize_layerN.param
        if headRest.hasPrefix("resize_layers.") {
            let suffix = String(headRest.dropFirst("resize_layers.".count))
            guard let (idxStr, param) = splitFirst(suffix) else { return nil }
            guard let idx = Int(idxStr) else { return nil }
            if idx == 2 { return nil } // Identity layer
            return "head.resize_layer\(idx).\(param)"
        }

        // Scratch layer adapters: scratch.layerN_rn.* -> head.scratch.layerN_rn.*
        if headRest.hasPrefix("scratch.layer") {
            let suffix = String(headRest.dropFirst("scratch.".count))
            return "head.scratch.\(suffix)"
        }

        // Aux output_conv1_aux (check before output_conv1 to avoid false match)
        if headRest.hasPrefix("scratch.output_conv1_aux.") {
            let suffix = String(headRest.dropFirst("scratch.output_conv1_aux.".count))
            return "head.output_conv1_aux.\(suffix)"
        }

        // Aux output_conv2_aux: sequential [0]=conv, [2]=LN, [5]=conv
        if headRest.hasPrefix("scratch.output_conv2_aux.") {
            let suffix = String(headRest.dropFirst("scratch.output_conv2_aux.".count))
            // suffix = "LEVEL.IDX.param"
            guard let (level, rest2) = splitFirst(suffix) else { return nil }
            guard let (idxStr, param) = splitFirst(rest2) else { return nil }
            guard let idx = Int(idxStr) else { return nil }
            if idx == 0 { return "head.output_conv2_aux.\(level).conv0.\(param)" }
            if idx == 2 { return "head.output_conv2_aux.\(level).ln.\(param)" }
            if idx == 5 { return "head.output_conv2_aux.\(level).conv1.\(param)" }
            return nil
        }

        // Main output_conv1: scratch.output_conv1.* -> head.output_conv1.*
        if headRest.hasPrefix("scratch.output_conv1.") {
            let suffix = String(headRest.dropFirst("scratch.output_conv1.".count))
            return "head.output_conv1.\(suffix)"
        }

        // Sky head: scratch.output_conv2_sky.N.* -> head.sky_conv2_N.*
        if headRest.hasPrefix("scratch.sky_output_conv2.") || headRest.hasPrefix("scratch.output_conv2_sky.") {
            let prefix = headRest.hasPrefix("scratch.sky_output_conv2.") ? "scratch.sky_output_conv2." : "scratch.output_conv2_sky."
            let suffix = String(headRest.dropFirst(prefix.count))
            guard let (idxStr, param) = splitFirst(suffix) else { return nil }
            guard let idx = Int(idxStr) else { return nil }
            if idx == 0 { return "head.sky_conv2_0.\(param)" }
            if idx == 2 { return "head.sky_conv2_1.\(param)" }
            return nil
        }

        // Main output_conv2: sequential 0=conv, 1=ReLU, 2=conv
        if headRest.hasPrefix("scratch.output_conv2.") {
            let suffix = String(headRest.dropFirst("scratch.output_conv2.".count))
            guard let (idxStr, param) = splitFirst(suffix) else { return nil }
            guard let idx = Int(idxStr) else { return nil }
            if idx == 0 { return "head.output_conv2_0.\(param)" }
            if idx == 2 { return "head.output_conv2_1.\(param)" }
            return nil // ReLU
        }

        // Fusion blocks: scratch.refinenetN.* -> head.refinenetN.*
        if headRest.hasPrefix("scratch.refinenet") {
            let suffix = String(headRest.dropFirst("scratch.".count))
            return "head.\(suffix)"
        }

        // Passthrough for anything else under head that isn't scratch
        return headRest.hasPrefix("scratch.") ? nil : rest
    }

    return nil
}

// MARK: - Conv Transposition

/// Check if a 4D weight tensor needs NCHW -> NHWC transposition.
func isConvWeight(key: String, shape: [Int]) -> Bool {
    key.hasSuffix(".weight") && shape.count == 4
}

/// Check if key belongs to a ConvTranspose2d layer.
public func isConvTranspose(key: String) -> Bool {
    key.contains("resize_layer0") || key.contains("resize_layer1")
}

/// Transpose Conv2d weight: PyTorch (O, I, kH, kW) -> MLX (O, kH, kW, I).
func transposeConv2d(_ value: MLXArray) -> MLXArray {
    value.transposed(axes: [0, 2, 3, 1])
}

/// Transpose ConvTranspose2d weight: PyTorch (I, O, kH, kW) -> MLX (O, kH, kW, I).
func transposeConvTranspose2d(_ value: MLXArray) -> MLXArray {
    value.transposed(axes: [1, 2, 3, 0])
}

// swiftlint:enable all
