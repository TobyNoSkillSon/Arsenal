// Vendored model implementation; source and adaptations: Workers/Perception/licenses/MLXRFDETR/PORTED-FROM.md
import Foundation
import MLX
// MARK: - Key sanitization

/// Remap a single HF safetensors key → (possibly multiple) Swift key(s).
/// Conv2d weights are transposed in-place. Fused QKV splits produce 3 entries.
///
/// Returns empty array for keys that should be skipped.
public func sanitized(key: String, value: MLXArray) -> [(String, MLXArray)] {
    // 1. Strip model. prefix
    var k = key.hasPrefix("model.") ? String(key.dropFirst("model.".count)) : key

    // 2. Skip mask_token and keypoint buffers we don't materialize.
    //    keypoint_class_mask is all-False for the preview schema (no-op); _kp_active_mask
    //    is recomputed from config in the model. Neither has a Swift parameter.
    if k.contains("mask_token") { return [] }
    if k.contains("keypoint_class_mask") || k.contains("_kp_active_mask") { return [] }

    // 3. Remap backbone key structure
    k = remapBackbone(k)

    // 4. Remap projector key structure (main + keypoint cross-attn projector)
    k = remapProjector(k)

    // 4b. nn.Embedding → bare ParameterInfo (Swift stores these as MLXArray, not Module)
    if k == "refpoint_embed.weight" { k = "refpoint_embed" }
    if k == "query_feat.weight" { k = "query_feat" }

    // 4c. Keypoint initializer AdaLN: PyTorch nn.Sequential indices are 0 (Linear),
    //     1 (GELU, no weights), 2 (Linear). Collapse index 2 → 1 to fit the Swift
    //     two-element `[Linear]` array (GELU applied in code).
    if k.contains(".adaLN_modulation.2.") {
        k = k.replacingOccurrences(of: ".adaLN_modulation.2.", with: ".adaLN_modulation.1.")
    }
    // Keypoint decoder positional embedding: bare ParameterInfo.
    if k == "transformer.decoder.keypoint_pos_embed" { /* already correct */  }

    // 5. Handle fused QKV split for decoder self-attention
    if k.contains("in_proj_weight") || k.contains("in_proj_bias") {
        return [(k.replacingOccurrences(of: "in_proj_weight", with: "in_proj.weight").replacingOccurrences(of: "in_proj_bias", with: "in_proj.bias"), value)]
    }

    // 6. Transpose Conv weights (PyTorch NCHW → MLX NHWC)
    var v = value
    if v.ndim == 4 {
        // ConvTranspose2d in stages_sampling: PyTorch (in, out, kH, kW) → MLX (out, kH, kW, in)
        if k.contains("stages_sampling") && !k.contains(".conv.") && k.hasSuffix(".weight") {
            v = v.transposed(1, 2, 3, 0)
        } else if k.lowercased().contains("conv") || k.contains("spatial_features_proj") || k.contains("patch_embed.proj") {
            v = v.transposed(0, 2, 3, 1) // Conv2d: (O, I, kH, kW) → (O, kH, kW, I)
        }
    }

    return [(k, v)]
}

// MARK: - Backbone remapping

private func remapBackbone(_ key: String) -> String {
    var k = key
    // DINOv2 backbone structural remapping
    // cls_token / position_embeddings live at backbone level in Swift, not patch_embed
    k = k.replacingOccurrences(
        of: "backbone.0.encoder.encoder.embeddings.cls_token",
        with: "backbone.cls_token"
    )
    k = k.replacingOccurrences(
        of: "backbone.0.encoder.encoder.embeddings.position_embeddings",
        with: "backbone.pos_embed"
    )
    k = k.replacingOccurrences(
        of: "backbone.0.encoder.encoder.embeddings.register_tokens",
        with: "backbone.register_tokens"
    )
    k = k.replacingOccurrences(
        of: "backbone.0.encoder.encoder.embeddings.",
        with: "backbone.patch_embed."
    )
    k = k.replacingOccurrences(
        of: "backbone.0.encoder.encoder.encoder.layer.",
        with: "backbone.blocks."
    )
    k = k.replacingOccurrences(
        of: "backbone.0.encoder.encoder.layernorm.",
        with: "backbone.norm."
    )
    // PatchEmbed: patch_embeddings.projection → proj
    k = k.replacingOccurrences(of: ".patch_embeddings.projection", with: ".proj")
    // Attention key remapping (HF → MLX)
    k = k.replacingOccurrences(of: ".attention.attention.query.", with: ".attn.q.")
    k = k.replacingOccurrences(of: ".attention.attention.key.", with: ".attn.k.")
    k = k.replacingOccurrences(of: ".attention.attention.value.", with: ".attn.v.")
    k = k.replacingOccurrences(of: ".attention.output.dense.", with: ".attn.out.")
    // LayerScale: strip only .lambda1 suffix, then rename .layer_scale1 → .ls1.gamma
    k = k.replacingOccurrences(of: ".layer_scale1.lambda1", with: ".layer_scale1")
    k = k.replacingOccurrences(of: ".layer_scale2.lambda1", with: ".layer_scale2")
    k = k.replacingOccurrences(of: ".layer_scale1", with: ".ls1.gamma")
    k = k.replacingOccurrences(of: ".layer_scale2", with: ".ls2.gamma")
    return k
}

// MARK: - Projector remapping

private func remapProjector(_ key: String) -> String {
    var k = key
    // Main projector and keypoint cross-attention projector (dual_projector).
    k = k.replacingOccurrences(of: "backbone.0.projector.", with: "projector.")
    k = k.replacingOccurrences(of: "backbone.0.cross_attn_projector.", with: "cross_attn_projector.")

    // MLX Swift treats integer key segments as array indices, so we remap integer
    // sub-module keys to named strings to match our @ModuleInfo declarations.
    //
    // stages.N.0.{rest}  → stages.N.c2f.{rest}   (ProjectorStage.c2f)
    // stages.N.1.{rest}  → stages.N.norm.{rest}   (ProjectorStage.norm)
    // stages_sampling.N.M.0.{rest} → stages_sampling.N.M.op.{rest}  (FeatureSamplerStep.op)
    var parts = k.components(separatedBy: ".")
    let isProjector = parts.first == "projector" || parts.first == "cross_attn_projector"
    if parts.count >= 4 && isProjector && parts[1] == "stages"
        && Int(parts[2]) != nil && (parts[3] == "0" || parts[3] == "1")
    {
        parts[3] = parts[3] == "0" ? "c2f" : "norm"
        k = parts.joined(separator: ".")
    } else if parts.count >= 5 && isProjector && parts[1] == "stages_sampling"
        && Int(parts[2]) != nil && Int(parts[3]) != nil && parts[4] == "0"
    {
        parts[4] = "op"
        k = parts.joined(separator: ".")
    }
    return k
}
