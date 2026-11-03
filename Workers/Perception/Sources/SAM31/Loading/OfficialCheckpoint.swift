// Vendored model implementation; source and adaptations: Workers/Perception/licenses/SAM31/PORTED-FROM.md
// Get-time conversion of Meta's facebook/sam3.1 checkpoint (`sam3.1_multiplex.pt`, FP32, Meta's own key names) to the
// tensor set the port loads. Stage 1 follows mlx_vlm/models/sam3_1/convert_weights.py (mlx-vlm 0.4.3); stage 2 renames
// to the module names of the mlx-community/sam3.1-bf16 layout the port was written against.
import Foundation
import MLX
import IntegrationKit
import ModelPrecision
import ArsenalWire

enum OfficialCheckpoint {
    static let checkpoint = "sam3.1_multiplex.pt"
    static let weights = "model.safetensors"

    struct Entry: Equatable {
        var shape: [Int]
        var dtype: DType
    }
    /// Where an output tensor's values come from.
    enum Source: Equatable {
        case tensor(String)
        /// One of three equal row blocks of a fused q/k/v projection (0 = q, 1 = k, 2 = v).
        case third(String, Int)
        /// Source tensors concatenated along axis 0.
        case stacked([String])
    }
    enum Layout: Equatable {
        case keep
        /// PyTorch Conv2d `[out, in, kh, kw]` to MLX `[out, kh, kw, in]`.
        case conv
        /// PyTorch ConvTranspose2d `[in, out, kh, kw]` to MLX `[out, kh, kw, in]`.
        case convTranspose
        /// Matrix transpose.
        case transpose
        /// The ViT position embedding without its class-token row.
        case dropFirstToken
    }
    struct Planned: Equatable {
        let source: Source
        let layout: Layout
        let shape: [Int]
    }

    // MARK: Plan

    private enum Target {
        case name(String)
        /// A fused q/k/v tensor: `<prefix>.{q,k,v}_proj.<suffix>`.
        case split(String, String)
    }

    /// Every output tensor of the converted snapshot, with its source and layout. Pure: reads names and shapes only.
    /// `dropping` names output tensors the port never reads.
    static func plan(_ entries: [String: Entry], dropping: Set<String> = []) throws -> [String: Planned] {
        var out: [String: Planned] = [:]
        func add(_ name: String, _ planned: Planned) throws {
            guard out.updateValue(planned, forKey: name) == nil else { throw PrecisionError("SAM 3.1 checkpoint maps two tensors to " + name) }
        }
        for (key, entry) in entries.sorted(by: { $0.key < $1.key }) {
            guard let target = try target(key) else { continue }
            guard [.float32, .float16, .bfloat16].contains(entry.dtype) else { throw PrecisionError("SAM 3.1 checkpoint tensor is not floating point: " + key) }
            switch target {
            case .split(let prefix, let suffix):
                guard let rows = entry.shape.first, rows > 0, rows % 3 == 0 else { throw PrecisionError("Fused q/k/v tensor does not split in three: " + key) }
                for (index, name) in ["q_proj", "k_proj", "v_proj"].enumerated() {
                    try add(rename("\(prefix).\(name).\(suffix)"), Planned(source: .third(key, index), layout: .keep, shape: [rows / 3] + entry.shape.dropFirst()))
                }
            case .name(let raw):
                let name = rename(raw)
                let (layout, shape) = try layout(key, name, entry.shape)
                try add(name, Planned(source: .tensor(key), layout: layout, shape: shape))
            }
        }
        // The four point embeddings become one table; the interactive encoder's Gaussian matrix also serves the shared
        // image embedding (convert_weights.py post-processing).
        for encoder in ["tracker_model.interactive_sam_prompt_encoder", "tracker_model.sam_prompt_encoder"] {
            let names = (0..<4).map { "\(encoder).point_embeddings.\($0).weight" }
            let parts = names.compactMap { out[$0] }
            guard !parts.isEmpty else { continue }
            guard parts.count == 4, parts.allSatisfy({ $0.layout == .keep && $0.shape == parts[0].shape && $0.shape.count >= 1 }) else {
                throw PrecisionError("SAM 3.1 point embeddings are incomplete in " + encoder)
            }
            let sources = try parts.map { part -> String in
                guard case .tensor(let key) = part.source else { throw PrecisionError("SAM 3.1 point embedding source") }
                return key
            }
            for name in names { out[name] = nil }
            try add("\(encoder).point_embed.weight", Planned(source: .stacked(sources), layout: .keep, shape: [4 * parts[0].shape[0]] + parts[0].shape.dropFirst()))
        }
        if let gaussian = out["tracker_model.interactive_sam_prompt_encoder.shared_embedding.positional_embedding"] {
            try add("tracker_model.shared_image_embedding.positional_embedding", gaussian)
        }
        for name in dropping { out[name] = nil }
        return out
    }

    private static func layout(_ key: String, _ name: String, _ shape: [Int]) throws -> (Layout, [Int]) {
        if shape.count == 4, !(name.contains("positional_encoding") || key.contains("tpos_enc") || key.contains("pos_embed")) {
            if key.contains("dconv_2x2") || key.contains("output_upscaling") { return (.convTranspose, [shape[1], shape[2], shape[3], shape[0]]) }
            return (.conv, [shape[0], shape[2], shape[3], shape[1]])
        }
        if name.hasSuffix("embeddings.position_embeddings"), shape.count >= 2, shape[shape.count - 2] == 577 {
            var out = shape; out[shape.count - 2] -= 1
            return (.dropFirstToken, out)
        }
        if name == "detector_model.text_encoder.text_projection.weight" {
            guard shape.count == 2 else { throw PrecisionError("SAM 3.1 text projection is not a matrix") }
            return (.transpose, [shape[1], shape[0]])
        }
        return (.keep, shape)
    }

    /// Stage 1: `convert_key` and `_resolve_qkv_prefix` of convert_weights.py. Nil drops the tensor.
    private static func target(_ key: String) throws -> Target? {
        if key.contains("freqs_cis") { return nil } // RoPE buffers (complex64), recomputed by the port
        if let rest = key.after("detector.") { return detector(rest) }
        if let rest = key.after("tracker.model.") { return tracker(rest) }
        throw PrecisionError("Unknown SAM 3.1 checkpoint tensor: " + key)
    }

    private static func fpn(_ text: String) -> String {
        var text = text
        for (old, new) in [("dconv_2x2_0", "scale_layers.0"), ("dconv_2x2_1", "scale_layers.2"), ("dconv_2x2", "scale_layers.0"), ("conv_1x1", "proj1"), ("conv_3x3", "proj2")] {
            text = text.replacingOccurrences(of: old, with: new)
        }
        return text
    }

    private static func detector(_ key: String) -> Target {
        let trunk = "backbone.vision_backbone.trunk.", vision = "detector_model.vision_encoder.backbone."
        if let g = key.groups(#"backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.(.+)"#) {
            let prefix = vision + "layers.\(g[0]).", rest = g[1]
            if let s = rest.after("attn.qkv.") { return .split(prefix + "attention", s) }
            if let s = rest.after("attn.proj.") { return .name(prefix + "attention.out_proj." + s) }
            if let s = rest.after("attn.") { return .name(prefix + "attention." + s) }
            for (old, new) in [("norm1.", "layer_norm1."), ("norm2.", "layer_norm2."), ("ls1.", "layer_scale1."), ("ls2.", "layer_scale2.")] {
                if let s = rest.after(old) { return .name(prefix + new + s) }
            }
            return .name(prefix + rest)
        }
        if let s = key.after(trunk + "ln_pre.") { return .name(vision + "layer_norm." + s) }
        if let s = key.after(trunk + "patch_embed.proj.") { return .name(vision + "embeddings.patch_embeddings.projection." + s) }
        if key == trunk + "pos_embed" { return .name(vision + "embeddings.position_embeddings") }
        if let s = key.after(trunk) { return .name(vision + s) }
        for convs in ["convs", "interactive_convs", "propagation_convs"] {
            if let s = key.after("backbone.vision_backbone.\(convs).") { return .name("detector_model.vision_encoder.neck.\(convs)." + fpn(s)) }
        }
        if let l = key.after("backbone.language_backbone.") {
            let text = "detector_model.text_encoder."
            if l == "text_projection" || l == "encoder.text_projection" { return .name(text + "text_projection.weight") }
            if let s = l.after("encoder.ln_final.") { return .name(text + "text_model.final_layer_norm." + s) }
            if let s = l.after("encoder.token_embedding.") { return .name(text + "text_model.embeddings.token_embedding." + s) }
            if l == "encoder.positional_embedding" { return .name(text + "text_model.embeddings.position_embedding.weight") }
            if let g = l.groups(#"encoder\.transformer\.resblocks\.(\d+)\.(.+)"#) {
                let prefix = text + "text_model.encoder.layers.\(g[0]).", rest = g[1]
                if let s = rest.after("attn.in_proj_") { return .split(prefix + "self_attn", s) }
                for (old, new) in [
                    ("attn.out_proj.", "self_attn.out_proj."), ("ln_1.", "layer_norm1."), ("ln_2.", "layer_norm2."), ("mlp.c_fc.", "mlp.fc1."), ("mlp.c_proj.", "mlp.fc2.")
                ] {
                    if let s = rest.after(old) { return .name(prefix + new + s) }
                }
                return .name(prefix + rest)
            }
            if let s = l.after("resizer.") { return .name("detector_model.text_projection." + s) }
            return .name(text + l)
        }
        if let g = key.groups(#"transformer\.encoder\.layers\.(\d+)\.(.+)"#) {
            let prefix = "detector_model.detr_encoder.layers.\(g[0]).", rest = g[1]
            if let s = rest.after("self_attn.in_proj_") { return .split(prefix + "self_attn", s) }
            if let s = rest.after("cross_attn_image.in_proj_") { return .split(prefix + "cross_attn", s) }
            var r = rest.replacingOccurrences(of: "cross_attn_image", with: "cross_attn")
            if let g = r.groups(#"norm(\d)(.*)"#) { r = "layer_norm" + g[0] + g[1] }
            return .name(prefix + r)
        }
        if let s = key.after("transformer.encoder.norm.") { return .name("detector_model.detr_encoder.layer_norm." + s) }
        if let d = key.after("transformer.decoder.") {
            let decoder = "detector_model.detr_decoder."
            if let g = d.groups(#"layers\.(\d+)\.(.+)"#) {
                let prefix = decoder + "layers.\(g[0]).", rest = g[1]
                for (old, new) in [("self_attn.in_proj_", "self_attn"), ("cross_attn.in_proj_", "vision_cross_attn"), ("ca_text.in_proj_", "text_cross_attn")] {
                    if let s = rest.after(old) { return .split(prefix + new, s) }
                }
                var r = rest.replacingOccurrences(of: "ca_text", with: "text_cross_attn").replacingOccurrences(of: "catext_norm", with: "text_cross_attn_layer_norm")
                for (old, new) in [("cross_attn", "vision_cross_attn"), ("norm1", "self_attn_layer_norm"), ("norm2", "vision_cross_attn_layer_norm"), ("norm3", "mlp_layer_norm")] {
                    r = r.replacingWord(old, with: new)
                }
                return .name(prefix + r)
            }
            for (pattern, head) in [
                (#"bbox_embed\.layers\.(\d+)\.(.+)"#, "box_head"), (#"boxRPB_embed_x\.layers\.(\d+)\.(.+)"#, "box_rpb_embed_x"),
                (#"boxRPB_embed_y\.layers\.(\d+)\.(.+)"#, "box_rpb_embed_y"), (#"presence_token_head\.layers\.(\d+)\.(.+)"#, "presence_head"),
                (#"ref_point_head\.layers\.(\d+)\.(.+)"#, "ref_point_head")
            ] {
                if let g = d.groups(pattern), let index = Int(g[0]) { return .name(decoder + head + ".layer\(index + 1)." + g[1]) }
            }
            if let s = d.after("presence_token_out_norm.") { return .name(decoder + "presence_layer_norm." + s) }
            if let s = d.after("norm.") { return .name(decoder + "output_layer_norm." + s) }
            return .name(decoder + d)
        }
        if let d = key.after("dot_prod_scoring.") {
            var r = d
            for (old, new) in [("prompt_mlp.out_norm", "text_mlp_out_norm"), ("prompt_mlp", "text_mlp"), ("prompt_proj", "text_proj"), ("hs_proj", "query_proj")] {
                r = r.replacingOccurrences(of: old, with: new)
            }
            return .name("detector_model.dot_product_scoring." + r)
        }
        if let g = key.after("geometry_encoder.") {
            let geometry = "detector_model.geometry_encoder."
            if let m = g.groups(#"(.+?)\.self_attn\.in_proj_(.+)"#) { return .split(geometry + m[0] + ".self_attn", m[1]) }
            if let m = g.groups(#"(.+?)\.cross_attn_image\.in_proj_(.+)"#) { return .split(geometry + m[0] + ".cross_attn", m[1]) }
            var r = g.replacingOccurrences(of: "cross_attn_image", with: "cross_attn")
            if let m = r.groups(#"(.*)\.norm(\d)\.(.*)"#) { r = m[0] + ".layer_norm" + m[1] + "." + m[2] }
            return .name(geometry + r)
        }
        if let s = key.after("segmentation_head.") {
            let mask = "detector_model.mask_decoder."
            if let r = s.after("cross_attend_prompt.in_proj_") { return .split(mask + "prompt_cross_attn", r) }
            for (old, new) in [
                ("pixel_decoder.", "pixel_decoder."), ("mask_predictor.mask_embed.", "mask_embedder."), ("cross_attend_prompt.", "prompt_cross_attn."),
                ("cross_attn_norm.", "prompt_cross_attn_norm."), ("instance_seg_head.", "instance_projection."), ("semantic_seg_head.", "semantic_projection.")
            ] {
                if let r = s.after(old) { return .name(mask + new + r) }
            }
            return .name(mask + s)
        }
        return .name("detector_model." + key)
    }

    private static func tracker(_ key: String) -> Target? {
        let tracker = "tracker_model."
        if let m = key.after("maskmem_backbone.") {
            let memory = tracker + "memory_encoder."
            if let g = m.groups(#"mask_downsampler\.encoder\.(\d+)\.(.+)"#), let index = Int(g[0]) {
                switch (index / 3, index % 3) {
                case (4, 0): return .name(memory + "mask_downsampler.final_conv." + g[1])
                case (let layer, 0): return .name(memory + "mask_downsampler.layers.\(layer).conv." + g[1])
                case (let layer, 1): return .name(memory + "mask_downsampler.layers.\(layer).layer_norm." + g[1])
                default: return nil
                }
            }
            if let g = m.groups(#"fuser\.layers\.(\d+)\.(.+)"#) {
                var r = g[1]
                for (old, new) in [("dwconv", "depthwise_conv"), ("gamma", "scale"), ("pwconv1", "pointwise_conv1"), ("pwconv2", "pointwise_conv2")] {
                    r = r.replacingOccurrences(of: old, with: new)
                }
                if let s = r.after("norm.") { r = "layer_norm." + s }
                return .name(memory + "memory_fuser.layers.\(g[0])." + r)
            }
            if let s = m.after("pix_feat_proj.") { return .name(memory + "feature_projection." + s) }
            return .name(memory + m)
        }
        if let s = key.after("maskmem_tpos_enc") { return .name(tracker + "memory_temporal_positional_encoding" + s) }
        if let s = key.after("obj_ptr_tpos_proj.") { return .name(tracker + "temporal_positional_encoding_projection_layer." + s) }
        if let g = key.groups(#"transformer\.encoder\.layers\.(\d+)\.(.+)"#) {
            let prefix = tracker + "memory_attention.layers.\(g[0]).", rest = g[1]
            for attention in ["self_attn", "cross_attention"] {
                if let s = rest.after(attention + ".in_proj_") { return .split(prefix + attention, s) }
            }
            return .name(prefix + rest)
        }
        if let s = key.after("transformer.encoder.norm.") { return .name(tracker + "memory_attention.layer_norm." + s) }
        for decoder in ["interactive_sam_mask_decoder", "sam_mask_decoder"] {
            if let rest = key.after(decoder + ".") {
                if let g = rest.groups(#"(.+?)\.in_proj_(.+)"#) { return .split(tracker + decoder + "." + g[0], g[1]) }
                return .name(tracker + decoder + "." + rest)
            }
        }
        return .name(tracker + key)
    }

    /// Stage 2: the module names of the layout the port loads (transformers-style SAM 3 names).
    private static func rename(_ name: String) -> String {
        var t = name
        let attention = [
            "detector_model.vision_encoder.backbone.", "detector_model.detr_encoder.", "detector_model.detr_decoder.", "detector_model.geometry_encoder.",
            "detector_model.mask_decoder.prompt_cross_attn.", "tracker_model.sam_mask_decoder.transformer.", "tracker_model.interactive_sam_mask_decoder.transformer."
        ]
        if attention.contains(where: t.hasPrefix) { t = t.replacingOccurrences(of: ".out_proj.", with: ".o_proj.") }
        if attention[1...3].contains(where: t.hasPrefix) {
            t = t.replacingOccurrences(of: ".linear1.", with: ".mlp.fc1.").replacingOccurrences(of: ".linear2.", with: ".mlp.fc2.")
        }
        if let g = t.groups(#"(detector_model\.dot_product_scoring\.text_mlp)\.layers\.(\d+)\.(.+)"#), let index = Int(g[1]) {
            t = g[0] + ".layer\(index + 1)." + g[2]
        }
        let geometry = "detector_model.geometry_encoder."
        for (old, new) in [("encode_norm.", "output_layer_norm."), ("img_pre_norm.", "vision_layer_norm."), ("norm.", "prompt_layer_norm."), ("encode.", "layers.")] {
            if let s = t.after(geometry + old) { t = geometry + new + s; break }
        }
        for decoder in ["tracker_model.sam_mask_decoder.", "tracker_model.interactive_sam_mask_decoder."] {
            guard var r = t.after(decoder) else { continue }
            if let g = r.groups(#"(iou_prediction_head|pred_obj_score_head|output_hypernetworks_mlps\.\d+)\.layers\.([012])\.(.+)"#), let index = Int(g[1]) {
                r = g[0] + "." + ["proj_in", "layers.0", "proj_out"][index] + "." + g[2]
            }
            if let g = r.groups(#"output_upscaling\.([013])\.(.+)"#) {
                r = ["0": "upscale_conv1.", "1": "upscale_layer_norm.", "3": "upscale_conv2."][g[0]]! + g[1]
            }
            if let g = r.groups(#"(transformer\.layers\.\d+)\.norm(\d)\.(.+)"#) { r = g[0] + ".layer_norm" + g[1] + "." + g[2] }
            if let g = r.groups(#"(transformer\.layers\.\d+)\.mlp\.lin([12])\.(.+)"#) { r = g[0] + ".mlp." + (g[1] == "1" ? "proj_in." : "proj_out.") + g[2] }
            r = r.replacingOccurrences(of: "transformer.norm_final_attn.", with: "transformer.layer_norm_final_attn.")
            t = decoder + r
        }
        let prompt = "tracker_model.interactive_sam_prompt_encoder."
        if let g = t.groups(NSRegularExpression.escapedPattern(for: prompt) + #"mask_downscaling\.([01346])\.(.+)"#) {
            t = prompt + "mask_embed." + ["0": "conv1", "1": "layer_norm1", "3": "conv2", "4": "layer_norm2", "6": "conv3"][g[0]]! + "." + g[1]
        }
        if t == prompt + "pe_layer.positional_encoding_gaussian_matrix" { t = prompt + "shared_embedding.positional_embedding" }
        if t == "tracker_model.image_pe_layer.positional_encoding_gaussian_matrix" { t = "tracker_model.image_pe_layer.positional_embedding" }
        return t
    }

    // MARK: Convert

    /// The planned output tensor, read from the checkpoint and laid out on the CPU stream (still in its source dtype).
    static func read(_ planned: Planned, from checkpoint: TorchCheckpoint, tensors: [String: TorchCheckpoint.Tensor]) throws -> MLXArray {
        func load(_ key: String) throws -> MLXArray {
            guard let tensor = tensors[key] else { throw PrecisionError("SAM 3.1 checkpoint lost tensor " + key) }
            return try checkpoint.array(tensor)
        }
        var value: MLXArray
        switch planned.source {
        case .tensor(let key): value = try load(key)
        case .third(let key, let index): value = split(try load(key), parts: 3, axis: 0, stream: .cpu)[index]
        case .stacked(let keys): value = concatenated(try keys.map(load), axis: 0, stream: .cpu)
        }
        switch planned.layout {
        case .keep: break
        case .conv: value = transposed(value, axes: [0, 2, 3, 1], stream: .cpu)
        case .convTranspose: value = transposed(value, axes: [1, 2, 3, 0], stream: .cpu)
        case .transpose: value = transposed(value, stream: .cpu)
        case .dropFirstToken: value = split(value, indices: [1], axis: value.ndim - 2, stream: .cpu)[1]
        }
        guard value.shape == planned.shape else { throw PrecisionError("SAM 3.1 tensor shape differs from its plan") }
        return value
    }

    /// Converts the verified source file by file: the checkpoint streams tensor by tensor into `model.safetensors` in the
    /// selected half, `config.json` gets the port's form, other files are copied. A source holding only config files
    /// (a per-file cast) yields no tensors.
    static func convert(_ context: NativeConversionContext, cancellation: NativeCancellation, dropping: Set<String>) throws -> JSONObject {
        for path in try NativeConversionContext.files(in: context.source) {
            try cancellation.check()
            let file = URL(fileURLWithPath: path).lastPathComponent
            if StoredNames.reserved.contains(file) { continue }
            let source = context.source.appendingPathComponent(path)
            if path == checkpoint {
                let target = context.destination.appendingPathComponent(weights)
                guard !FileManager.default.fileExists(atPath: target.path) else { throw PrecisionError("Duplicate conversion output: " + weights) }
                let reader = try TorchCheckpoint(source)
                let tensors = try reader.tensors()
                let planned = try plan(tensors.mapValues { Entry(shape: $0.shape, dtype: $0.dtype) }, dropping: dropping)
                guard !planned.isEmpty else { throw PrecisionError("SAM 3.1 checkpoint has no tensors") }
                _ = try StreamedPrecision.convert(
                    keys: planned.keys.sorted(),
                    read: { name in
                        try cancellation.check()
                        return try read(planned[name]!, from: reader, tensors: tensors)
                    }, to: target, half: context.half, eligible: { _, _ in false })
                continue
            }
            guard !["safetensors", "pt", "pth", "bin"].contains(source.pathExtension) else {
                throw PrecisionError("This recipe converts Meta's " + checkpoint + ", not " + path)
            }
            let target = context.destination.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            guard !FileManager.default.fileExists(atPath: target.path) else { throw PrecisionError("Duplicate conversion output: " + path) }
            if path == "config.json" {
                try config(Data(contentsOf: source)).write(to: target)
            } else {
                try FileManager.default.copyItem(at: source, to: target)
            }
        }
        return try context.receipt(cancellation: cancellation)
    }

    /// Meta's config.json in the form the port reads: the checkpoint's FPN necks have no 0.5x level (the port builds one
    /// level per scale factor), and the model type names SAM 3.1's video model. Idempotent.
    static func config(_ data: Data) throws -> Data {
        guard var root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw PrecisionError("SAM 3.1 config.json is not an object") }
        root["model_type"] = "sam3.1_video"
        for model in ["detector_config", "tracker_config"] {
            guard var section = root[model] as? [String: Any], var vision = section["vision_config"] as? [String: Any],
                let factors = vision["scale_factors"] as? [Double]
            else { continue }
            let kept = factors.filter { $0 >= 1 }
            guard !kept.isEmpty else { throw PrecisionError("SAM 3.1 config.json has no upsampling FPN level") }
            vision["scale_factors"] = kept; section["vision_config"] = vision; root[model] = section
        }
        return try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
    }
}

private extension String {
    /// The remainder after `prefix`, or nil.
    func after(_ prefix: String) -> String? { hasPrefix(prefix) ? String(dropFirst(prefix.count)) : nil }
    /// Capture groups of a whole-string match of `pattern`.
    func groups(_ pattern: String) -> [String]? {
        guard let regex = try? NSRegularExpression(pattern: "^(?:" + pattern + ")$"),
            let match = regex.firstMatch(in: self, range: NSRange(startIndex..., in: self))
        else { return nil }
        return (1..<match.numberOfRanges).map { Range(match.range(at: $0), in: self).map { String(self[$0]) } ?? "" }
    }
    /// Replaces a leading `word` followed by a non-word character or the end (Python's `^word\b`).
    func replacingWord(_ word: String, with new: String) -> String {
        guard hasPrefix(word) else { return self }
        let next = dropFirst(word.count).first
        guard next.map({ !($0.isLetter || $0.isNumber || $0 == "_") }) ?? true else { return self }
        return new + dropFirst(word.count)
    }
}
