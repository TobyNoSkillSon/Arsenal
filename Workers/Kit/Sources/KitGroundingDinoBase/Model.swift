// SPDX-License-Identifier: MIT
// Grounding DINO Base (IDEA-Research/grounding-dino-base) open-set detection, ported from transformers'
// modeling_grounding_dino.py (GroundingDinoForObjectDetection), image_processing_grounding_dino.py (torchvision backend)
// and processing_grounding_dino.py. One image, no padding, so every pixel and text mask is all-valid.
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN
import Tokenizers

let modelKind = "perception"
let modelOps = ["detect"]

private let dModel = 256, levels = 4, points = 4, heads = 8, queriesCount = 900, maxText = 256
private let specialTokens: Set<Int32> = [101, 102, 1012, 1029]
private let mean: [Float] = [0.485, 0.456, 0.406], std: [Float] = [0.229, 0.224, 0.225]

final class Model {
    let net: GroundingDino
    let tokenizer: Tokenizer

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        net = GroundingDino()
        var weights = try Weights.load(weightsDir)
        for (key, value) in weights where value.ndim == 4 { weights[key] = value.transposed(0, 2, 3, 1) }
        try Weights.apply(weights, to: net) { key in
            if key.hasSuffix("relative_position_index") { return nil }  // a buffer, rebuilt in Backbones.swift
            // input_proj_vision.N is an nn.Sequential (conv, GroupNorm); numeric child keys would unflatten as arrays.
            if key.hasPrefix("model.input_proj_vision.") {
                var parts = key.split(separator: ".").map(String.init)
                parts[3] = parts[3] == "0" ? "conv" : "norm"
                return parts.joined(separator: ".")
            }
            return key
        }
        tokenizer = try Text.tokenizer(weightsDir)
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        guard op == "detect" else { throw RunnerError.invalid("Unknown op \(op).") }
        let queries = try Queries.from(params, limit: 128)
        let threshold = Float(params.double("threshold", default: 0.3))
        // "." and "?" separate labels in the prompt, so a label holding one ("st. bernard") would split into two
        // label groups and shift every later label: they are dropped from each label before tokenizing.
        let cleaned = queries.map { query in
            query.lowercased().filter { $0 != "." && $0 != "?" }.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        }
        if let empty = cleaned.firstIndex(where: \.isEmpty) {
            throw RunnerError.invalid("The label \"\(queries[empty])\" holds no words to look for.")
        }
        // GroundingDinoProcessor._merge_candidate_labels_text: lowercase, "a. b."
        let prompt = cleaned.joined(separator: ". ") + "."
        let ids = tokenizer.encode(text: prompt).map(Int32.init)
        guard ids.count <= maxText else { throw RunnerError.invalid("The labels are too long (at most \(maxText) tokens in all).") }
        guard Labels.groups(ids).max() == queries.count else {
            throw RunnerError.invalid("A label holds punctuation the model reads as a separator; use plain words.")
        }
        let file = try params.file("image")
        let header = try checkImageHeader(file)
        // An extreme aspect (4200x100) leaves fewer than the 900 feature positions the decoder's queries need.
        let (resizedH, resizedW) = Preprocess.targetSize(height: header.height, width: header.width)
        guard min(resizedH, resizedW) >= 33 else {
            throw RunnerError.invalid("The image is \(header.width)x\(header.height); its aspect is too extreme for this model (at most about 40:1).")
        }
        let photo = try Images.load(file)
        let pixels = Preprocess.groundingDino(photo.pixels, width: photo.width, height: photo.height)
        try params.checkCancelled()

        let (logits, boxes, extra) = try net(pixels: pixels.asType(.float16), ids: ids, checkpoint: { try params.checkCancelled() })
        eval(logits, boxes)
        if let dump = ProcessInfo.processInfo.environment["ARSENAL_PARITY_DIR"] {
            var arrays = extra
            arrays["pixel_values"] = pixels; arrays["logits"] = logits; arrays["pred_boxes"] = boxes
            arrays["input_ids"] = MLXArray(ids)
            try save(arrays: arrays, url: URL(fileURLWithPath: dump).appendingPathComponent("gdino-\(UUID().uuidString.prefix(8)).safetensors"))
        }

        // post_process_grounded_object_detection: score = max token probability; boxes cx cy w h of the photo.
        let probs = sigmoid(logits).asArray(Float.self)  // [900, L]
        let b = boxes.asArray(Float.self)
        let groups = Labels.groups(ids)
        let length = ids.count, w = Float(photo.width), h = Float(photo.height)
        var found: [Detection] = []
        for q in 0..<queriesCount {
            let row = Array(probs[(q * length)..<((q + 1) * length)])
            let score = row.max() ?? 0
            guard score > threshold else { continue }
            let cx = b[q * 4] * w, cy = b[q * 4 + 1] * h, bw = b[q * 4 + 2] * w, bh = b[q * 4 + 3] * h
            found.append(Detection(
                label: Labels.best(row, groups: groups, queries: queries), score: score,
                x0: cx - bw / 2, y0: cy - bh / 2, x1: cx + bw / 2, y1: cy + bh / 2))
        }
        return Reply.detections(found, width: photo.width, height: photo.height)
    }
}

// MARK: - Labels

enum Labels {
    /// build_label_maps: a group number per token, 0 for delimiters ([CLS], [SEP], ".", "?").
    static func groups(_ ids: [Int32]) -> [Int] {
        var count = 0, result: [Int] = []
        for id in ids {
            if specialTokens.contains(id) || id == 0 { count += 1; result.append(0) } else { result.append(count) }
        }
        // renumber the non-zero groups 1, 2, 3 ... in order
        var order: [Int: Int] = [:]
        for g in result where g != 0 && order[g] == nil { order[g] = order.count + 1 }
        return result.map { $0 == 0 ? 0 : order[$0]! }
    }

    /// The query whose tokens hold the highest probability for this box.
    static func best(_ row: [Float], groups: [Int], queries: [String]) -> String {
        var bestGroup = 0, bestValue = -Float.infinity
        for (i, g) in groups.enumerated() where g > 0 && row[i] > bestValue { bestValue = row[i]; bestGroup = g }
        guard bestGroup > 0 else { return queries[0] }
        return queries[min(bestGroup, queries.count) - 1]
    }
}

// MARK: - Preprocessing

enum Preprocess {
    /// get_size_with_aspect_ratio(size=800, max_size=1333).
    static func targetSize(height: Int, width: Int) -> (Int, Int) {
        var size = 800
        var raw: Double? = nil
        let minO = Double(min(height, width)), maxO = Double(max(height, width))
        if maxO / minO * Double(size) > 1333 {
            raw = 1333 * minO / maxO
            size = Int(raw!.rounded(.toNearestOrEven))
        }
        if (height <= width && height == size) || (width <= height && width == size) { return (height, width) }
        if width < height {
            let oh = raw.map { Int($0 * Double(height) / Double(width)) } ?? Int(Double(size) * Double(height) / Double(width))
            return (oh, size)
        }
        let ow = raw.map { Int($0 * Double(width) / Double(height)) } ?? Int(Double(size) * Double(width) / Double(height))
        return (size, ow)
    }

    /// Resize (uint8 antialiased bilinear), rescale, ImageNet normalize: [h, w, 3] float32.
    static func groundingDino(_ image: MLXArray, width: Int, height: Int) -> MLXArray {
        let (oh, ow) = targetSize(height: height, width: width)
        let bytes = (image * 255).round().asType(.uint8).asArray(UInt8.self)
        let resized = pilBilinear(bytes, width: width, height: height, outWidth: ow, outHeight: oh)
        let pixels = MLXArray(resized, [oh, ow, 3]).asType(.float32) / 255
        return Images.normalize(pixels, mean: mean, std: std)
    }
}

// MARK: - Positional encodings

/// encode_sinusoidal_position_embedding: coords [N, n] → [N, n * feats]; x and y swapped (DETR order y, x, w, h).
func sinusoidal(_ coords: MLXArray, feats: Int, temperature: Float = 10000) -> MLXArray {
    let dimT = MLXArray((0..<feats).map { pow(temperature, Float(2 * ($0 / 2)) / Float(feats)) })
    let e = coords.asType(.float32).expandedDimensions(axis: -1) * (2 * Float.pi) / dimT  // [N, n, feats]
    let odd = MLXArray((0..<feats).map { $0 % 2 == 1 })
    var parts = which(odd, cos(e), sin(e))
    let n = coords.dim(-1)
    if n >= 2 {
        let order = [1, 0] + Array(2..<n)
        parts = take(parts, MLXArray(order.map(Int32.init)), axis: 1)
    }
    return parts.reshaped([coords.dim(0), n * feats])
}

/// GroundingDinoSinePositionEmbedding with an all-valid mask, temperature 20: [H * W, 256].
func imageSine(height: Int, width: Int) -> MLXArray {
    let half = dModel / 2
    let dimT = (0..<half).map { pow(Float(20), Float(2 * ($0 / 2)) / Float(half)) }
    var out = [Float](repeating: 0, count: height * width * dModel)
    for y in 0..<height {
        let ye = Float(y + 1) / (Float(height) + 1e-6) * 2 * Float.pi
        for x in 0..<width {
            let xe = Float(x + 1) / (Float(width) + 1e-6) * 2 * Float.pi
            let base = (y * width + x) * dModel
            for k in 0..<half {
                let py = ye / dimT[k], px = xe / dimT[k]
                out[base + k] = k % 2 == 0 ? sin(py) : cos(py)
                out[base + half + k] = k % 2 == 0 ? sin(px) : cos(px)
            }
        }
    }
    return MLXArray(out, [height * width, dModel])
}

func logitEps(_ x: MLXArray, eps: Float = 1e-5) -> MLXArray {
    let c = clip(x, min: eps, max: 1 - eps)
    return log(c / (1 - c))
}

// MARK: - Parity probe (ARSENAL_PARITY_DIR only)

nonisolated(unsafe) var probes: [String: MLXArray] = [:]
let probing = ProcessInfo.processInfo.environment["ARSENAL_PARITY_DIR"] != nil
func probe(_ name: String, _ x: MLXArray) { if probing { probes[name] = x } }

// MARK: - Network modules (checkpoint names)

final class MLPHead: Module {
    var layers: [Linear]
    init(_ dims: [Int]) { layers = zip(dims.dropLast(), dims.dropFirst()).map { Linear($0, $1) }; super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for (i, layer) in layers.enumerated() { x = i < layers.count - 1 ? relu(layer(x)) : layer(x) }
        return x
    }
}

/// GroundingDinoMultiheadAttention (query/key/value/out_proj).
final class GDAttention: Module {
    var query = Linear(dModel, dModel), key = Linear(dModel, dModel), value = Linear(dModel, dModel), out_proj = Linear(dModel, dModel)
    let heads: Int
    init(heads: Int) { self.heads = heads; super.init() }
    /// q [Nq, 256], k/v [Nk, 256]; mask additive [Nq, Nk] or nil.
    func callAsFunction(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let hd = dModel / heads
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([1, t.dim(0), heads, hd]).transposed(0, 2, 1, 3) }
        let out = MLXFast.scaledDotProductAttention(
            queries: split(query(q)), keys: split(key(k)), values: split(value(v)), scale: 1 / sqrt(Float(hd)),
            mask: mask.map { .array($0.asType(q.dtype).reshaped([1, 1, $0.dim(0), $0.dim(1)])) } ?? .none)
        return out_proj(out.transposed(0, 2, 1, 3).reshaped([q.dim(0), dModel]))
    }
}

/// GroundingDinoMultiscaleDeformableAttention with MultiScaleDeformableAttention's grid_sample (bilinear, zeros,
/// align_corners=False) done as gathers.
final class DeformableAttention: Module {
    var sampling_offsets = Linear(dModel, heads * levels * points * 2)
    var attention_weights = Linear(dModel, heads * levels * points)
    var value_proj = Linear(dModel, dModel)
    var output_proj = Linear(dModel, dModel)

    /// hidden [Q, 256] (position already added); source [N, 256]; reference [Q, 2] (x, y) or [Q, 4] (cx, cy, w, h).
    func callAsFunction(_ hidden: MLXArray, source: MLXArray, reference: MLXArray, shapes: [(Int, Int)]) -> MLXArray {
        let q = hidden.dim(0), hd = dModel / heads
        let value = value_proj(source).reshaped([source.dim(0), heads, hd])
        let offsets = sampling_offsets(hidden).asType(.float32).reshaped([q, heads, levels, points, 2])
        let attention = softmax(attention_weights(hidden).asType(.float32).reshaped([q, heads, levels * points]), axis: -1)
            .reshaped([q, heads, levels, points])
        let ref = reference.asType(.float32)
        var output: MLXArray? = nil
        var start = 0
        for (level, (h, w)) in shapes.enumerated() {
            let off = offsets[0..., 0..., level, 0..., 0...]  // [Q, heads, P, 2]
            var locX: MLXArray, locY: MLXArray
            if ref.dim(-1) == 2 {
                locX = ref[0..., 0].reshaped([q, 1, 1]) + off[0..., 0..., 0..., 0] / Float(w)
                locY = ref[0..., 1].reshaped([q, 1, 1]) + off[0..., 0..., 0..., 1] / Float(h)
            } else {
                locX = ref[0..., 0].reshaped([q, 1, 1]) + off[0..., 0..., 0..., 0] / Float(points) * ref[0..., 2].reshaped([q, 1, 1]) * 0.5
                locY = ref[0..., 1].reshaped([q, 1, 1]) + off[0..., 0..., 0..., 1] / Float(points) * ref[0..., 3].reshaped([q, 1, 1]) * 0.5
            }
            let px = locX * Float(w) - 0.5, py = locY * Float(h) - 0.5
            let x0 = floor(px), y0 = floor(py)
            let fx = px - x0, fy = py - y0
            let flat = value[start..<(start + h * w)].transposed(1, 0, 2).reshaped([heads * h * w, hd])
            let headBase = (MLXArray(Int32(0)..<Int32(heads)) * Int32(h * w)).reshaped([1, heads, 1])
            let weight = attention[0..., 0..., level, 0...]  // [Q, heads, P]
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let cx = x0 + Float(dx), cy = y0 + Float(dy)
                let bilinear = (dx == 0 ? 1 - fx : fx) * (dy == 0 ? 1 - fy : fy)
                let valid = (cx .>= 0) .&& (cx .<= Float(w - 1)) .&& (cy .>= 0) .&& (cy .<= Float(h - 1))
                let xi = clip(cx, min: 0, max: Float(w - 1)).asType(.int32), yi = clip(cy, min: 0, max: Float(h - 1)).asType(.int32)
                let index = headBase + yi * Int32(w) + xi  // [Q, heads, P]
                let sampled = take(flat, index.flattened(), axis: 0).reshaped([q, heads, points, hd])
                let k = (bilinear * valid.asType(.float32) * weight).expandedDimensions(axis: -1)
                let term = (sampled.asType(.float32) * k).sum(axis: 2)  // [Q, heads, hd]
                output = output.map { $0 + term } ?? term
            }
            start += h * w
        }
        return output_proj(output!.reshaped([q, dModel]).asType(hidden.dtype))
    }
}

final class BiAttention: Module {
    var vision_proj = Linear(dModel, 1024), text_proj = Linear(dModel, 1024)
    var values_vision_proj = Linear(dModel, 1024), values_text_proj = Linear(dModel, 1024)
    var out_vision_proj = Linear(1024, dModel), out_text_proj = Linear(1024, dModel)

    /// GroundingDinoBiMultiHeadAttention, 4 heads of 256, no masks (one image, one prompt).
    func callAsFunction(_ v: MLXArray, _ t: MLXArray) -> (MLXArray, MLXArray) {
        let nh = 4, hd = 256, nv = v.dim(0), nt = t.dim(0)
        func split(_ x: MLXArray, _ n: Int) -> MLXArray { x.reshaped([n, nh, hd]).transposed(1, 0, 2) }
        let vq = split(vision_proj(v) * pow(Float(hd), -0.5), nv)
        let tk = split(text_proj(t), nt)
        let vv = split(values_vision_proj(v), nv), tv = split(values_text_proj(t), nt)
        var attn = matmul(vq, tk.transposed(0, 2, 1)).asType(.float32)  // [4, Nv, Nt]
        attn = clip(attn - attn.max(), min: -50000, max: 50000)
        var textAttn = attn.transposed(0, 2, 1)
        textAttn = clip(textAttn - textAttn.max(axis: -1, keepDims: true), min: -50000, max: 50000)
        let textProbs = softmax(textAttn, axis: -1).asType(v.dtype)  // [4, Nt, Nv]
        let visionProbs = softmax(attn, axis: -1).asType(v.dtype)  // [4, Nv, Nt]
        let visionOut = matmul(visionProbs, tv).transposed(1, 0, 2).reshaped([nv, 1024])
        let textOut = matmul(textProbs, vv).transposed(1, 0, 2).reshaped([nt, 1024])
        return (out_vision_proj(visionOut), out_text_proj(textOut))
    }
}

final class FusionLayer: Module {
    var layer_norm_vision = LayerNorm(dimensions: dModel, eps: 1e-5), layer_norm_text = LayerNorm(dimensions: dModel, eps: 1e-5)
    var attn = BiAttention()
    var vision_param = MLXArray.zeros([dModel]), text_param = MLXArray.zeros([dModel])
    func callAsFunction(_ vision: MLXArray, _ text: MLXArray) -> (MLXArray, MLXArray) {
        let v = layer_norm_vision(vision), t = layer_norm_text(text)
        let (dv, dt) = attn(v, t)
        return (v + vision_param * dv, t + text_param * dt)
    }
}

final class TextEnhancerLayer: Module {
    var self_attn = GDAttention(heads: 4)
    var fc1 = Linear(dModel, 1024), fc2 = Linear(1024, dModel)
    var layer_norm_before = LayerNorm(dimensions: dModel, eps: 1e-5), layer_norm_after = LayerNorm(dimensions: dModel, eps: 1e-5)
    func callAsFunction(_ t: MLXArray, mask: MLXArray, position: MLXArray) -> MLXArray {
        let qk = t + position
        var h = layer_norm_before(t + self_attn(qk, qk, t, mask: mask))
        h = layer_norm_after(h + fc2(relu(fc1(h))))
        return h
    }
}

final class DeformableLayer: Module {
    var self_attn = DeformableAttention()
    var self_attn_layer_norm = LayerNorm(dimensions: dModel, eps: 1e-5)
    var fc1 = Linear(dModel, 2048), fc2 = Linear(2048, dModel)
    var final_layer_norm = LayerNorm(dimensions: dModel, eps: 1e-5)
    func callAsFunction(_ x: MLXArray, position: MLXArray, reference: MLXArray, shapes: [(Int, Int)]) -> MLXArray {
        var h = self_attn_layer_norm(x + self_attn(x + position, source: x, reference: reference, shapes: shapes))
        h = final_layer_norm(h + fc2(relu(fc1(h))))
        return h
    }
}

final class EncoderLayer: Module {
    var text_enhancer_layer = TextEnhancerLayer()
    var fusion_layer = FusionLayer()
    var deformable_layer = DeformableLayer()
}

final class Encoder: Module { var layers: [EncoderLayer] = (0..<6).map { _ in EncoderLayer() } }

final class DecoderLayer: Module {
    var self_attn = GDAttention(heads: heads)
    var self_attn_layer_norm = LayerNorm(dimensions: dModel, eps: 1e-5)
    var encoder_attn_text = GDAttention(heads: heads)
    var encoder_attn_text_layer_norm = LayerNorm(dimensions: dModel, eps: 1e-5)
    var encoder_attn = DeformableAttention()
    var encoder_attn_layer_norm = LayerNorm(dimensions: dModel, eps: 1e-5)
    var fc1 = Linear(dModel, 2048), fc2 = Linear(2048, dModel)
    var final_layer_norm = LayerNorm(dimensions: dModel, eps: 1e-5)

    func callAsFunction(_ x: MLXArray, position: MLXArray, reference: MLXArray, vision: MLXArray, text: MLXArray, shapes: [(Int, Int)]) -> MLXArray {
        let qk = x + position
        var h = self_attn_layer_norm(x + self_attn(qk, qk, x))
        h = encoder_attn_text_layer_norm(h + encoder_attn_text(h + position, text, text))
        h = encoder_attn_layer_norm(h + encoder_attn(h + position, source: vision, reference: reference, shapes: shapes))
        h = final_layer_norm(h + fc2(relu(fc1(h))))
        return h
    }
}

final class Decoder: Module {
    var layers: [DecoderLayer] = (0..<6).map { _ in DecoderLayer() }
    var layer_norm = LayerNorm(dimensions: dModel, eps: 1e-5)
    var reference_points_head = MLPHead([2 * dModel, dModel, dModel])
    var bbox_embed: [MLPHead] = [MLPHead([dModel, dModel, dModel, 4])]
}

final class ConvModel: Module { var conv_encoder = ConvEncoder() }
final class ConvEncoder: Module { var model = SwinBackbone() }

final class InputProjection: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "norm") var norm: GroupNorm
    init(_ inC: Int, kernel: Int, stride: Int, padding: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: inC, outputChannels: dModel, kernelSize: .init(kernel), stride: .init(stride), padding: .init(padding))
        _norm.wrappedValue = GroupNorm(groupCount: 32, dimensions: dModel, pytorchCompatible: true)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { norm(conv(x.expandedDimensions(axis: 0)))[0] }
}

final class GDModel: Module {
    var backbone = ConvModel()
    var input_proj_vision: [InputProjection] = [
        InputProjection(256, kernel: 1, stride: 1, padding: 0), InputProjection(512, kernel: 1, stride: 1, padding: 0),
        InputProjection(1024, kernel: 1, stride: 1, padding: 0), InputProjection(1024, kernel: 3, stride: 2, padding: 1),
    ]
    var text_backbone = BertModel()
    var text_projection = Linear(768, dModel)
    var query_position_embeddings = Embedding(embeddingCount: queriesCount, dimensions: dModel)
    var encoder = Encoder()
    var decoder = Decoder()
    var level_embed = MLXArray.zeros([levels, dModel])
    var enc_output = Linear(dModel, dModel)
    var enc_output_norm = LayerNorm(dimensions: dModel, eps: 1e-5)
    var encoder_output_bbox_embed = MLPHead([dModel, dModel, dModel, 4])
}

final class GroundingDino: Module {
    var model = GDModel()

    /// generate_masks_with_special_tokens_and_transfer_map: per-phrase attention (additive, [L, L]) and position ids.
    static func textMasks(_ ids: [Int32]) -> (mask: MLXArray, positions: [Int32]) {
        let n = ids.count
        let special = ids.map { specialTokens.contains($0) }
        var prev = [Int](repeating: -1, count: n), next = [Int](repeating: n, count: n)
        var last = -1
        for i in 0..<n { if special[i] { last = i }; prev[i] = last }
        var upcoming = n
        for i in stride(from: n - 1, through: 0, by: -1) { if special[i] { upcoming = i }; next[i] = upcoming }
        let valid = (0..<n).map { next[$0] != 0 && next[$0] != n - 1 && next[$0] != n }
        var mask = [Float](repeating: -65504, count: n * n)
        for i in 0..<n {
            for j in 0..<n where i == j || (next[i] == next[j] && valid[j]) { mask[i * n + j] = 0 }
        }
        let positions = (0..<n).map { i -> Int32 in valid[i] ? Int32(max(i - prev[i] - 1, 0)) : 0 }
        return (MLXArray(mask, [n, n]), positions)
    }

    func callAsFunction(pixels: MLXArray, ids: [Int32], checkpoint: () throws -> Void) throws -> (MLXArray, MLXArray, [String: MLXArray]) {
        let m = model
        // Text: BERT with the phrase mask, projected to 256.
        let (textMask, positions) = GroundingDino.textMasks(ids)
        let bert = m.text_backbone(MLXArray(ids), positions: MLXArray(positions), mask: textMask)
        var text = m.text_projection(bert)  // [L, 256]
        probe("bert", bert); probe("text_proj", text)
        // The maker's encode_sinusoidal_position_embedding casts its result back to the dtype of the int64 position ids,
        // so every value truncates toward zero: only the cos(0) = 1 entries survive. Reproduced on purpose.
        let textPosition = sinusoidal(MLXArray(positions).asType(.float32).reshaped([ids.count, 1]), feats: dModel)
            .asType(.int32).asType(text.dtype)

        // Vision: Swin stages 2-4, input projections, a fourth level from stage 4.
        let features = try m.backbone.conv_encoder.model(pixels, checkpoint: checkpoint)
        var maps = zip(features, m.input_proj_vision.prefix(3)).map { $1($0) }
        maps.append(m.input_proj_vision[3](features[2]))
        let shapes = maps.map { ($0.dim(0), $0.dim(1)) }
        for (i, f) in features.enumerated() { probe("backbone\(i)", f) }
        for (i, f) in maps.enumerated() { probe("proj\(i)", f) }
        var source = concatenated(maps.map { $0.reshaped([-1, dModel]) }, axis: 0)
        let position = concatenated(shapes.enumerated().map { level, s in
            imageSine(height: s.0, width: s.1) + m.level_embed[level].asType(.float32)
        }, axis: 0).asType(source.dtype)
        // Encoder reference points: patch centres, the same for every level.
        var refs = [Float]()
        for (h, w) in shapes {
            for y in 0..<h { for x in 0..<w { refs += [(Float(x) + 0.5) / Float(w), (Float(y) + 0.5) / Float(h)] } }
        }
        let encoderRef = MLXArray(refs, [refs.count / 2, 2])
        eval(source, text)
        try checkpoint()

        probe("vision_pos", position)
        for (i, layer) in m.encoder.layers.enumerated() {
            (source, text) = layer.fusion_layer(source, text)
            probe("enc\(i)_fusion_v", source); probe("enc\(i)_fusion_t", text)
            text = layer.text_enhancer_layer(text, mask: textMask, position: textPosition)
            probe("enc\(i)_text", text)
            source = layer.deformable_layer(source, position: position, reference: encoderRef, shapes: shapes)
            probe("enc\(i)_vision", source)
            eval(source, text)
            try checkpoint()
        }

        // Two-stage query selection.
        var proposals = [Float](), validity = [Bool]()
        for (level, (h, w)) in shapes.enumerated() {
            let size = 0.05 * pow(2, Float(level))
            for y in 0..<h {
                for x in 0..<w {
                    let p = [(Float(x) + 0.5) / Float(w), (Float(y) + 0.5) / Float(h), size, size]
                    let ok = p.allSatisfy { $0 > 0.01 && $0 < 0.99 }
                    validity.append(ok)
                    proposals += ok ? p.map { log($0 / (1 - $0)) } : [Float](repeating: .infinity, count: 4)
                }
            }
        }
        let valid = MLXArray(validity).reshaped([-1, 1])
        let objectQuery = m.enc_output_norm(m.enc_output(which(valid, source, MLXArray.zeros(like: source))))
        // Scores in the model's float16, as the maker computes them; torch.topk keeps equal scores in index order.
        let encScore = matmul(objectQuery, text.transposed(1, 0)).max(axis: -1).asType(.float32).asArray(Float.self)  // [N]
        let order = encScore.indices.sorted { encScore[$0] != encScore[$1] ? encScore[$0] > encScore[$1] : $0 < $1 }
        guard order.count >= queriesCount else {
            throw RunnerError.invalid("The image's aspect is too extreme: it gives \(order.count) feature positions, fewer than the \(queriesCount) the model needs.")
        }
        let top = MLXArray(order.prefix(queriesCount).map(Int32.init))
        let encCoords = m.encoder_output_bbox_embed(objectQuery).asType(.float32) + MLXArray(proposals, [proposals.count / 4, 4])
        var reference = sigmoid(take(encCoords, top, axis: 0))  // [900, 4]
        var hidden = m.query_position_embeddings.weight.asType(source.dtype)

        // Decoder with iterative box refinement.
        var lastInput = reference
        for layer in m.decoder.layers {
            lastInput = reference
            let queryPos = m.decoder.reference_points_head(sinusoidal(reference, feats: dModel / 2).asType(hidden.dtype))
            hidden = layer(hidden, position: queryPos, reference: reference, vision: source, text: text, shapes: shapes)
            reference = sigmoid(m.decoder.bbox_embed[0](hidden).asType(.float32) + logitEps(reference))
            eval(hidden, reference)
            try checkpoint()
        }
        let final = m.decoder.layer_norm(hidden)
        let logits = matmul(final, text.transposed(1, 0)).asType(.float32)  // [900, L]
        let boxes = sigmoid(m.decoder.bbox_embed[0](final).asType(.float32) + logitEps(lastInput))
        var extra = probes
        extra["enc_vision"] = source; extra["enc_text"] = text; extra["topk"] = top.asType(.int32)
        probes = [:]
        return (logits, boxes, extra)
    }
}

/// Reads the image's pixel size from its header, before decoding, and refuses a side over `maxSide` pixels
/// (so a huge or extreme-aspect file is refused without allocating its pixels).
private func checkImageHeader(_ url: URL, maxSide: Int = 8192) throws -> (width: Int, height: Int) {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
          let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
          let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue
    else { throw RunnerError.invalid("\(url.lastPathComponent) is not a readable image.") }
    guard width >= 1, height >= 1, width <= maxSide, height <= maxSide else {
        throw RunnerError.invalid("The image is \(width)x\(height); each side must be 1 to \(maxSide) pixels.")
    }
    return (width, height)
}
