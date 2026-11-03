// SPDX-License-Identifier: MIT
// OWLv2 (google/owlv2-base-patch16-ensemble) open-vocabulary detection, ported from transformers' modeling_owlv2.py
// (Owlv2ForObjectDetection) and image_processing_owlv2.py (TorchvisionBackend): pad to a square with black, Gaussian
// anti-alias blur when shrinking, bilinear resize to 960, CLIP normalization; one box and one score per 16-pixel patch.
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN

let modelKind = "perception"
let modelOps = ["detect"]

private let side = 960, patch = 16, grid = 60
private let visionWidth = 768, visionHeads = 12, visionMLP = 3072, visionLayers = 12
private let textWidth = 512, textHeads = 8, textMLP = 2048, textLayers = 12, textContext = 16
private let projection = 512
private let mean: [Float] = [0.48145466, 0.4578275, 0.40821073], std: [Float] = [0.26862954, 0.26130258, 0.27577711]

final class Model {
    let net: OwlV2Detector
    let tokenizer: CLIPTokenizer
    let boxBias: MLXArray

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        net = OwlV2Detector()
        var weights = try Weights.load(weightsDir)
        let key = "owlv2.vision_model.embeddings.patch_embedding.weight"
        if let w = weights[key] { weights[key] = w.transposed(0, 2, 3, 1) }
        try Weights.apply(weights, to: net)
        tokenizer = try CLIPTokenizer(weightsDir)
        boxBias = OwlV2Detector.boxBias()
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        guard op == "detect" else { throw RunnerError.invalid("Unknown op \(op).") }
        let queries = try Queries.from(params, limit: 512)
        let threshold = Float(params.double("threshold", default: 0.3))
        let file = try params.file("image")
        _ = try checkImageHeader(file)
        let photo = try Images.load(file)
        let pixels = Preprocess.owlv2(photo.pixels)
        try params.checkCancelled()

        let ids = queries.map { tokenizer.encode($0, maxLength: textContext) }
        let queryEmbeds = net.textEmbeds(ids)  // [Q, 512], normalized
        let feats = try net.imageFeatures(pixels.asType(.float16).expandedDimensions(axis: 0), checkpoint: { try params.checkCancelled() })
        let logits = net.classLogits(feats, queries: queryEmbeds)[0]  // [3600, Q] float32
        let boxes = net.boxes(feats, bias: boxBias)[0]  // [3600, 4] cx cy w h of the padded square
        eval(logits, boxes)
        if let dump = ProcessInfo.processInfo.environment["ARSENAL_PARITY_DIR"] {
            try save(arrays: ["pixel_values": pixels, "logits": logits, "pred_boxes": boxes],
                     url: URL(fileURLWithPath: dump).appendingPathComponent("owlv2-\(UUID().uuidString.prefix(8)).safetensors"))
        }
        // post_process_object_detection: sigmoid of the best query's logit per box, boxes scaled by the padded side.
        let best = logits.max(axis: -1), label = argMax(logits, axis: -1)
        let scores = sigmoid(best).asArray(Float.self), labels = label.asArray(Int32.self)
        let b = boxes.asArray(Float.self)
        let scale = Float(max(photo.width, photo.height))
        var found: [Detection] = []
        for i in 0..<scores.count where scores[i] > threshold {
            let cx = b[i * 4], cy = b[i * 4 + 1], w = b[i * 4 + 2], h = b[i * 4 + 3]
            found.append(Detection(
                label: queries[Int(labels[i])], score: scores[i],
                x0: (cx - w / 2) * scale, y0: (cy - h / 2) * scale, x1: (cx + w / 2) * scale, y1: (cy + h / 2) * scale))
        }
        return Reply.detections(found, width: photo.width, height: photo.height)
    }
}

// MARK: - Preprocessing (Owlv2ImageProcessor, torchvision backend)

enum Preprocess {
    /// [H, W, 3] 0...1 → [960, 960, 3] normalized float32.
    static func owlv2(_ image: MLXArray) -> MLXArray {
        let h = image.dim(0), w = image.dim(1), s = max(h, w)
        var square = image.asType(.float32)
        if w < s { square = concatenated([square, MLXArray.zeros([h, s - w, 3])], axis: 1) }
        if h < s { square = concatenated([square, MLXArray.zeros([s - h, s, 3])], axis: 0) }
        // resize(): anti_aliasing sigma = max(0, (factor - 1) / 2), kernel 2 * ceil(3 sigma) + 1, reflect padding,
        // then bilinear (align_corners=False, antialias=False). Blur and resample are folded into one gather per tap.
        let factor = Float(s) / Float(side)
        let sigma = max(0, (factor - 1) / 2)
        var kernel: [Float] = [1]
        if sigma > 0 {
            let size = 2 * Int((3 * sigma).rounded(.up)) + 1
            let half = Float(size - 1) * 0.5
            let pdf = (0..<size).map { i -> Float in
                let x = -half + Float(i) * (size > 1 ? 2 * half / Float(size - 1) : 0)
                return exp(-0.5 * (x / sigma) * (x / sigma))
            }
            let total = pdf.reduce(0, +)
            kernel = pdf.map { $0 / total }
        }
        let rows = filterResample(square, axis: 0, from: s, to: side, kernel: kernel)
        let both = filterResample(rows, axis: 1, from: s, to: side, kernel: kernel)
        return Images.normalize(both, mean: mean, std: std)
    }

    /// Along `axis`: Gaussian blur (reflect padding) sampled at the bilinear taps of an n → m resize.
    static func filterResample(_ x: MLXArray, axis: Int, from n: Int, to m: Int, kernel: [Float]) -> MLXArray {
        let half = kernel.count / 2
        func reflect(_ i: Int) -> Int32 {
            var j = i
            if j < 0 { j = -j }
            if j >= n { j = 2 * (n - 1) - j }
            return Int32(min(max(j, 0), n - 1))
        }
        let scale = Double(n) / Double(m)
        var lower = [Int](), upper = [Int](), t = [Float]()
        for i in 0..<m {
            let source = max((Double(i) + 0.5) * scale - 0.5, 0)
            let lo = min(Int(source), n - 1)
            lower.append(lo); upper.append(min(lo + 1, n - 1)); t.append(Float(source - Double(lo)))
        }
        var shape = [Int](repeating: 1, count: x.ndim); shape[axis] = m
        let tt = MLXArray(t).reshaped(shape)
        var out: MLXArray? = nil
        for (k, weight) in kernel.enumerated() {
            let lo = take(x, MLXArray(lower.map { reflect($0 + k - half) }), axis: axis)
            let hi = take(x, MLXArray(upper.map { reflect($0 + k - half) }), axis: axis)
            let term = (lo * (1 - tt) + hi * tt) * weight
            out = out.map { $0 + term } ?? term
            if k % 4 == 3 { eval(out!) }
        }
        return out!
    }
}

// MARK: - Network (modeling_owlv2.py), channels-last

private func quickGelu(_ x: MLXArray) -> MLXArray { x * sigmoid(1.702 * x) }

final class Attention: Module {
    var q_proj: Linear, k_proj: Linear, v_proj: Linear, out_proj: Linear
    let heads: Int
    init(_ width: Int, heads: Int) {
        q_proj = Linear(width, width); k_proj = Linear(width, width); v_proj = Linear(width, width); out_proj = Linear(width, width)
        self.heads = heads
        super.init()
    }
    func callAsFunction(_ x: MLXArray, causal: Bool = false) -> MLXArray {
        let b = x.dim(0), n = x.dim(1), width = x.dim(2), headDim = width / heads
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([b, n, heads, headDim]).transposed(0, 2, 1, 3) }
        let out = MLXFast.scaledDotProductAttention(
            queries: split(q_proj(x)), keys: split(k_proj(x)), values: split(v_proj(x)),
            scale: pow(Float(headDim), -0.5), mask: causal ? .causal : .none)
        return out_proj(out.transposed(0, 2, 1, 3).reshaped([b, n, width]))
    }
}

final class MLP: Module {
    var fc1: Linear, fc2: Linear
    init(_ width: Int, _ hidden: Int) { fc1 = Linear(width, hidden); fc2 = Linear(hidden, width); super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(quickGelu(fc1(x))) }
}

final class Block: Module {
    var self_attn: Attention, mlp: MLP, layer_norm1: LayerNorm, layer_norm2: LayerNorm
    init(_ width: Int, heads: Int, hidden: Int) {
        self_attn = Attention(width, heads: heads); mlp = MLP(width, hidden)
        layer_norm1 = LayerNorm(dimensions: width, eps: 1e-5); layer_norm2 = LayerNorm(dimensions: width, eps: 1e-5)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, causal: Bool = false) -> MLXArray {
        let h = x + self_attn(layer_norm1(x), causal: causal)
        return h + mlp(layer_norm2(h))
    }
}

final class Encoder: Module {
    var layers: [Block]
    init(count: Int, width: Int, heads: Int, hidden: Int) {
        layers = (0..<count).map { _ in Block(width, heads: heads, hidden: hidden) }
        super.init()
    }
}

final class VisionEmbeddings: Module {
    var class_embedding = MLXArray.zeros([visionWidth])
    var patch_embedding = Conv2d(inputChannels: 3, outputChannels: visionWidth, kernelSize: .init(patch), stride: .init(patch), bias: false)
    var position_embedding = Embedding(embeddingCount: grid * grid + 1, dimensions: visionWidth)
}

final class VisionModel: Module {
    var embeddings = VisionEmbeddings()
    var pre_layernorm = LayerNorm(dimensions: visionWidth, eps: 1e-5)
    var encoder = Encoder(count: visionLayers, width: visionWidth, heads: visionHeads, hidden: visionMLP)
    var post_layernorm = LayerNorm(dimensions: visionWidth, eps: 1e-5)
}

final class TextEmbeddings: Module {
    var token_embedding = Embedding(embeddingCount: 49408, dimensions: textWidth)
    var position_embedding = Embedding(embeddingCount: textContext, dimensions: textWidth)
}

final class TextModel: Module {
    var embeddings = TextEmbeddings()
    var encoder = Encoder(count: textLayers, width: textWidth, heads: textHeads, hidden: textMLP)
    var final_layer_norm = LayerNorm(dimensions: textWidth, eps: 1e-5)
}

final class OwlV2: Module {
    var text_model = TextModel()
    var vision_model = VisionModel()
    var text_projection = Linear(textWidth, projection, bias: false)
    var visual_projection = Linear(visionWidth, projection, bias: false)
    var logit_scale = MLXArray(Float(0))
}

final class PredictionHead: Module {
    var dense0: Linear, dense1: Linear, dense2: Linear
    init(out: Int) { dense0 = Linear(visionWidth, visionWidth); dense1 = Linear(visionWidth, visionWidth); dense2 = Linear(visionWidth, out); super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { dense2(gelu(dense1(gelu(dense0(x))))) }
}

final class ClassHead: Module {
    var dense0 = Linear(visionWidth, textWidth)
    var logit_shift = Linear(visionWidth, 1)
    var logit_scale = Linear(visionWidth, 1)
}

final class OwlV2Detector: Module {
    var owlv2 = OwlV2()
    var class_head = ClassHead()
    var box_head = PredictionHead(out: 4)
    var objectness_head = PredictionHead(out: 1)  // loaded with the checkpoint; text-query detection does not use it
    var layer_norm = LayerNorm(dimensions: visionWidth, eps: 1e-5)

    /// get_text_features per query, then text_embeds / ||text_embeds||: [Q, 512] float32.
    func textEmbeds(_ ids: [[Int]]) -> MLXArray {
        let length = ids.map(\.count).max()!
        let padded = ids.flatMap { row in row.map(Int32.init) + [Int32](repeating: 0, count: length - row.count) }
        let tokens = MLXArray(padded, [ids.count, length])
        let text = owlv2.text_model
        var x = text.embeddings.token_embedding(tokens) + text.embeddings.position_embedding(MLXArray(Int32(0)..<Int32(length)))
        for layer in text.encoder.layers { x = layer(x, causal: true) }  // causal: padding after EOS never reaches EOS
        x = text.final_layer_norm(x)
        // pooled at the first end-of-text token (the largest id in the row)
        let eos = ids.enumerated().map { Int32($0.offset * length + $0.element.firstIndex(of: $0.element.max()!)!) }
        let pooled = take(x.reshaped([ids.count * length, textWidth]), MLXArray(eos), axis: 0)
        let embeds = owlv2.text_projection(pooled).asType(.float32)
        return embeds / sqrt((embeds * embeds).sum(axis: -1, keepDims: true))
    }

    /// image_text_embedder's image side: post_layernorm, patch tokens times the class token, layer_norm. [1, 3600, 768].
    func imageFeatures(_ pixels: MLXArray, checkpoint: () throws -> Void) rethrows -> MLXArray {
        let vision = owlv2.vision_model
        let b = pixels.dim(0)
        let patches = vision.embeddings.patch_embedding(pixels).reshaped([b, grid * grid, visionWidth])
        let cls = broadcast(vision.embeddings.class_embedding.reshaped([1, 1, visionWidth]), to: [b, 1, visionWidth])
        var x = concatenated([cls, patches], axis: 1) + vision.embeddings.position_embedding.weight.expandedDimensions(axis: 0)
        x = vision.pre_layernorm(x)
        for layer in vision.encoder.layers {
            x = layer(x)
            eval(x)
            try checkpoint()
        }
        x = vision.post_layernorm(x)
        return layer_norm(x[0..., 1..., 0...] * x[0..., ..<1, 0...])
    }

    /// Owlv2ClassPredictionHead with every query valid: [1, 3600, Q] float32.
    func classLogits(_ feats: MLXArray, queries: MLXArray) -> MLXArray {
        let image = class_head.dense0(feats).asType(.float32)
        let imageNorm = image / (sqrt((image * image).sum(axis: -1, keepDims: true)) + 1e-6)
        let queryNorm = queries / (sqrt((queries * queries).sum(axis: -1, keepDims: true)) + 1e-6)
        let logits = matmul(imageNorm, queryNorm.transposed(1, 0))
        let shift = class_head.logit_shift(feats).asType(.float32)
        let scale = elu(class_head.logit_scale(feats).asType(.float32)) + 1
        return (logits + shift) * scale
    }

    /// box_predictor: sigmoid(box_head + grid bias), [1, 3600, 4] cx cy w h in 0...1 of the padded square.
    func boxes(_ feats: MLXArray, bias: MLXArray) -> MLXArray {
        sigmoid(box_head(feats).asType(.float32) + bias)
    }

    /// compute_box_bias for the 60×60 grid: logit of each patch's corner position and of the patch size.
    static func boxBias() -> MLXArray {
        var values = [Float]()
        func logit(_ v: Float) -> Float { log(v + 1e-4) - log1p(-v + 1e-4) }
        let size = logit(1 / Float(grid))
        for y in 1...grid {
            for x in 1...grid {
                values += [logit(min(max(Float(x) / Float(grid), 0), 1)), logit(min(max(Float(y) / Float(grid), 0), 1)), size, size]
            }
        }
        return MLXArray(values, [1, grid * grid, 4])
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
