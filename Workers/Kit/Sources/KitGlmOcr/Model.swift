// SPDX-License-Identifier: MIT
// GLM-OCR (zai-org/GLM-OCR): a CogViT vision encoder (24 blocks, 2-D axial RoPE, q/k RMSNorm), a 2x2 conv downsample and
// SwiGLU merger into a 16-layer GLM decoder with 3-D M-RoPE. The network follows mlx-swift-lm's GlmOcr.swift at the app's
// pinned revision bd4b7434 (MIT, itself a port of mlx-vlm's glm_ocr), made self-contained for the kit runner and checked
// against transformers 5.22's modeling_glm_ocr.py. Preprocessing follows Glm46VImageProcessor (smart_resize, bicubic,
// CLIP mean/std, patchify); the prompt is the repo's chat template for the card's "Text Recognition:" task; decoding is
// greedy (generation_config.json: do_sample false) with up to 8192 new tokens.
import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

let modelKind = "local.document"
let modelOps = ["read"]

final class Model {
    let net: GlmOcrNet
    let tokenizer: Tokenizer
    let imageTokenId: Int
    let eosIds: Set<Int>
    let patch: Int, merge: Int, temporal: Int, minPixels: Int, maxPixels: Int
    let mean: [Float], std: [Float]
    let maxNewTokens = 8192
    let special: [String: Int]

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        let config = try Files.json(weightsDir)
        guard let text = config["text_config"] as? [String: Any], let vision = config["vision_config"] as? [String: Any] else {
            throw RunnerError.failed("config.json is not a GLM-OCR config.")
        }
        net = GlmOcrNet(vision: vision, text: text)
        var weights = try Weights.load(weightsDir)
        let layers = text["num_hidden_layers"] as? Int ?? 16
        if let w = weights["model.visual.downsample.weight"] { weights["model.visual.downsample.weight"] = w.transposed(0, 2, 3, 1) }
        // The checkpoint's extra decoder layer (index num_hidden_layers) is the multi-token-prediction head, unused here.
        try Weights.apply(weights, to: net) { key in
            key.hasPrefix("model.language_model.layers.\(layers).") ? nil : key
        }
        let data = try JSONSerialization.jsonObject(with: Data(contentsOf: weightsDir.appendingPathComponent("tokenizer.json")))
        guard let tokenizerData = data as? [NSString: Any] else { throw RunnerError.failed("tokenizer.json is not a JSON object.") }
        tokenizer = try AutoTokenizer.from(
            tokenizerConfig: Config(["tokenizer_class": "PreTrainedTokenizer", "clean_up_tokenization_spaces": false] as [NSString: Any]),
            tokenizerData: Config(tokenizerData))
        var special: [String: Int] = [:]
        for name in ["[gMASK]", "<sop>", "<|user|>", "<|assistant|>", "<|begin_of_image|>", "<|end_of_image|>", "<|image|>"] {
            guard let id = tokenizer.convertTokenToId(name) else { throw RunnerError.failed("tokenizer.json has no \(name).") }
            special[name] = id
        }
        self.special = special
        imageTokenId = config["image_token_id"] as? Int ?? special["<|image|>"]!
        eosIds = Set((text["eos_token_id"] as? [Int]) ?? [59246, 59253])
        let pre = (try? Files.json(weightsDir, "preprocessor_config.json")) ?? [:]
        patch = pre["patch_size"] as? Int ?? 14
        merge = pre["merge_size"] as? Int ?? 2
        temporal = pre["temporal_patch_size"] as? Int ?? 2
        let size = pre["size"] as? [String: Any] ?? [:]
        minPixels = size["shortest_edge"] as? Int ?? 12544
        maxPixels = size["longest_edge"] as? Int ?? 9_633_792
        mean = ((pre["image_mean"] as? [Double]) ?? [0.48145466, 0.4578275, 0.40821073]).map(Float.init)
        std = ((pre["image_std"] as? [Double]) ?? [0.26862954, 0.26130258, 0.27577711]).map(Float.init)
    }

    /// Glm46VImageProcessor smart_resize with num_frames = temporal_factor = 2.
    func targetSize(height h0: Int, width w0: Int) throws -> (Int, Int) {
        let factor = patch * merge
        var height = Double(h0), width = Double(w0)
        if h0 < factor || w0 < factor {
            let scale = max(Double(factor) / height, Double(factor) / width)
            height = Double(Int(height * scale)); width = Double(Int(width * scale))
        }
        guard max(height, width) / min(height, width) <= 200 else {
            throw RunnerError.invalid("image aspect ratio must be under 200:1.")
        }
        func round_(_ x: Double) -> Int { Int((x).rounded(.toNearestOrEven)) }
        var hBar = round_(height / Double(factor)) * factor
        var wBar = round_(width / Double(factor)) * factor
        let t = Double(temporal)
        if t * Double(hBar) * Double(wBar) > Double(maxPixels) {
            let beta = (t * height * width / Double(maxPixels)).squareRoot()
            hBar = max(factor, Int((height / beta / Double(factor)).rounded(.down)) * factor)
            wBar = max(factor, Int((width / beta / Double(factor)).rounded(.down)) * factor)
        } else if t * Double(hBar) * Double(wBar) < Double(minPixels) {
            let beta = (Double(minPixels) / (t * height * width)).squareRoot()
            hBar = Int((height * beta / Double(factor)).rounded(.up)) * factor
            wBar = Int((width * beta / Double(factor)).rounded(.up)) * factor
        }
        return (hBar, wBar)
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        let url = try params.file("image")
        let source = try Pixels.load(url)
        let (rh, rw) = try targetSize(height: source.height, width: source.width)
        let resized = source.resized(width: rw, height: rh, filter: .bicubic)
        let gh = rh / patch, gw = rw / patch
        // Patchify: [H, W, C] -> (gh/m, gw/m, m, m, C, T, p, p) flattened to [gh*gw, C*T*p*p].
        let img = MLXArray(resized.normalized(mean: mean, std: std), [rh, rw, 3])
        var x = img.reshaped([gh / merge, merge, patch, gw / merge, merge, patch, 3]).transposed(0, 3, 1, 4, 6, 2, 5)
        x = expandedDimensions(x, axis: 5)  // (gh/m, gw/m, m, m, C, 1, p, p)
        x = broadcast(x, to: [gh / merge, gw / merge, merge, merge, 3, temporal, patch, patch])
        let pixelValues = x.reshaped([gh * gw, 3 * temporal * patch * patch]).asType(.float16)

        let features = net.visual(pixelValues, gridH: gh, gridW: gw, merge: merge)
        eval(features)
        try params.checkCancelled()

        // Chat template for [image, "Text Recognition:"] with add_generation_prompt.
        let nImage = (gh / merge) * (gw / merge)
        var ids = [special["[gMASK]"]!, special["<sop>"]!, special["<|user|>"]!] + encodePlain("\n")
        ids += [special["<|begin_of_image|>"]!] + Array(repeating: imageTokenId, count: nImage) + [special["<|end_of_image|>"]!]
        ids += encodePlain("Text Recognition:") + [special["<|assistant|>"]!] + encodePlain("\n")
        let start = ids.firstIndex(of: imageTokenId)!

        // 3-D M-RoPE positions: text sequential on all three axes; image tokens (t, row, col) offset by the text before.
        var pt = [Int32](), ph = [Int32](), pw = [Int32]()
        for i in 0 ..< start { pt.append(Int32(i)); ph.append(Int32(i)); pw.append(Int32(i)) }
        let lh = gh / merge, lw = gw / merge
        for r in 0 ..< lh { for c in 0 ..< lw { pt.append(Int32(start)); ph.append(Int32(start + r)); pw.append(Int32(start + c)) } }
        var next = start + max(lh, lw)
        for _ in (start + nImage) ..< ids.count { pt.append(Int32(next)); ph.append(Int32(next)); pw.append(Int32(next)); next += 1 }
        let delta = next - ids.count
        let positions = stacked([MLXArray(pt), MLXArray(ph), MLXArray(pw)], axis: 0).reshaped([3, 1, ids.count])

        let lm = net.languageModel
        var embeds = lm.embedTokens(MLXArray(ids.map(Int32.init)).reshaped([1, -1]))
        embeds = concatenated([embeds[0..., ..<start, 0...], features.expandedDimensions(axis: 0).asType(embeds.dtype),
                               embeds[0..., (start + nImage)..., 0...]], axis: 1)
        var cache: [(MLXArray, MLXArray)?] = Array(repeating: nil, count: lm.layers.count)
        var logits = net.logits(lm(embeds, positions: positions, cache: &cache))
        let firstLogits = logits
        var tokens: [Int] = []
        while tokens.count < maxNewTokens {
            try params.checkCancelled()
            let token = argMax(logits.asType(.float32), axis: -1).item(Int.self)
            tokens.append(token)
            if eosIds.contains(token) { break }
            let p = Int32(ids.count + tokens.count - 1 + delta)
            let pos = broadcast(MLXArray([p]).reshaped([1, 1, 1]), to: [3, 1, 1])
            logits = net.logits(lm(lm.embedTokens(MLXArray([Int32(token)]).reshaped([1, 1])), positions: pos, cache: &cache))
        }
        if let dump = ProcessInfo.processInfo.environment["ARSENAL_PARITY_DIR"] {
            try save(arrays: ["pixel_values": pixelValues.asType(.float32), "image_features": features.asType(.float32),
                              "first_logits": firstLogits.asType(.float32), "input_ids": MLXArray(ids.map(Int32.init)),
                              "tokens": MLXArray(tokens.map(Int32.init))],
                     url: URL(fileURLWithPath: dump).appendingPathComponent("mlx-\(url.deletingPathExtension().lastPathComponent).safetensors"))
        }
        let kept = tokens.filter { !eosIds.contains($0) }
        return ["text": tokenizer.decode(tokens: kept, skipSpecialTokens: true)]
    }

    /// Plain text between special tokens, tokenized without any added BOS/EOS.
    func encodePlain(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }
}

// MARK: network (names follow the checkpoint: model.visual.*, model.language_model.*, lm_head)

func cfg(_ c: [String: Any], _ key: String, _ value: Int) -> Int { c[key] as? Int ?? value }
func epsOf(_ c: [String: Any]) -> Float { Float(c["rms_norm_eps"] as? Double ?? 1e-5) }

final class GlmOcrInner: Module {
    @ModuleInfo(key: "visual") var visual: VisionModel
    @ModuleInfo(key: "language_model") var languageModel: TextModel
    init(vision: [String: Any], text: [String: Any]) {
        _visual.wrappedValue = VisionModel(config: vision)
        _languageModel.wrappedValue = TextModel(config: text)
    }
}

final class GlmOcrNet: Module {
    @ModuleInfo(key: "model") var model: GlmOcrInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    var visual: VisionModel { model.visual }
    var languageModel: TextModel { model.languageModel }
    init(vision: [String: Any], text: [String: Any]) {
        _model.wrappedValue = GlmOcrInner(vision: vision, text: text)
        _lmHead.wrappedValue = Linear(cfg(text, "hidden_size", 1536), cfg(text, "vocab_size", 59392), bias: false)
    }
    /// Last-position logits [vocab].
    func logits(_ hidden: MLXArray) -> MLXArray {
        lmHead(hidden[0..., (hidden.dim(1) - 1)..., 0...]).reshaped([-1])
    }
}

// MARK: vision

func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
}

final class PatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: PatchConv
    init(inChannels: Int, temporal: Int, patch: Int, dim: Int) {
        _proj.wrappedValue = PatchConv(dim: dim, inChannels: inChannels, temporal: temporal, patch: patch)
    }
}

/// Conv3d with kernel = stride = the whole patch: a linear map over each flattened (C, T, p, p) patch.
final class PatchConv: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    init(dim: Int, inChannels: Int, temporal: Int, patch: Int) {
        _weight.wrappedValue = MLXArray.zeros([dim, inChannels, temporal, patch, patch])
        _bias.wrappedValue = MLXArray.zeros([dim])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        matmul(x, weight.reshaped([weight.dim(0), -1]).T) + bias
    }
}

final class VisionAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let heads: Int, headDim: Int
    init(dim: Int, heads: Int, eps: Float) {
        self.heads = heads; headDim = dim / heads
        _qkv.wrappedValue = Linear(dim, dim * 3, bias: true)
        _proj.wrappedValue = Linear(dim, dim, bias: true)
        _qNorm.wrappedValue = RMSNorm(dimensions: dim / heads, eps: eps)
        _kNorm.wrappedValue = RMSNorm(dimensions: dim / heads, eps: eps)
    }
    /// x [N, D]; cos/sin [N, 1, headDim] float32.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let n = x.dim(0)
        let parts = qkv(x).reshaped([n, 3, heads, headDim])
        var q = qNorm(parts[0..., 0, 0..., 0...]), k = kNorm(parts[0..., 1, 0..., 0...])
        let v = parts[0..., 2, 0..., 0...]
        // apply_rotary_pos_emb_vision: in float32, then back to the input dtype.
        q = ((q.asType(.float32) * cos) + (rotateHalf(q.asType(.float32)) * sin)).asType(x.dtype)
        k = ((k.asType(.float32) * cos) + (rotateHalf(k.asType(.float32)) * sin)).asType(x.dtype)
        let o = MLXFast.scaledDotProductAttention(
            queries: q.transposed(1, 0, 2).expandedDimensions(axis: 0), keys: k.transposed(1, 0, 2).expandedDimensions(axis: 0),
            values: v.transposed(1, 0, 2).expandedDimensions(axis: 0), scale: 1 / Float(headDim).squareRoot(), mask: nil)
        return proj(o[0].transposed(1, 0, 2).reshaped([n, -1]))
    }
}

final class VisionMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(dim: Int, hidden: Int, bias: Bool) {
        _gate.wrappedValue = Linear(dim, hidden, bias: bias)
        _up.wrappedValue = Linear(dim, hidden, bias: bias)
        _down.wrappedValue = Linear(hidden, dim, bias: bias)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class VisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "attn") var attn: VisionAttention
    @ModuleInfo(key: "mlp") var mlp: VisionMLP
    init(dim: Int, hidden: Int, heads: Int, eps: Float) {
        _norm1.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        _norm2.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        _attn.wrappedValue = VisionAttention(dim: dim, heads: heads, eps: eps)
        _mlp.wrappedValue = VisionMLP(dim: dim, hidden: hidden, bias: true)
    }
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let x = x + attn(norm1(x), cos: cos, sin: sin)
        return x + mlp(norm2(x))
    }
}

final class PatchMerger: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "post_projection_norm") var norm: LayerNorm
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(dim: Int, context: Int) {
        _proj.wrappedValue = Linear(dim, dim, bias: false)
        _norm.wrappedValue = LayerNorm(dimensions: dim)
        _gate.wrappedValue = Linear(dim, context, bias: false)
        _up.wrappedValue = Linear(dim, context, bias: false)
        _down.wrappedValue = Linear(context, dim, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = gelu(norm(proj(x)))
        return down(silu(gate(h)) * up(h))
    }
}

final class VisionModel: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
    @ModuleInfo(key: "blocks") var blocks: [VisionBlock]
    @ModuleInfo(key: "post_layernorm") var postLayernorm: RMSNorm
    @ModuleInfo(key: "downsample") var downsample: Conv2d
    @ModuleInfo(key: "merger") var merger: PatchMerger
    let headDim: Int, theta: Float, outDim: Int
    init(config c: [String: Any]) {
        let dim = cfg(c, "hidden_size", 1024), heads = cfg(c, "num_heads", 16), eps = epsOf(c)
        let merge = cfg(c, "spatial_merge_size", 2), out = cfg(c, "out_hidden_size", 1536), inCh = cfg(c, "in_channels", 3)
        headDim = dim / heads; outDim = out
        theta = Float(((c["rope_parameters"] as? [String: Any])?["rope_theta"] as? Double) ?? 10000)
        _patchEmbed.wrappedValue = PatchEmbed(inChannels: inCh, temporal: cfg(c, "temporal_patch_size", 2), patch: cfg(c, "patch_size", 14), dim: dim)
        _blocks.wrappedValue = (0 ..< cfg(c, "depth", 24)).map { _ in VisionBlock(dim: dim, hidden: cfg(c, "intermediate_size", 4096), heads: heads, eps: eps) }
        _postLayernorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        _downsample.wrappedValue = Conv2d(inputChannels: dim, outputChannels: out, kernelSize: .init(merge), stride: .init(merge), bias: true)
        _merger.wrappedValue = PatchMerger(dim: out, context: out * inCh)
    }

    /// pixel patches [gh*gw, C*T*p*p] in merge-block order -> image features [gh*gw/m^2, out].
    func callAsFunction(_ pixels: MLXArray, gridH gh: Int, gridW gw: Int, merge m: Int) -> MLXArray {
        var x = patchEmbed.proj(pixels)
        // Axial 2-D RoPE: per patch (row, col) in merge-block order; head_dim/4 frequencies for each axis.
        var rows = [Float](), cols = [Float]()
        for br in 0 ..< gh / m { for bc in 0 ..< gw / m { for i in 0 ..< m { for j in 0 ..< m {
            rows.append(Float(br * m + i)); cols.append(Float(bc * m + j))
        } } } }
        let spatial = headDim / 2
        let invFreq = MLXArray(stride(from: 0, to: spatial, by: 2).map { 1 / pow(theta, Float($0) / Float(spatial)) })
        let fh = MLXArray(rows).reshaped([-1, 1]) * invFreq, fw = MLXArray(cols).reshaped([-1, 1]) * invFreq
        let fhw = concatenated([fh, fw], axis: -1)
        let emb = concatenated([fhw, fhw], axis: -1).expandedDimensions(axis: 1)  // [N, 1, headDim]
        let cos = MLX.cos(emb), sin = MLX.sin(emb)
        for block in blocks { x = block(x, cos: cos, sin: sin); eval(x) }
        x = postLayernorm(x)
        x = downsample(x.reshaped([-1, m, m, x.dim(-1)])).reshaped([-1, outDim])
        return merger(x)
    }
}

// MARK: text

final class TextAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var o: Linear
    let heads: Int, kvHeads: Int, headDim: Int
    init(dim: Int, heads: Int, kvHeads: Int, headDim: Int) {
        self.heads = heads; self.kvHeads = kvHeads; self.headDim = headDim
        _q.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _k.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _v.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _o.wrappedValue = Linear(heads * headDim, dim, bias: false)
    }
}

final class TextMLP: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUp: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(dim: Int, hidden: Int) {
        _gateUp.wrappedValue = Linear(dim, hidden * 2, bias: false)
        _down.wrappedValue = Linear(hidden, dim, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(gateUp(x), parts: 2, axis: -1)
        return down(silu(parts[0]) * parts[1])
    }
}

final class TextLayer: Module {
    @ModuleInfo(key: "self_attn") var attn: TextAttention
    @ModuleInfo(key: "mlp") var mlp: TextMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_self_attn_layernorm") var postSelfAttnNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttnNorm: RMSNorm
    @ModuleInfo(key: "post_mlp_layernorm") var postMlpNorm: RMSNorm
    init(dim: Int, hidden: Int, heads: Int, kvHeads: Int, headDim: Int, eps: Float) {
        _attn.wrappedValue = TextAttention(dim: dim, heads: heads, kvHeads: kvHeads, headDim: headDim)
        _mlp.wrappedValue = TextMLP(dim: dim, hidden: hidden)
        _inputNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        _postSelfAttnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        _postAttnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        _postMlpNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
    }
}

/// rotate_half_llm: interleaved pairs, (-x2, x1) for x1 = x[0::2], x2 = x[1::2].
func rotateHalfInterleaved(_ x: MLXArray) -> MLXArray {
    let d = x.dim(-1)
    let pairs = x.reshaped(Array(x.shape.dropLast()) + [d / 2, 2])
    let x1 = pairs[.ellipsis, 0], x2 = pairs[.ellipsis, 1]
    return stacked([-x2, x1], axis: -1).reshaped(x.shape)
}

final class TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [TextLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    let invFreq: [Float]  // plain floats: an MLXArray property would count as a weight
    let section: [Int]
    init(config c: [String: Any]) {
        let dim = cfg(c, "hidden_size", 1536), eps = epsOf(c), headDim = cfg(c, "head_dim", 128)
        let rope = c["rope_parameters"] as? [String: Any] ?? [:]
        section = rope["mrope_section"] as? [Int] ?? [16, 24, 24]
        let theta = Float(rope["rope_theta"] as? Double ?? 10000)
        let rotary = Int(Double(headDim) * (rope["partial_rotary_factor"] as? Double ?? 1.0))
        invFreq = stride(from: 0, to: rotary, by: 2).map { 1 / pow(theta, Float($0) / Float(rotary)) }
        _embedTokens.wrappedValue = Embedding(embeddingCount: cfg(c, "vocab_size", 59392), dimensions: dim)
        _layers.wrappedValue = (0 ..< cfg(c, "num_hidden_layers", 16)).map { _ in
            TextLayer(dim: dim, hidden: cfg(c, "intermediate_size", 4608), heads: cfg(c, "num_attention_heads", 16),
                      kvHeads: cfg(c, "num_key_value_heads", 8), headDim: headDim, eps: eps)
        }
        _norm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
    }

    /// M-RoPE cos/sin [1, 1, L, rotary] from positions [3, 1, L]: frequency chunks by mrope_section take the T, H, W axes
    /// in turn, then each frequency is repeated twice (interleaved layout).
    func rope(_ positions: MLXArray, dtype: DType) -> (MLXArray, MLXArray) {
        let freqs = positions.asType(.float32).expandedDimensions(axis: -1) * MLXArray(invFreq)  // [3, 1, L, F]
        var chunks: [MLXArray] = []
        var start = 0
        for (i, s) in section.enumerated() {
            chunks.append(freqs[i % 3, 0..., 0..., start ..< (start + s)])
            start += s
        }
        let f = concatenated(chunks, axis: -1)  // [1, L, F]
        let rep = stacked([f, f], axis: -1).reshaped([f.dim(0), f.dim(1), f.dim(2) * 2])  // repeat_interleave(2)
        return (MLX.cos(rep).asType(dtype).expandedDimensions(axis: 1), MLX.sin(rep).asType(dtype).expandedDimensions(axis: 1))
    }

    /// embeds [1, L, D] at `positions` [3, 1, L], causal with the cache -> final-norm hidden states [1, L, D].
    func callAsFunction(_ embeds: MLXArray, positions: MLXArray, cache: inout [(MLXArray, MLXArray)?]) -> MLXArray {
        var h = embeds
        let l = h.dim(1)
        let (cos, sin) = rope(positions, dtype: h.dtype)
        let past = cache[0]?.0.dim(2) ?? 0
        var mask: MLXArray? = nil
        if l > 1 {
            let rows = MLXArray((0 ..< l).map { Int32($0 + past) }).reshaped([l, 1])
            let cols = MLXArray((0 ..< (past + l)).map(Int32.init)).reshaped([1, past + l])
            mask = MLX.where(cols .<= rows, MLXArray(Float(0)), MLXArray(-Float.infinity)).asType(h.dtype)
        }
        for (i, layer) in layers.enumerated() {
            let a = layer.attn
            let x = layer.inputNorm(h)
            var q = a.q(x).reshaped([1, l, a.heads, a.headDim]).transposed(0, 2, 1, 3)
            var k = a.k(x).reshaped([1, l, a.kvHeads, a.headDim]).transposed(0, 2, 1, 3)
            var v = a.v(x).reshaped([1, l, a.kvHeads, a.headDim]).transposed(0, 2, 1, 3)
            q = q * cos + rotateHalfInterleaved(q) * sin
            k = k * cos + rotateHalfInterleaved(k) * sin
            if let (pk, pv) = cache[i] { k = concatenated([pk, k], axis: 2); v = concatenated([pv, v], axis: 2) }
            cache[i] = (k, v)
            let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / Float(a.headDim).squareRoot(), mask: mask)
            h = h + layer.postSelfAttnNorm(a.o(o.transposed(0, 2, 1, 3).reshaped([1, l, -1])))
            h = h + layer.postMlpNorm(layer.mlp(layer.postAttnNorm(h)))
        }
        return norm(h)
    }
}
