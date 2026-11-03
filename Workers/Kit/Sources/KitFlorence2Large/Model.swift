// SPDX-License-Identifier: MIT
// Florence-2 (microsoft/Florence-2-base and -large): a DaViT image encoder, a learned 2-D position embedding and a
// linear projection into a BART encoder-decoder. Ported from the maker's repo code at the pinned revision
// (modeling_florence2.py: DaViT, Florence2ForConditionalGeneration._encode_image, Florence2Encoder/Decoder;
// processing_florence2.py: task prompts, added tokens, pure-text post-processing). Decoding is the card's
// generate(num_beams=3, max_new_tokens=1024) with the language model's generation settings from config.json
// (no_repeat_ngram_size 3, early_stopping, forced BOS/EOS), as transformers 4.41 applied them.
import Foundation
import MLX
import MLXNN
import Hub
import Tokenizers

let modelKind = "local.vision"
let modelOps = ["caption"]

final class Model {
    let net: FlorenceNet
    let tokenizer: Tokenizer
    let imageSize: Int
    let bosId = 0, padId = 1, eosId = 2
    let startId: Int, numBeams: Int, noRepeatNgram: Int, maxLength: Int, lengthPenalty: Float, earlyStopping: Bool
    let baseVocab: Int
    /// The processor's added special tokens (ids from 50265): <od>, </od>, <ocr>, </ocr>, <loc_0..999>, <cap>, ...
    let addedTokens: [String]

    /// processing_florence2.py task_prompts_without_inputs for the pure-text tasks.
    static let taskPrompts = [
        "<CAPTION>": "What does the image describe?",
        "<DETAILED_CAPTION>": "Describe in detail what is shown in the image.",
        "<MORE_DETAILED_CAPTION>": "Describe with a paragraph what is shown in the image.",
        "<OCR>": "What is the text in the image?",
    ]

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        let config = try Files.json(weightsDir)
        guard let text = config["text_config"] as? [String: Any], let vision = config["vision_config"] as? [String: Any] else {
            throw RunnerError.failed("config.json is not a Florence-2 config.")
        }
        net = FlorenceNet(vision: vision, text: text)
        var weights = try Weights.load(weightsDir)
        for (k, v) in weights where v.ndim == 4 { weights[k] = v.transposed(0, 2, 3, 1) }  // Conv2d [out, in, kh, kw] -> [out, kh, kw, in]
        // vision_tower.blocks.<stage>.<j>. -> vision_tower.layers.<flat index>. (stages run in order)
        let depths = (vision["depths"] as? [Int]) ?? [1, 1, 9, 1]
        let offsets = depths.indices.map { depths[..<$0].reduce(0, +) }
        try Weights.apply(weights, to: net) { key in
            guard key.hasPrefix("vision_tower.blocks.") else { return key }
            let parts = key.split(separator: ".", omittingEmptySubsequences: false)
            guard parts.count > 4, let s = Int(parts[2]), let j = Int(parts[3]) else { return key }
            return (["vision_tower", "layers", String(offsets[s] + j)] + parts[4...].map(String.init)).joined(separator: ".")
        }
        // tokenizer_config.json names no class and config.json says florence2: the processor's tokenizer is
        // BartTokenizerFast, a RoBERTa-style byte-level BPE, which is how tokenizer.json is read here.
        let data = try JSONSerialization.jsonObject(with: Data(contentsOf: weightsDir.appendingPathComponent("tokenizer.json")))
        guard let tokenizerData = data as? [NSString: Any] else { throw RunnerError.failed("tokenizer.json is not a JSON object.") }
        tokenizer = try AutoTokenizer.from(
            tokenizerConfig: Config(["tokenizer_class": "RobertaTokenizer", "clean_up_tokenization_spaces": false] as [NSString: Any]),
            tokenizerData: Config(tokenizerData))
        baseVocab = 50265
        addedTokens = ["<od>", "</od>", "<ocr>", "</ocr>"] + (0 ..< 1000).map { "<loc_\($0)>" } + [
            "<cap>", "</cap>", "<ncap>", "</ncap>", "<dcap>", "</dcap>", "<grounding>", "</grounding>", "<seg>", "</seg>", "<sep>",
            "<region_cap>", "</region_cap>", "<region_to_desciption>", "</region_to_desciption>", "<proposal>", "</proposal>",
            "<poly>", "</poly>", "<and>",
        ]
        let pre = (try? Files.json(weightsDir, "preprocessor_config.json")) ?? [:]
        imageSize = ((pre["size"] as? [String: Any])?["height"] as? Int) ?? 768
        startId = text["decoder_start_token_id"] as? Int ?? 2
        numBeams = Int(ProcessInfo.processInfo.environment["ARSENAL_PARITY_BEAMS"] ?? "") ?? 3
        noRepeatNgram = text["no_repeat_ngram_size"] as? Int ?? 3
        earlyStopping = text["early_stopping"] as? Bool ?? true
        lengthPenalty = 1.0
        // max_new_tokens=1024 after the start token; the decoder has 1024 positions, enough for exactly that.
        maxLength = min(1 + 1024, (text["max_position_embeddings"] as? Int ?? 1024) + 1)
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        let url = try params.file("image")
        let prompt = params.string("prompt", default: "")
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        // A Florence task token picks that task; any other prompt is the start of a <CAPTION> answer.
        var task = "<CAPTION>"
        var prefix: [Int] = []
        if Model.taskPrompts[trimmed] != nil {
            task = trimmed
        } else if !trimmed.isEmpty {
            // swift-transformers' RobertaProcessing adds <s> ... </s> even without special tokens: drop them.
            prefix = tokenizer.encode(text: prompt)
            if prefix.first == bosId { prefix.removeFirst() }
            if prefix.last == eosId { prefix.removeLast() }
            guard !prefix.isEmpty, prefix.count <= 100 else { throw RunnerError.invalid("prompt must be 1 to 100 tokens long.") }
            guard prefix.allSatisfy({ $0 > 3 && $0 < baseVocab }) else { throw RunnerError.invalid("prompt must be plain text.") }
        }
        let inputIds = tokenizer.encode(text: Model.taskPrompts[task]!)  // <s> ... </s>

        // CLIPImageProcessor: convert RGB, PIL bicubic resize to 768x768, /255, ImageNet mean and std.
        let pixels = try Pixels.load(url).resized(width: imageSize, height: imageSize, filter: .bicubic)
        let input = MLXArray(pixels.normalized(mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]),
                             [1, imageSize, imageSize, 3]).asType(.float16)
        let features = net.encodeImage(input)
        try params.checkCancelled()
        let lm = net.languageModel.model
        let embeds = concatenated([features, lm.shared(MLXArray(inputIds.map(Int32.init)).reshaped([1, -1]))], axis: 1)
        let memory = lm.encoder(embeds)
        eval(memory)
        try params.checkCancelled()

        var firstLogits: MLXArray? = nil
        let tokens = try beamSearch(memory: memory, prefix: [startId, bosId] + prefix, params: params) {
            if firstLogits == nil { firstLogits = $0 }
        }
        if let dump = ProcessInfo.processInfo.environment["ARSENAL_PARITY_DIR"] {
            let tag = url.deletingPathExtension().lastPathComponent
                + (prompt.isEmpty ? "" : "-" + String(prompt.lowercased().unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }))
            try save(arrays: ["pixel_values": input.asType(.float32), "image_features": features.asType(.float32),
                              "encoder": memory.asType(.float32), "first_logits": firstLogits!.asType(.float32),
                              "input_ids": MLXArray(inputIds.map(Int32.init)), "tokens": MLXArray(tokens.map(Int32.init))],
                     url: URL(fileURLWithPath: dump).appendingPathComponent("mlx-\(tag).safetensors"))
        }
        // post_process_generation('pure_text'): the decoded text with <s> and </s> removed.
        return ["caption": decode(tokens).replacingOccurrences(of: "<s>", with: "").replacingOccurrences(of: "</s>", with: "")]
    }

    /// transformers GenerationMixin._beam_search (5.x) for one image, with NoRepeatNGram, ForcedBOS and ForcedEOS
    /// logits processors and EOS / max_length stopping. Log-probabilities and scores are float32 like the reference.
    /// `prefix` is the decoder prompt: [decoder_start] alone (forced BOS follows), or [start, bos, prompt tokens].
    func beamSearch(memory: MLXArray, prefix: [Int], params: Params, onFirstLogits: (MLXArray) -> Void) throws -> [Int] {
        let beams = max(1, numBeams), keep = 2 * beams
        let lm = net.languageModel
        let dec = lm.model.decoder
        let cross = dec.crossCaches(memory, beams: beams)
        var cache: [(MLXArray, MLXArray)?] = Array(repeating: nil, count: dec.layers.count)
        // The forced BOS step is part of generation when only the start token is given.
        let promptIds = prefix.count == 2 && prefix[1] == bosId ? [prefix[0]] : prefix
        let promptLen = promptIds.count
        var running = Array(repeating: promptIds, count: beams)
        var runningScores = [Float](repeating: -1e9, count: beams); runningScores[0] = 0
        var finished = Array(repeating: [Int](), count: beams)
        var finishedScores = [Float](repeating: -1e9, count: beams)
        var isFinished = [Bool](repeating: false, count: beams)
        var improvable = true
        var curLen = promptLen
        var logits = lm.prefill(tokens: promptIds, beams: beams, cross: cross, cache: &cache)
        while true {
            try params.checkCancelled()
            if curLen == promptLen { onFirstLogits(logits[0]) }
            var logProbs = logSoftmax(logits.asType(.float32), axis: -1)
            eval(logProbs)
            let vocabSize = logProbs.dim(1)
            var flat = logProbs.asArray(Float.self)
            logProbs = MLXArray(0)
            for b in 0 ..< beams {
                let row = b * vocabSize
                if curLen == 1 || curLen == maxLength - 1 {
                    // ForcedBOSTokenLogitsProcessor (cur_len 1) and ForcedEOSTokenLogitsProcessor (max_length - 1).
                    let forced = curLen == 1 ? bosId : eosId
                    for t in 0 ..< vocabSize { flat[row + t] = t == forced ? 0 : -Float.infinity }
                }
                for t in bannedTokens(running[b]) { flat[row + t] = -Float.infinity }
                let base = runningScores[b]
                for t in 0 ..< vocabSize { flat[row + t] += base }
            }
            let top = topK(flat, keep)
            let candidates = top.map { (beam: $0 / vocabSize, token: $0 % vocabSize, score: flat[$0]) }
            let hits = candidates.map { $0.token == eosId || curLen + 1 >= maxLength }
            let runScores = candidates.indices.map { candidates[$0].score + (hits[$0] ? -1e9 : 0) }
            let nextRun = topK(runScores, beams)
            let allFull = isFinished.allSatisfy { $0 } && earlyStopping
            var mergedSeqs = finished, mergedScores = finishedScores, mergedDone = isFinished
            for (k, c) in candidates.enumerated() {
                var score = c.score / pow(Float(curLen + 1 - promptLen), lengthPenalty)
                if allFull { score += -1e9 }
                if !improvable { score += -1e9 }
                let justDone = hits[k] && k < beams
                if !justDone { score += -1e9 }
                mergedSeqs.append(running[c.beam] + [c.token]); mergedScores.append(score); mergedDone.append(justDone)
            }
            let best = topK(mergedScores, beams)
            finished = best.map { mergedSeqs[$0] }; finishedScores = best.map { mergedScores[$0] }; isFinished = best.map { mergedDone[$0] }
            let order = nextRun.map { candidates[$0].beam }
            running = nextRun.map { running[candidates[$0].beam] + [candidates[$0].token] }
            runningScores = nextRun.map { runScores[$0] }
            if beams > 1 {
                let index = MLXArray(order.map(Int32.init))
                cache = cache.map { $0.map { (k, v) in (k[index], v[index]) } }
            }
            curLen += 1
            let bestPossible = runningScores[0] / pow(Float(curLen - promptLen), lengthPenalty)
            let worst = finishedScores.min()!
            improvable = improvable && isFinished.indices.contains { bestPossible > (isFinished[$0] ? worst : -1e9) }
            let open = !(isFinished.allSatisfy { $0 } && earlyStopping)
            let valid = !hits.allSatisfy { $0 }
            if !(improvable && open && valid) { break }
            logits = lm.step(tokens: running.map { $0.last! }, position: curLen - 1, cross: cross, cache: &cache)
        }
        return finished[0]
    }

    /// NoRepeatNGramLogitsProcessor: tokens that would repeat an n-gram already in the sequence.
    func bannedTokens(_ seq: [Int]) -> [Int] {
        let n = noRepeatNgram
        guard n > 0, seq.count >= n else { return [] }
        let prefix = Array(seq[(seq.count - n + 1)...])
        var banned: [Int] = []
        for i in 0 ... (seq.count - n) where Array(seq[i ..< (i + n - 1)]) == prefix { banned.append(seq[i + n - 1]) }
        return banned
    }

    /// Indices of the k largest values, largest first (ties: lower index first).
    func topK(_ values: [Float], _ k: Int) -> [Int] {
        var best: [Int] = []
        for i in values.indices {
            if best.count == k, values[i] <= values[best[k - 1]] { continue }
            var pos = best.count
            while pos > 0 && values[i] > values[best[pos - 1]] { pos -= 1 }
            best.insert(i, at: pos)
            if best.count > k { best.removeLast() }
        }
        return best
    }

    /// processor.batch_decode(skip_special_tokens=False) as transformers 4.41 ran it: byte-level BPE decode, special
    /// and added tokens written as their text, then clean_up_tokenization_spaces (on by default in 4.41).
    func decode(_ ids: [Int]) -> String {
        var out = ""
        var run: [Int] = []
        func flush() {
            if !run.isEmpty { out += tokenizer.decode(tokens: run, skipSpecialTokens: false); run = [] }
        }
        for id in ids {
            if id >= baseVocab {
                flush()
                if id - baseVocab < addedTokens.count { out += addedTokens[id - baseVocab] }
            } else if id <= 3 {
                flush()
                out += ["<s>", "<pad>", "</s>", "<unk>"][id]
            } else {
                run.append(id)
            }
        }
        flush()
        for (from, to) in [(" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"), (" n't", "n't"), (" 'm", "'m"),
                           (" 's", "'s"), (" 've", "'ve"), (" 're", "'re")] {
            out = out.replacingOccurrences(of: from, with: to)
        }
        return out
    }
}

// MARK: network (names follow the checkpoint)

func cfg(_ c: [String: Any], _ key: String, _ value: Int) -> Int { c[key] as? Int ?? value }

/// [B, L, H*D] -> [B, H, L, D]
func heads(_ x: MLXArray, _ n: Int) -> MLXArray {
    x.reshaped([x.dim(0), x.dim(1), n, x.dim(2) / n]).transposed(0, 2, 1, 3)
}
func merge(_ x: MLXArray) -> MLXArray {
    x.transposed(0, 2, 1, 3).reshaped([x.dim(0), x.dim(2), x.dim(1) * x.dim(3)])
}

final class FlorenceNet: Module {
    @ModuleInfo(key: "vision_tower") var visionTower: DaViT
    @ParameterInfo(key: "image_projection") var imageProjection: MLXArray
    @ModuleInfo(key: "image_proj_norm") var imageProjNorm: LayerNorm
    @ModuleInfo(key: "image_pos_embed") var imagePosEmbed: PosEmbed2D
    @ModuleInfo(key: "visual_temporal_embed") var temporalEmbed: TemporalEmbed
    @ModuleInfo(key: "language_model") var languageModel: BartForGeneration

    init(vision: [String: Any], text: [String: Any]) {
        let dims = (vision["dim_embed"] as? [Int]) ?? [128, 256, 512, 1024]
        let proj = cfg(vision, "projection_dim", 768)
        let maxPos = ((vision["image_pos_embed"] as? [String: Any])?["max_pos_embeddings"] as? Int) ?? 50
        let maxT = ((vision["visual_temporal_embedding"] as? [String: Any])?["max_temporal_embeddings"] as? Int) ?? 100
        _visionTower.wrappedValue = DaViT(config: vision)
        _imageProjection.wrappedValue = MLXArray.zeros([dims.last!, proj])
        _imageProjNorm.wrappedValue = LayerNorm(dimensions: proj)
        _imagePosEmbed.wrappedValue = PosEmbed2D(dim: dims.last!, numPos: maxPos)
        _temporalEmbed.wrappedValue = TemporalEmbed(dim: dims.last!, maxLen: maxT)
        _languageModel.wrappedValue = BartForGeneration(config: text)
    }

    /// Florence2ForConditionalGeneration._encode_image for one still image (T = 1): DaViT features plus the 2-D and
    /// temporal position embeddings, then [spatial average token, all tokens] through the projection and its norm.
    func encodeImage(_ pixels: MLXArray) -> MLXArray {
        let (x0, h, w) = visionTower(pixels)
        let c = x0.dim(2)
        let pos = imagePosEmbed(h: h, w: w).reshaped([1, h * w, c])
        let x = x0 + pos + temporalEmbed.posIdxToEmbed[0 ..< 1].reshaped([1, 1, c])
        let pooled = x.mean(axis: 1, keepDims: true)
        let tokens = concatenated([pooled, x], axis: 1)
        return imageProjNorm(matmul(tokens, imageProjection))
    }
}

final class PosEmbed2D: Module {
    @ModuleInfo(key: "row_embeddings") var rows: Embedding
    @ModuleInfo(key: "column_embeddings") var columns: Embedding
    init(dim: Int, numPos: Int) {
        _rows.wrappedValue = Embedding(embeddingCount: numPos, dimensions: dim / 2)
        _columns.wrappedValue = Embedding(embeddingCount: numPos, dimensions: dim - dim / 2)
    }
    /// [h, w, dim]: column embedding of x, then row embedding of y.
    func callAsFunction(h: Int, w: Int) -> MLXArray {
        let x = columns(MLXArray((0 ..< w).map(Int32.init)))  // [w, d1]
        let y = rows(MLXArray((0 ..< h).map(Int32.init)))  // [h, d2]
        let xb = broadcast(x.expandedDimensions(axis: 0), to: [h, w, x.dim(1)])
        let yb = broadcast(y.expandedDimensions(axis: 1), to: [h, w, y.dim(1)])
        return concatenated([xb, yb], axis: -1)
    }
}

final class TemporalEmbed: Module {
    @ParameterInfo(key: "pos_idx_to_embed") var posIdxToEmbed: MLXArray
    init(dim: Int, maxLen: Int) { _posIdxToEmbed.wrappedValue = MLXArray.zeros([maxLen, dim]) }
}

// MARK: DaViT

final class ConvEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    let preNorm: Bool
    init(patch: Int, inChans: Int, dim: Int, stride: Int, padding: Int, preNorm: Bool) {
        self.preNorm = preNorm
        _proj.wrappedValue = Conv2d(inputChannels: inChans, outputChannels: dim, kernelSize: .init(patch), stride: .init(stride), padding: .init(padding))
        _norm.wrappedValue = LayerNorm(dimensions: preNorm ? inChans : dim)
    }
    /// x: [B, H, W, C] -> [B, H', W', dim]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        if preNorm { x = norm(x) }
        x = proj(x)
        if !preNorm { x = norm(x) }
        return x
    }
}

final class DepthWise: Module {
    @ModuleInfo(key: "dw") var dw: Conv2d
    init(dim: Int) {
        _dw.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 3, stride: 1, padding: 1, groups: dim)
    }
}

/// PreNorm(None, DepthWiseConv2d): x + dw(x).
final class ConvResidual: Module {
    @ModuleInfo(key: "fn") var fn: DepthWise
    init(dim: Int) { _fn.wrappedValue = DepthWise(dim: dim) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x + fn.dw(x) }
}

final class MlpNet: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    init(dim: Int, hidden: Int) {
        _fc1.wrappedValue = Linear(dim, hidden)
        _fc2.wrappedValue = Linear(hidden, dim)
    }
}

final class Mlp: Module {
    @ModuleInfo(key: "net") var net: MlpNet
    init(dim: Int, hidden: Int) { _net.wrappedValue = MlpNet(dim: dim, hidden: hidden) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { net.fc2(gelu(net.fc1(x))) }
}

final class FfnResidual: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "fn") var fn: Mlp
    init(dim: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: dim)
        _fn.wrappedValue = Mlp(dim: dim, hidden: dim * 4)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x + fn(norm(x)) }
}

final class WindowAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    let nHeads: Int, window: Int
    init(dim: Int, nHeads: Int, window: Int) {
        self.nHeads = nHeads; self.window = window
        _qkv.wrappedValue = Linear(dim, dim * 3)
        _proj.wrappedValue = Linear(dim, dim)
    }
    /// x: [B, H, W, C]; non-overlapping windows, zero-padded at the bottom and right like the maker's F.pad.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3), ws = window
        let padB = (ws - h % ws) % ws, padR = (ws - w % ws) % ws
        var y = x
        if padB > 0 || padR > 0 { y = padded(y, widths: [.init(0), .init((0, padB)), .init((0, padR)), .init(0)]) }
        let hp = h + padB, wp = w + padR
        y = y.reshaped([b, hp / ws, ws, wp / ws, ws, c]).transposed(0, 1, 3, 2, 4, 5).reshaped([-1, ws * ws, c])
        let n = y.dim(0), d = c / nHeads
        let qkv = qkv(y).reshaped([n, ws * ws, 3, nHeads, d]).transposed(2, 0, 3, 1, 4)
        let o = MLXFast.scaledDotProductAttention(queries: qkv[0], keys: qkv[1], values: qkv[2], scale: 1 / Float(d).squareRoot(), mask: nil)
        y = proj(merge(o))
        y = y.reshaped([b, hp / ws, wp / ws, ws, ws, c]).transposed(0, 1, 3, 2, 4, 5).reshaped([b, hp, wp, c])
        if padB > 0 || padR > 0 { y = y[0..., 0 ..< h, 0 ..< w, 0...] }
        return y
    }
}

final class ChannelAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    let groups: Int
    init(dim: Int, groups: Int) {
        self.groups = groups
        _qkv.wrappedValue = Linear(dim, dim * 3)
        _proj.wrappedValue = Linear(dim, dim)
    }
    /// x: [B, N, C]. Attention across channels within each group: softmax((q * N^-0.5)^T k) applied to v^T.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), n = x.dim(1), c = x.dim(2), g = c / groups
        let qkv = qkv(x).reshaped([b, n, 3, groups, g]).transposed(2, 0, 3, 1, 4)  // [3, B, G, N, g]
        let q = qkv[0] * MLXArray(1 / Float(n).squareRoot()).asType(x.dtype)
        let k = qkv[1], v = qkv[2]
        let attention = softmax(matmul(q.transposed(0, 1, 3, 2), k), axis: -1, precise: true)  // [B, G, g, g]
        let y = matmul(attention, v.transposed(0, 1, 3, 2)).transposed(0, 1, 3, 2)  // [B, G, N, g]
        return proj(y.transposed(0, 2, 1, 3).reshaped([b, n, c]))
    }
}

final class WindowResidual: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "fn") var fn: WindowAttention
    init(dim: Int, nHeads: Int, window: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: dim)
        _fn.wrappedValue = WindowAttention(dim: dim, nHeads: nHeads, window: window)
    }
}

final class ChannelResidual: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "fn") var fn: ChannelAttention
    init(dim: Int, groups: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: dim)
        _fn.wrappedValue = ChannelAttention(dim: dim, groups: groups)
    }
}

final class SpatialBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: ConvResidual
    @ModuleInfo(key: "window_attn") var windowAttn: WindowResidual
    @ModuleInfo(key: "conv2") var conv2: ConvResidual
    @ModuleInfo(key: "ffn") var ffn: FfnResidual
    init(dim: Int, nHeads: Int, window: Int) {
        _conv1.wrappedValue = ConvResidual(dim: dim)
        _windowAttn.wrappedValue = WindowResidual(dim: dim, nHeads: nHeads, window: window)
        _conv2.wrappedValue = ConvResidual(dim: dim)
        _ffn.wrappedValue = FfnResidual(dim: dim)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = conv1(x)
        x = x + windowAttn.fn(windowAttn.norm(x))
        x = conv2(x)
        return ffn(x)
    }
}

final class ChannelBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: ConvResidual
    @ModuleInfo(key: "channel_attn") var channelAttn: ChannelResidual
    @ModuleInfo(key: "conv2") var conv2: ConvResidual
    @ModuleInfo(key: "ffn") var ffn: FfnResidual
    init(dim: Int, groups: Int) {
        _conv1.wrappedValue = ConvResidual(dim: dim)
        _channelAttn.wrappedValue = ChannelResidual(dim: dim, groups: groups)
        _conv2.wrappedValue = ConvResidual(dim: dim)
        _ffn.wrappedValue = FfnResidual(dim: dim)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
        var x = conv1(x)
        let t = x.reshaped([b, h * w, c])
        x = (t + channelAttn.fn(channelAttn.norm(t))).reshaped([b, h, w, c])
        x = conv2(x)
        return ffn(x)
    }
}

final class DaViTLayer: Module {
    @ModuleInfo(key: "spatial_block") var spatial: SpatialBlock
    @ModuleInfo(key: "channel_block") var channel: ChannelBlock
    init(dim: Int, nHeads: Int, groups: Int, window: Int) {
        _spatial.wrappedValue = SpatialBlock(dim: dim, nHeads: nHeads, window: window)
        _channel.wrappedValue = ChannelBlock(dim: dim, groups: groups)
    }
}

final class DaViT: Module {
    @ModuleInfo(key: "convs") var convs: [ConvEmbed]
    @ModuleInfo(key: "layers") var layers: [DaViTLayer]
    let depths: [Int]
    init(config c: [String: Any]) {
        let dims = (c["dim_embed"] as? [Int]) ?? [128, 256, 512, 1024]
        let heads = (c["num_heads"] as? [Int]) ?? [4, 8, 16, 32]
        let groups = (c["num_groups"] as? [Int]) ?? [4, 8, 16, 32]
        let patch = (c["patch_size"] as? [Int]) ?? [7, 3, 3, 3]
        let stride = (c["patch_stride"] as? [Int]) ?? [4, 2, 2, 2]
        let pad = (c["patch_padding"] as? [Int]) ?? [3, 1, 1, 1]
        let pre = (c["patch_prenorm"] as? [Bool]) ?? [false, true, true, true]
        let window = cfg(c, "window_size", 12)
        let depths = (c["depths"] as? [Int]) ?? [1, 1, 9, 1]
        self.depths = depths
        _convs.wrappedValue = dims.indices.map { i in
            ConvEmbed(patch: patch[i], inChans: i == 0 ? 3 : dims[i - 1], dim: dims[i], stride: stride[i], padding: pad[i], preNorm: pre[i])
        }
        _layers.wrappedValue = dims.indices.flatMap { i in
            (0 ..< depths[i]).map { _ in DaViTLayer(dim: dims[i], nHeads: heads[i], groups: groups[i], window: window) }
        }
    }
    /// forward_features_unpool: [B, H, W, 3] -> tokens [B, h*w, C] and the final grid size.
    func callAsFunction(_ pixels: MLXArray) -> (MLXArray, Int, Int) {
        var x = pixels
        var k = 0
        for (i, conv) in convs.enumerated() {
            x = conv(x)
            for _ in 0 ..< depths[i] {
                x = layers[k].channel(layers[k].spatial(x))
                k += 1
            }
            eval(x)
        }
        return (x.reshaped([x.dim(0), x.dim(1) * x.dim(2), x.dim(3)]), x.dim(1), x.dim(2))
    }
}

// MARK: BART encoder-decoder (post-norm, learned positions with offset 2, GELU)

final class BartAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let nHeads: Int
    init(dim: Int, nHeads: Int) {
        self.nHeads = nHeads
        _qProj.wrappedValue = Linear(dim, dim)
        _kProj.wrappedValue = Linear(dim, dim)
        _vProj.wrappedValue = Linear(dim, dim)
        _outProj.wrappedValue = Linear(dim, dim)
    }
    func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, mask: MLXArray?) -> MLXArray {
        let d = q.dim(3)
        return outProj(merge(MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / Float(d).squareRoot(), mask: mask)))
    }
}

final class BartEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: BartAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm
    init(dim: Int, ffn: Int, nHeads: Int) {
        _selfAttn.wrappedValue = BartAttention(dim: dim, nHeads: nHeads)
        _selfAttnNorm.wrappedValue = LayerNorm(dimensions: dim)
        _fc1.wrappedValue = Linear(dim, ffn)
        _fc2.wrappedValue = Linear(ffn, dim)
        _finalNorm.wrappedValue = LayerNorm(dimensions: dim)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = selfAttn
        let h = a.nHeads
        var x = selfAttnNorm(x + a.attend(heads(a.qProj(x), h), heads(a.kProj(x), h), heads(a.vProj(x), h), mask: nil))
        x = finalNorm(x + fc2(gelu(fc1(x))))
        return x
    }
}

final class BartDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: BartAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnNorm: LayerNorm
    @ModuleInfo(key: "encoder_attn") var encoderAttn: BartAttention
    @ModuleInfo(key: "encoder_attn_layer_norm") var encoderAttnNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm
    init(dim: Int, ffn: Int, nHeads: Int) {
        _selfAttn.wrappedValue = BartAttention(dim: dim, nHeads: nHeads)
        _selfAttnNorm.wrappedValue = LayerNorm(dimensions: dim)
        _encoderAttn.wrappedValue = BartAttention(dim: dim, nHeads: nHeads)
        _encoderAttnNorm.wrappedValue = LayerNorm(dimensions: dim)
        _fc1.wrappedValue = Linear(dim, ffn)
        _fc2.wrappedValue = Linear(ffn, dim)
        _finalNorm.wrappedValue = LayerNorm(dimensions: dim)
    }
}

final class BartEncoder: Module {
    @ModuleInfo(key: "embed_positions") var embedPositions: Embedding
    @ModuleInfo(key: "layernorm_embedding") var layernormEmbedding: LayerNorm
    @ModuleInfo(key: "layers") var layers: [BartEncoderLayer]
    init(dim: Int, ffn: Int, nHeads: Int, nLayers: Int, maxPos: Int) {
        _embedPositions.wrappedValue = Embedding(embeddingCount: maxPos + 2, dimensions: dim)
        _layernormEmbedding.wrappedValue = LayerNorm(dimensions: dim)
        _layers.wrappedValue = (0 ..< nLayers).map { _ in BartEncoderLayer(dim: dim, ffn: ffn, nHeads: nHeads) }
    }
    /// inputs_embeds [1, L, D] (embed_scale 1.0) -> last_hidden_state.
    func callAsFunction(_ embeds: MLXArray) -> MLXArray {
        let n = embeds.dim(1)
        var x = layernormEmbedding(embeds + embedPositions(MLXArray((0 ..< n).map { Int32($0 + 2) })).expandedDimensions(axis: 0))
        for l in layers { x = l(x) }
        return x
    }
}

final class BartDecoder: Module {
    @ModuleInfo(key: "embed_positions") var embedPositions: Embedding
    @ModuleInfo(key: "layernorm_embedding") var layernormEmbedding: LayerNorm
    @ModuleInfo(key: "layers") var layers: [BartDecoderLayer]
    let nHeads: Int
    init(dim: Int, ffn: Int, nHeads: Int, nLayers: Int, maxPos: Int) {
        self.nHeads = nHeads
        _embedPositions.wrappedValue = Embedding(embeddingCount: maxPos + 2, dimensions: dim)
        _layernormEmbedding.wrappedValue = LayerNorm(dimensions: dim)
        _layers.wrappedValue = (0 ..< nLayers).map { _ in BartDecoderLayer(dim: dim, ffn: ffn, nHeads: nHeads) }
    }

    func crossCaches(_ memory: MLXArray, beams: Int) -> [(MLXArray, MLXArray)] {
        let cache = layers.map { l -> (MLXArray, MLXArray) in
            let k = heads(l.encoderAttn.kProj(memory), nHeads), v = heads(l.encoderAttn.vProj(memory), nHeads)
            return (repeated(k, count: beams, axis: 0), repeated(v, count: beams, axis: 0))
        }
        eval(cache.flatMap { [$0.0, $0.1] })
        return cache
    }

    /// Runs ids [beams, n] at positions start..<start+n with the cache; returns last-position logits [beams, vocab].
    /// `shared` is language_model.model.shared (token embedding and tied LM head); `bias` is final_logits_bias.
    func forward(_ ids: MLXArray, start: Int, shared: Embedding, bias: MLXArray, cross: [(MLXArray, MLXArray)],
                 cache: inout [(MLXArray, MLXArray)?]) -> MLXArray {
        let n = ids.dim(1)
        let pos = embedPositions(MLXArray((start ..< start + n).map { Int32($0 + 2) })).expandedDimensions(axis: 0)
        var x = layernormEmbedding(shared(ids) + pos)
        var mask: MLXArray? = nil
        if n > 1 {
            let rows = MLXArray((0 ..< n).map(Int32.init)).reshaped([n, 1]) + Int32(start)
            let cols = MLXArray((0 ..< start + n).map(Int32.init)).reshaped([1, start + n])
            mask = MLX.where(cols .<= rows, MLXArray(Float(0)), MLXArray(-Float.infinity)).asType(x.dtype)
        }
        for (i, l) in layers.enumerated() {
            let a = l.selfAttn
            let q = heads(a.qProj(x), nHeads)
            var k = heads(a.kProj(x), nHeads), v = heads(a.vProj(x), nHeads)
            if let (pk, pv) = cache[i] { k = concatenated([pk, k], axis: 2); v = concatenated([pv, v], axis: 2) }
            cache[i] = (k, v)
            x = l.selfAttnNorm(x + a.attend(q, k, v, mask: mask))
            let c = l.encoderAttn
            x = l.encoderAttnNorm(x + c.attend(heads(c.qProj(x), nHeads), cross[i].0, cross[i].1, mask: nil))
            x = l.finalNorm(x + l.fc2(gelu(l.fc1(x))))
        }
        let last = x[0..., (n - 1) ..< n, 0...].reshaped([x.dim(0), -1])
        return matmul(last, shared.weight.T) + bias.reshaped([1, -1])
    }

}

final class BartModel: Module {
    @ModuleInfo(key: "shared") var shared: Embedding
    @ModuleInfo(key: "encoder") var encoder: BartEncoder
    @ModuleInfo(key: "decoder") var decoder: BartDecoder
    init(config c: [String: Any]) {
        let dim = cfg(c, "d_model", 768), maxPos = cfg(c, "max_position_embeddings", 1024)
        _shared.wrappedValue = Embedding(embeddingCount: cfg(c, "vocab_size", 51289), dimensions: dim)
        _encoder.wrappedValue = BartEncoder(dim: dim, ffn: cfg(c, "encoder_ffn_dim", 3072), nHeads: cfg(c, "encoder_attention_heads", 12),
                                            nLayers: cfg(c, "encoder_layers", 6), maxPos: maxPos)
        _decoder.wrappedValue = BartDecoder(dim: dim, ffn: cfg(c, "decoder_ffn_dim", 3072), nHeads: cfg(c, "decoder_attention_heads", 12),
                                            nLayers: cfg(c, "decoder_layers", 6), maxPos: maxPos)
    }
}

final class BartForGeneration: Module {
    @ModuleInfo(key: "model") var model: BartModel
    @ParameterInfo(key: "final_logits_bias") var finalLogitsBias: MLXArray
    init(config c: [String: Any]) {
        _model.wrappedValue = BartModel(config: c)
        _finalLogitsBias.wrappedValue = MLXArray.zeros([1, cfg(c, "vocab_size", 51289)])
    }

    func prefill(tokens: [Int], beams: Int, cross: [(MLXArray, MLXArray)], cache: inout [(MLXArray, MLXArray)?]) -> MLXArray {
        let ids = broadcast(MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count]), to: [beams, tokens.count])
        return model.decoder.forward(ids, start: 0, shared: model.shared, bias: finalLogitsBias, cross: cross, cache: &cache)
    }

    func step(tokens: [Int], position: Int, cross: [(MLXArray, MLXArray)], cache: inout [(MLXArray, MLXArray)?]) -> MLXArray {
        model.decoder.forward(MLXArray(tokens.map(Int32.init)).reshaped([tokens.count, 1]), start: position, shared: model.shared,
                              bias: finalLogitsBias, cross: cross, cache: &cache)
    }
}
