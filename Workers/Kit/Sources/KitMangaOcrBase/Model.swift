// SPDX-License-Identifier: MIT
// Manga OCR (kha-white/manga-ocr-base): a VisionEncoderDecoder with a ViT/DeiT-base image encoder and a 2-layer
// character-level BERT-Japanese decoder with cross-attention. Ported from transformers' modeling_vit.py,
// modeling_bert.py (BertLMHeadModel) and the maker's manga_ocr/ocr.py (preprocessing, decoding, post-processing).
import Foundation
import MLX
import MLXNN

let modelKind = "local.document"
let modelOps = ["read"]

final class Model {
    let net: MangaOcrNet
    let vocab: [String]
    let special: Set<Int>
    let startId: Int, eosId: Int, maxLength: Int
    let numBeams: Int, noRepeatNgram: Int, lengthPenalty: Float, earlyStopping: Bool

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        let config = try Files.json(weightsDir)
        guard let enc = config["encoder"] as? [String: Any], let dec = config["decoder"] as? [String: Any] else {
            throw RunnerError.failed("config.json is not a VisionEncoderDecoder config.")
        }
        net = MangaOcrNet(enc: enc, dec: dec)
        // The ViT pooler and position_ids buffer are not used by the encoder-decoder; the LM head's weight is tied
        // to the decoder's word embeddings (BertLMHeadModel with tie_word_embeddings).
        var weights = try Weights.load(weightsDir)
        let conv = "encoder.embeddings.patch_embeddings.projection.weight"
        if let w = weights[conv] { weights[conv] = w.transposed(0, 2, 3, 1) }  // [out, in, kh, kw] -> [out, kh, kw, in]
        try Weights.apply(weights, to: net) { key in
            key.hasPrefix("encoder.pooler.") || key.hasSuffix("position_ids") ? nil : key
        }
        let lines = try String(contentsOf: weightsDir.appendingPathComponent("vocab.txt"), encoding: .utf8)
            .components(separatedBy: "\n").map { $0.trimmingCharacters(in: CharacterSet(charactersIn: "\r")) }
        vocab = lines
        // [PAD] [UNK] [CLS] [SEP] [MASK]: skip_special_tokens drops all of them.
        special = Set(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"].compactMap { lines.firstIndex(of: $0) })
        startId = config["decoder_start_token_id"] as? Int ?? 2
        eosId = config["eos_token_id"] as? Int ?? 3
        // manga_ocr calls generate(max_length=300): the start token plus at most 299 generated tokens.
        maxLength = min(300, dec["max_position_embeddings"] as? Int ?? 512)
        // generate() takes its decoding settings from config.json (transformers copies them into generation_config):
        // beam search with 4 beams, no repeated 3-grams, length_penalty 2.0, early_stopping.
        let beams = config["num_beams"] as? Int ?? 1
        numBeams = Int(ProcessInfo.processInfo.environment["ARSENAL_PARITY_BEAMS"] ?? "") ?? beams
        noRepeatNgram = config["no_repeat_ngram_size"] as? Int ?? 0
        lengthPenalty = Float(config["length_penalty"] as? Double ?? 1.0)
        earlyStopping = config["early_stopping"] as? Bool ?? false
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        let url = try params.file("image")
        // manga_ocr: img.convert("L").convert("RGB"), then ViTImageProcessor (PIL bilinear 224x224, /255, mean 0.5, std 0.5).
        let pixels = try Pixels.load(url).grayRGB().resized(width: 224, height: 224, filter: .bilinear)
        let input = MLXArray(pixels.normalized(mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5]), [1, 224, 224, 3]).asType(.float16)
        let dump = ProcessInfo.processInfo.environment["ARSENAL_PARITY_DIR"]

        let memory = net.encoder(input)
        var firstLogits: MLXArray? = nil
        let tokens = try beamSearch(memory: memory, params: params) { if firstLogits == nil { firstLogits = $0 } }
        if let dump {
            try save(arrays: ["pixel_values": input.asType(.float32), "first_logits": firstLogits!.asType(.float32),
                              "tokens": MLXArray(tokens.map(Int32.init))],
                     url: URL(fileURLWithPath: dump).appendingPathComponent("mlx-\(url.deletingPathExtension().lastPathComponent).safetensors"))
        }
        let text = Model.postProcess(decode(tokens))
        return ["text": text]
    }

    /// transformers GenerationMixin._beam_search (5.x) for one image, with NoRepeatNGramLogitsProcessor, EOS and
    /// max_length stopping. Log-probabilities and scores are float32 like the reference; with one beam it is greedy.
    func beamSearch(memory: MLXArray, params: Params, onFirstLogits: (MLXArray) -> Void) throws -> [Int] {
        let beams = max(1, numBeams), keep = 2 * beams
        let dec = net.decoder
        let cross = dec.crossCaches(memory, beams: beams)
        var cache: [(MLXArray, MLXArray)?] = Array(repeating: nil, count: dec.bert.encoder.layer.count)
        var running = Array(repeating: [startId], count: beams)
        var runningScores = [Float](repeating: -1e9, count: beams); runningScores[0] = 0
        var finished = Array(repeating: [Int](), count: beams)
        var finishedScores = [Float](repeating: -1e9, count: beams)
        var isFinished = [Bool](repeating: false, count: beams)
        var improvable = true
        var curLen = 1
        while true {
            try params.checkCancelled()
            let logits = dec.step(tokens: running.map { $0.last! }, position: curLen - 1, cross: cross, cache: &cache).asType(.float32)
            if curLen == 1 { onFirstLogits(logits[0]) }
            let logProbs = logSoftmax(logits, axis: -1)
            eval(logProbs)
            let vocabSize = logProbs.dim(1)
            var flat = logProbs.asArray(Float.self)
            for b in 0 ..< beams {
                for t in bannedTokens(running[b]) { flat[b * vocabSize + t] = -Float.infinity }
                let base = runningScores[b]
                for t in 0 ..< vocabSize { flat[b * vocabSize + t] += base }
            }
            // torch.topk(keep): the best `keep` continuations over all beams, best first.
            let top = topK(flat, keep)
            let candidates = top.map { (beam: $0 / vocabSize, token: $0 % vocabSize, score: flat[$0]) }
            let hits = candidates.map { $0.token == eosId || curLen + 1 >= maxLength }
            // Running beams for the next step: the best `beams` candidates that did not stop.
            let runScores = candidates.indices.map { candidates[$0].score + (hits[$0] ? -1e9 : 0) }
            let nextRun = topK(runScores, beams)
            // Finished hypotheses: merge the kept ones with candidates among the top `beams` that just stopped.
            let allFull = isFinished.allSatisfy { $0 } && earlyStopping
            var mergedSeqs = finished, mergedScores = finishedScores, mergedDone = isFinished
            for (k, c) in candidates.enumerated() {
                var score = c.score / pow(Float(curLen), lengthPenalty)
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
            // _check_early_stop_heuristic with early_stopping True (or False): the best running score at this length.
            let bestPossible = runningScores[0] / pow(Float(curLen - 1), lengthPenalty)
            let worst = finishedScores.min()!
            improvable = improvable && isFinished.indices.contains { bestPossible > (isFinished[$0] ? worst : -1e9) }
            let open = !(isFinished.allSatisfy { $0 } && earlyStopping)
            let valid = !hits.allSatisfy { $0 }
            if !(improvable && open && valid) { break }
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

    /// BertJapaneseTokenizer.decode(skip_special_tokens=True): tokens joined with spaces, " ##" removed, stripped.
    func decode(_ ids: [Int]) -> String {
        let pieces = ids.filter { !special.contains($0) && $0 >= 0 && $0 < vocab.count }.map { vocab[$0] }
        return pieces.joined(separator: " ").replacingOccurrences(of: " ##", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// manga_ocr.ocr.post_process: drop whitespace, "…" to "...", runs of ・/. to dots, jaconv.h2z(ascii, digit, kana).
    static func postProcess(_ raw: String) -> String {
        var text = String(String.UnicodeScalarView(raw.unicodeScalars.filter { !$0.properties.isWhitespace }))
        text = text.replacingOccurrences(of: "…", with: "...")
        var out = ""
        var run = 0
        var pending: [Character] = []
        for ch in text {
            if ch == "・" || ch == "." {
                pending.append(ch); run += 1; continue
            }
            out += run >= 2 ? String(repeating: ".", count: run) : String(pending)
            pending = []; run = 0
            out.append(ch)
        }
        out += run >= 2 ? String(repeating: ".", count: run) : String(pending)
        return h2z(out)
    }

    static let halfKana = Array("ｧｱｨｲｩｳｪｴｫｵｶｷｸｹｺｻｼｽｾｿﾀﾁｯﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓｬﾔｭﾕｮﾖﾗﾘﾙﾚﾛﾜｦﾝｰヮヰヱヵヶヽヾ･｢｣｡､".unicodeScalars)
    static let fullKana = Array("ァアィイゥウェエォオカキクケコサシスセソタチッツテトナニヌネノハヒフヘホマミムメモャヤュユョヨラリルレロワヲンーヮヰヱヵヶヽヾ・「」。、".unicodeScalars)
    static let dakuten: [(String, String)] = [
        ("ｶﾞ", "ガ"), ("ｷﾞ", "ギ"), ("ｸﾞ", "グ"), ("ｹﾞ", "ゲ"), ("ｺﾞ", "ゴ"), ("ｻﾞ", "ザ"), ("ｼﾞ", "ジ"), ("ｽﾞ", "ズ"),
        ("ｾﾞ", "ゼ"), ("ｿﾞ", "ゾ"), ("ﾀﾞ", "ダ"), ("ﾁﾞ", "ヂ"), ("ﾂﾞ", "ヅ"), ("ﾃﾞ", "デ"), ("ﾄﾞ", "ド"), ("ﾊﾞ", "バ"),
        ("ﾋﾞ", "ビ"), ("ﾌﾞ", "ブ"), ("ﾍﾞ", "ベ"), ("ﾎﾞ", "ボ"), ("ﾊﾟ", "パ"), ("ﾋﾟ", "ピ"), ("ﾌﾟ", "プ"), ("ﾍﾟ", "ペ"),
        ("ﾎﾟ", "ポ"), ("ｳﾞ", "ヴ"),
    ]

    /// jaconv.h2z(text, ascii=True, digit=True): half-width kana (with voiced marks), ASCII and digits to full width.
    static func h2z(_ input: String) -> String {
        var text = input
        for (half, full) in dakuten { text = text.replacingOccurrences(of: half, with: full) }
        var map: [UInt32: Unicode.Scalar] = [:]
        for (h, f) in zip(halfKana, fullKana) { map[h.value] = f }
        var scalars = String.UnicodeScalarView()
        for s in text.unicodeScalars {
            if s.value == 0x20 { scalars.append(Unicode.Scalar(0x3000)!) }
            else if s.value >= 0x21 && s.value <= 0x7E { scalars.append(Unicode.Scalar(s.value + 0xFEE0)!) }
            else if let f = map[s.value] { scalars.append(f) }
            else { scalars.append(s) }
        }
        return String(scalars)
    }
}

// MARK: network (names follow the checkpoint)

final class MangaOcrNet: Module {
    @ModuleInfo(key: "encoder") var encoder: ViTModel
    @ModuleInfo(key: "decoder") var decoder: BertLMHead
    init(enc: [String: Any], dec: [String: Any]) {
        _encoder.wrappedValue = ViTModel(config: enc)
        _decoder.wrappedValue = BertLMHead(config: dec)
    }
}

func cfg(_ c: [String: Any], _ key: String, _ value: Int) -> Int { c[key] as? Int ?? value }

final class QKV: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    init(_ hidden: Int) {
        _query.wrappedValue = Linear(hidden, hidden)
        _key.wrappedValue = Linear(hidden, hidden)
        _value.wrappedValue = Linear(hidden, hidden)
    }
}

final class DenseOnly: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    init(_ i: Int, _ o: Int) { _dense.wrappedValue = Linear(i, o) }
}

final class DenseNorm: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm
    init(_ i: Int, _ o: Int, eps: Float) {
        _dense.wrappedValue = Linear(i, o)
        _norm.wrappedValue = LayerNorm(dimensions: o, eps: eps)
    }
}

/// [B, L, H*D] -> [B, H, L, D]
func heads(_ x: MLXArray, _ n: Int) -> MLXArray {
    x.reshaped([x.dim(0), x.dim(1), n, x.dim(2) / n]).transposed(0, 2, 1, 3)
}
func merge(_ x: MLXArray) -> MLXArray {
    x.transposed(0, 2, 1, 3).reshaped([x.dim(0), x.dim(2), x.dim(1) * x.dim(3)])
}

// ViT (transformers ViTModel, pre-norm), last_hidden_state after the final LayerNorm.
final class ViTPatch: Module {
    @ModuleInfo(key: "projection") var projection: Conv2d
    init(hidden: Int, patch: Int, channels: Int) {
        _projection.wrappedValue = Conv2d(inputChannels: channels, outputChannels: hidden, kernelSize: .init(patch), stride: .init(patch))
    }
}

final class ViTEmbeddings: Module {
    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    @ParameterInfo(key: "position_embeddings") var positionEmbeddings: MLXArray
    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: ViTPatch
    init(hidden: Int, patch: Int, channels: Int, image: Int) {
        let n = (image / patch) * (image / patch)
        _clsToken.wrappedValue = MLXArray.zeros([1, 1, hidden])
        _positionEmbeddings.wrappedValue = MLXArray.zeros([1, n + 1, hidden])
        _patchEmbeddings.wrappedValue = ViTPatch(hidden: hidden, patch: patch, channels: channels)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let p = patchEmbeddings.projection(x)  // [B, h, w, C]
        let tokens = p.reshaped([p.dim(0), p.dim(1) * p.dim(2), p.dim(3)])
        let cls = broadcast(clsToken, to: [tokens.dim(0), 1, tokens.dim(2)])
        return concatenated([cls, tokens], axis: 1) + positionEmbeddings
    }
}

final class ViTAttentionInner: Module {
    @ModuleInfo(key: "attention") var attention: QKV
    @ModuleInfo(key: "output") var output: DenseOnly
    init(_ hidden: Int) {
        _attention.wrappedValue = QKV(hidden)
        _output.wrappedValue = DenseOnly(hidden, hidden)
    }
}

final class ViTLayer: Module {
    @ModuleInfo(key: "attention") var attention: ViTAttentionInner
    @ModuleInfo(key: "intermediate") var intermediate: DenseOnly
    @ModuleInfo(key: "output") var output: DenseOnly
    @ModuleInfo(key: "layernorm_before") var before: LayerNorm
    @ModuleInfo(key: "layernorm_after") var after: LayerNorm
    let nHeads: Int
    init(hidden: Int, inter: Int, nHeads: Int, eps: Float) {
        self.nHeads = nHeads
        _attention.wrappedValue = ViTAttentionInner(hidden)
        _intermediate.wrappedValue = DenseOnly(hidden, inter)
        _output.wrappedValue = DenseOnly(inter, hidden)
        _before.wrappedValue = LayerNorm(dimensions: hidden, eps: eps)
        _after.wrappedValue = LayerNorm(dimensions: hidden, eps: eps)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = before(x)
        let a = attention.attention
        let q = heads(a.query(h), nHeads), k = heads(a.key(h), nHeads), v = heads(a.value(h), nHeads)
        let scale = 1 / Float(q.dim(3)).squareRoot()
        let o = merge(MLX.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil))
        let x = x + attention.output.dense(o)
        return x + output.dense(gelu(intermediate.dense(after(x))))
    }
}

final class ViTEncoderStack: Module {
    @ModuleInfo(key: "layer") var layer: [ViTLayer]
    init(_ layers: [ViTLayer]) { _layer.wrappedValue = layers }
}

final class ViTModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: ViTEmbeddings
    @ModuleInfo(key: "encoder") var encoder: ViTEncoderStack
    @ModuleInfo(key: "layernorm") var layernorm: LayerNorm
    init(config c: [String: Any]) {
        let hidden = cfg(c, "hidden_size", 768)
        let eps = Float(c["layer_norm_eps"] as? Double ?? 1e-12)
        _embeddings.wrappedValue = ViTEmbeddings(hidden: hidden, patch: cfg(c, "patch_size", 16), channels: cfg(c, "num_channels", 3), image: cfg(c, "image_size", 224))
        _encoder.wrappedValue = ViTEncoderStack((0 ..< cfg(c, "num_hidden_layers", 12)).map { _ in
            ViTLayer(hidden: hidden, inter: cfg(c, "intermediate_size", 3072), nHeads: cfg(c, "num_attention_heads", 12), eps: eps)
        })
        _layernorm.wrappedValue = LayerNorm(dimensions: hidden, eps: eps)
    }
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        var x = embeddings(pixels)
        for l in encoder.layer { x = l(x) }
        return layernorm(x)
    }
}

// BERT decoder (BertLMHeadModel, post-norm, causal self-attention plus cross-attention), one token per step.
final class BertEmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var word: Embedding
    @ModuleInfo(key: "position_embeddings") var position: Embedding
    @ModuleInfo(key: "token_type_embeddings") var tokenType: Embedding
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm
    init(vocab: Int, hidden: Int, positions: Int, types: Int, eps: Float) {
        _word.wrappedValue = Embedding(embeddingCount: vocab, dimensions: hidden)
        _position.wrappedValue = Embedding(embeddingCount: positions, dimensions: hidden)
        _tokenType.wrappedValue = Embedding(embeddingCount: types, dimensions: hidden)
        _norm.wrappedValue = LayerNorm(dimensions: hidden, eps: eps)
    }
}

final class BertAttention: Module {
    @ModuleInfo(key: "self") var selfAttention: QKV
    @ModuleInfo(key: "output") var output: DenseNorm
    init(_ hidden: Int, eps: Float) {
        _selfAttention.wrappedValue = QKV(hidden)
        _output.wrappedValue = DenseNorm(hidden, hidden, eps: eps)
    }
}

final class BertLayer: Module {
    @ModuleInfo(key: "attention") var attention: BertAttention
    @ModuleInfo(key: "crossattention") var cross: BertAttention
    @ModuleInfo(key: "intermediate") var intermediate: DenseOnly
    @ModuleInfo(key: "output") var output: DenseNorm
    init(hidden: Int, inter: Int, eps: Float) {
        _attention.wrappedValue = BertAttention(hidden, eps: eps)
        _cross.wrappedValue = BertAttention(hidden, eps: eps)
        _intermediate.wrappedValue = DenseOnly(hidden, inter)
        _output.wrappedValue = DenseNorm(inter, hidden, eps: eps)
    }
}

final class BertEncoderStack: Module {
    @ModuleInfo(key: "layer") var layer: [BertLayer]
    init(_ layers: [BertLayer]) { _layer.wrappedValue = layers }
}

final class BertBody: Module {
    @ModuleInfo(key: "embeddings") var embeddings: BertEmbeddings
    @ModuleInfo(key: "encoder") var encoder: BertEncoderStack
    init(embeddings: BertEmbeddings, layers: [BertLayer]) {
        _embeddings.wrappedValue = embeddings
        _encoder.wrappedValue = BertEncoderStack(layers)
    }
}

final class BertPredictionHead: Module {
    @ModuleInfo(key: "transform") var transform: DenseNorm
    @ParameterInfo(key: "bias") var bias: MLXArray
    init(hidden: Int, vocab: Int, eps: Float) {
        _transform.wrappedValue = DenseNorm(hidden, hidden, eps: eps)
        _bias.wrappedValue = MLXArray.zeros([vocab])
    }
}

final class BertCls: Module {
    @ModuleInfo(key: "predictions") var predictions: BertPredictionHead
    init(hidden: Int, vocab: Int, eps: Float) { _predictions.wrappedValue = BertPredictionHead(hidden: hidden, vocab: vocab, eps: eps) }
}

final class BertLMHead: Module {
    @ModuleInfo(key: "bert") var bert: BertBody
    @ModuleInfo(key: "cls") var cls: BertCls
    let nHeads: Int
    init(config c: [String: Any]) {
        let hidden = cfg(c, "hidden_size", 768), vocab = cfg(c, "vocab_size", 6144)
        let eps = Float(c["layer_norm_eps"] as? Double ?? 1e-12)
        nHeads = cfg(c, "num_attention_heads", 12)
        _bert.wrappedValue = BertBody(
            embeddings: BertEmbeddings(vocab: vocab, hidden: hidden, positions: cfg(c, "max_position_embeddings", 512), types: cfg(c, "type_vocab_size", 2), eps: eps),
            layers: (0 ..< cfg(c, "num_hidden_layers", 2)).map { _ in BertLayer(hidden: hidden, inter: cfg(c, "intermediate_size", 3072), eps: eps) })
        _cls.wrappedValue = BertCls(hidden: hidden, vocab: vocab, eps: eps)
    }

    /// Cross-attention keys and values for every layer, computed once per image.
    func crossCaches(_ memory: MLXArray, beams: Int) -> [(MLXArray, MLXArray)] {
        let cache = bert.encoder.layer.map { l -> (MLXArray, MLXArray) in
            let k = heads(l.cross.selfAttention.key(memory), nHeads), v = heads(l.cross.selfAttention.value(memory), nHeads)
            return (repeated(k, count: beams, axis: 0), repeated(v, count: beams, axis: 0))
        }
        eval(cache.flatMap { [$0.0, $0.1] })
        return cache
    }

    /// One decoder step for every beam: logits [beams, vocab] for the token after `tokens` at `position`.
    func step(tokens: [Int], position: Int, cross: [(MLXArray, MLXArray)], cache: inout [(MLXArray, MLXArray)?]) -> MLXArray {
        let e = bert.embeddings
        let ids = MLXArray(tokens.map(Int32.init)).reshaped([tokens.count, 1])
        var x = e.word(ids) + e.tokenType(MLXArray([Int32(0)]).reshaped([1, 1])) + e.position(MLXArray([Int32(position)]).reshaped([1, 1]))
        x = e.norm(x)
        for (i, l) in bert.encoder.layer.enumerated() {
            let sa = l.attention.selfAttention
            let q = heads(sa.query(x), nHeads)
            var k = heads(sa.key(x), nHeads), v = heads(sa.value(x), nHeads)
            if let (pk, pv) = cache[i] {
                k = concatenated([pk, k], axis: 2); v = concatenated([pv, v], axis: 2)
            }
            cache[i] = (k, v)
            let scale = 1 / Float(q.dim(3)).squareRoot()
            let a = merge(MLX.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil))
            x = l.attention.output.norm(l.attention.output.dense(a) + x)
            let cq = heads(l.cross.selfAttention.query(x), nHeads)
            let c = merge(MLX.scaledDotProductAttention(queries: cq, keys: cross[i].0, values: cross[i].1, scale: scale, mask: nil))
            x = l.cross.output.norm(l.cross.output.dense(c) + x)
            x = l.output.norm(l.output.dense(gelu(l.intermediate.dense(x))) + x)
        }
        let t = cls.predictions.transform
        let h = t.norm(gelu(t.dense(x)))
        // Decoder weight tied to the word embeddings.
        let logits = matmul(h, e.word.weight.T) + cls.predictions.bias
        return logits.reshaped([tokens.count, -1])
    }
}
