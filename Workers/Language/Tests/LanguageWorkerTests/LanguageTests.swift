import Foundation
import Testing
import ArsenalWire
import MLX
@testable import LanguageWorker

struct LanguageTests {
    init() { MLXReady.ensure() }
    @Test func groundingMakerFormats() throws {
        let point = try LanguageOutput.ground(
            id: "groundgui-8b", text: "<tool_call>{\"name\":\"computer_use\",\"arguments\":{\"action\":\"left_click\",\"coordinate\":[250,750]}}</tool_call>", label: "Save",
            size: CGSize(width: 2000, height: 1000))
        #expect(point["detections"]?.array?.first?.object?["point"]?.object?["x"]?.number == 500)
        // Past the 0–1000 grid the makers do not reject; the point is clamped to the image.
        let clamped = try LanguageOutput.ground(id: "ui-venus-2-9b", text: "[1001,0]", label: "Save", size: CGSize(width: 2000, height: 1000))
        #expect(clamped["detections"]?.array?.first?.object?["point"]?.object?["x"]?.number == 2000)
        // ui_venus2_gd.py takes the first point when two unpaired points appear.
        let first = try LanguageOutput.ground(id: "ui-venus-2-9b", text: "[100,200] and [300,400]", label: "Save", size: CGSize(width: 2000, height: 1000))
        #expect(first["detections"]?.array?.first?.object?["point"]?.object?["x"]?.number == 200)
        #expect(throws: (any Error).self) { try LanguageOutput.ground(id: "ui-venus-2-9b", text: "no coordinates", label: "Save", size: CGSize(width: 2000, height: 1000)) }
    }
    @Test func indexConstraints() throws {
        let prompt = try LanguagePrompts.translation(id: "index-translate-2b", text: "bank", target: "pl", input: ["glossary": ["bank": "bank"], "tone": "formal"])
        #expect(prompt.contains("波兰语")); #expect(prompt.contains("bank→bank")); #expect(prompt.contains("【注意】formal"))
    }
    @Test func malformedJSONIsVisible() { #expect(throws: (any Error).self) { try LanguageOutput.json("not json") } }
}

extension LanguageTests {
    @Test func teleOCRSpanningTable() throws {
        let output = try OTSLOutput.html("<fcel>A & B<lcel><nl><ucel><xcel><nl>")
        #expect(output == "<table><tr><td rowspan=\"2\" colspan=\"2\">A &amp; B</td></tr><tr></tr></table>")
    }
    @Test func venusAbstention() throws {
        let result = try LanguageOutput.ground(id: "ui-venus-2-9b", text: "[-1,-1]", label: "Missing", size: CGSize(width: 1000, height: 500))
        #expect(result["detections"]?.array?.isEmpty == true)
    }
}

extension LanguageTests {
    @Test func infinityLayoutCoordinates() throws {
        let output = try LanguageOutput.document(
            id: "infinity-parser2-flash", index: 0, text: "[{\"category\":\"table\",\"text\":\"<table></table>\",\"bbox\":[100,200,600,800]}]",
            size: CGSize(width: 2000, height: 1000))
        #expect(output.object?["blocks"]?.array?.first?.object?["box"]?.object?["xmin"]?.number == 200)
        #expect(output.object?["tables"]?.array?.count == 1)
    }
}

extension LanguageTests {
    /// Hy-MT2 card instruction examples, English wording.
    @Test func hunyuanPromptTypes() throws {
        func prompt(_ input: [String: ArsenalWire.JSONValue]) throws -> String { try LanguagePrompts.translation(id: "hy-mt2-7b", text: "Hello", target: "pl", input: input) }
        #expect(try prompt([:]) == "Translate the following text into Polish. Note that you should only output the translated result without any additional explanation:\n\nHello")
        #expect(
            try prompt(["glossary": ["bank": "bank"]]).hasPrefix(
                "Reference the following translations:\nbank translates to bank\n\nTranslate the following text into Polish. Note that you must ONLY"))
        #expect(try prompt(["tone": "formal"]).contains("strictly conform to [formal]:\n\nHello"))
        #expect(try prompt(["context": "Bank app"]).hasPrefix("[Background Information]\nBank app\n\n"))
        #expect(try prompt(["format": "json"]).hasPrefix("### Task\nTranslate the user-facing text within the following JSON data into Polish."))
        #expect(try prompt(["instructions": "Short\nBritish"]).hasSuffix("[Translation Tasks]\n1. Short\n2. British\n3. Translate the [Source Text] into Polish."))
        #expect(try prompt(["preserve_delimiters": true]).hasPrefix("Please accurately translate the following text into Polish.\n"))
        #expect(throws: (any Error).self) { try prompt(["tone": "formal", "context": "x"]) }
    }
}

/// Index-Translate inference_llm_translate.py trans_prompt and inference_llm_call_api.py image requests.
extension LanguageTests {
    @Test func indexLanguageNames() throws {
        func prompt(_ target: String, _ input: [String: ArsenalWire.JSONValue] = [:]) throws -> String {
            try LanguagePrompts.translation(id: "index-translate-2b", text: "Hello", target: target, input: input)
        }
        #expect(try prompt("uk").hasPrefix("请将以下文本翻译为乌克兰语，"))
        #expect(try prompt("EN-us").hasPrefix("请将以下文本翻译为英语，"))
        #expect(try prompt("no").hasPrefix("请将以下文本翻译为书面挪威语，"))
        #expect(try prompt("zh-Hant").hasPrefix("请将以下文本翻译为繁体中文，"))
        // Codes outside the maker table use the full Chinese language name, never the raw code.
        let hebrew = try prompt("he")
        #expect(!hebrew.hasPrefix("请将以下文本翻译为he，")); #expect(hebrew.hasPrefix("请将以下文本翻译为"))
        #expect(try prompt("pl", ["source_lang": "AUTO"]).hasPrefix("请将以下文本翻译为波兰语，"))
        #expect(try prompt("pl", ["source_lang": "cs"]).hasPrefix("请将以下捷克语文本翻译为波兰语，"))
        #expect(throws: (any Error).self) { try prompt("x1") }
    }
    @Test func indexConstraintNormalization() throws {
        let prompt = try LanguagePrompts.translation(
            id: "index-translate-9b", text: " {\"title\": \"用户协议\"} ", target: "en",
            input: ["hard_constraints": ["1. 保留占位符\n2. 【硬性要求】不要翻译品牌"], "soft_constraints": ["正式"], "glossary": [" 协议 ": " agreement "], "format": "json"])
        #expect(prompt.hasPrefix("请将以下文本翻译成英语，并且严格遵循所有约束要求。\n\n【源文】\n{\"title\": \"用户协议\"}\n\n【约束要求】\n"))
        #expect(prompt.contains("1. 【硬性要求】保留占位符\n2. 【硬性要求】不要翻译品牌\n3. 【硬性要求】仅翻译键值内容，严格保留合法 JSON 语法格式与键名\n4. 【硬性要求】专名/术语对照: 协议→agreement\n5. 【注意】正式"))
        #expect(prompt.hasSuffix("请以相同的 JSON 格式输出翻译结果，key 保持不变，value 为对应译文。只输出 JSON，不要有任何额外说明。"))
        #expect(!prompt.contains("Preserve"))
    }
    @Test func indexImageOrderAndThinkStrip() throws {
        let messages = LanguagePrompts.textThenImage(prompt: "x")
        let parts = try #require(messages.first?["content"] as? [[String: String]])
        #expect(parts.map { $0["type"] } == ["text", "image"])
        let prompt = try LanguagePrompts.imageTranslation(text: "", target: "de", input: ["glossary": ["a": "b"]])
        #expect(prompt.hasPrefix("请将图片中的文本翻译为德语，")); #expect(prompt.contains("【术语对照】a→b"))
        #expect(LanguagePrompts.indexAnswer("<think>\nx\n</think>\n\nHallo") == "Hallo")
        #expect(LanguagePrompts.indexAnswer("Hallo") == "Hallo")
    }
}

/// Hy-MT2 card supported languages and recommended 7B inference parameters.
extension LanguageTests {
    @Test func hunyuanLanguagesAndSampling() throws {
        #expect(try LanguagePrompts.hunyuanLanguage("zh-Hant") == "Traditional Chinese")
        #expect(try LanguagePrompts.hunyuanLanguage("zh-TW") == "Traditional Chinese")
        #expect(try LanguagePrompts.hunyuanLanguage("yue") == "Cantonese")
        #expect(try LanguagePrompts.hunyuanLanguage("fil") == "Filipino")
        #expect(try LanguagePrompts.hunyuanLanguage("tl") == "Filipino")
        #expect(try LanguagePrompts.hunyuanLanguage("pt-BR") == "Portuguese")
        #expect(try LanguagePrompts.hunyuanLanguage("polish") == "Polish")
        #expect(throws: (any Error).self) { try LanguagePrompts.hunyuanLanguage("sv") }
        let spec: JSONObject = ["sampling": ["temperature": 0.7, "top_p": 0.6, "top_k": 20, "repetition_penalty": 1.05]]
        let sampled = LanguageDecoding.make(spec: spec, input: [:], thinking: false)
        #expect(sampled.temperature == 0.7); #expect(sampled.topP == 0.6); #expect(sampled.topK == 20); #expect(sampled.repetition == 1.05)
        let greedy = LanguageDecoding.make(spec: spec, input: ["temperature": 0], thinking: false)
        #expect(greedy == LanguageDecoding())
    }
}

/// Maker decoding settings as declared in each folder's runners/native/language.json.
extension LanguageTests {
    private func spec(_ model: String) throws -> JSONObject {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        // Retired chat models keep their folders under retired-models/; their decoding defaults still exercise shipped code.
        let shipping = root.appendingPathComponent("models/\(model)")
        let folder = FileManager.default.fileExists(atPath: shipping.path) ? shipping : root.appendingPathComponent("retired-models/\(model)")
        return try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("runners/native/language.json")))
    }
    @Test func makerSamplingDefaults() throws {
        let qwen = try spec("qwen3.5-2b")
        let plain = LanguageDecoding.make(spec: qwen, input: [:], thinking: false)
        #expect(plain.temperature == 0.7); #expect(plain.topP == 0.8); #expect(plain.topK == 20); #expect(plain.presence == 1.5)
        let thinking = LanguageDecoding.make(spec: qwen, input: [:], thinking: true)
        #expect(thinking.temperature == 1.0); #expect(thinking.topP == 0.95); #expect(thinking.presence == 1.5)
        let lfm = LanguageDecoding.make(spec: try spec("lfm2.5-vl-450m"), input: [:], thinking: false)
        #expect(lfm.temperature == 0.1); #expect(lfm.minP == 0.15); #expect(lfm.repetition == 1.05)
        let gemma = LanguageDecoding.make(spec: try spec("gemma-4-e4b-it"), input: ["temperature": 0.5], thinking: false)
        #expect(gemma.temperature == 0.5); #expect(gemma.topP == 0.95); #expect(gemma.topK == 64)
        // Greedy describe/tag: no sampler filters, but at least a 1.05 repetition penalty.
        let describe = LanguageDecoding.make(spec: try spec("lfm2.5-vl-450m"), input: ["temperature": 0], thinking: false, repetitionFloor: 1.05)
        #expect(describe.temperature == 0); #expect(describe.minP == 0); #expect(describe.repetition == 1.05)
    }
    /// NaviDC TeleOCR_client.py DEFAULT_SAMPLING_PARAMS: greedy, repetition 1.0, `no_repeat_ngram_size` 100 for every
    /// kind; presence/frequency for recognition kinds only, none for layout.
    @Test func teleOCRClientPenalties() throws {
        let tele = try spec("teleocr")
        let table = LanguageDecoding.make(spec: tele, input: [:], thinking: false, kind: "table")
        #expect(table.temperature == 0); #expect(table.presence == 1.0); #expect(table.frequency == 0.005); #expect(table.repetition == nil)
        #expect(table.noRepeatNgram == 100)
        let text = LanguageDecoding.make(spec: tele, input: [:], thinking: false, kind: "title")
        #expect(text.presence == 1.0); #expect(text.frequency == 0.05); #expect(text.repetition == nil); #expect(text.noRepeatNgram == 100)
        let layout = LanguageDecoding.make(spec: tele, input: [:], thinking: false, kind: "layout")
        #expect(layout.presence == nil); #expect(layout.frequency == nil); #expect(layout.repetition == nil); #expect(layout.noRepeatNgram == 100)
        #expect(layout.processor(promptTokens: 10, maxTokens: 16) != nil)
        #expect(TeleDocument.kind(mode: "scientific_figure") == "char"); #expect(TeleDocument.kind(mode: "ocr") == nil)
    }
    /// mineru-vl-utils vllm_v1_no_repeat_ngram.py, transcribed: the banned next tokens after `output`.
    private func referenceBans(_ output: [Int], size: Int) -> Set<Int> {
        guard output.count >= size else { return [] }
        var cached: [[Int]: [Int]] = [:]
        for end in (size - 1)..<output.count { cached[Array(output[(end - size + 1)..<end]), default: []].append(output[end]) }
        return Set(cached[Array(output[(output.count - size + 1)...])] ?? [])
    }
    @Test(.heavy) func noRepeatNGramMatchesTheMakersProcessor() {
        MLXReady.ensure()
        var generator = SystemRandomNumberGenerator()
        // Small vocabularies force repeats; every prefix of each sequence is checked, with counts first.
        var checked = 0, banning = 0
        for (size, vocabulary, length) in [(2, 3, 40), (3, 4, 60), (4, 2, 50), (5, 3, 80), (100, 2, 260)] {
            var ngram = NoRepeatNGram(size: size, capacity: length)
            var output: [Int] = []
            for step in 0..<length {
                let logits = ngram.process(logits: MLXArray.zeros([1, vocabulary], type: Float.self))
                let banned = Set(logits.reshaped(-1).asArray(Float.self).enumerated().filter { $0.element == -.infinity }.map(\.offset))
                #expect(banned == referenceBans(output, size: size), "size \(size) step \(step)")
                checked += 1; if !banned.isEmpty { banning += 1 }
                // Mostly repeat the cycle so long n-grams recur; a random token sometimes breaks it.
                let token = Int.random(in: 0..<10, using: &generator) < 8 && output.count >= 3 ? output[output.count - 3] : Int.random(in: 0..<vocabulary, using: &generator)
                output.append(token); ngram.didSample(token: MLXArray([Int32(token)]))
            }
        }
        #expect(checked == 490); #expect(banning > 50)
    }
}

/// Streaming deltas never end inside a multi-byte character.
extension LanguageTests {
    @Test func streamHoldsSplitCharacter() {
        // Byte-level tokens: "a", the first two bytes of "€", its last byte, a newline, "b".
        let bytes: [Int: [UInt8]] = [1: [0x61], 2: [0xE2, 0x82], 3: [0xAC], 4: [0x0A], 5: [0x62]]
        let decode = { (ids: [Int]) in String(decoding: ids.flatMap { bytes[$0]! }, as: UTF8.self) }
        var detokenizer = StreamDetokenizer()
        let tokens = [1, 2, 3, 4, 5]
        var deltas: [String] = []
        for count in 1...tokens.count { deltas.append(detokenizer.next(Array(tokens[..<count]), decode: decode)) }
        #expect(deltas == ["a", "", "€", "\n", "b"])
        #expect(!deltas.joined().contains("\u{FFFD}"))
    }
}

/// Qwen <think> and Gemma thought-channel output become reasoning and answer, streamed or whole.
extension LanguageTests {
    @Test func qwenThinkingSplit() throws {
        let splitter = try #require(ThinkingSplitter.make(architecture: "qwen3_5", thinking: true))
        let whole = ThinkingSplitter.split("The user greets.\n</think>\n\nHello!", splitter: splitter)
        #expect(whole.answer == "Hello!"); #expect(whole.reasoning == "The user greets.")
        // Chunks that cut the closing marker apart.
        var streaming = splitter
        var parts: [ThinkingSplitter.Part] = []
        for chunk in ["The user", " greets.\n</th", "ink>", "\n\nHel", "lo!"] { parts += streaming.append(chunk) }
        parts += streaming.finish()
        #expect(parts.filter(\.reasoning).map(\.text).joined() == "The user greets.\n")
        #expect(parts.filter { !$0.reasoning }.map(\.text).joined() == "Hello!")
        #expect(ThinkingSplitter.frames(parts).first == .json(["event": "reasoning", "text": "The user"]))
        // Thinking off: the template already closed the block; text is the answer.
        let plain = ThinkingSplitter.split("Hi", splitter: try #require(ThinkingSplitter.make(architecture: "qwen3_vl", thinking: false)))
        #expect(plain.answer == "Hi"); #expect(plain.reasoning == nil)
    }
    @Test func gemmaThoughtChannelSplit() throws {
        let splitter = try #require(ThinkingSplitter.make(architecture: "gemma4", thinking: true))
        #expect(ThinkingSplitter.keepsSpecialTokens("gemma4"))
        let whole = ThinkingSplitter.split("<|channel>thought\nCount the cats.<channel|>Two cats.", splitter: splitter)
        #expect(whole.answer == "Two cats."); #expect(whole.reasoning == "Count the cats.")
        let empty = ThinkingSplitter.split("<|channel>thought\n<channel|>Two cats.", splitter: splitter)
        #expect(empty.answer == "Two cats."); #expect(empty.reasoning == nil)
        var streaming = splitter
        var parts: [ThinkingSplitter.Part] = []
        for chunk in ["<|chan", "nel>tho", "ught\nCount.", "<chan", "nel|>Two."] { parts += streaming.append(chunk) }
        parts += streaming.finish()
        #expect(parts == [.init(reasoning: true, text: "Count."), .init(reasoning: false, text: "Two.")])
        #expect(ThinkingSplitter.make(architecture: "lfm2_vl", thinking: true) == nil)
    }
}
