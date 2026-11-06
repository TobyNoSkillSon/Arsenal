// SPDX-License-Identifier: MIT
// The image path against the maker (OneJev qev @81ce62f + serve.patch, train.common.batch_scores for multimodal rows;
// transformers 5.19.0 Qwen3VLProcessor). Fixtures: maker-mm.json (Fixtures/maker_mm_fixture.py from
// Fixtures/maker_capture_mm.py) and maker-decode.json (Fixtures/maker_decode_fixture.py: PIL decode, torchvision resize).
// Weight-free: decode, resize, prompts, grids, pixels, token ids, M-RoPE positions, placeholder rules.
// Heavy: answers through ElephantEngine.decide, vision tower and per-layer states against the maker's capture.
@testable import ElephantEngine
import CryptoKit
import Foundation
import MLX
import Testing

struct MakerMMBranch: Decodable {
    let prompt: String
    let input_ids: [Int]
    let prefix_tokens: Int
    let image_grid_thw: [[Int]]
    let positions: [[Int]]
    let rope_delta: Int
}

struct MakerMMCase: Decodable {
    struct Pixels: Decodable { let count: Int; let stride: Int; let values: [Double] }
    let id: String
    let request: String
    let branches: [MakerMMBranch]
    let slot_logits_bf16: [[Double]]
    let pixels: Pixels
}

func makerMM() throws -> [MakerMMCase] {
    struct File: Decodable { let cases: [MakerMMCase] }
    let url = try #require(Bundle.module.url(forResource: "maker-mm", withExtension: "json", subdirectory: "Fixtures"))
    return try JSONDecoder().decode(File.self, from: Data(contentsOf: url)).cases
}

func fixtureMedia() throws -> URL { try #require(Bundle.module.url(forResource: "media", withExtension: nil, subdirectory: "Fixtures")) }

func mmRequest(_ text: String, media: URL) throws -> (state: PyJSON?, questions: [(String, Question)], images: [ImageSource]) {
    let request = try PyJSONParser.parse(Data(text.utf8))
    let images = try (request["media"]?.array ?? []).map { try ImageInput.probe(media.appendingPathComponent(try #require($0["path"]?.string))) }
    return (request["state"].flatMap { $0.isNull ? nil : $0 }, try Requests.questions(request["questions"]), images)
}

/// A request's media list as the served path reads it: images and frame lists (fps default 1.0).
func mmItems(_ text: String, media: URL) throws -> (state: PyJSON?, questions: [(String, Question)], items: [MediaItem]) {
    let request = try PyJSONParser.parse(Data(text.utf8))
    let items = try (request["media"]?.array ?? []).map { item -> MediaItem in
        if item["type"]?.string == "video" {
            let frames = try (item["frames"]?.array ?? []).map { media.appendingPathComponent(try #require($0.string)) }
            return .video(try VideoInput.probe(frames, fps: item["fps"].flatMap { Double($0.compact()) } ?? 1.0))
        }
        return .image(try ImageInput.probe(media.appendingPathComponent(try #require(item["path"]?.string))))
    }
    return (request["state"].flatMap { $0.isNull ? nil : $0 }, try Requests.questions(request["questions"]), items)
}

func makeElephantProcessor() throws -> ImageProcessorConfig {
    try ImageProcessorConfig(json: [
        "image_processor": [
            "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2, "size": ["shortest_edge": 65536, "longest_edge": 16_777_216],
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "rescale_factor": 1.0 / 255, "resample": 3
        ]
    ])
}

func sha256(_ bytes: [UInt8]) -> String { SHA256.hash(data: Data(bytes)).map { String(format: "%02x", $0) }.joined() }

struct DecodeEntry: Decodable {
    struct Resized: Decodable { let width: Int; let height: Int; let sha256: String }
    struct Sample: Decodable { let stride: Int; let values: [Int] }
    let width: Int, height: Int, rgb_sha256: String, resized: Resized
    let rgb: [Int]?
    let rgb_sample: Sample?
}

func makerDecode() throws -> [String: DecodeEntry] {
    let url = try #require(Bundle.module.url(forResource: "maker-decode", withExtension: "json", subdirectory: "Fixtures"))
    return try JSONDecoder().decode([String: DecodeEntry].self, from: Data(contentsOf: url))
}

@Suite struct MediaTests {
    @Test func decodeMatchesPIL() throws {
        let entries = try makerDecode()
        #expect(entries.count == 10)
        for (name, e) in entries.sorted(by: { $0.key < $1.key }) {
            let image = try ImageInput.load(try fixtureMedia().appendingPathComponent(name))
            #expect(image.width == e.width && image.height == e.height, "\(name) size")
            if let sample = e.rgb_sample {
                // JPEG: ImageIO's decoder against PIL's libjpeg-turbo, measured.
                var worst = 0, total = 0
                for (i, v) in sample.values.enumerated() {
                    let d = Swift.abs(Int(image.bytes[i * sample.stride]) - v)
                    worst = Swift.max(worst, d); total += d
                }
                let mean = Double(total) / Double(sample.values.count)
                // Measured 2 Nov 2026: up to 62 levels at sharp colour edges (4:2:0 chroma upsampling differs), mean 1.30.
                // A regression guard at that level, not a parity claim; the effect on answers is in MediaParityTests.
                #expect(worst <= 64 && mean <= 1.5, "\(name) JPEG decode differs from PIL by up to \(worst) levels, mean \(mean)")
            } else {
                #expect(sha256(image.bytes) == e.rgb_sha256, "\(name) decode")
                if let rgb = e.rgb { #expect(image.bytes.map(Int.init) == rgb, "\(name) bytes") }
            }
        }
    }

    @Test func resizeMatchesTorchvisionExactly() throws {
        // Every PNG fixture decodes exactly as PIL does, so its resize must equal torchvision's on PIL's bytes.
        for (name, e) in try makerDecode() where name.hasSuffix(".png") {
            let image = try ImageInput.load(try fixtureMedia().appendingPathComponent(name))
            let resized = Resample.bicubic(image, width: e.resized.width, height: e.resized.height)
            #expect(sha256(resized.bytes) == e.resized.sha256, "\(name) resize to \(e.resized.width)x\(e.resized.height)")
        }
    }

    @Test func promptsGridsAndPixelsMatchTheMaker() throws {
        let cases = try makerMM()
        #expect(cases.count == 5 && cases.map(\.branches.count) == [3, 2, 1, 2, 1])
        let processor = try makeElephantProcessor()
        for c in cases {
            let (state, questions, images) = try mmRequest(c.request, media: try fixtureMedia())
            let plan = try ElephantEngine.branches(questions.map(\.1), maxSlots: 256)
            for (r, b) in zip(plan.rendered, c.branches) {
                #expect(same(try Prompt.chat(state: state, suffix: r.suffix, images: images.count), b.prompt), "\(c.id) prompt")
            }
            let grids = try images.map { try ImageInput.tokens(height: $0.height, width: $0.width, config: processor).grid }
            #expect(grids.map { [$0.t, $0.h, $0.w] } == c.branches[0].image_grid_thw, "\(c.id) grids")
            var pixels: [Float] = []
            for image in images { pixels += try ImageInput.pixels(try ImageInput.load(image.url), config: processor).0 }
            #expect(pixels.count == c.pixels.count, "\(c.id) pixel count")
            var worst = 0.0
            for (i, v) in c.pixels.values.enumerated() { worst = Swift.max(worst, Swift.abs(Double(pixels[i * c.pixels.stride]) - v)) }
            // PNG pictures: identical to the processor's float32 values. JPEG: ImageIO's decode, measured in decodeMatchesPIL.
            let bound = c.id == "jpeg" ? 64.0 / 127.5 : 1e-6
            #expect(worst <= bound, "\(c.id) max normalised pixel difference \(worst)")
        }
    }

    @Test(.makerTokenizer) func tokenIdsAndPositionsMatchTheMaker() throws {
        let dir = try #require(makerTokenizerDirectory())
        let tokenizer = try ElephantEngine.tokenizer(dir)
        let processor = try makeElephantProcessor()
        let imageToken = 248056
        for c in try makerMM() {
            let (state, questions, images) = try mmRequest(c.request, media: try fixtureMedia())
            let plan = try ElephantEngine.branches(questions.map(\.1), maxSlots: 256)
            let layout = try ElephantEngine.layout(tokenizer, state, plan.rendered, images: images.count)
            let grids = try images.map { try ImageInput.tokens(height: $0.height, width: $0.width, config: processor).grid }
            let prefix = try ElephantEngine.expand(layout.prefix, grids: grids, imageTokenId: imageToken, merge: 2)
            let rope = try MultimodalPositions.rope(prefix, grids: grids, imageTokenId: imageToken, merge: 2)
            for (suffix, b) in zip(layout.suffixes, c.branches) {
                #expect(prefix.count == b.prefix_tokens, "\(c.id) prefix length")
                #expect(prefix + suffix == b.input_ids, "\(c.id) ids")
                #expect(rope.delta == b.rope_delta, "\(c.id) delta")
                for axis in 0..<3 {
                    #expect(rope.positions[axis].map(Int.init) == Array(b.positions[axis][0..<prefix.count]), "\(c.id) prefix positions")
                    #expect(Array(b.positions[axis][prefix.count...]) == Array((prefix.count + rope.delta)..<(b.input_ids.count + rope.delta)))
                }
            }
        }
    }

    @Test func placeholderRules() throws {
        let block = Prompt.visionBlock
        #expect(try Prompt.visionUser("a <image:1> b <image:2>", images: 2) == "a \(block) b \(block)")
        #expect(try Prompt.visionUser("<<image:1>>", images: 1) == "<\(block)>")
        // Python's \d is any Unicode decimal digit and int() reads it: <image:١> (Arabic-Indic one) is image 1.
        #expect(try Prompt.visionUser("<image:\u{0661}>", images: 1) == block)
        #expect(throws: PromptError.self) { try Prompt.visionUser("<image:2> <image:1>", images: 2) } // out of order
        #expect(throws: PromptError.self) { try Prompt.visionUser("<image:1>", images: 2) } // one unreferenced
        #expect(throws: PromptError.self) { try Prompt.visionUser("<image:1> <image:1>", images: 1) } // twice
        #expect(throws: PromptError.self) { try Prompt.visionUser("<video:1>", images: 1) } // videos are not ported
        #expect(throws: PromptError.self) { try Prompt.visionUser("<image:3>", images: 1) }
        // A number past Int is still a placeholder, which matches no item (Python: int() then a refusal).
        #expect(throws: PromptError.self) { try Prompt.visionUser("<image:1> <image:99999999999999999999999>", images: 1) }
        #expect(Prompt.hasPlaceholder("see <image:1>") && !Prompt.hasPlaceholder("see <image:x>"))
        // Without media the placeholder text is left as written, as the maker renders it.
        #expect(try Prompt.chat(state: .string("<image:1>"), suffix: "Q").contains("<state>\n<image:1>\n</state>"))
    }

    @Test func visionTokensInAQuestionAreRefused() throws {
        let vision = try VisionConfig(json: [
            "image_token_id": 248056, "video_token_id": 248057, "vision_start_token_id": 248053, "vision_end_token_id": 248054,
            "vision_config": [
                "depth": 1, "hidden_size": 8, "num_heads": 2, "intermediate_size": 8, "patch_size": 16, "temporal_patch_size": 2,
                "spatial_merge_size": 2, "out_hidden_size": 8, "num_position_embeddings": 4
            ]
        ])
        #expect(throws: PromptError.self) { try ElephantEngine.refuseVisionTokens([[1, 2], [3, 248056]], vision: vision) }
        try ElephantEngine.refuseVisionTokens([[1, 2], [3, 4]], vision: vision)
    }

    @Test func smartResizeAndSizeChecks() throws {
        // transformers smart_resize(h, w, 32, 65536, 16777216): (240, 320) -> (256, 320); (300, 400) -> (288, 384);
        // (40, 50) -> (256, 288); (5000, 4000) -> (4576, 3648).
        func r(_ h: Int, _ w: Int) throws -> [Int] {
            let s = try ImageInput.smartResize(height: h, width: w, factor: 32, minPixels: 65536, maxPixels: 16_777_216)
            return [s.height, s.width]
        }
        #expect(try r(240, 320) == [256, 320] && r(300, 400) == [288, 384] && r(40, 50) == [256, 288])
        #expect(try r(5000, 4000) == [4576, 3648])
        #expect(throws: PromptError.self) { try r(1, 300) }
        // probe reads the stored size without decoding.
        let source = try ImageInput.probe(try fixtureMedia().appendingPathComponent("textured-300x260.png"))
        #expect(source.width == 300 && source.height == 260)
    }
}

@Suite(.serialized) struct MediaParityTests {
    init() { if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) } }

    /// Requests: the tracked fixtures, or ELEPHANT_LAB_REQUESTS with pictures in ELEPHANT_LAB_MEDIA (adds the lab photo).
    func requests() throws -> (ids: [String], texts: [String], media: URL) {
        let env = ProcessInfo.processInfo.environment
        if let path = env["ELEPHANT_LAB_REQUESTS"], let media = env["ELEPHANT_LAB_MEDIA"] {
            // Parsed in key order (PyJSON), each request rewritten as JSON text with state, questions and media.
            let list = try #require(try PyJSONParser.parse(Data(contentsOf: URL(fileURLWithPath: path))).array)
            let texts = list.map { item in PyJSON.object((item.fields ?? []).filter { ["state", "questions", "media"].contains($0.0) }).compact() }
            return (list.map { $0["id"]?.string ?? "" }, texts, URL(fileURLWithPath: media))
        }
        let cases = try makerMM()
        return (cases.map(\.id), cases.map(\.request), try fixtureMedia())
    }

    @Test(.heavy) func imagePathMatchesTheMaker() throws {
        let env = ProcessInfo.processInfo.environment
        let weights = URL(fileURLWithPath: try #require(env["ELEPHANT_WEIGHTS"]))
        let evidence = URL(fileURLWithPath: try #require(env["ELEPHANT_MM_EVIDENCE"]))
        struct Ref: Decodable {
            struct Case: Decodable { struct B: Decodable { let slot_logits_blocks: [Double] }; let id: String; let branches: [B] }
            let cases: [Case]
        }
        func ref(_ dtype: String) throws -> [String: Ref.Case] {
            let cases = try JSONDecoder().decode(Ref.self, from: Data(contentsOf: evidence.appendingPathComponent("mm-reference-\(dtype).json"))).cases
            return Dictionary(uniqueKeysWithValues: cases.map { ($0.id, $0) })
        }
        let bf16 = try ref("bfloat16"), fp16 = try ref("float16")
        func comps(_ dtype: String) throws -> [String: [Double]] {
            try #require(
                try JSONSerialization.jsonObject(with: Data(contentsOf: evidence.appendingPathComponent("mm-components-\(dtype).json"))) as? [String: [Double]])
        }
        let componentsFP16 = try comps("float16"), componentsBF16 = try comps("bfloat16")
        let engine = try ElephantEngine(weights: weights)
        let (ids, texts, mediaDir) = try requests()
        func cosine(_ a: [Double], _ b: [Double]) -> Double {
            let dot = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
            return dot / (a.reduce(0) { $0 + $1 * $1 }.squareRoot() * b.reduce(0) { $0 + $1 * $1 }.squareRoot())
        }
        /// An answer's probabilities in option order (noul: yes, no).
        func probabilities(_ answer: PyJSON) -> [Double] {
            if let p = answer["noul"], let v = Double(p.compact()) { return [v, 1 - v] }
            return (answer["probabilities"]?.fields ?? []).map { Double($0.1.compact()) ?? .nan }
        }
        var worstBF16 = 0.0, worstFP16 = 0.0, flips = 0, branches = 0, worstCosine = 1.0, worstName = "", maxAbs = 0.0
        var report: [[String: Any]] = []
        for (id, text) in zip(ids, texts) {
            let (state, questions, items) = try mmItems(text, media: mediaDir)
            // ImageIO's JPEG decode differs from PIL's (MediaTests.decodeMatchesPIL), so a JPEG request checks answers
            // only; its vision and decoder cosines are recorded. Feed PIL-decoded PNGs to check the model itself.
            let urls = items.flatMap { item -> [URL] in
                switch item {
                case .image(let i): return [i.url]
                case .video(let v): return v.frames
                }
            }
            let identicalPixels = !urls.contains { ["jpg", "jpeg"].contains($0.pathExtension.lowercased()) }
            let rb = try #require(bf16[id]), rf = try #require(fp16[id])
            // Answers through the served path.
            let decided = try engine.decide(state: state, questions: questions, media: items)
            #expect(decided.answers.count == rb.branches.count)
            var caseWorst = 0.0
            for ((_, answer), (lb, lf)) in zip(decided.answers, zip(rb.branches, rf.branches)) {
                let p = probabilities(answer)
                let pb = Answers.softmax(lb.slot_logits_blocks), pf = Answers.softmax(lf.slot_logits_blocks)
                #expect(p.count == pb.count, "\(id) option count")
                // Served probabilities are rounded to six places: 5e-7 of the difference is rounding.
                let db = zip(p, pb).map { Swift.abs($0 - $1) }.max()!, df = zip(p, pf).map { Swift.abs($0 - $1) }.max()!
                worstBF16 = Swift.max(worstBF16, db); worstFP16 = Swift.max(worstFP16, df); caseWorst = Swift.max(caseWorst, Swift.max(db, df))
                let a = p.firstIndex(of: p.max()!)!
                if a != pb.firstIndex(of: pb.max()!)! || a != pf.firstIndex(of: pf.max()!)! { flips += 1 }
                branches += 1
            }
            // Components: the first question's prefix and suffix pass, compared at the suffix's last token.
            let plan = try engine.branches([questions[0].1])
            let (layout, media) = try engine.prepare(state, plan.rendered, media: items)
            var captured: [String: [Double]] = [:], pass = -1
            _ = try engine.run(layout, plan.rendered, media: media) { name, value in
                if name == "embed" { pass += 1 }
                if name == "vision" || pass == 1 { captured[name] = value.asType(.float32).asArray(Float.self).map(Double.init) }
            }
            var rows: [String: Double] = [:]
            for (name, value) in captured {
                let finite = value.allSatisfy(\.isFinite)
                #expect(finite, "\(id).\(name) finite")
                maxAbs = Swift.max(maxAbs, value.map(Swift.abs).max() ?? 0)
                let key = "\(id).\(name)"
                guard let maker = componentsFP16[key] else { continue } // the maker captured no embedding row
                guard maker.count == value.count else {
                    Issue.record("\(key): \(maker.count) maker values, \(value.count) here")
                    continue
                }
                let cs = cosine(value, maker)
                rows[name] = cs
                if name == "vision" {
                    let makerBF16 = try #require(componentsBF16[key])
                    let floor = cosine(makerBF16, maker)
                    rows["vision_floor"] = floor
                    rows["vision_vs_maker_bf16"] = cosine(value, makerBF16)
                    if let fp32 = componentsBF16["\(id).vision_fp32"] {
                        // Diagnostic: the maker's vision tower in float32 on the GPU, same pixels.
                        rows["vision_vs_fp32"] = cosine(value, fp32)
                        rows["maker_fp16_vs_fp32"] = cosine(maker, fp32)
                        rows["maker_bf16_vs_fp32"] = cosine(makerBF16, fp32)
                    }
                    if identicalPixels { #expect(cs >= Swift.min(0.999, floor), "\(key) \(cs) below the maker's own floor \(floor)") }
                    continue
                }
                if identicalPixels, cs < worstCosine { worstCosine = cs; worstName = key }
            }
            report.append([
                "id": id, "max_abs_dp": caseWorst, "components": rows, "identical_pixels": identicalPixels, "answers": sortedCompact(.object(decided.answers))
            ])
        }
        #expect(flips == 0)
        #expect(worstBF16 <= 0.025 && worstFP16 <= 0.025, "bf16 \(worstBF16) fp16 \(worstFP16)")
        #expect(worstCosine >= 0.999, "worst \(worstName) \(worstCosine)")
        if let out = env["ELEPHANT_MM_PARITY_OUT"] {
            let summary: [String: Any] = [
                "branches": branches, "argmax_flips": flips, "max_abs_dp_vs_maker_bf16": worstBF16, "max_abs_dp_vs_maker_fp16": worstFP16,
                "worst_decoder_cosine": worstCosine, "worst_decoder": worstName, "max_abs_activation": maxAbs, "cases": report,
                "peak_mlx_mb": Double(Memory.peakMemory) / 1e6
            ]
            try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: out))
        }
    }
}
