import ArsenalWire
import AVFoundation
import CryptoKit
import Foundation
import IntegrationKit
import PDFKit
import MLX
import Testing

@testable import EmbedWorker

/// EmbeddingGemma 2's processor and tower layout rules against Transformers 5.22.1, captured without weights
/// (`Fixtures/eg2_capture.py`, `Fixtures/eg2-reference.json`, `Fixtures/eg2-picture.png`). MLX work runs on the CPU.
private struct EG2Reference: Decodable {
    struct Layout: Decodable {
        let prompt: String?
        let content, kinds: [String]
        let text: String
    }
    struct Size: Decodable { let h, w, tokens: Int; let out: [Int] }
    struct PictureSet: Decodable {
        struct Processed: Decodable {
            let patches, columns: Int
            let levelsSha256: String
            let firstPatch: [Int]
            enum CodingKeys: String, CodingKey { case patches, columns, levelsSha256 = "levels_sha256", firstPatch = "first_patch" }
        }
        let sha256: String
        let image, video: Processed
    }
    struct Orientation: Decodable { let orientation, width, height: Int; let pixels: [Int] }
    struct Frames: Decodable { let total: Int; let fps: Double; let indices: [Int] }
    struct Mel: Decodable {
        let samples, frames, valid, every, tokens: Int
        let rows: [[Float]]
        let filterColumnSums: [Double]
        enum CodingKeys: String, CodingKey { case samples, frames, valid, every, tokens, rows, filterColumnSums = "filter_column_sums" }
    }
    struct AudioTokens: Decodable { let samples, tokens: Int }
    struct AudioMask: Decodable { let valid, allowed: [Bool]; let shape: [Int] }
    struct RelShift: Decodable { let input, output: [Float]; let shape: [Int]; let context: Int }
    struct Rope: Decodable {
        let columns, rows: Int
        let cos, sin: [Float]
    }
    struct Pool: Decodable { let columns, rows: Int; let hidden, pooled: [Float] }
    let layouts: [Layout]
    let sizes: [Size]
    let picture: PictureSet
    let orientation: [Orientation]
    let frames: [Frames]
    let mel: Mel
    let audioTokens: [AudioTokens]
    let audioMask: AudioMask
    let relShift: RelShift
    let visionRope: Rope
    let visionPool: Pool
    enum CodingKeys: String, CodingKey {
        case layouts, sizes, picture, orientation, frames, mel, audioTokens = "audio_tokens", audioMask = "audio_mask", relShift = "rel_shift"
        case visionRope = "vision_rope", visionPool = "vision_pool"
    }
}

private func fixtureURL(_ name: String, _ ext: String) throws -> URL {
    try #require(Bundle.module.url(forResource: name, withExtension: ext, subdirectory: "Fixtures"))
}
private func reference() throws -> EG2Reference {
    try JSONDecoder().decode(EG2Reference.self, from: Data(contentsOf: try fixtureURL("eg2-reference", "json")))
}
/// The no-fast-math metallib scripts/test.sh lays out beside the native helpers; MLX's CPU stream still loads it.
private func cpuMLX() {
    if let path = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] { GPU.metallib = URL(fileURLWithPath: path) }
}
private func sha256(_ bytes: [UInt8]) -> String { SHA256.hash(data: Data(bytes)).map { String(format: "%02x", $0) }.joined() }

/// A tiny vision layout for the weight-free table and pooling checks (hidden 64, two heads of 32).
private func tinyVision() throws -> EmbeddingGemma2VisionTower {
    let json = """
        {"model_type":"gemma4_vision","hidden_size":64,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":32,
        "intermediate_size":64,"patch_size":16,"pooling_kernel_size":3,"position_embedding_size":16,"rms_norm_eps":1e-6,
        "rope_parameters":{"rope_theta":100.0,"rope_type":"axial"},"standardize":false,"use_clipped_linears":false}
        """
    return EmbeddingGemma2VisionTower(try JSONDecoder().decode(EmbeddingGemma2VisionConfiguration.self, from: Data(json.utf8)))
}

@Test func eg2LayoutsMatchTheChatTemplate() throws {
    let fixture = try reference()
    #expect(fixture.layouts.count == 12)
    for layout in fixture.layouts {
        let parts: [EG2Part] = zip(layout.kinds, layout.content).map { kind, value in
            let url = URL(fileURLWithPath: "/tmp/x")
            switch kind {
            case "image": return .image(url)
            case "video": return .video(url)
            case "audio": return .audio(url)
            default: return .text(value)
            }
        }
        let rendered = try EG2Prompt.render(parts, prefix: layout.prompt).map { piece -> String in
            switch piece {
            case .text(let text): return text
            case .image: return EG2Prompt.image
            case .video: return EG2Prompt.video
            case .audio: return EG2Prompt.audio
            }
        }
        #expect(rendered.joined() == layout.text, "layout \(layout.kinds)")
    }
}

@Test func eg2PlaceholderCountsMustMatchTheMedia() {
    let url = URL(fileURLWithPath: "/tmp/x")
    #expect(throws: (any Error).self) { try EG2Prompt.render([.text("one <|image|> two <|image|>"), .image(url)], prefix: nil) }
    #expect(throws: (any Error).self) { try EG2Prompt.render([.text("<|audio|>")], prefix: nil) }
}

@Test func eg2TaskPrefixesAreTheCards() throws {
    #expect(try EG2Prompt.prefix(task: "query", title: nil, instruction: nil) == "task: search result | query: ")
    #expect(try EG2Prompt.prefix(task: "document", title: nil, instruction: nil) == "title: none | text: ")
    #expect(try EG2Prompt.prefix(task: "document", title: "Aurora", instruction: nil) == "title: Aurora | text: ")
    #expect(try EG2Prompt.prefix(task: "similarity", title: nil, instruction: nil) == "task: sentence similarity | query: ")
    #expect(try EG2Prompt.prefix(task: "code_retrieval", title: nil, instruction: nil) == "task: code retrieval | query: ")
    #expect(try EG2Prompt.prefix(task: "query", title: nil, instruction: "fact checking") == "task: fact checking | query: ")
}

@Test func eg2ResizeMatchesTheProcessorBudget() throws {
    let fixture = try reference()
    #expect(fixture.sizes.count == 18)
    for size in fixture.sizes {
        let out = try EG2Vision.targetSize(height: size.h, width: size.w, maxSoftTokens: size.tokens)
        #expect([out.height, out.width] == size.out, "\(size.h)×\(size.w) at \(size.tokens)")
    }
}

@Test func eg2PicturePatchesMatchTheProcessorExactly() throws {
    let fixture = try reference()
    let url = try fixtureURL("eg2-picture", "png")
    #expect(sha256([UInt8](try Data(contentsOf: url))) == fixture.picture.sha256)
    let picture = try EG2Picture.load(url)
    cpuMLX()
    try Device.withDefaultDevice(.cpu) {
        for (budget, expected) in [(EG2Vision.imageSoftTokens, fixture.picture.image), (EG2Vision.videoSoftTokens, fixture.picture.video)] {
            let patches = try EG2Vision.patches(picture, maxSoftTokens: budget)
            #expect(patches.rows * patches.columns == expected.patches && patches.columns == expected.columns)
            let levels = (patches.pixels * 255).round().asType(.uint8).asArray(UInt8.self)
            #expect(levels.prefix(768).map(Int.init) == expected.firstPatch)
            #expect(sha256(levels) == expected.levelsSha256)
        }
    }
}

@Test func eg2OrientationMatchesPillowsExifTranspose() throws {
    let fixture = try reference()
    #expect(fixture.orientation.count == 7)
    let small = EG2Picture(width: 3, height: 2, pixels: (0..<18).map { UInt8($0 * 10) })
    for case_ in fixture.orientation {
        let out = small.oriented(case_.orientation)
        #expect(out.width == case_.width && out.height == case_.height && out.pixels.map(Int.init) == case_.pixels, "orientation \(case_.orientation)")
    }
}

@Test func eg2FrameSamplingMatchesTheVideoProcessor() throws {
    let fixture = try reference()
    #expect(fixture.frames.count == 7)
    for frames in fixture.frames {
        #expect(EG2Vision.sampleFrames(total: frames.total, fps: frames.fps) == frames.indices, "\(frames.total) frames at \(frames.fps)")
    }
}

@Test func eg2LogMelMatchesTheFeatureExtractor() throws {
    let mel = try reference().mel
    let wave = (0..<mel.samples).map { i -> Float in
        let t = Double(i) / 16000
        return Float(0.4 * sin(2 * Double.pi * (300 * t + 900 * t * t)) + 0.1 * sin(2 * Double.pi * 3000 * t))
    }
    let sums = EG2Mel.filters().reduce(into: [Double](repeating: 0, count: EG2Mel.bins)) { acc, row in for m in 0..<EG2Mel.bins { acc[m] += row[m] } }
    #expect(zip(sums, mel.filterColumnSums).allSatisfy { abs($0 - $1) < 1e-6 })
    let (values, frames, mask) = EG2Mel.features(wave)
    #expect(frames == mel.frames && mask.filter { $0 }.count == mel.valid)
    #expect(EG2Mel.tokens(samples: mel.samples) == mel.tokens)
    var largest: Float = 0
    for (row, expected) in zip(stride(from: 0, to: frames, by: mel.every), mel.rows) {
        for m in 0..<EG2Mel.bins { largest = max(largest, abs(values[row * EG2Mel.bins + m] - expected[m])) }
    }
    #expect(largest < 1e-3, "largest log-mel difference \(largest)")
}

@Test func eg2AudioTokenCountsMatchTheProcessor() throws {
    let counts = try reference().audioTokens
    #expect(counts.count == 6)
    for count in counts { #expect(EG2Mel.tokens(samples: count.samples) == count.tokens, "\(count.samples) samples") }
}

@Test func eg2AudioKeyMaskMatchesTheBlockedMask() throws {
    let fixture = try reference().audioMask
    let config = try JSONDecoder().decode(
        EmbeddingGemma2AudioConfiguration.self,
        from: Data(
            """
            {"model_type":"gemma4_audio","hidden_size":64,"num_attention_heads":2,"num_hidden_layers":1,"output_proj_dims":32,
            "attention_chunk_size":12,"attention_context_left":13,"attention_context_right":0,"attention_logit_cap":50.0,
            "attention_invalid_logits_value":-1e9,"residual_weight":0.5,"conv_kernel_size":5,"subsampling_conv_channels":[16,16],
            "rms_norm_eps":1e-6,"gradient_clipping":1e10,"use_clipped_linears":true,"hidden_act":"silu"}
            """.utf8))
    cpuMLX()
    try Device.withDefaultDevice(.cpu) {
        let tower = EmbeddingGemma2AudioTower(config)
        let allowed = tower.allowedKeys(valid: fixture.valid)
        #expect(allowed.shape == fixture.shape)
        #expect(allowed.flattened().asArray(Bool.self) == fixture.allowed)
        _ = try #require(Optional(tower))
    }
}

@Test func eg2RelativeShiftMatchesTheMakersRelShift() throws {
    let fixture = try reference().relShift
    cpuMLX()
    try Device.withDefaultDevice(.cpu) {
        let shifted = EG2AudioAttention.relativeShift(MLXArray(fixture.input, fixture.shape), context: fixture.context)
        let values = shifted.flattened().asArray(Float.self)
        #expect(values.count == fixture.output.count)
        #expect(zip(values, fixture.output).allSatisfy { $0 == $1 })
        _ = try #require(Optional(values))
    }
}

@Test func eg2VisionRopeAndPoolingMatchTheMaker() throws {
    let fixture = try reference()
    let tower = try tinyVision()
    cpuMLX()
    try Device.withDefaultDevice(.cpu) {
        let rope = fixture.visionRope
        let (cos, sin) = tower.rotary(columns: rope.columns, rows: rope.rows, dtype: .float32)
        #expect(cos.shape == [rope.columns * rope.rows, 32])
        // The tiny tower uses head dim 32; the maker table (head dim 64) repeats each axis' frequencies, so compare its
        // own-size table by recomputing the maker's for 32: the first 8 frequencies of each 16-wide axis block agree.
        let theirs = MLXArray(rope.cos, [rope.columns * rope.rows, 64]), theirsSin = MLXArray(rope.sin, [rope.columns * rope.rows, 64])
        let wide = tower.rotary(columns: rope.columns, rows: rope.rows, dtype: .float32, headDim: 64)
        #expect(abs(wide.0 - theirs).max().item(Float.self) < 1e-5)
        #expect(abs(wide.1 - theirsSin).max().item(Float.self) < 1e-5)
        let pool = fixture.visionPool
        let pooled = tower.pool(MLXArray(pool.hidden, [pool.columns * pool.rows, 4]), columns: pool.columns, rows: pool.rows)
        let unscaled = pooled / Float(64).squareRoot()
        #expect(abs(unscaled - MLXArray(pool.pooled, [pool.columns * pool.rows / 9, 4])).max().item(Float.self) < 1e-6)
        _ = try #require(Optional(sin))
    }
}

/// Review fix (2 Nov 2026): an out-of-range `dimensions` (1e300) trapped in Int(Double) before the enum check.
@Test func eg2OptionsRefuseBadValuesWithoutTrapping() throws {
    #expect(try EmbeddingGemma2Adapter.options(["dimensions": .number(256)], width: 768).dimension == 256)
    for bad: JSONValue in [.number(1e300), .number(-1e300), .number(300), .number(256.5), .number(.nan)] {
        #expect(throws: (any Error).self) { try EmbeddingGemma2Adapter.options(["dimensions": bad], width: 768) }
    }
    #expect(throws: (any Error).self) { try EmbeddingGemma2Adapter.options(["title": .string("")], width: 768) }
    #expect(throws: (any Error).self) { try EmbeddingGemma2Adapter.options(["task": .string("rerank")], width: 768) }
    let titled = try EmbeddingGemma2Adapter.options(["task": .string("document"), "title": .string("Aurora")], width: 768)
    #expect(titled.prefix == "title: Aurora | text: " && titled.normalize && !titled.truncate)
}

/// Review fix (2 Nov 2026): 5.1 audio failed because AVAudioFormat has no plain 6-channel PCM target; it is now the mean of
/// its channels, as librosa's mono load gives.
@Test func eg2SurroundAudioIsTheMeanOfItsChannels() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("eg2-surround-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    for rate in [16_000.0, 48_000.0] {
        let layout = try #require(AVAudioChannelLayout(layoutTag: kAudioChannelLayoutTag_MPEG_5_1_A))
        let format = AVAudioFormat(standardFormatWithSampleRate: rate, channelLayout: layout)
        let url = directory.appendingPathComponent("six-\(Int(rate)).wav")
        let frames = Int(rate / 2)
        do {
            let file = try AVAudioFile(forWriting: url, settings: format.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
            let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)))
            buffer.frameLength = AVAudioFrameCount(frames)
            for c in 0..<6 { for i in 0..<frames { buffer.floatChannelData![c][i] = Float(c + 1) / 20 } }
            try file.write(from: buffer)
        }
        let mono = try EG2AudioIO.mono(url, maximumSamples: EG2AudioIO.decodeLimit)
        #expect(abs(mono.count - 8_000) <= 2, "\(rate) Hz gave \(mono.count) samples")
        let middle = mono[1000..<7000]
        #expect(middle.allSatisfy { abs($0 - 0.175) < 2e-3 }, "\(rate) Hz mean \(middle.first ?? .nan)")
    }
}

/// Review fix (2 Nov 2026): PDF pages were drawn from the media box without /Rotate, so rotated scans came out sideways and
/// crop-box margins were kept. Review fix (M1): the page was then drawn at 1× in the middle of its 2× canvas, because
/// `getDrawingTransform` never scales up; the ink must now reach the canvas edges.
@Test func eg2RotatedPDFPagesRenderUpright() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("eg2-pdf-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let url = try rotatedCroppedPDF(in: directory)
    let pages = try EmbeddingGemma2Adapter.renderPDF(url, cancellation: NativeCancellation())
    #expect(pages.count == 1)
    let rendered = try #require(pages.first)
    // Crop 300×200 turned upright is 200 wide, 300 tall, at the 2× render scale.
    expectRotatedPageFillsTheCanvas(width: rendered.width, height: rendered.height) { x, y in
        let i = 3 * (y * rendered.width + x)
        return (rendered.pixels[i], rendered.pixels[i + 1], rendered.pixels[i + 2])
    }
}

/// The long-text path: blocked sliding-window attention (taken once 3·blocks·window² < L²) equals full attention under
/// the maker's inclusive |i - j| ≤ window mask. The parity inputs were too short to reach it.
@Test func eg2WindowAttentionMatchesMaskedFullAttention() throws {
    cpuMLX()
    try Device.withDefaultDevice(.cpu) {
        MLXRandom.seed(3)
        let (length, radius) = (41, 8)
        let q = MLXRandom.normal([1, 2, length, 16]), k = MLXRandom.normal([1, 2, length, 16]), v = MLXRandom.normal([1, 2, length, 16])
        #expect(3 * ((length + radius - 1) / radius) * radius * radius < length * length)
        let blocked = EmbeddingGemma2Span.window(radius: radius, validKeys: nil).attention(queries: q, keys: k, values: v)
        let pattern = EmbeddingGemma2Masks.slidingWindowPattern(seqLen: length, radius: radius)
        let full = EmbeddingGemma2Span.full(pattern).attention(queries: q, keys: k, values: v)
        #expect(blocked.shape == full.shape)
        let difference = abs(blocked - full).max().item(Float.self)
        #expect(difference < 1e-5, "largest difference \(difference)")
        _ = try #require(Optional(difference))
    }
}
