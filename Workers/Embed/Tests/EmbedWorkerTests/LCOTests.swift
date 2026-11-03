import Foundation
import MLX
import Testing
@testable import EmbedWorker

/// LCO-Embedding Omni's processor and layout rules against Transformers 5.22.1 / qwen-omni-utils, captured without
/// weights (`Fixtures/lco_capture.py`, `Fixtures/lco-reference.json`).
private struct LCOReference: Decodable {
    struct Mel: Decodable {
        let samples, frames, every: Int
        let rows: [[Float]]
        let filterColumnSums: [Double]
        let chirp: [Double]
        enum CodingKeys: String, CodingKey { case samples, frames, every, rows, filterColumnSums = "filter_column_sums", chirp }
    }
    struct Rope: Decodable {
        let ids: [Int]
        let image, video: [Int]
        let audioFrames, audioTokens: Int
        let positions: [[Int]]
        enum CodingKeys: String, CodingKey { case ids, image, video, audioFrames = "audio_frames", audioTokens = "audio_tokens", positions }
    }
    struct Window: Decodable { let grid: [Int]; let index, cu: [Int] }
    struct Resize: Decodable { let h, w, min, max: Int; let out: [Int] }
    struct Frames: Decodable { let total: Int; let fps: Double; let frames: Int }
    struct Budget: Decodable { let frames: Int; let max: Double }
    struct Prompt: Decodable { let content, kinds: [String]; let text: String }
    let mel: Mel
    let rope: Rope
    let windows: [Window]
    let resize: [Resize]
    let nframes: [Frames]
    let videoBudget: [Budget]
    let prompts: [Prompt]
    enum CodingKeys: String, CodingKey { case mel, rope, windows, resize, nframes, videoBudget = "video_budget", prompts }
}

private func reference() throws -> LCOReference {
    let url = try #require(Bundle.module.url(forResource: "lco-reference", withExtension: "json", subdirectory: "Fixtures"))
    return try JSONDecoder().decode(LCOReference.self, from: Data(contentsOf: url))
}

@Test func lcoPromptMatchesTheSentenceTransformersTemplate() throws {
    let fixture = try reference()
    #expect(fixture.prompts.count == 7)
    for prompt in fixture.prompts {
        let parts: [LCOPart] = zip(prompt.kinds, prompt.content).map { kind, value in
            let url = URL(fileURLWithPath: "/tmp/x")
            switch kind {
            case "image": return .image(url)
            case "video": return .video(url)
            case "audio": return .audio(url)
            default: return .text(value)
            }
        }
        let rendered = LCOPrompt.render(parts).map { piece -> String in
            switch piece {
            case .text(let text): return text
            case .image: return "<|IMAGE|>"
            case .video: return "<|VIDEO|>"
            case .audio: return "<|AUDIO|>"
            }
        }.joined()
        #expect(rendered == prompt.text)
    }
}

@Test func lcoPositionsMatchGetRopeIndex() throws {
    let rope = try reference().rope
    // The fixture: text, image (1,4,6), text, audio, text, video (2,4,4) at 1 s per grid, text.
    #expect(LCOAudioEncoder.outputTokens(frames: rope.audioFrames) == rope.audioTokens)
    var sequence = LCOSequence()
    let ids = rope.ids
    var i = 0
    while i < ids.count {
        switch ids[i] {
        case 151655:
            sequence.vision(token: 151655, grid: LCOGrid(t: rope.image[0], h: rope.image[1], w: rope.image[2]), merge: 2, temporalStep: 25, kind: "image")
            i += rope.image[0] * rope.image[1] * rope.image[2] / 4
        case 151656:
            sequence.vision(token: 151656, grid: LCOGrid(t: rope.video[0], h: rope.video[1], w: rope.video[2]), merge: 2, temporalStep: 25, kind: "video")
            i += rope.video[0] * rope.video[1] * rope.video[2] / 4
        case 151646:
            sequence.audio(token: 151646, count: rope.audioTokens); i += rope.audioTokens
        default:
            sequence.text([ids[i]]); i += 1
        }
    }
    #expect(sequence.ids == ids)
    #expect(sequence.positions == rope.positions)
    #expect(sequence.media.map(\.kind) == ["image", "audio", "video"])
}

@Test func lcoWindowIndexMatchesTheVisionEncoder() throws {
    let windows = try reference().windows
    #expect(windows.count == 4)
    for case let window in windows {
        let grid = LCOGrid(t: window.grid[0], h: window.grid[1], w: window.grid[2])
        let (index, cu) = LCOVisionLayout.windows([grid], merge: 2, windowSize: 112, patchSize: 14)
        #expect(index == window.index)
        #expect(cu == window.cu)
    }
    #expect(LCOVisionLayout.frames([LCOGrid(t: 2, h: 4, w: 6)]) == [0, 24, 48])
}

@Test func lcoResizeAndFrameRulesMatchQwenOmniUtils() throws {
    let fixture = try reference()
    for r in fixture.resize {
        let (h, w) = try LCOMedia.smartResize(height: r.h, width: r.w, minPixels: r.min, maxPixels: r.max)
        #expect([h, w] == r.out, "\(r.h)x\(r.w)")
    }
    for f in fixture.nframes { #expect(LCOMedia.frameCount(total: f.total, fps: f.fps) == f.frames, "\(f.total)@\(f.fps)") }
    for b in fixture.videoBudget { #expect(LCOMedia.videoMaxPixels(frames: b.frames) == Int(b.max)) }
    #expect(LCOMedia.sampleIndices(total: 8, count: 4) == [0, 2, 5, 7])
}

@Test func lcoAudioChunksFollowTheEncoder() {
    #expect(LCOAudioEncoder.chunkLengths(frames: 437, window: 100) == [200, 200, 37])
    #expect(LCOAudioEncoder.chunkLengths(frames: 400, window: 100) == [200, 200])
    #expect(LCOMel.frames(samples: 21_937) == 138)
    #expect(LCOMel.frames(samples: 10_000_000) == 30_000)
}

/// Whisper log-mel with Accelerate (no MLX): filter bank and features on a fixed chirp.
@Test func lcoMelFeaturesMatchWhisperFeatureExtractor() throws {
    let mel = try reference().mel
    let sums = LCOMel.filters().reduce(into: [Double](repeating: 0, count: 128)) { acc, row in for (i, v) in row.enumerated() { acc[i] += v } }
    for (a, b) in zip(sums, mel.filterColumnSums) { #expect(abs(a - b) < 1e-6) }
    let c = mel.chirp
    let pcm = (0..<mel.samples).map { i -> Float in
        let t = Double(i) / 16000
        return Float(c[2] * sin(2 * Double.pi * (c[0] * t + c[1] * t * t)) + c[4] * sin(2 * Double.pi * c[3] * t))
    }
    let (values, frames) = LCOMel.features(pcm)
    #expect(frames == mel.frames && values.count == 128 * frames)
    let rows = stride(from: 0, to: frames, by: mel.every).flatMap { t in (0..<128).map { values[$0 * frames + t] } }
    let expected = mel.rows.flatMap { $0 }
    #expect(rows.count == expected.count)
    let worst = zip(rows, expected).map { abs($0 - $1) }.max() ?? 1
    #expect(worst < 2e-4, "max |Δ| \(worst)")
}

/// Needs MLX arrays (the Metal library), so it runs with the heavy suite although it evaluates nothing.
@Test(.heavy) func lcoCheckpointMapSkipsUnusedTensorsAndReordersConvolutions() {
    #expect(LCOGraph.map("lm_head.weight", MLXArray.zeros([2, 2])) == nil)
    #expect(LCOGraph.map("audio_tower.audio_bos_eos_token.weight", MLXArray.zeros([2, 2])) == nil)
    #expect(LCOGraph.map("audio_tower.conv1.weight", MLXArray.zeros([1280, 128, 3]))?.1.shape == [1280, 3, 128])
    #expect(LCOGraph.map("visual.patch_embed.proj.weight", MLXArray.zeros([1280, 3, 2, 14, 14]))?.1.shape == [1280, 1176])
    #expect(LCOGraph.map("model.layers.0.self_attn.q_proj.bias", MLXArray.zeros([4]))?.0 == "model.layers.0.self_attn.q_proj.bias")
}
