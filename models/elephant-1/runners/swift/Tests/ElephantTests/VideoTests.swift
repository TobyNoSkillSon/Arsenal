// SPDX-License-Identifier: MIT
// Video frame lists against the maker (OneJev qev @81ce62f + serve.patch, train.common.batch_scores for multimodal rows;
// transformers 5.19.0 Qwen3VLVideoProcessor with the makers' videos_kwargs and Qwen3VLProcessor). Fixtures:
// requests-video.json, media/ball-*.png and light-*.png (maker_make_frames.py), maker-video.json (maker_mm_fixture.py from
// maker_capture_mm.py). Weight-free; the heavy check is MediaParityTests run on these requests.
@testable import ElephantEngine
import Foundation
import Testing

func makerVideo() throws -> [MakerVideoCase] {
    struct File: Decodable { let cases: [MakerVideoCase] }
    let url = try #require(Bundle.module.url(forResource: "maker-video", withExtension: "json", subdirectory: "Fixtures"))
    return try JSONDecoder().decode(File.self, from: Data(contentsOf: url)).cases
}

struct MakerVideoCase: Decodable {
    struct Branch: Decodable {
        let prompt: String
        let input_ids: [Int]
        let prefix_tokens: Int
        let image_grid_thw: [[Int]]
        let video_grid_thw: [[Int]]
        let positions: [[Int]]
        let rope_delta: Int
    }
    struct Pixels: Decodable { let count: Int; let stride: Int; let values: [Double] }
    let id: String
    let request: String
    let branches: [Branch]
    let pixels: Pixels
    let video_pixels: Pixels
}

func makeVideoProcessor() throws -> VideoProcessorConfig {
    try VideoProcessorConfig(json: [
        "video_processor": [
            "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2, "size": ["shortest_edge": 4096, "longest_edge": 25_165_824],
            "max_video_tokens": 768, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "rescale_factor": 1.0 / 255, "resample": 3
        ]
    ])
}

@Suite struct VideoTests {
    @Test func sizeAndTimestampsMatchTransformers() throws {
        let c = try makeVideoProcessor()
        // Qwen3VLVideoProcessor.resize with cap_pixels_per_frame, transformers 5.19.0:
        // (4, 240, 320) -> (256, 320); (3, 260, 200) -> (256, 192); (2, 20, 700) -> (32, 1120);
        // (768, 1080, 1920) -> (128, 224); (10, 4000, 3000) -> (1024, 768).
        func r(_ n: Int, _ h: Int, _ w: Int) throws -> [Int] {
            let s = try VideoInput.size(frames: n, height: h, width: w, config: c)
            return [s.height, s.width]
        }
        #expect(try r(4, 240, 320) == [256, 320] && r(3, 260, 200) == [256, 192] && r(2, 20, 700) == [32, 1120])
        #expect(try r(768, 1080, 1920) == [128, 224] && r(10, 4000, 3000) == [1024, 768])
        #expect(throws: PromptError.self) { try r(1, 240, 320) } // t:1 must be larger than temporal_factor:2
        // Qwen3VLProcessor._calculate_timestamps: [0.25, 1.25]; [0.5, 2.0]; [1/6, 5/6, 4/3]; "%.1f" of 0.25 is "0.2".
        #expect(VideoInput.timestamps(frames: 4, fps: 2, temporal: 2) == [0.25, 1.25])
        #expect(VideoInput.timestamps(frames: 3, fps: 1, temporal: 2) == [0.5, 2.0])
        #expect(VideoInput.timestamps(frames: 5, fps: 3, temporal: 2).map { ($0 * 1e9).rounded() } == [166_666_667, 833_333_333, 1_333_333_333])
        #expect(VideoInput.label(0.25) == "<0.2 seconds>" && VideoInput.label(1.25) == "<1.2 seconds>" && VideoInput.label(2.35) == "<2.4 seconds>")
    }

    @Test func promptsGridsAndPixelsMatchTheMaker() throws {
        let cases = try makerVideo()
        #expect(cases.count == 3 && cases.map(\.branches.count) == [2, 1, 1])
        let images = try makeElephantProcessor(), videos = try makeVideoProcessor()
        for c in cases {
            let (state, questions, items) = try mmItems(c.request, media: try fixtureMedia())
            let plan = try ElephantEngine.branches(questions.map(\.1), maxSlots: 256)
            for (r, b) in zip(plan.rendered, c.branches) {
                #expect(same(try Prompt.chat(state: state, suffix: r.suffix, media: items.map(\.kind)), b.prompt), "\(c.id) prompt")
            }
            var pixels: [Float] = [], videoPixels: [Float] = [], grids: [[Int]] = [], videoGrids: [[Int]] = []
            for item in items {
                switch item {
                case .image(let i):
                    let (v, g) = try ImageInput.pixels(try ImageInput.load(i.url), config: images)
                    pixels += v; grids.append([g.t, g.h, g.w])
                case .video(let source):
                    let g = try VideoInput.grid(source, config: videos)
                    let (v, actual) = try VideoInput.pixels(source, config: videos)
                    #expect(actual == g)
                    videoPixels += v; videoGrids.append([g.t, g.h, g.w])
                }
            }
            #expect(grids == c.branches[0].image_grid_thw && videoGrids == c.branches[0].video_grid_thw, "\(c.id) grids")
            #expect(pixels.count == c.pixels.count && videoPixels.count == c.video_pixels.count, "\(c.id) pixel counts")
            for (mine, maker) in [(pixels, c.pixels), (videoPixels, c.video_pixels)] {
                for (i, v) in maker.values.enumerated() { #expect(Swift.abs(Double(mine[i * maker.stride]) - v) <= 1e-6, "\(c.id) pixel \(i)") }
            }
        }
    }

    @Test(.makerTokenizer) func tokenIdsAndPositionsMatchTheMaker() throws {
        let dir = try #require(makerTokenizerDirectory())
        let tokenizer = try ElephantEngine.tokenizer(dir)
        let vision = try VisionConfig(json: [
            "image_token_id": 248056, "video_token_id": 248057, "vision_start_token_id": 248053, "vision_end_token_id": 248054,
            "vision_config": [
                "depth": 27, "hidden_size": 1152, "num_heads": 16, "intermediate_size": 4304, "patch_size": 16, "temporal_patch_size": 2,
                "spatial_merge_size": 2, "out_hidden_size": 4096, "num_position_embeddings": 2304
            ]
        ])
        let images = try makeElephantProcessor(), videos = try makeVideoProcessor()
        for c in try makerVideo() {
            let (state, questions, items) = try mmItems(c.request, media: try fixtureMedia())
            let plan = try ElephantEngine.branches(questions.map(\.1), maxSlots: 256)
            let layout = try ElephantEngine.layout(tokenizer, state, plan.rendered, media: items.map(\.kind))
            let expansion = try items.map { item -> ElephantEngine.Expansion in
                switch item {
                case .image(let i): return .image(try ImageInput.tokens(height: i.height, width: i.width, config: images).grid)
                case .video(let v):
                    let labels = VideoInput.timestamps(frames: v.frames.count, fps: v.fps, temporal: 2)
                    return .video(try VideoInput.grid(v, config: videos), labels.map { tokenizer.encode(text: VideoInput.label($0), addSpecialTokens: false) })
                }
            }
            let prefix = try ElephantEngine.expand(layout.prefix, items: expansion, vision: vision)
            let rope = try MultimodalPositions.rope(prefix, blocks: ElephantEngine.blocks(expansion, vision: vision), merge: 2)
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

    @Test func placeholdersAndFrameRules() throws {
        let video = Prompt.videoBlock
        #expect(try Prompt.visionUser("clip <video:1>", media: [.video]) == "clip \(video)")
        #expect(try Prompt.visionUser("<image:1> <video:2>", media: [.image, .video]) == "\(Prompt.visionBlock) \(video)")
        #expect(throws: PromptError.self) { try Prompt.visionUser("<image:1>", media: [.video]) } // kind must match
        #expect(throws: PromptError.self) { try Prompt.visionUser("<video:1>", media: [.image]) }
        let media = try fixtureMedia()
        #expect(throws: PromptError.self) { try VideoInput.probe([media.appendingPathComponent("ball-0.png")], fps: 2) } // one frame
        #expect(throws: PromptError.self) {
            try VideoInput.probe([media.appendingPathComponent("ball-0.png"), media.appendingPathComponent("light-0.png")], fps: 2) // sizes differ
        }
        #expect(throws: PromptError.self) { try VideoInput.probe([media.appendingPathComponent("ball-0.png"), media.appendingPathComponent("ball-1.png")], fps: 0) }
    }
}
