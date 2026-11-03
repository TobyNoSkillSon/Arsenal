import ArsenalWire
import Foundation
import IntegrationKit
import MLX
import Testing
@testable import EmbedWorker

/// Lab parity against the maker's code (models/lco-embedding-omni-7b/README.md): needs the pinned weights and the
/// captured reference, so it runs only when LCO_WEIGHTS (snapshot), LCO_EVIDENCE (maker capture directory),
/// LCO_FIXTURES (its input files) and LCO_OUT (result JSON) are set, with the heavy suite.
@Suite(.serialized) struct LCOParityTests {
    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["LCO_WEIGHTS"] != nil))
    func lcoMatchesTheMakersComponentsAndEmbeddings() throws {
        let env = ProcessInfo.processInfo.environment
        let weights = URL(fileURLWithPath: try #require(env["LCO_WEIGHTS"])), evidence = URL(fileURLWithPath: try #require(env["LCO_EVIDENCE"]))
        let fixtures = URL(fileURLWithPath: try #require(env["LCO_FIXTURES"])), output = URL(fileURLWithPath: try #require(env["LCO_OUT"]))
        let precision = env["LCO_PRECISION"] ?? "fp16"
        NativeRuntime.prepare()
        let reference = try MLX.loadArrays(url: evidence.appendingPathComponent("reference.safetensors"))
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: evidence.appendingPathComponent("reference.json"))) as? [String: Any]
        let cases = try #require(meta?["cases"] as? [String: [String: Any]])
        let adapter = LCOAdapter()
        try adapter.load(
            NativeLoadContext(["weights_dir": .string(weights.path), "model_dir": .string(weights.path), "dtype": .string(precision)]), cancellation: NativeCancellation())
        let graph = try #require(adapter.graph)
        graph.language.traceMaximum = true
        func compare(_ a: MLXArray, _ b: MLXArray) -> [String: Double] {
            let x = a.asType(.float32).flattened(), y = b.asType(.float32).flattened()
            let cosine = (sum(x * y) / (sqrt(sum(x * x)) * sqrt(sum(y * y)))).item(Float.self)
            let rel = (sqrt(sum((x - y) * (x - y))) / sqrt(sum(y * y))).item(Float.self)
            return ["cosine": Double(cosine), "rel_l2": Double(rel), "max_abs": Double(abs(x - y).max().item(Float.self)), "finite": MLX.all(isFinite(x)).item(Bool.self) ? 1 : 0]
        }
        var report: [String: Any] = ["precision": precision]
        var components: [String: Any] = [:]
        // Vision tower on the maker's own pixel patches and grid.
        for name in ["image", "video"] {
            let pixels = try #require(reference["\(name).pixels"]), expected = try #require(reference["\(name).vision"])
            let grid = try #require(cases[name]?["grid"] as? [Int])
            let actual = try graph.vision(pixels, grids: [LCOGrid(t: grid[0], h: grid[1], w: grid[2])], cancellation: NativeCancellation())
            #expect(actual.shape == expected.shape)
            components["\(name).vision"] = compare(actual, expected)
        }
        // Audio: Arsenal's log-mel against the processor's, then the tower on the maker's features.
        let pcm = try NativeMedia.audio(fixtures.appendingPathComponent("audio.wav"), sampleRate: 16000).samples
        let (values, frames) = LCOMel.features(pcm)
        let mel = try #require(reference["audio.mel"])
        #expect(mel.shape == [128, frames])
        components["audio.mel"] = compare(MLXArray(values, [128, frames]), mel)
        let audio = try graph.audio(mel, cancellation: NativeCancellation())
        let expectedAudio = try #require(reference["audio.audio"])
        #expect(audio.shape == expectedAudio.shape)
        components["audio.tower"] = compare(audio, expectedAudio)
        // Language model on the maker's ids, positions and tower outputs: the pooled hidden state alone.
        var maxima: [String: Double] = [:]
        for name in ["query", "everest", "image", "audio", "video"] {
            let ids = try #require(cases[name]?["ids"] as? [Int]), positions = try #require(reference["\(name).positions"])
            var embeds = graph.language.embedTokens(MLXArray(ids.map(Int32.init)).reshaped(1, -1))
            let feature = name == "audio" ? reference["audio.audio"] : reference["\(name).vision"]
            if let feature {
                let token = name == "audio" ? graph.config.audioToken : name == "image" ? graph.config.imageToken : graph.config.videoToken
                let at = ids.indices.filter { ids[$0] == token }
                #expect(at.count == feature.dim(0))
                embeds[0, MLXArray(at.map(Int32.init))] = feature.asType(embeds.dtype)
            }
            let rows = (0..<3).map { positions[$0].asArray(Int32.self).map(Int.init) }
            let hidden = try graph.language.lastHidden(embeds, positions: rows, cancellation: NativeCancellation())
            components["\(name).language"] = compare(hidden, try #require(reference["\(name).hidden"]))
            maxima[name] = Double(graph.language.observedMaximum)
        }
        // Inputs: Arsenal's own decode and resize against the maker's processor output.
        let image = try LCOImageIO.load(fixtures.appendingPathComponent("image.png"))
        let (ih, iw) = try LCOMedia.smartResize(height: image.height, width: image.width, minPixels: LCOMedia.imageMinPixels, maxPixels: LCOMedia.imageMaxPixels)
        components["image.pixels"] = compare(LCOMedia.patches([image], height: ih, width: iw).0, try #require(reference["image.pixels"]))
        let videoURL = fixtures.appendingPathComponent("video.mp4")
        let track = try LCOVideoIO.track(videoURL, cancellation: NativeCancellation())
        let count = try #require(LCOMedia.frameCount(total: track.frames, fps: track.fps))
        let (rh, rw) = try LCOMedia.smartResize(height: track.height, width: track.width, minPixels: LCOMedia.videoMinPixels, maxPixels: LCOMedia.videoMaxPixels(frames: count))
        let videoFrames = try LCOVideoIO.frames(
            videoURL, track: track, indices: LCOMedia.sampleIndices(total: track.frames, count: count), width: rw, height: rh, cancellation: NativeCancellation())
        components["video.pixels"] = compare(LCOMedia.patches(videoFrames, height: rh, width: rw).0, try #require(reference["video.pixels"]))
        if let dump = env["LCO_DUMP"] {
            let videoPixels = LCOMedia.patches(videoFrames, height: rh, width: rw).0
            try MLX.save(arrays: ["audio.tower": audio.asType(.float32), "video.pixels": videoPixels.asType(.float32)], url: URL(fileURLWithPath: dump))
        }
        report["components"] = components
        report["language_max_abs_hidden"] = maxima
        // End to end through the served operation: Arsenal's own tokenizer, resize, decoding and features.
        func unit(_ v: MLXArray) -> MLXArray { let f = v.asType(.float32); return f / sqrt(sum(f * f)) }
        let texts = try #require(meta?["cases"] as? [String: [String: Any]])
        var end: [String: Any] = [:], vectors: [String: MLXArray] = [:]
        let items: [(String, ArsenalWire.JSONValue)] =
            try ["query", "everest", "k2", "kilimanjaro"].map { name in
                let prompt = try #require(texts[name]?["prompt"] as? String)
                let start = try #require(prompt.range(of: "<|im_start|>user\n")).upperBound, stop = try #require(prompt.range(of: "\nSummarize the above")).lowerBound
                return (name, .string(String(prompt[start..<stop])))
            } + [
                ("image", ["type": "image", "image": ["path": .string(fixtures.appendingPathComponent("image.png").path)]]),
                ("audio", ["type": "audio", "audio": ["path": .string(fixtures.appendingPathComponent("audio.wav").path)]]),
                ("video", ["type": "video", "video": ["path": .string(fixtures.appendingPathComponent("video.mp4").path)]])
            ]
        for (name, item) in items {
            let result = try adapter.run("embed", input: ["input": .array([item])], cancellation: NativeCancellation()) { _ in }
            let row = try #require(result["embeddings"]?.array?.first?.array).map { Float($0.number ?? .nan) }
            let vector = MLXArray(row), expected = unit(try #require(reference["\(name).hidden"]))
            vectors[name] = vector
            var entry = compare(vector, expected)
            entry["ids_equal"] = adapter.lastIDs == (texts[name]?["ids"] as? [Int] ?? []) ? 1 : 0
            entry["tokens"] = Double(adapter.lastIDs.count)
            end[name] = entry
        }
        let query = try #require(vectors["query"])
        report["card_text_check"] = ["everest", "k2", "kilimanjaro"].map { Double(sum(query * vectors[$0]!).item(Float.self)) }
        report["end_to_end"] = end
        report["peak_bytes"] = Memory.peakMemory
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: output)
        // The report is the record; these are the pass conditions (the real check's lowest item, video, is 0.99916).
        for (name, entry) in end {
            let values = try #require(entry as? [String: Double])
            #expect((values["cosine"] ?? 0) >= 0.999 && values["finite"] == 1, "\(name): end-to-end cosine \(values["cosine"] ?? .nan)")
            if texts[name]?["ids"] != nil { #expect(values["ids_equal"] == 1, "\(name): token ids differ from the maker's") }
        }
    }
}
