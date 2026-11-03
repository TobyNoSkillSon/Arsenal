import ArsenalWire
import Foundation
import IntegrationKit
import MLX
import Testing
@testable import EmbedWorker

/// Lab parity against the maker's code (models/embeddinggemma-2/README.md): needs the pinned weights and the captured
/// reference, so it runs only with the heavy suite when EG2_WEIGHTS (snapshot), EG2_EVIDENCE (maker capture directory),
/// EG2_FIXTURES (its input files) and EG2_OUT (result JSON) are set. EG2_PRECISION picks fp16 (default) or bf16.
@Suite(.serialized) struct EmbeddingGemma2ParityTests {
    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["EG2_WEIGHTS"] != nil))
    func embeddingGemma2MatchesTheMakersComponentsAndEmbeddings() throws {
        let env = ProcessInfo.processInfo.environment
        let weights = URL(fileURLWithPath: try #require(env["EG2_WEIGHTS"])), evidence = URL(fileURLWithPath: try #require(env["EG2_EVIDENCE"]))
        let fixtures = URL(fileURLWithPath: try #require(env["EG2_FIXTURES"])), output = URL(fileURLWithPath: try #require(env["EG2_OUT"]))
        let precision = env["EG2_PRECISION"] ?? "fp16"
        NativeRuntime.prepare()
        let reference = try MLX.loadArrays(url: evidence.appendingPathComponent("reference.safetensors"))
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: evidence.appendingPathComponent("reference.json"))) as? [String: Any]
        let cases = try #require(meta?["cases"] as? [String: [String: Any]])
        let adapter = EmbeddingGemma2Adapter()
        try adapter.load(
            NativeLoadContext(["weights_dir": .string(weights.path), "model_dir": .string(weights.path), "dtype": .string(precision)]), cancellation: NativeCancellation())
        let graph = try #require(adapter.graph), vision = try #require(graph.vision), audio = try #require(graph.audio)
        let embedVision = try #require(graph.embedVision), embedAudio = try #require(graph.embedAudio)
        let cancel = NativeCancellation()
        func compare(_ a: MLXArray, _ b: MLXArray) -> [String: Double] {
            let x = a.asType(.float32).flattened(), y = b.asType(.float32).flattened()
            guard x.size == y.size else { return ["size_mismatch": Double(x.size - y.size)] }
            let cosine = (sum(x * y) / (sqrt(sum(x * x)) * sqrt(sum(y * y)))).item(Float.self)
            let rel = (sqrt(sum((x - y) * (x - y))) / sqrt(sum(y * y))).item(Float.self)
            return ["cosine": Double(cosine), "rel_l2": Double(rel), "max_abs": Double(abs(x - y).max().item(Float.self)), "finite": MLX.all(isFinite(x)).item(Bool.self) ? 1 : 0]
        }
        func ref(_ key: String) throws -> MLXArray { try #require(reference[key], "missing \(key)") }
        var components: [String: Any] = [:]

        // Vision: Arsenal's decode, resize and patches against the processor; the tower on the processor's own patches.
        func grid(_ positions: MLXArray) -> (Int, Int, Int) {
            let p = positions.reshaped(-1, 2).asArray(Int64.self)
            let real = stride(from: 0, to: p.count, by: 2).filter { p[$0] >= 0 }.count
            let columns = Int(stride(from: 0, to: p.count, by: 2).map { p[$0] }.max()! + 1)
            return (real, columns, real / columns)
        }
        var visionFeatures: [String: MLXArray] = [:]
        for name in ["image_coast", "image_recipe"] {
            let pixels = try ref("\(name).pixel_values")[0], (real, columns, rows) = grid(try ref("\(name).image_position_ids")[0])
            let file = name == "image_coast" ? "coast.png" : "recipe.png"
            let mine = try EG2Vision.patches(try EG2Picture.load(fixtures.appendingPathComponent(file)), maxSoftTokens: EG2Vision.imageSoftTokens)
            #expect(mine.columns == columns && mine.rows == rows)
            var pixelReport = compare(mine.pixels, pixels[..<real])
            pixelReport["levels_max_abs"] = Double(abs(mine.pixels - pixels[..<real]).max().item(Float.self) * 255)
            components["\(name).pixels"] = pixelReport
            let patches = EG2Patches(pixels: pixels[..<real], columns: columns, rows: rows)
            let pooled = vision.pool(try vision.encode(patches, cancellation: cancel), columns: columns, rows: rows)
            components["\(name).vision_tower"] = compare(pooled, try ref("\(name).vision_tower"))
            let embedded = embedVision(pooled)
            components["\(name).embed_vision"] = compare(embedded, try ref("\(name).embed_vision"))
            components["\(name).vision_max_abs"] = Double(abs(pooled).max().item(Float.self))
            visionFeatures[name] = embedded
        }
        // Video: Arsenal's reader and sampling against the processor's frames; the tower per frame on the maker's frames.
        do {
            let frames = try ref("video_ball.pixel_values_videos"), positions = try ref("video_ball.video_position_ids")
            let url = fixtures.appendingPathComponent("ball.mp4")
            let track = try EG2VideoIO.track(url, cancellation: cancel)
            let indices = EG2Vision.sampleFrames(total: track.frames, fps: track.fps)
            #expect(indices.count == frames.dim(0))
            let decoded = try EG2VideoIO.frames(url, track: track, indices: indices, cancellation: cancel) { $0 }
            var features: [MLXArray] = [], mineFrames: [MLXArray] = [], theirs: [MLXArray] = []
            for f in 0..<frames.dim(0) {
                let (real, columns, rows) = grid(positions[f])
                let patches = EG2Patches(pixels: frames[f][..<real], columns: columns, rows: rows)
                features.append(vision.pool(try vision.encode(patches, cancellation: cancel), columns: columns, rows: rows))
                mineFrames.append(try EG2Vision.patches(decoded[f], maxSoftTokens: EG2Vision.videoSoftTokens).pixels)
                theirs.append(frames[f][..<real])
            }
            var pixelReport = compare(concatenated(mineFrames, axis: 0), concatenated(theirs, axis: 0))
            pixelReport["levels_max_abs"] = Double(abs(concatenated(mineFrames, axis: 0) - concatenated(theirs, axis: 0)).max().item(Float.self) * 255)
            pixelReport["frames"] = Double(indices.count)
            components["video_ball.pixels"] = pixelReport
            let pooled = concatenated(features, axis: 0)
            components["video_ball.vision_tower"] = compare(pooled, try ref("video_ball.vision_tower"))
            let embedded = embedVision(pooled)
            components["video_ball.embed_vision"] = compare(embedded, try ref("video_ball.embed_vision"))
            visionFeatures["video_ball"] = embedded
        }
        // Audio: Arsenal's decode and log-mel against the extractor; the tower on the maker's features.
        let makerFeatures = try ref("audio_speech.input_features")[0], makerMask = try ref("audio_speech.input_features_mask")[0]
        let pcm = try EG2AudioIO.mono(fixtures.appendingPathComponent("speech.wav"), maximumSamples: EG2AudioIO.decodeLimit)
        let mel = EG2Mel.features(pcm)
        #expect(mel.frames == makerFeatures.dim(0))
        #expect(mel.mask == makerMask.asArray(UInt8.self).map { $0 != 0 })
        components["audio_speech.features"] = compare(MLXArray(mel.values, [mel.frames, EG2Mel.bins]), makerFeatures)
        let tower = try audio(
            makerFeatures.asArray(Float.self), frames: makerFeatures.dim(0), mask: makerMask.asArray(UInt8.self).map { $0 != 0 }, cancellation: cancel)
        components["audio_speech.audio_tower"] = compare(tower, try ref("audio_speech.audio_tower")[0])
        let audioFeatures = embedAudio(tower)
        components["audio_speech.embed_audio"] = compare(audioFeatures, try ref("audio_speech.embed_audio")[0])

        // Text model on the maker's ids and the maker's own soft tokens: per-token outputs, hidden layers, pooled vector.
        let config = graph.config
        graph.text.traceMaximum = true
        graph.text.captureLayers = [1, 6, 12, 18]
        var maxima: [String: Double] = [:]
        for (name, info) in cases.sorted(by: { $0.key < $1.key }) {
            let ids = try #require(info["ids"] as? [Int])
            let placeholders: Set<Int> = [config.imageToken, config.videoToken, config.audioToken]
            var embeds = graph.embed(ids.map { placeholders.contains($0) ? 0 : $0 })
            for (token, key) in [(config.imageToken, "embed_vision"), (config.videoToken, "embed_vision"), (config.audioToken, "embed_audio")] {
                let at = ids.indices.filter { ids[$0] == token }
                guard !at.isEmpty else { continue }
                var feature = try ref("\(name).\(key)")
                if feature.ndim == 3 { feature = feature[0] }
                #expect(feature.dim(0) == at.count)
                embeds[MLXArray(at.map(Int32.init))] = feature.asType(embeds.dtype)
            }
            if let first = reference["\(name).hidden.0"] { components["\(name).inputs_embeds"] = compare(embeds, first) }
            let tokens = try graph.tokens(embeds, cancellation: cancel)
            components["\(name).tokens"] = compare(tokens, try ref("\(name).tokens"))
            components["\(name).pooled"] = compare(tokens.asType(.float32).mean(axis: 0), try ref("\(name).embedding"))
            for layer in [1, 6, 12, 18] {
                if let expected = reference["\(name).hidden.\(layer)"], let mine = graph.text.captured[layer] {
                    components["\(name).hidden.\(layer)"] = compare(mine[0], expected)
                }
            }
            maxima[name] = Double(graph.text.observedMaximum)
        }
        graph.text.traceMaximum = false; graph.text.captureLayers = []

        // End to end through the served operation: Arsenal's own tokenizer, prompt, decoding, resize and features.
        func unit(_ v: MLXArray) -> MLXArray { let f = v.asType(.float32); return f / sqrt(sum(f * f)) }
        let tasks = ["task: search result | query: ": "query", "title: none | text: ": "document", "task: sentence similarity | query: ": "similarity"]
        func path(_ file: String) -> JSONValue { .string(fixtures.appendingPathComponent(file).path) }
        var end: [String: Any] = [:], vectors: [String: MLXArray] = [:]
        for (name, info) in cases.sorted(by: { $0.key < $1.key }) {
            var request: JSONObject
            if let text = info["input"] as? String {
                let prompt = try #require(info["prompt"] as? String)
                request = ["input": .array([.string(text)]), "task": .string(try #require(tasks[prompt]))]
            } else {
                let input = try #require(info["input"] as? [String: String])
                var parts: [JSONValue] = []
                if let text = input["text"] { parts.append(.string(text)) }
                if let file = input["image"] { parts.append(["type": "image", "image": ["path": path(file)]]) }
                if let file = input["audio"] { parts.append(["type": "audio", "audio": ["path": path(file)]]) }
                if let file = input["video"] { parts.append(["type": "video", "video": ["path": path(file)]]) }
                request = ["input": .array([parts.count == 1 ? parts[0] : ["type": "content", "content": .array(parts)]])]
            }
            let result = try adapter.run("embed", input: request, cancellation: cancel) { _ in }
            let row = try #require(result["embeddings"]?.array?.first?.array).map { Float($0.number ?? .nan) }
            let vector = MLXArray(row)
            vectors[name] = vector
            var entry = compare(vector, unit(try ref("\(name).embedding")))
            let expectedIDs = try #require(info["ids"] as? [Int])
            entry["ids_equal"] = adapter.lastIDs == expectedIDs ? 1 : 0
            entry["tokens"] = Double(adapter.lastIDs.count)
            end[name] = entry
        }
        // Retrieval order the maker's vectors give: each query's best match.
        var ranking: [String: Any] = [:]
        for query in ["query", "q_coast", "q_dog", "q_pasta", "q_ball"] {
            let q = try #require(vectors[query])
            let scored = vectors.filter { !$0.key.hasPrefix("q") && $0.key != "query" }.map { ($0.key, Double(sum(q * $0.value).item(Float.self))) }
            ranking[query] = Dictionary(uniqueKeysWithValues: scored.map { ($0.0, ($0.1 * 10000).rounded() / 10000) })
        }
        let report: [String: Any] = [
            "precision": precision, "components": components, "end_to_end": end, "language_max_abs_hidden": maxima, "similarity": ranking,
            "peak_bytes": Memory.peakMemory
        ]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: output)
        // The report is the record; these are the pass conditions (the real checks give 0.9999 or better, BF16 included).
        for (name, entry) in end {
            let values = try #require(entry as? [String: Double])
            #expect((values["cosine"] ?? 0) >= 0.999 && values["finite"] == 1, "\(name): end-to-end cosine \(values["cosine"] ?? .nan)")
            #expect(values["ids_equal"] == 1, "\(name): token ids differ from the maker's")
        }
    }
}
