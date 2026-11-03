// SPDX-License-Identifier: MIT
import CryptoKit
import Foundation
import IntegrationKit
import MLX
import MLXNN
import ArsenalWire
import Testing
@testable import AudioWorker

@Suite struct MelBandRoformerTests {
    /// The maker's band layout, captured from its own module (`freq_indices`, `num_bands_per_freq`, librosa 1.0.0) and
    /// the band_split input widths of the pinned checkpoint.
    @Test func melLayoutMatchesTheMaker() {
        let layout = MelBandLayout(MelBandRoformerConfig())
        func digest(_ values: [Int32]) -> String {
            values.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
        }
        #expect(layout.indices.count == 3958 && layout.covered)
        #expect(Array(layout.indices.prefix(20)) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 8, 9, 10, 11, 12, 13])
        #expect(digest(layout.indices) == "c087bfc8e1a110a16a7aa998de5fe43b025ea08de0e4606c7b80e258b1ed5ecc")
        #expect(digest(layout.bandsPerFreq.map(Int32.init)) == "41947c540f2511f98bb2530176d9a9a3576e5a954135dbf5ef207247e0933683")
        #expect(
            layout.bandInputs == [
                28, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 28, 28, 28, 36, 36, 36, 40, 40, 44, 52, 52, 52, 60, 64, 68, 76, 80, 80, 88,
                96, 104, 112, 116, 124, 132, 144, 156, 164, 176, 188, 200, 216, 228, 244, 264, 284, 304, 320, 344, 372, 396, 420, 452, 488, 520
            ])
    }

    @Test func mapsEveryCheckpointNameOnce() throws {
        #expect(try MelBandRoformerConversion.name("layers.3.0.layers.0.0.to_qkv.weight") == "layers.3.time.layers.0.attn.to_qkv.weight")
        #expect(try MelBandRoformerConversion.name("layers.5.1.layers.0.0.to_out.0.weight") == "layers.5.freq.layers.0.attn.to_out.weight")
        #expect(try MelBandRoformerConversion.name("layers.0.1.layers.0.1.net.4.bias") == "layers.0.freq.layers.0.ff.proj_out.bias")
        #expect(try MelBandRoformerConversion.name("layers.2.0.norm.gamma") == "layers.2.time.norm.gamma")
        #expect(try MelBandRoformerConversion.name("layers.0.0.layers.0.0.rotary_embed.freqs") == nil)
        #expect(try MelBandRoformerConversion.name("band_split.to_features.59.1.weight") == "band_split.59.linear.weight")
        #expect(try MelBandRoformerConversion.name("mask_estimators.0.to_freqs.7.0.4.weight") == "mask_estimator.7.layers.2.weight")
        #expect(throws: (any Error).self) { try MelBandRoformerConversion.name("mask_estimators.1.to_freqs.7.0.2.weight") }
        #expect(throws: (any Error).self) { try MelBandRoformerConversion.name("final_norm.gamma") }
        // 228 203 172 checkpoint values, 12 rotary buffers of 32 among them (recomputed); the graph holds the rest.
        let config = MelBandRoformerConfig(), d = config.dim, h = config.heads * config.dimHead, hidden = 4 * d
        let attention = d + 3 * h * d + (config.heads * d + config.heads) + h * d
        let feedForward = d + (4 * d * d + 4 * d) + (4 * d * d + d)
        let trunk = 2 * config.depth * (attention + feedForward + d)
        // Mask MLPs: dim → hidden → hidden → 2 × inputs (depth 2 means three linears upstream).
        let bands = MelBandLayout(config).bandInputs.reduce(0) {
            $0 + ($1 + $1 * d + d) + (d * hidden + hidden) + (hidden * hidden + hidden) + (hidden * 2 * $1 + 2 * $1)
        }
        #expect(trunk + bands == 228_203_172 - 12 * 32)
    }

    /// `demix_track` @25f44ffb: windows, padding and chunk starts.
    @Test func demixFollowsTheMaker() {
        let demix = MelBandRoformerAdapter.Demix(MelBandRoformerConfig())
        #expect(demix.chunk == 352_800 && demix.step == 176_400 && demix.fade == 35_280 && demix.border == 176_400)
        #expect(MelBandRoformerAdapter.Demix.reflect([1, 2, 3, 4], left: 2, right: 3) == [3, 2, 1, 2, 3, 4, 3, 2, 1])
        let total = 900_000
        let first = demix.window(start: 0, total: total), middle = demix.window(start: 176_400, total: total)
        let last = demix.window(start: 705_600, total: total)
        #expect(first[0] == 1 && first[352_799] == 0 && abs(first[352_800 - 35_280] - 1) < 1e-6)
        #expect(middle[0] == 0 && middle[35_279] == 1 && middle[352_799] == 0)
        #expect(last[0] == 0 && last[352_799] == 1)
        #expect(demix.starts(total) == [0, 176_400, 352_800, 529_200, 705_600, 882_000])
        #expect(demix.pads(352_801) && !demix.pads(352_800))
        let four = MelBandRoformerAdapter.Demix(MelBandRoformerConfig(), overlap: 4), one = MelBandRoformerAdapter.Demix(MelBandRoformerConfig(), overlap: 1)
        #expect(four.step == 88_200 && four.border == 264_600 && one.border == 0 && !one.pads(10_000_000))
        // A short tail is zero-padded; one longer than half a chunk is reflect-padded.
        let short = demix.part([Float](repeating: 1, count: 1000), start: 0)
        #expect(short.count == 352_800 && short[999] == 1 && short[1000] == 0)
        let long = demix.part((0..<200_000).map(Float.init), start: 0)
        #expect(long.count == 352_800 && long[200_000] == 199_998)
    }

    /// Review fix (2 Nov): overlap 1 is refused, in the schema and in the adapter, because the maker's fades leave two
    /// silent-vocal samples at every seam without overlap.
    @Test func overlapBoundsAgreeWithTheSchema() throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let schema =
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: repo.appendingPathComponent("models/melband-roformer-kim/ops/separate_audio.input.json"))) as? [String: Any]
        let overlap = try #require((schema?["properties"] as? [String: Any])?["overlap"] as? [String: Any])
        #expect(overlap["minimum"] as? Int == 2 && overlap["maximum"] as? Int == 8 && overlap["default"] as? Int == 2)
        let source = try String(
            contentsOf: repo.appendingPathComponent("Workers/Audio/Sources/AudioWorker/MelBandRoformer/MelBandRoformerAdapter.swift"), encoding: .utf8)
        #expect(source.contains("(2...8).contains(n)") && MelBandRoformerConfig().overlap == 2)
        let windows = MelBandRoformerAdapter.Demix(MelBandRoformerConfig(), overlap: 2)
        let first = windows.window(start: 0, total: 900_000), second = windows.window(start: windows.step, total: 900_000)
        // With overlap 2 every sample of the first two chunks' shared span has positive total weight.
        #expect((0..<(windows.chunk - windows.step)).allSatisfy { first[windows.step + $0] + second[$0] > 0 })
    }

    /// Trained weights: Get's conversion of the pinned checkpoint, per-component parity against the maker's code
    /// (captured once on MPS, FP16 network), then the maker's demix on the 12 s fixture through the adapter.
    @Test(.heavy) func trainedWeightsMatchTheMaker() throws {
        let lab = try audioPortsLab()
        let source = lab.appendingPathComponent("src/melband-roformer"), snapshot = lab.appendingPathComponent("src/melband-roformer-fp16")
        if !FileManager.default.fileExists(atPath: snapshot.appendingPathComponent(MelBandRoformerConversion.marker).path) {
            try? FileManager.default.removeItem(at: snapshot)
            _ = try MelBandRoformerConversion.base.run(
                [
                    "src_dir": .string(source.path), "dst_dir": .string(snapshot.path), "dtype": "fp16", "group_size": 64,
                    "recipe": .string(MelBandRoformerConversion.recipe)
                ], cancellation: NativeCancellation())
        }
        let adapter = MelBandRoformerAdapter()
        try adapter.load(
            try NativeLoadContext(["model_dir": .string(lab.path), "weights_dir": .string(snapshot.path), "dtype": "fp16"]),
            cancellation: NativeCancellation())
        let model = try #require(adapter.graph as? MelBandRoformerModel)
        let ref = try loadArrays(url: lab.appendingPathComponent("fixtures/melband/reference.safetensors"))
        var results: [String: Float] = [:]
        let (real, imag) = model.spectrum(ref["chunk_in"]!)
        results["features"] = relativeL2(model.features(real: real, imag: imag, dtype: .float16)[0], ref["features"]!)
        let refFeatures = ref["features"]!.asType(.float16).expandedDimensions(axis: 0)
        results["layer0"] = relativeL2(model.layers[0](refFeatures)[0], ref["layer0"]!)
        results["trunk"] = relativeL2(model.trunk(refFeatures)[0], ref["trunk"]!)
        let refTrunk = ref["trunk"]!.asType(.float16).expandedDimensions(axis: 0)
        results["mask"] = relativeL2(model.mask(fromTrunk: refTrunk)[0], ref["mask"]!)
        results["render_from_ref_mask"] = relativeL2(model.render(real: real, imag: imag, mask: ref["mask"]!), ref["chunk_out"]!)
        results["chunk_out"] = relativeL2(model(ref["chunk_in"]!), ref["chunk_out"]!)

        let fixture = lab.appendingPathComponent("fixtures/melband/fixture-song.wav")
        let pcm = try NativeMedia.audio(fixture, sampleRate: 44_100, channels: 2)
        let frames = pcm.samples.count / 2
        let planar = (0..<2).map { c in (0..<frames).map { pcm.samples[$0 * 2 + c] } }
        let vocals = try adapter.separate(planar, model: model, cancellation: NativeCancellation())
        results["demix_vocals"] = relativeL2(MLXArray(vocals.flatMap { $0 }).reshaped(2, frames), ref["demix_vocals"]!)
        print("melband-roformer parity", results.sorted { $0.key < $1.key })
        #expect(frames == 529_200)
        #expect(results["render_from_ref_mask"]! < 1e-4)
        #expect(results["features"]! < 5e-3 && results["layer0"]! < 1e-2 && results["trunk"]! < 3e-2 && results["mask"]! < 5e-2)
        #expect(results["chunk_out"]! < 5e-2 && results["demix_vocals"]! < 5e-2)

        let out = lab.appendingPathComponent("out/melband-roformer")
        try? FileManager.default.removeItem(at: out)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let result = try adapter.run(
            "separate_audio", input: ["audio": ["path": .string(fixture.path)], "out_dir": .string(out.path)], cancellation: NativeCancellation(), emit: { _ in })
        #expect(result["files"]?.array?.compactMap { $0["id"]?.string } == ["vocals", "instrumental"])
        try WireJSON.encode(["parity": .object(results.mapValues { .number(Double($0)) })]).write(to: out.appendingPathComponent("summary.json"))
    }
}
