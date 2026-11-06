// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN
import ArsenalWire
import Testing
@testable import AudioWorker

/// Lab paths for the opt-in trained-weight checks: `ARSENAL_AUDIO_PORTS_LAB` holds `src/`, `fixtures/` and `out/`.
func audioPortsLab() throws -> URL {
    guard let path = ProcessInfo.processInfo.environment["ARSENAL_AUDIO_PORTS_LAB"] else { throw audioFailure("Set ARSENAL_AUDIO_PORTS_LAB.") }
    if let metallib = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] ?? ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] {
        setTestMetallib(URL(fileURLWithPath: metallib))
    }
    return URL(fileURLWithPath: path)
}

/// ‖a − b‖ / ‖b‖ over float32 copies.
func relativeL2(_ a: MLXArray, _ b: MLXArray) -> Float {
    let x = a.asType(.float32), y = b.asType(.float32)
    return (sqrt(((x - y) * (x - y)).sum()) / sqrt((y * y).sum())).item(Float.self)
}

@Suite struct BSRoformerTests {
    static let yaml = """
        audio:
          chunk_size: 352800
          hop_length: 441 # don't work (use in model)
          sample_rate: 44100
        model:
          dim: 512
          depth: 12
          stereo: true
          num_stems: 1
          time_transformer_depth: 1
          freq_transformer_depth: 1
          freqs_per_bands: !!python/tuple
            - 2
            - 1023
          dim_head: 64
          heads: 8
          flash_attn: true
          stft_n_fft: 2048
          stft_hop_length: 441
          stft_win_length: 2048
          stft_normalized: false
          mask_estimator_depth: 2
          multi_stft_resolutions_window_sizes: !!python/tuple
          - 4096
          - 256
        training:
          instruments:
          - Vocals
          - Instrumental
          target_instrument: Vocals
          lr: 5.0e-05
        inference:
          dim_t: 801
          num_overlap: 4
        """

    @Test func parsesTheReleaseYAMLSubset() throws {
        let object = try SimpleYAML.parse(Self.yaml)
        let config = try BSRoformerConfig.parse(object)
        #expect(config.freqsPerBands == [2, 1023])
        #expect(config.dim == 512 && config.depth == 12 && config.hop == 441 && config.dimT == 801 && config.stereo)
        #expect(config.bandInputs == [8, 4092])
        #expect((object["training"] as? [String: Any])?["instruments"] as? [String] == ["Vocals", "Instrumental"])
        #expect(throws: (any Error).self) { try SimpleYAML.parse("a: &anchor 1") }
        var stems = object
        stems["model"] = (object["model"] as? [String: Any] ?? [:]).merging(["num_stems": 2]) { _, new in new }
        #expect(throws: (any Error).self) { try BSRoformerConfig.parse(stems) }
    }

    @Test func mapsEveryCheckpointNameOnce() throws {
        #expect(try BSRoformerConversion.name("layers.3.0.layers.0.0.to_qkv.weight") == "layers.3.time.0.attn.to_qkv.weight")
        #expect(try BSRoformerConversion.name("layers.3.1.layers.0.0.to_out.0.weight") == "layers.3.freq.0.attn.to_out.weight")
        #expect(try BSRoformerConversion.name("layers.0.1.layers.0.1.net.4.bias") == "layers.0.freq.0.ff.proj_out.bias")
        #expect(try BSRoformerConversion.name("layers.0.0.layers.0.0.rotary_embed.freqs") == nil)
        #expect(try BSRoformerConversion.name("band_split.to_features.61.0.gamma") == "band_split.61.norm.gamma")
        #expect(try BSRoformerConversion.name("mask_estimators.0.to_freqs.7.0.2.weight") == "mask_estimator.7.layers.1.weight")
        #expect(throws: (any Error).self) { try BSRoformerConversion.name("mask_estimators.1.to_freqs.7.0.2.weight") }
        // The release's checkpoint holds 159 758 796 values, 24 rotary buffers of 32 among them (recomputed, not loaded);
        // the graph's declared layers account for exactly the rest. (Module shapes are checked against the
        // checkpoint by the conversion itself.)
        let config = BSRoformerConfig(), d = config.dim, h = config.heads * config.dimHead
        let attention = d + 3 * h * d + (config.heads * d + config.heads) + h * d
        let feedForward = d + (4 * d * d + 4 * d) + (4 * d * d + d)
        let trunk = 2 * config.depth * (attention + feedForward) + d
        let bands = config.bandInputs.reduce(0) { $0 + ($1 + $1 * d + d) + (d * 4 * d + 4 * d) + (4 * d * 2 * $1 + 2 * $1) }
        #expect(trunk + bands == 159_758_796 - 24 * 32)
    }

    /// A mono clip reaches the stereo model on both channels (audio-separator duplicates mono as well).
    @Test func monoInputFillsBothChannels() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let samples = (0..<4410).map { Float(0.5 * sin(Double($0) * 0.07)) }
        let file = try audioWave(samples, rate: 44_100, channels: 1, id: "mono", input: ["out_dir": .string(directory.path)], cancellation: NativeCancellation())
        let pcm = try NativeMedia.audio(URL(fileURLWithPath: try #require(file["path"]?.string)), sampleRate: 44_100, channels: 2)
        let left = stride(from: 0, to: pcm.samples.count, by: 2).map { pcm.samples[$0] }
        let right = stride(from: 1, to: pcm.samples.count, by: 2).map { pcm.samples[$0] }
        #expect(left.count == 4410 && left == right && left.contains { abs($0) > 0.4 })
    }

    @Test func hostilePicklesFailCleanly() {
        // 40 nested TUPLE1s, then STOP: refused for depth rather than recursing.
        let deep = Data([0x80, 0x02, 0x4E] + Array(repeating: 0x85, count: 40) + [0x2E])
        #expect(throws: (any Error).self) { try TorchPickle.parse(deep) }
        #expect(throws: (any Error).self) { try TorchPickle.parse(Data([0x80, 0x02, 0x2E])) } // STOP on an empty stack
        #expect(throws: (any Error).self) { try TorchPickle.parse(Data([0x80, 0x02, 0x68, 0x05, 0x2E])) } // unknown memo
    }

    @Test func chunkPlanFollowsAudioSeparator() {
        let config = BSRoformerConfig()
        let short = BSRoformerAdapter.plan(frames: 176_400, config: config) // 4 s: segment 256 frames
        #expect(short.chunk == 441 * 255 && short.starts == [0, 176_400 - 441 * 255])
        let long = BSRoformerAdapter.plan(frames: 44_100 * 20, config: config)
        #expect(long.chunk == 352_800 && long.starts == [0, 352_800, 44_100 * 20 - 352_800])
        #expect(BSRoformerAdapter.plan(frames: 1000, config: config).starts == [0])
    }

    @Test func pickleReaderBuildsTensorDescriptorsWithoutExecuting() throws {
        // OrderedDict({"w": _rebuild_tensor_v2(storage("0", Float), 4, (2, 3), (3, 1), False, OrderedDict())}), protocol 2.
        var p: [UInt8] = [0x80, 0x02]
        p += [0x63] + Array("collections\nOrderedDict\n".utf8) + [0x29, 0x52]
        p += [0x58, 1, 0, 0, 0] + Array("w".utf8)
        p += [0x63] + Array("torch._utils\n_rebuild_tensor_v2\n".utf8)
        p += [0x28, 0x28, 0x58, 7, 0, 0, 0] + Array("storage".utf8) + [0x63] + Array("torch\nFloatStorage\n".utf8)
        p += [0x58, 1, 0, 0, 0] + Array("0".utf8) + [0x58, 3, 0, 0, 0] + Array("cpu".utf8) + [0x4B, 10, 0x74, 0x51]
        p += [0x4B, 4, 0x4B, 2, 0x4B, 3, 0x86, 0x4B, 3, 0x4B, 1, 0x86, 0x89]
        p += [0x63] + Array("collections\nOrderedDict\n".utf8) + [0x29, 0x52, 0x74, 0x52, 0x73, 0x2E]
        let root = try TorchPickle.parse(Data(p))
        guard case .dict(let box) = root, box.items.count == 1, case .tensor(let tensor) = box.items[0].1 else {
            Issue.record("Expected one tensor"); return
        }
        #expect(tensor == TorchCheckpoint.Tensor(storage: "0", dtype: .float32, offset: 4, shape: [2, 3], stride: [3, 1]))
        // An unknown callable stays an opaque value; an unknown opcode fails.
        let opaque = try TorchPickle.parse(Data([0x80, 0x02, 0x63] + Array("os\nsystem\n".utf8) + [0x29, 0x52, 0x2E]))
        #expect(opaque == .opaque("os.system"))
        #expect(throws: (any Error).self) { try TorchPickle.parse(Data([0x80, 0x02, 0x69, 0x2E])) }
    }

    /// Trained weights: conversion of the pinned release, per-component parity against the captured FP16 MPS reference,
    /// then one end-to-end clip through the adapter's run path.
    @Test(.heavy) func trainedWeightsMatchTheReference() throws {
        let lab = try audioPortsLab()
        let source = lab.appendingPathComponent("src/bs-roformer"), snapshot = lab.appendingPathComponent("src/bs-roformer-fp16")
        if !FileManager.default.fileExists(atPath: snapshot.appendingPathComponent(BSRoformerConversion.marker).path) {
            try? FileManager.default.removeItem(at: snapshot)
            let start = Date()
            _ = try BSRoformerConversion.base.run(
                [
                    "src_dir": .string(source.path), "dst_dir": .string(snapshot.path), "dtype": "fp16", "group_size": 64,
                    "recipe": .string(BSRoformerConversion.recipe)
                ], cancellation: NativeCancellation())
            print("bs-roformer conversion seconds", Date().timeIntervalSince(start))
        }
        let adapter = BSRoformerAdapter()
        try adapter.load(
            try NativeLoadContext(["model_dir": .string(lab.path), "weights_dir": .string(snapshot.path), "dtype": "fp16"]),
            cancellation: NativeCancellation())
        let model = try #require(adapter.graph as? BSRoformerModel)
        let ref = try loadArrays(url: lab.appendingPathComponent("fixtures/roformer/components.safetensors"))
        var results: [String: Float] = [:]
        let (real, imag) = model.spectrum(ref["input"]!)
        results["stft"] = relativeL2(stacked([real, imag]), stacked([ref["stft_real"]!, ref["stft_imag"]!]))
        let features = model.features(real: ref["stft_real"]!, imag: ref["stft_imag"]!, dtype: .float16)
        results["band_split"] = relativeL2(features[0], ref["band_split"]!)
        let refFeatures = ref["band_split"]!.asType(.float16).expandedDimensions(axis: 0)
        results["layer0"] = relativeL2(model.layers[0](refFeatures)[0], ref["layer0"]!)
        let mask = model.mask(refFeatures)
        results["mask"] = relativeL2(mask[0], ref["mask"]!)
        let rendered = model.render(real: ref["stft_real"]!, imag: ref["stft_imag"]!, mask: ref["mask"]!.expandedDimensions(axis: 0))
        results["istft_from_ref_mask"] = relativeL2(rendered, ref["output"]!)
        results["chunk_output"] = relativeL2(model(ref["input"]!), ref["output"]!)
        print("bs-roformer parity", results.sorted { $0.key < $1.key })
        #expect(results["stft"]! < 1e-4 && results["istft_from_ref_mask"]! < 1e-4)
        #expect(results["band_split"]! < 5e-3 && results["layer0"]! < 1e-2 && results["mask"]! < 5e-2 && results["chunk_output"]! < 5e-2)

        let out = lab.appendingPathComponent("out/bs-roformer")
        try? FileManager.default.removeItem(at: out)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        Memory.peakMemory = 0
        let start = Date()
        let result = try adapter.run(
            "separate_audio",
            input: ["audio": ["path": .string(lab.appendingPathComponent("fixtures/roformer-clip.wav").path)], "out_dir": .string(out.path)],
            cancellation: NativeCancellation(), emit: { _ in })
        let seconds = Date().timeIntervalSince(start)
        let files = result["files"]?.array ?? []
        #expect(files.compactMap { $0["id"]?.string } == ["vocals", "instrumental"])
        let summary: JSONObject = [
            "parity": .object(results.mapValues { .number(Double($0)) }), "run_seconds": .number(seconds),
            "mlx_peak_bytes": .number(Double(Memory.peakMemory)), "files": .array(files)
        ]
        try WireJSON.encode(summary).write(to: out.appendingPathComponent("summary.json"))
        // Clips shorter than one FFT frame and than one chunk run padded and come back at their own length.
        for frames in [500, 57_000] {
            let clip = try audioWave(
                (0..<(frames * 2)).map { Float(0.3 * sin(Double($0) * 0.01)) }, rate: 44_100, channels: 2, id: "short", input: ["out_dir": .string(out.path)],
                cancellation: NativeCancellation())
            let short = try adapter.run(
                "separate_audio", input: ["audio": ["path": clip["path"]!], "out_dir": .string(out.path), "stems": ["vocals"]], cancellation: NativeCancellation(),
                emit: { _ in })
            let pcm = try NativeMedia.audio(URL(fileURLWithPath: try #require(short["files"]?.array?.first?["path"]?.string)), sampleRate: 44_100, channels: 2)
            #expect(pcm.samples.count == frames * 2)
        }
        // A 60 s clip, when the lab provides one: wall time and MLX peak only.
        let long = lab.appendingPathComponent("fixtures/roformer-60s.wav")
        if FileManager.default.fileExists(atPath: long.path) {
            Memory.peakMemory = 0
            let begin = Date()
            _ = try adapter.run("separate_audio", input: ["audio": ["path": .string(long.path)], "out_dir": .string(out.path)], cancellation: NativeCancellation(), emit: { _ in })
            try "60s seconds \(Date().timeIntervalSince(begin)) mlx_peak_bytes \(Memory.peakMemory)\n".write(
                to: out.appendingPathComponent("long.txt"), atomically: true, encoding: .utf8)
        }
        print("bs-roformer e2e seconds", seconds, "mlx peak GB", Double(Memory.peakMemory) / 1e9)
    }
}
