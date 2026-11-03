// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN
import MLXRandom
import ArsenalWire
import Testing
@testable import AudioWorker

@Suite struct AudioSRTests {
    static let signal: [Double] = (0..<400).map { (n: Int) -> Double in
        let t = Double(n)
        let a: Double = sin(0.05 * t), b: Double = 0.3 * sin(1.3 * t), c: Double = 0.1 * cos(2.9 * t)
        return a + b + c
    }

    @Test func filterTypeFollowsNumpyLegacySeeding() {
        // np.random.seed(s); np.random.choice(["butter", "cheby1", "ellip", "bessel"]) (numpy 2.4.6).
        #expect(AudioSRDSP.filter(seed: 42) == .ellip)
        #expect(AudioSRDSP.filter(seed: 0) == .butter)
        #expect(AudioSRDSP.filter(seed: 1) == .cheby1)
        #expect(AudioSRDSP.filter(seed: 7) == .bessel)
    }

    /// audiosr.lowpass.lowpass(x, highcut=8000, fs=48000, order=8, _type) under SciPy 1.17.1: samples 0, 57, 199, 399 and Σy².
    @Test func lowpassMatchesScipy() {
        let expected: [(AudioSRDSP.Filter, [Double], Double)] = [
            (.butter, [0.05641141964283304, 0.2878193684394464, -0.502002388771267, 0.5819087441528199], 195.9309604839088),
            (.cheby1, [0.054636508776601325, 0.2760562923688348, -0.4819342445913236, 0.5567718152523132], 180.61386601030418),
            (.ellip, [0.05563094216399167, 0.27514793811622634, -0.4808164002383865, 0.5506720670805885], 179.7979204674445),
            (.bessel, [0.06862413519438228, 0.2851142752123244, -0.4972832758393064, 0.566110188838995], 191.70005674429314)
        ]
        for (filter, samples, energy) in expected {
            let y = AudioSRDSP.lowpass(Self.signal, highcut: 8000, filter: filter)
            #expect(y.count == 400)
            for (index, value) in zip([0, 57, 199, 399], samples) { #expect(abs(y[index] - value) < 1e-9, "\(filter) [\(index)]") }
            #expect(abs(y.reduce(0) { $0 + $1 * $1 } - energy) < 1e-7)
        }
        #expect(AudioSRDSP.lowpass(Self.signal, highcut: 24_000, filter: .ellip) == Self.signal)
    }

    @Test func resamplePolyMatchesScipy() {
        let down = AudioSRDSP.resamplePoly(Self.signal, up: 3, down: 7)
        #expect(down.count == 172)
        for (index, value) in zip([0, 50, 100, 171], [0.20031127415960917, -0.27847040051256816, -0.5817227194463785, 0.751396341701193]) {
            #expect(abs(down[index] - value) < 1e-10)
        }
        let up = AudioSRDSP.resamplePoly(Self.signal, up: 7, down: 3)
        #expect(up.count == 934)
        for (index, value) in zip([0, 300, 600, 933], [0.10006626053203592, -0.07927065304711921, 0.5259379274433789, 0.08751579024630973]) {
            #expect(abs(up[index] - value) < 1e-10)
        }
    }

    /// librosa.filters.mel(sr=48000, n_fft=2048, n_mels=256, fmin=20, fmax=24000) (librosa 0.9.2), [mel, bin] entries.
    @Test func melBasisMatchesLibrosa() {
        let w = AudioSRDSP.melWeights()
        #expect(w.count == 1025 * 256)
        #expect(abs(w[1 * 256 + 0] - 0.01376257836818695) < 1e-7)
        #expect(w[9 * 256 + 10] == 0)
        #expect(abs(w[1000 * 256 + 255] - 0.0014104993315413594) < 1e-8)
        #expect(abs(w.reduce(0, +) - 10.916974067687988) < 1e-4)
    }

    /// Whole 5.12 s blocks in integers; the reference's float arithmetic gives 511 frames for 1186 samples.
    @Test func paddingUsesWholeBlocks() {
        #expect(AudioSRPipeline.padding(frames: 1186) == (512, 245_760))
        #expect(AudioSRPipeline.padding(frames: 245_760) == (512, 245_760))
        #expect(AudioSRPipeline.padding(frames: 245_761) == (1024, 491_520))
        #expect(AudioSRPipeline.padding(frames: 48_000 * 60) == (6144, 2_949_120))
    }

    @Test func scheduleAndNamesFollowTheReference() throws {
        let alphas = AudioSRConversion.alphasCumprod()
        #expect(alphas.count == 1000 && alphas[0] > 0.9999 && alphas[999] < 1e-4)
        let schedule = AudioSRPipeline.schedule(steps: 50)
        #expect(schedule.count == 50 && schedule.first?.t == 981 && schedule.last?.t == 1)
        #expect(AudioSRPipeline.schedule(steps: 300).count == 334) // range(0, 1000, 1000 // 300)
        #expect(AudioSRConversion.name("model.diffusion_model.input_blocks.4.1.norm.weight", rank: 1)?.0 == "unet.input_blocks.4.layers.1.norm.weight")
        let middle = AudioSRConversion.name("model.diffusion_model.middle_block.0.in_layers.2.weight", rank: 4)
        #expect(middle?.0 == "unet.middle_block.layers.0.in_layers.2.weight" && middle?.1 == [0, 2, 3, 1])
        let up = AudioSRConversion.name("first_stage_model.vocoder.ups.0.weight", rank: 3)
        #expect(up?.0 == "vocoder.ups.0.weight" && up?.1 == [1, 2, 0])
        let res = AudioSRConversion.name("first_stage_model.vocoder.resblocks.3.convs1.2.weight", rank: 3)
        #expect(res?.0 == "vocoder.resblocks.3.convs1.2.weight" && res?.1 == [0, 2, 1])
        #expect(AudioSRConversion.name("first_stage_model.encoder.down.0.block.1.conv1.bias", rank: 1)?.0 == "vae.encoder.down.0.block.1.conv1.bias")
        #expect(AudioSRConversion.name("cond_stage_models.0.vae.encoder.conv_in.weight", rank: 4) == nil)
        #expect(AudioSRConversion.name("clap.model.text_branch.embeddings.word_embeddings.weight", rank: 2) == nil)
    }

    /// The shipped schema and the adapter agree: `stream` is accepted (the benchmark sends it), steps stop at 100.
    @Test func requestBoundsMatchTheShippedSchema() throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: repo.appendingPathComponent("models/audiosr-basic/ops/enhance.input.json"))
        let properties = try #require((try JSONSerialization.jsonObject(with: data) as? [String: Any])?["properties"] as? [String: Any])
        let steps = try #require(properties["steps"] as? [String: Any])
        #expect(steps["maximum"] as? Int == AudioSRAdapter.stepRange.upperBound && steps["minimum"] as? Int == AudioSRAdapter.stepRange.lowerBound)
        #expect((properties["stream"] as? [String: Any])?["type"] as? String == "boolean")
        let adapter = AudioSRAdapter()
        func message(_ input: JSONObject) -> String? {
            do { _ = try adapter.run("enhance", input: input, cancellation: NativeCancellation(), emit: { _ in }); return nil } catch let failure as WorkerFailure {
                return failure.message
            } catch { return "\(error)" }
        }
        let audio: JSONObject = ["path": "/unused.wav"]
        // Valid requests reach the load check (nothing is loaded here).
        #expect(message(["audio": .object(audio), "stream": false]) == "AudioSR is not loaded.")
        #expect(message(["audio": .object(audio), "steps": 100]) == "AudioSR is not loaded.")
        #expect(message(["audio": .object(audio), "steps": 101]) == "steps must be an integer in 10...100.")
        #expect(message(["audio": .object(audio), "stream": "yes"]) == "stream must be a boolean.")
    }

    /// Trained weights: conversion, per-component parity against the provider's pipeline captured on MPS (FP16
    /// autocast), and the full pipeline replaying the reference's noise.
    @Test(.heavy) func trainedWeightsMatchTheReference() throws {
        let lab = try audioPortsLab()
        let source = lab.appendingPathComponent("src/audiosr"), snapshot = lab.appendingPathComponent("src/audiosr-fp16")
        if !FileManager.default.fileExists(atPath: snapshot.appendingPathComponent(AudioSRConversion.marker).path) {
            try? FileManager.default.removeItem(at: snapshot)
            let start = Date()
            _ = try AudioSRConversion.base.run(
                ["src_dir": .string(source.path), "dst_dir": .string(snapshot.path), "dtype": "fp16", "group_size": 64, "recipe": .string(AudioSRConversion.recipe)],
                cancellation: NativeCancellation())
            print("audiosr conversion seconds", Date().timeIntervalSince(start))
        }
        let adapter = AudioSRAdapter()
        try adapter.load(
            try NativeLoadContext(["model_dir": .string(lab.path), "weights_dir": .string(snapshot.path), "dtype": "fp16"]), cancellation: NativeCancellation())
        let model = try #require(adapter.graph as? AudioSRModel)
        let ref = try loadArrays(url: lab.appendingPathComponent("fixtures/audiosr/components.safetensors"))
        func nhwc(_ x: MLXArray) -> MLXArray { x.transposed(0, 2, 3, 1) }
        var results: [String: Float] = [:]

        // DSP: low-pass and mel features of the reference's normalized, padded waveform.
        let wave = ref["lowpass_in"]!.asArray(Float.self)
        let pipeline = AudioSRPipeline(model: model, scale: adapter.scale) { MLXRandom.normal($0) }
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: lab.appendingPathComponent("fixtures/audiosr/meta.json"))) as? [String: Any]
        let highcut = Int(try #require(meta?["highcut"] as? Double))
        #expect(meta?["type"] as? String == "ellip")
        let lowpassed = AudioSRDSP.lowpass(wave.map(Double.init), highcut: highcut, filter: AudioSRDSP.filter(seed: 42))
        results["lowpass"] = relativeL2(MLXArray(lowpassed.map(Float.init)), ref["lowpass_out"]!)
        let prepared = pipeline.prepare(wave, seed: 42)
        #expect(prepared.highcut == highcut && prepared.filter == .ellip)
        results["lowpass_mel"] = relativeL2(prepared.lowpassMel, ref["lowpass_mel"]![0])
        // Networks on the reference's own inputs.
        let posterior = model.vae.encode(nhwc(ref["encode_in"]!).asType(.float16))
        results["vae_encode_mean"] = relativeL2(posterior.mean, nhwc(ref["encode_mean"]!))
        results["vae_encode_std"] = relativeL2(posterior.std, nhwc(ref["encode_std"]!))
        let tRef = ref["unet_t"]!.asType(.int32)
        let unetIn = nhwc(ref["unet_x"]!)
        let unetOut = model.unet(unetIn[.ellipsis, ..<16].asType(.float16), condition: unetIn[.ellipsis, 16...].asType(.float16), timesteps: tRef)
        results["unet"] = relativeL2(unetOut, nhwc(ref["unet_out"]!))
        results["vae_decode"] = relativeL2(model.vae.decode(nhwc(ref["decode_in"]!).asType(.float16)), nhwc(ref["decode_out"]!))
        let vocoderIn = ref["vocoder_in"]!.transposed(0, 2, 1)
        results["vocoder"] = relativeL2(model.vocoder(vocoderIn.asType(.float16)), ref["vocoder_out"]![0..., 0])
        let post = AudioSRPipeline.replaceLowBand(ref["post_in"]![0, 0].asArray(Float.self), lowpassed: ref["post_lowpass"]![0, 0].asArray(Float.self))
        results["low_band_replacement"] = relativeL2(MLXArray(post), ref["post_out"]![0, 0])

        // Full pipeline with the reference's noise: [0] target-latent sample (unused), [1] condition sample, [2] x_T,
        // then one per DDIM step.
        var next = 1
        let replay = AudioSRPipeline(model: model, scale: adapter.scale) { shape in
            defer { next += 1 }
            return nhwc(ref[String(format: "noise_%03d", next)]!).reshaped(shape)
        }
        let samples = try audioReadMono(lab.appendingPathComponent("fixtures/audiosr-clip.wav"))
        Memory.peakMemory = 0
        let start = Date()
        let out = try replay.run(samples, seed: 42, steps: 50, guidance: 3.5, cancellation: NativeCancellation())
        let seconds = Date().timeIntervalSince(start)
        results["final_waveform"] = relativeL2(MLXArray(out), ref["final"]![..<out.count])
        print("audiosr parity", results.sorted { $0.key < $1.key }, "seconds", seconds, "mlx peak GB", Double(Memory.peakMemory) / 1e9)
        let dir = lab.appendingPathComponent("out/audiosr")
        try? FileManager.default.removeItem(at: dir)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let replayed = try audioWave(out, rate: 48_000, channels: 1, id: "replay", input: ["out_dir": .string(dir.path)], cancellation: NativeCancellation())

        // Production path: the adapter's own seeded noise, as served.
        Memory.peakMemory = 0
        let served = Date()
        let result = try adapter.run(
            "enhance", input: ["audio": ["path": .string(lab.appendingPathComponent("fixtures/audiosr-clip.wav").path)], "out_dir": .string(dir.path)],
            cancellation: NativeCancellation(), emit: { _ in })
        let summary: JSONObject = [
            "parity": .object(results.mapValues { .number(Double($0)) }), "replay_seconds": .number(seconds), "replay": replayed,
            "served_seconds": .number(Date().timeIntervalSince(served)), "served_mlx_peak_bytes": .number(Double(Memory.peakMemory)),
            "served": result["files"] ?? .null
        ]
        try WireJSON.encode(summary).write(to: dir.appendingPathComponent("summary.json"))
        // The 60 s maximum, when the lab provides a clip: wall time and MLX peak only.
        let long = lab.appendingPathComponent("fixtures/audiosr-60s.wav")
        if FileManager.default.fileExists(atPath: long.path) {
            Memory.peakMemory = 0
            let begin = Date()
            _ = try adapter.run("enhance", input: ["audio": ["path": .string(long.path)], "out_dir": .string(dir.path)], cancellation: NativeCancellation(), emit: { _ in })
            let line = "60s seconds \(Date().timeIntervalSince(begin)) mlx_peak_bytes \(Memory.peakMemory)\n"
            try line.write(to: dir.appendingPathComponent("long.txt"), atomically: true, encoding: .utf8)
        }
        #expect(results["lowpass"]! < 1e-4 && results["low_band_replacement"]! < 1e-3)
        #expect(results["unet"]! < 2e-2 && results["vae_decode"]! < 2e-2 && results["vocoder"]! < 5e-2)
    }
}

func audioReadMono(_ url: URL) throws -> [Float] { try NativeMedia.audio(url, sampleRate: 48_000, channels: 1).samples }
