// SPDX-License-Identifier: MIT
import AVFoundation
import Foundation
import IntegrationKit
import MLX
import ArsenalWire
@testable import RestorationWorker
import Testing

/// The `interpolate_video` driver RIFE and GIMM-VFI-R share, with a recording stand-in for the model: endpoints pass
/// through, a pair is prepared once and only when needed, video pairs are reset per source pair, and mismatched frames
/// are refused. CPU only, no weights.
@Suite(.serialized)
struct InterpolationDriverTests {
    init() { NativeRuntime.prepare() }

    final class Recorder {
        var pairs: [(Float, Float)] = []
        var times: [Double] = []
        /// Returns a constant frame whose value encodes t, so outputs show which path produced them.
        var pair: InterpolationDriver.Pair {
            { a, b in
                self.pairs.append((a[0, 0, 0, 0].item(Float.self), b[0, 0, 0, 0].item(Float.self)))
                return { t in
                    self.times.append(t); return MLXArray.zeros(a.shape, dtype: .float32) + Float(t)
                }
            }
        }
    }

    static func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("interp-driver-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func frame(_ value: Float, width: Int = 16, height: Int = 16) -> MLXArray { MLXArray.zeros([1, height, width, 3], dtype: .float16) + value }

    @Test func framesPassEndpointsThroughAndPrepareOnce() throws {
        try Device.withDefaultDevice(.cpu) {
            let root = try Self.directory()
            defer { try? FileManager.default.removeItem(at: root) }
            let first = root.appendingPathComponent("a.png"), second = root.appendingPathComponent("b.png")
            try MediaBridge.png(Self.frame(0.2), at: first); try MediaBridge.png(Self.frame(0.8), at: second)
            let recorder = Recorder()
            let times: [JSONValue] = [0, 0.00005, 0.5, 0.99995, 1, 0.25].map { .number($0) }
            let out = try InterpolationDriver.run(
                ["frames": [["path": .string(first.path)], ["path": .string(second.path)]], "t": .array(times), "out_dir": .string(root.appendingPathComponent("out").path)],
                cancellation: NativeCancellation(), emit: { _ in }, pair: recorder.pair)
            #expect(out["files"]?.array?.count == 6)
            #expect(recorder.pairs.count == 1)
            #expect(recorder.times == [0.5, 0.25])
            // t = 0 and t within 1e-4 of 0 are the first image's pixels; near 1 and 1 the second's.
            func pixel(_ i: Int) throws -> Float {
                try MediaBridge.tensor(NativeMedia.image(root.appendingPathComponent("out/frame-\(i).png")))[0, 0, 0, 0].item(Float.self)
            }
            let a = try pixel(0), b = try pixel(4)
            #expect(abs(a - 0.2) < 0.01 && abs(b - 0.8) < 0.01)
            #expect(try pixel(1) == a && pixel(3) == b)
            #expect(abs(try pixel(2) - 0.5) < 0.01)
        }
    }

    @Test func framesOfDifferentSizesAreRefusedBeforeTheModel() throws {
        try Device.withDefaultDevice(.cpu) {
            let root = try Self.directory()
            defer { try? FileManager.default.removeItem(at: root) }
            let first = root.appendingPathComponent("a.png"), second = root.appendingPathComponent("b.png")
            try MediaBridge.png(Self.frame(0.2), at: first); try MediaBridge.png(Self.frame(0.8, width: 20), at: second)
            let recorder = Recorder()
            #expect(throws: WorkerFailure.self) {
                try InterpolationDriver.run(
                    ["frames": [["path": .string(first.path)], ["path": .string(second.path)]], "out_dir": .string(root.path)],
                    cancellation: NativeCancellation(), emit: { _ in }, pair: recorder.pair)
            }
            #expect(recorder.pairs.isEmpty)
        }
    }

    /// Six frames at NTSC 30000/1001 timing, doubled: every other output lands on a source frame a few microseconds
    /// off in floating point and must pass through; each in-between frame comes from its own pair, prepared once.
    @Test func ntscVideoPreparesEachPairOnceAndPassesSourceFramesThrough() throws {
        try Device.withDefaultDevice(.cpu) {
            let root = try Self.directory()
            defer { try? FileManager.default.removeItem(at: root) }
            let source = root.appendingPathComponent("ntsc.mp4")
            let writer = try NativeVideoWriter(url: source, width: 16, height: 16)
            for index in 0..<6 {
                try writer.append(
                    MediaBridge.buffer(Self.frame(Float(index + 1) / 8)), time: CMTime(value: Int64(index) * 1001, timescale: 30000),
                    cancellation: NativeCancellation())
            }
            try MediaBridge.wait { try await writer.finish() }
            let recorder = Recorder()
            let out = try InterpolationDriver.run(
                ["video": ["path": .string(source.path)], "factor": 2, "out_dir": .string(root.appendingPathComponent("out").path)],
                cancellation: NativeCancellation(), emit: { _ in }, pair: recorder.pair)
            #expect(out["files"]?.array?.count == 1)
            #expect(recorder.pairs.count == 5)
            #expect(recorder.times.count == 5 && recorder.times.allSatisfy { abs($0 - 0.5) < 0.01 })
            // Consecutive source pairs, in order: each pair's second frame is the next pair's first.
            for (p, q) in zip(recorder.pairs, recorder.pairs.dropFirst()) { #expect(abs(p.1 - q.0) < 0.01 && q.0 > p.0) }
        }
    }

    /// Both interpolation models through their own `run` with random weights (GPU): endpoints are the inputs exactly,
    /// the in-between frame has the input size, and mismatched frames are refused.
    @Test(.heavy) func rifeAndGIMMServeTheSharedContract() throws {
        NativeRuntime.prepare()
        setenv("ARSENAL_NATIVE_LAB", "1", 1)
        let root = try Self.directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = root.appendingPathComponent("a.png"), second = root.appendingPathComponent("b.png"), odd = root.appendingPathComponent("c.png")
        let ramp = MLXArray((0..<(48 * 72 * 3)).map { Float($0 % 251) / 250 }).reshaped([1, 48, 72, 3])
        try MediaBridge.png(ramp, at: first); try MediaBridge.png(1 - ramp, at: second); try MediaBridge.png(Self.frame(0.5, width: 40), at: odd)
        let models: [(String, any NativeModel)] = [("rife", RIFEAdapter()), ("gimm_vfi_r", GIMMAdapter())]
        for (architecture, model) in models {
            try model.load(
                NativeLoadContext([
                    "model_dir": .string(root.path), "weights_dir": .string(root.path), "dtype": "fp16",
                    "config": ["architecture": .string(architecture), "random_weights": true]
                ]), cancellation: NativeCancellation())
            let out = root.appendingPathComponent("out-" + architecture)
            let reply: JSONObject
            do {
                reply = try model.run(
                    "interpolate_video", input: ["frames": [["path": .string(first.path)], ["path": .string(second.path)]], "t": [0, 0.5, 1], "out_dir": .string(out.path)],
                    cancellation: NativeCancellation(), emit: { _ in })
            } catch {
                Issue.record("\(architecture): \(error)"); continue
            }
            #expect(reply["files"]?.array?.count == 3, "\(architecture)")
            func image(_ url: URL) throws -> MLXArray { try MediaBridge.tensor(NativeMedia.image(url)) }
            #expect(MLX.arrayEqual(try image(out.appendingPathComponent("frame-0.png")), try image(first)).item(Bool.self), "\(architecture) t=0")
            #expect(MLX.arrayEqual(try image(out.appendingPathComponent("frame-2.png")), try image(second)).item(Bool.self), "\(architecture) t=1")
            #expect(try image(out.appendingPathComponent("frame-1.png")).shape == [1, 48, 72, 3], "\(architecture) t=0.5")
            #expect(throws: WorkerFailure.self, "\(architecture) size mismatch") {
                try model.run(
                    "interpolate_video", input: ["frames": [["path": .string(first.path)], ["path": .string(odd.path)]], "out_dir": .string(out.path)],
                    cancellation: NativeCancellation(), emit: { _ in })
            }
            model.unload()
        }
    }
}
