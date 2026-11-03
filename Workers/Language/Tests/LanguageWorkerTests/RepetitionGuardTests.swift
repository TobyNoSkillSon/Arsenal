// SPDX-License-Identifier: MIT
import Foundation
import Testing
import MLX
import MLXNN
import IntegrationKit
import ArsenalWire
@testable import LanguageWorker

struct RepetitionGuardTests {
    init() { MLXReady.ensure() }

    /// Tokens appended until the guard stops, or nil when it never does.
    private func stop(_ tokens: [Int], span: Int) -> Int? {
        var loop = RepetitionGuard(span: span)
        for (index, token) in tokens.enumerated() where loop.append(token) { return index + 1 }
        return nil
    }

    @Test func periodicLoopsStopOnceTheyCoverTheSpan() {
        // Chandra's rule: a block of p tokens needs more than ⌊4·(1 + 3/p)⌋ repeats.
        #expect(RepetitionGuard.copies(period: 1) == 17); #expect(RepetitionGuard.copies(period: 3) == 9); #expect(RepetitionGuard.copies(period: 12) == 6)
        let prefix = Array(1000..<1040)
        for (period, span) in [(1, 48), (2, 48), (7, 256), (40, 256), (100, 512), (128, 512)] {
            let block = Array(2000..<(2000 + period))
            let output = prefix + Array(repeating: block, count: 1000 / period + 1).flatMap { $0 }
            let tail = max(span, period * RepetitionGuard.copies(period: period))
            // A block of one or two tokens trips the no-progress rule first, once the span holds at most 3 distinct
            // tokens (one or two prefix tokens still in it).
            let early = tail == span ? max(0, 3 - period) : 0
            #expect(stop(output, span: span) == prefix.count + tail - early, "period \(period) span \(span)")
        }
    }

    @Test func legitimateRepeatsShorterThanTheSpanContinue() {
        // A 40-token dot leader, ten identical 45-token empty table rows (450 tokens) and a long separator line, between
        // ordinary text, never reach the 512-token document span.
        var output = Array(1..<300) + Array(repeating: 7, count: 40) + Array(300..<400)
        let row = [3000] + Array(repeating: [3001, 3002], count: 21).flatMap { $0 } + [3003, 3004]
        output += Array(repeating: row, count: 10).flatMap { $0 } + Array(400..<700) + Array(repeating: 9, count: 200) + Array(700..<900)
        #expect(stop(output, span: RepetitionGuard.span(operation: "read_document")) == nil)
        // Varied text over a large vocabulary never stops.
        var generator = SystemRandomNumberGenerator()
        let prose = (0..<6000).map { _ in Int.random(in: 0..<5000, using: &generator) }
        for operation in ["translate", "read_document", "extract", "ground_screen"] { #expect(stop(prose, span: RepetitionGuard.span(operation: operation)) == nil) }
    }

    @Test func noProgressOverThreeTokensStops() {
        // Not periodic, but only three distinct tokens: the Qwen3-ASR backstop shape over the span.
        var generator = SystemRandomNumberGenerator()
        let output = Array(1..<100) + (0..<2000).map { _ in [5, 6, 7].randomElement(using: &generator)! }
        let at = stop(output, span: 256)
        #expect(at != nil && at! <= 99 + 256)
    }

    /// A shrunk random Index-Translate graph whose tied embedding is all zeros generates token 0 forever (every logit
    /// ties, never EOS). The worker stops it at the translate span with finish_reason `repetition`, not at max_tokens,
    /// and the folder's 32,768-token serving context rejects what the checkpoint's 262,144 positions would admit.
    @Test(.heavy) func workerStopsALoopingDecoderAndHoldsTheServingContext() throws {
        setenv("ARSENAL_NATIVE_LAB", "1", 1)
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let folder = root.appendingPathComponent("models/index-translate-2b")
        let fixture = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("gate/no-weights.json")))
        var config = try #require(fixture["config"]?.object)
        var text = try #require(config["text_config"]?.object)
        text["max_position_embeddings"] = 262_144; config["text_config"] = .object(text)
        let weights = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-loop-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: weights) }
        try WireJSON.encode(config).write(to: weights.appendingPathComponent("config.json"))
        let native = LanguageNative()
        try native.load(
            NativeLoadContext([
                "weights_dir": .string(weights.path), "model_dir": .string(folder.path), "dtype": "fp16", "config": ["random_weights": true, "architecture": "qwen3_5"]
            ]), cancellation: NativeCancellation())
        defer { native.unload() }
        let graph = try #require(native.model)
        let (key, table) = try #require(graph.parameters().flattened().first { $0.0.hasSuffix("embed_tokens.weight") })
        try graph.update(parameters: ModuleParameters.unflattened([key: MLXArray.zeros(like: table)]), verify: .none)

        var lengths: [Int] = []
        let result = try native.run(
            "translate", input: ["text": ["Hello"], "target_lang": "pl", "max_tokens": 1024], cancellation: NativeCancellation()
        ) { _ in }
        let item = try #require(result["translations"]?.array?.first?.object)
        #expect(item["finish_reason"]?.string == "repetition")
        lengths.append((item["text"]?.string ?? "").utf8.count)
        #expect(lengths == [RepetitionGuard.span(operation: "translate")])

        #expect(throws: WorkerFailure.self) {
            try native.run("translate", input: ["text": ["Hello"], "target_lang": "pl", "max_tokens": 32_768], cancellation: NativeCancellation()) { _ in }
        }
    }
}
