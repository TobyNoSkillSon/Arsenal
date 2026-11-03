// SPDX-License-Identifier: MIT
import Testing
import Foundation
import ArsenalWire
import MLX
import MLXNN
import VireoCore
import VireoEngine
@testable import VoicesWorker

@Suite(.serialized) struct VoicesTests {
    @Test func namedSpeechOperationsRespectCapabilities() throws {
        try SpeechOperation.check(["op": "design_voice", "description": "A calm narrator"], modelID: "qwen3-tts-voicedesign")
        try SpeechOperation.check(["op": "clone_voice", "audio": ["path": "/tmp/reference.wav"], "transcript": "Hello", "name": "Test"], modelID: "pocket-tts")
        try SpeechOperation.check(["op": "clone_voice", "audio": ["path": "/tmp/reference.wav"], "transcript": "", "name": "Test"], modelID: "pocket-tts")
        #expect(throws: (any Error).self) {
            try SpeechOperation.check(["op": "clone_voice", "audio": ["path": "/tmp/reference.wav"], "transcript": "", "name": "Test"], modelID: "qwen3-tts-base")
        }
        #expect(throws: (any Error).self) { try SpeechOperation.check(["op": "clone_voice"], modelID: "kokoro-82m") }
        #expect(throws: (any Error).self) { try SpeechOperation.check(["op": "design_voice", "description": ""], modelID: "breeze-tts-2") }
        #expect(throws: (any Error).self) { try SpeechOperation.check(["op": "clone_voice", "audio": ["path": ""], "transcript": "Hello", "name": "Test"], modelID: "pocket-tts") }
        #expect(SpeechOperation.language("en") == "english")
        #expect(SpeechOperation.language("ZH") == "chinese")
        #expect(SpeechOperation.language("auto") == "auto")
    }
    @Test func savedVoicesPersistAndStayModelScoped() throws {
        let support = URL(fileURLWithPath: "/tmp/arsenal-v2-voices/saved-" + UUID().uuidString)
        let store = SavedVoices(model: "qwen3-tts-voicedesign", support: support)
        let saved = try store.save(name: "Narrator", description: "A calm narrator", language: "english")
        let reopened = SavedVoices(model: "qwen3-tts-voicedesign", support: support)
        #expect(try reopened.list().map(\.id) == [saved.id])
        let resolved = try reopened.resolve(["voice": .string(saved.id), "text": "Hello"])
        #expect(resolved["instructions"] == "A calm narrator")
        #expect(resolved["language"] == "english")
        #expect(try SavedVoices(model: "breeze-tts-2", support: support).list().isEmpty)
        #expect(throws: (any Error).self) { try SavedVoices(model: "breeze-tts-2", support: support).resolve(["voice": .string(saved.id)]) }
    }
    @Test func designPreviewsAreTransientUntilNamed() throws {
        let support = URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-f/design-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        let preview = support.appendingPathComponent("preview.wav")
        try Data("chosen preview".utf8).write(to: preview)
        let design: JSONObject = ["seed": 24301, "model_revision": "revision-one", "preview_text": "Hello world.", "dtype": "bf16", "fast": true, "language": "english"]
        let store = SavedVoices(model: "qwen3-tts-voicedesign", support: support)
        for _ in 0..<3 {
            let result = try store.designResult(request: ["description": "Warm narrator"], preview: ["path": .string(preview.path)], design: design)
            #expect(result["saved"] == false && result["voice_id"]?.string?.hasPrefix("preview-") == true)
            #expect(result["design"] == .object(design))
        }
        #expect(try store.list().isEmpty)
        let result = try store.designResult(request: ["name": "Narrator", "description": "Warm narrator"], preview: ["path": .string(preview.path)], design: design)
        #expect(result["saved"] == true)
        let saved = try #require(store.list().first)
        #expect(saved.name == "Narrator" && saved.seed == 24301 && saved.modelRevision == "revision-one")
        #expect(try Data(contentsOf: URL(fileURLWithPath: #require(saved.preview))) == Data("chosen preview".utf8))
        let reopened = SavedVoices(model: "qwen3-tts-voicedesign", support: support)
        let resolved = try reopened.resolve(["voice": .string(saved.id), "text": "Hello world.", "seed": 123, "instructions": "Different voice"], modelRevision: "revision-one")
        #expect(resolved["seed"] == 24301 && resolved["instructions"] == "Warm narrator" && resolved["language"] == "english")
        #expect(throws: (any Error).self) { try reopened.resolve(["voice": .string(saved.id)], modelRevision: "revision-two") }
        #expect(throws: (any Error).self) { try store.designResult(request: ["name": "Bad"], preview: ["path": .string(preview.path)], design: [:]) }
    }
    @Test func testExactTurnsOffInexactOnly() {
        let overrides = FastPathComponents.overrides(ModelCell(tier: "16", flavour: "optimized", mode: "exact"))
        #expect(overrides["VIREO_EXACT"] == "1")
        #expect(overrides["VIREO_KOKORO_HALF"] == "0")
        #expect(overrides["VIREO_KOKORO_ALIGN"] == nil)
    }
    @Test func testCatalogPrecisionMatchesVireo() throws {
        #expect(SpeechCatalog.all.count == 5)
        let kokoro = try SpeechCatalog.model("kokoro-82m")
        #expect(kokoro.voices.contains("af_heart")); #expect(kokoro.voices.count == 28)
        #expect(throws: (any Error).self) { try SpeechCatalog.checkPrecision("bf16", spec: kokoro) }
        try SpeechCatalog.checkPrecision("int4", spec: kokoro)
        try SpeechCatalog.checkPrecision("int8", spec: kokoro)
    }
    @Test func testPCMRejectsNonfiniteAndClamps() throws {
        #expect(try PCM.encode([-2, 0, 2]).count == 6)
        #expect(PCM.header(rate: 24000, bytes: 6).count == 44)
        #expect(throws: (any Error).self) { try PCM.encode([.nan]) }
    }
    @Test func testQuantizationRulesProtectCodec() {
        #expect(QuantRules.pocket.selects("flow_lm.transformer.layers.0.linear1"))
        #expect(!QuantRules.pocket.selects("flow_lm.flow_net.linear1"))
        #expect(!QuantRules.qwen3Talker.selects("speaker_encoder.linear"))
        #expect(!QuantRules.breeze.selects("audio_tokenizer.decoder.linear"))
    }
    @Test(.heavy) func testSyntheticCastRoundTrip() async throws {
        RuntimeCaches.useBundledMetallib()
        let root = URL(fileURLWithPath: "/tmp/arsenal-voices/cast-" + UUID().uuidString)
        let base = root.appendingPathComponent("base"), output = root.appendingPathComponent("out")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Device.withDefaultDevice(.cpu) {
            let values = MLXArray([Float(0.125), 0.5, 1, -0.25])
            try MLX.save(arrays: ["tiny.weight": values], url: base.appendingPathComponent("tiny.safetensors"), stream: .cpu)
        }
        let source = ModelSource(id: "kokoro-82m", engine: "mlx-audio", directory: base, precision: "fp32")
        let report = try await Engines.castTo16(source, files: ["tiny.safetensors"], into: output)
        #expect(report.mode == "cast"); #expect(report.parameters == 4)
        let result = try MLX.loadArrays(url: output.appendingPathComponent("tiny.safetensors"), stream: .cpu)
        #expect(result["tiny.weight"]?.dtype == .float16)
        try Device.withDefaultDevice(.cpu) {
            #expect(result["tiny.weight"]?.asArray(Float.self) == [0.125, 0.5, 1, -0.25])
        }
    }
    @Test(.heavy) func testSyntheticAffineRoundTrip() throws {
        RuntimeCaches.useBundledMetallib()
        try Device.withDefaultDevice(.cpu) {
            let skeleton = Sequential { Linear(64, 2, bias: false) }
            let rule = QuantRule(id: "tiny", version: 1, include: [".*"], exclude: [], note: "tiny test")
            let plan = QuantizedWeights.plan(skeleton, rule: rule)
            let weights = MLXArray((0..<128).map { Float($0) / 128 }).reshaped([2, 64], stream: .cpu).asType(.bfloat16, stream: .cpu)
            let (arrays, parameters) = try QuantizedWeights.quantize([plan.selected[0].path + ".weight": weights], selected: plan.selected, bits: 8)
            #expect(parameters == 128)
            #expect(QuantizedWeights.bits(of: arrays) == 8)
            #expect(throws: (any Error).self) { try QuantizedWeights.quantize(arrays, selected: plan.selected, bits: 8) }
        }
    }
}
