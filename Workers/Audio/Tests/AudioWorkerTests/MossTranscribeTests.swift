// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN
import ArsenalWire
import Testing
@testable import AudioWorker

@Suite struct MossTranscribeTests {
    /// The maker's parse_transcript (transcript_parser.py @79c675f0) on awkward transcripts, captured once.
    @Test func parserMatchesTheMaker() {
        let cases: [(String, [MossSegment])] = [
            (
                "[0.48][S01]Welcome everyone[1.66][12.26][S02]The new pipeline is ready[13.81][14.36][S01]Great[18.76]",
                [
                    MossSegment(start: 0.48, end: 1.66, speaker: "S01", text: "Welcome everyone"),
                    MossSegment(start: 12.26, end: 13.81, speaker: "S02", text: "The new pipeline is ready"), MossSegment(start: 14.36, end: 18.76, speaker: "S01", text: "Great")
                ]
            ),
            (
                "noise [1.0][S01] a [b] c [2.0] [3.0][S02]x[2.5][4.0][S03]tail[5.0]",
                [MossSegment(start: 1.0, end: 2.0, speaker: "S01", text: "a [b] c"), MossSegment(start: 3.0, end: 4.0, speaker: "S02", text: "x[2.5]")]
            ),
            ("[1][S1]x[2]junk[3][S02]y[4]", [MossSegment(start: 1.0, end: 3.0, speaker: "S1", text: "x[2]junk")]),
            ("[1.0][S01]  [2.0][3.0][S02]ok[4.0]", [MossSegment(start: 3.0, end: 4.0, speaker: "S02", text: "ok")]),
            ("[1.0][X1]bad[2.0][2.0][S07]good[3.5", []),
            ("[0.5][S01]日本語のテスト[1.5]", [MossSegment(start: 0.5, end: 1.5, speaker: "S01", text: "日本語のテスト")]),
            ("[1..0][S01]a[2]", []),
            ("[3][S01]back[2][3][S02]z[4]", [MossSegment(start: 3.0, end: 3.0, speaker: "S01", text: "back[2]")])
        ]
        for (text, expected) in cases { #expect(MossTranscriptParser.parse(text) == expected, "\(text)") }
        // Streaming in pieces gives the same segments as one pass.
        var parser = MossTranscriptParser()
        let text = cases[0].0
        let pieces = stride(from: 0, to: text.count, by: 7).map { String(text.dropFirst($0).prefix(7)) }
        #expect(pieces.flatMap { parser.feed($0) } + parser.close() == cases[0].1)
    }

    /// Review fixes (2 Nov): code points rather than grapheme clusters, the maker's duplicated 33rd timestamp character,
    /// times capped at 86,400 s; maker values captured from parse_transcript.
    @Test func parserMatchesTheMakerOnEdgeCases() {
        let ones33 = String(repeating: "1", count: 33), ones40 = String(repeating: "1", count: 40), twos = String(repeating: "2", count: 32)
        let cases: [(String, [MossSegment])] = [
            ("[1.0][S01]ab[" + ones33 + "x[2.0]", [MossSegment(start: 1, end: 2, speaker: "S01", text: "ab[" + ones33 + "1x")]),
            ("[1.0][S01]ab[" + ones40 + "[2.0]", [MossSegment(start: 1, end: 2, speaker: "S01", text: "ab[" + ones33 + "1" + String(repeating: "1", count: 7))]),
            ("[1.0][S01]x[\u{301}[2.0]", [MossSegment(start: 1, end: 2, speaker: "S01", text: "x[\u{301}")]),
            (
                "[1.0][S01]e\u{301} caf\u{e9}[2.0][3.0][S02]\u{1F1F5}\u{1F1F1}[4.0]",
                [MossSegment(start: 1, end: 2, speaker: "S01", text: "e\u{301} caf\u{e9}"), MossSegment(start: 3, end: 4, speaker: "S02", text: "\u{1F1F5}\u{1F1F1}")]
            ),
            ("[1.0][S01]a[" + twos + "]b[3.0]", [MossSegment(start: 1, end: 3, speaker: "S01", text: "a[" + twos + "]b")]),
            ("[90000][S01]late[90001]", [MossSegment(start: 86_400, end: 86_400, speaker: "S01", text: "late")])
        ]
        for (text, expected) in cases { #expect(MossTranscriptParser.parse(text) == expected, "\(text)") }
        #expect(MossProcessor.instruction(prompt: "  ", hotwords: ["MLX"]) == MossProcessor.defaultPrompt + "热词提示：MLX")
    }

    /// Review fix: streaming decodes only new tokens, holds a split character back, and hands over the rest at the end;
    /// fed to the parser and closed, the stream gives the same segments as the final parse.
    /// Bugs (2 Nov 2026 review): a transcript cut by max_new_tokens came back as a normal reply, and diarize refused
    /// music or noise that the model transcribed without timestamps under the default instruction.
    @Test func replySaysTruncatedAndDiarizesUntimedTextAsEmpty() throws {
        let timed = "[0.48][S01]Welcome everyone[1.66][12.26][S02]The new"
        for operation in ["transcribe", "diarize"] {
            let cut = try MossTranscribeAdapter.reply(operation, transcript: timed, truncated: true, customPrompt: false, duration: 20)
            #expect(cut["truncated"]?.bool == true && cut["segments"]?.array?.count == 1)
            let whole = try MossTranscribeAdapter.reply(operation, transcript: timed, truncated: false, customPrompt: false, duration: 20)
            #expect(whole["truncated"]?.bool == false)
        }
        let music = try MossTranscribeAdapter.reply("diarize", transcript: "(music)", truncated: false, customPrompt: false, duration: 20)
        #expect(music["segments"]?.array?.isEmpty == true && music["rttm"]?.string == "" && music["transcript"]?.string == "(music)")
        // A custom instruction that asked for something without timestamps is still refused with a 422.
        let custom = #expect(throws: WorkerFailure.self) {
            try MossTranscribeAdapter.reply("diarize", transcript: "[S01] hello", truncated: false, customPrompt: true, duration: 20)
        }
        #expect(custom?.code == "invalid_request")
        // Release review E (L-E14): an empty or blank prompt runs the default instruction, so it is not a custom one and
        // untimed music diarizes as empty rather than 422.
        for blank in [nil, "", "  \n"] as [String?] { #expect(MossProcessor.ownPrompt(blank) == nil, "\(String(describing: blank))") }
        #expect(MossProcessor.ownPrompt(" List the speakers ") == "List the speakers")
        let blankMusic = try MossTranscribeAdapter.reply(
            "diarize", transcript: "(music)", truncated: false, customPrompt: MossProcessor.ownPrompt("") != nil, duration: 20)
        #expect(blankMusic["segments"]?.array?.isEmpty == true)
    }

    @Test func streamHandsOverEveryTokenOnce() {
        let pieces = ["[0.5]", "[S01]", " Hello", " world", ".", "[1.5]", "[2.0]", "[S02]", " Za", "\u{FFFD}", "ż", "ółć", "[3.0]"]
        var decoder = MossStreamDecoder(), parser = MossTranscriptParser(), handed = "", segments: [MossSegment] = []
        func decode(_ tokens: [Int]) -> String {
            // Token 9 alone is a split character; joined with token 10 it decodes whole.
            tokens.map { $0 == 9 && tokens.last == 9 ? "\u{FFFD}" : $0 == 9 ? "" : pieces[$0] }.joined()
        }
        for token in pieces.indices {
            if let text = decoder.push(token, every: 3, decode: decode) { handed += text; segments += parser.feed(text) }
        }
        let rest = decoder.finish(decode: decode)
        handed += rest
        segments += parser.feed(rest) + parser.close()
        #expect(!handed.unicodeScalars.contains("\u{FFFD}"))
        #expect(segments == MossTranscriptParser.parse(handed) && segments.count == 2 && segments.last?.text == "Zażółć")
        let source =
            (try? String(
                contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
                    .deletingLastPathComponent().appendingPathComponent("Sources/AudioWorker/MossTranscribe/MossTranscribeAdapter.swift"), encoding: .utf8)) ?? ""
        #expect(source.contains("if token == MossProcessor.imEnd { break }"), "generation stops only on <|im_end|>, as the maker's generation_config")
    }

    /// The maker's `_audio_span_ids` (5 s markers at 12.5 tokens/s) and `_compute_audio_token_length`, captured once.
    @Test func audioSpanAndTokenCountsMatchTheMaker() throws {
        var config = MossConfig(text: Qwen3TextConfig())
        config.audioTokenID = -1
        let digits = Dictionary(uniqueKeysWithValues: "0123456789".enumerated().map { ($1, 100 + $0) })
        let spans: [(Int, Int, [Int], [Int])] = [
            (1, 1, [], []),
            (61, 61, [], []),
            (62, 62, [], []),
            (63, 64, [62], [105]),
            (124, 125, [62], [105]),
            (364, 373, [62, 125, 126, 189, 190, 253, 254, 317, 318], [105, 101, 100, 101, 105, 102, 100, 102, 105]),
            (
                1000, 1031,
                [62, 125, 126, 189, 190, 253, 254, 317, 318, 381, 382, 445, 446, 509, 510, 573, 574, 637, 638, 701, 702, 765, 766, 829, 830, 893, 894, 957, 958, 1021, 1022],
                [105, 101, 100, 101, 105, 102, 100, 102, 105, 103, 100, 103, 105, 104, 100, 104, 105, 105, 100, 105, 105, 106, 100, 106, 105, 107, 100, 107, 105, 108, 100]
            )
        ]
        for (tokens, length, positions, values) in spans {
            let span = MossProcessor.span(tokens: tokens, config: config, digits: digits)
            #expect(span.count == length && span.filter { $0 == -1 }.count == tokens, "\(tokens)")
            #expect(span.indices.filter { span[$0] != -1 } == positions && span.filter { $0 != -1 } == values, "\(tokens)")
        }
        for (samples, tokens) in [(1, 1), (1280, 1), (1281, 2), (16000, 13), (465760, 364), (480000, 375)] {
            #expect(MossProcessor.tokenCount(samples: samples, config: config) == tokens, "\(samples)")
        }
        #expect(MossProcessor.instruction(prompt: nil, hotwords: []) == MossProcessor.defaultPrompt)
        #expect(MossProcessor.instruction(prompt: nil, hotwords: ["Arsenal", "MLX"]) == MossProcessor.defaultPrompt + "热词提示：Arsenal, MLX")
        #expect(MossProcessor.instruction(prompt: "Transcribe.", hotwords: ["MLX"]) == "Transcribe. Hotwords: MLX")
        #expect(MossProcessor.instruction(prompt: "  ", hotwords: []) == MossProcessor.defaultPrompt)
    }

    /// Trained weights: Get's FP16 cast of the pinned release, then per component and end to end against the maker's code
    /// (captured once on MPS in FP16): features, Whisper encoder, adaptor, prompt ids, first-step logits, and the
    /// greedy transcript token for token.
    @Test(.heavy) func trainedWeightsMatchTheMaker() throws {
        let lab = try audioPortsLab()
        let source = lab.appendingPathComponent("src/moss"), snapshot = lab.appendingPathComponent("src/moss-fp16")
        if !FileManager.default.fileExists(atPath: snapshot.appendingPathComponent("config.json").path) {
            try? FileManager.default.removeItem(at: snapshot)
            _ = try NativeConversion(recipe: "cast:audio-moss-transcribe-diarize-v1", eligible: audioEligible).run(
                [
                    "src_dir": .string(source.path), "dst_dir": .string(snapshot.path), "dtype": "fp16", "group_size": 64,
                    "recipe": "cast:audio-moss-transcribe-diarize-v1"
                ], cancellation: NativeCancellation())
        }
        let adapter = MossTranscribeAdapter()
        try adapter.load(
            try NativeLoadContext(["model_dir": .string(lab.path), "weights_dir": .string(snapshot.path), "dtype": "fp16"]), cancellation: NativeCancellation())
        let model = try #require(adapter.graph as? MossTranscriber).model
        let fixtures = lab.appendingPathComponent("fixtures/moss")
        let reference = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: fixtures.appendingPathComponent("reference.json"))) as? [String: Any])
        let ref = try loadArrays(url: fixtures.appendingPathComponent("ref-default.safetensors"))
        let pcm = try NativeMedia.audio(fixtures.appendingPathComponent("fixture-2spk.wav"), sampleRate: 16000)
        var results: [String: Float] = [:]
        let features = MossTranscribeAdapter.features(pcm.samples, config: model.config)
        results["features"] = relativeL2(features, ref["input_features"]![0].transposed(1, 0))
        let whisper = try model.whisper(ref["input_features"]!.transposed(0, 2, 1))
        results["whisper"] = relativeL2(whisper, ref["whisper"]!)
        let (audio, tokens) = try adapter.audioEmbeddings(pcm.samples, model: model, cancellation: NativeCancellation())
        results["adaptor"] = relativeL2(audio, ref["adaptor"]![0])
        let maker = try #require(reference["default"] as? [String: Any])
        let result = try adapter.generate(pcm.samples, instruction: MossProcessor.defaultPrompt, maxNewTokens: 1024, cancellation: NativeCancellation())
        print("moss parity", results.sorted { $0.key < $1.key }, "tokens", result.tokens.count, (maker["generated_ids"] as? [Int])?.count ?? 0)
        print("moss transcript", result.transcript)
        #expect(tokens == 364 && result.promptTokens == (maker["input_ids"] as? [Int])?.count)
        #expect(results["features"]! < 1e-3 && results["whisper"]! < 2e-2 && results["adaptor"]! < 2e-2)
        let makerIDs = try #require(maker["generated_ids"] as? [Int]).filter { $0 != MossProcessor.imEnd && $0 != MossProcessor.endOfText }
        #expect(result.tokens == makerIDs)
        #expect(result.transcript == maker["text"] as? String)
        let out = lab.appendingPathComponent("out/moss")
        try? FileManager.default.removeItem(at: out)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        try WireJSON.encode([
            "parity": .object(results.mapValues { .number(Double($0)) }), "transcript": .string(result.transcript), "tokens": .array(result.tokens.map { .number(Double($0)) })
        ]).write(to: out.appendingPathComponent("summary.json"))
    }
}
