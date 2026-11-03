// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore
import MCP
@testable import ArsenalAPI

/// How each kind of output reaches the agent, file inputs, the table and settings, and the HTTP hints.
@Suite struct MCPAgentOutputTests {
    func render(_ result: JSONObject, operation: String, input: JSONObject = [:]) throws -> (String, URL) {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-mcp-render-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let router = APIRouter(service: StubModelsService(), support: directory)
        let entry = CatalogEntry(row: ModelRow(id: "m", name: "Model", kind: "audio"))
        let op = ModelOperation(name: operation, title: operation, inputSchema: ["type": "object"], output: "json", stream: false)
        let rendered = try router.mcpRenderText(result, entry: entry, operation: op, input: input, seconds: 4, directory: directory)
        #expect(try String(contentsOf: directory.appendingPathComponent("output.txt"), encoding: .utf8) == rendered.text, "output.txt holds the whole text")
        return (rendered.text, directory)
    }

    @Test func everyOutputTypeHasItsOwnReadableText() throws {
        let transcript: JSONObject = ["text": "Hello there.", "duration": 192, "language": "en", "segments": [["start": 1.2, "end": 4.5, "text": " Hello there."]]]
        var (text, folder) = try render(transcript, operation: "transcribe")
        #expect(text == "Transcript from m (3 min 12 s of audio, English, took 4 s). Full text: \(folder.path)/output.txt\nHello there.")
        var undetermined = transcript; undetermined["language"] = "und"
        (text, _) = try render(undetermined, operation: "transcribe")
        #expect(text.hasPrefix("Transcript from m (3 min 12 s of audio, took 4 s)."), "no raw placeholder code: \(text)")
        (text, _) = try render(transcript, operation: "transcribe", input: ["word_timestamps": true])
        #expect(text.hasSuffix("\n[00:01.2–00:04.5] Hello there."))
        (text, _) = try render(
            ["segments": [["start": 0, "end": 2.25, "speaker": "SPEAKER_00", "text": "Hi"], ["start": 2.25, "end": 3, "speaker": "SPEAKER_01"]]], operation: "diarize")
        #expect(text.hasPrefix("Speakers from m (2 speakers, 2 segments, took 4 s).") && text.hasSuffix("\n[00:00.0–00:02.3] SPEAKER_00: Hi\n[00:02.3–00:03.0] SPEAKER_01:"))
        (text, _) = try render(["prediction": 1, "probability": 0.93], operation: "end_of_turn")
        #expect(text == "End of turn (probability 0.93), from m.")
        (text, folder) = try render(["embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "input_tokens": 4], operation: "embed")
        #expect(text == "Embedded 2 inputs with m (took 4 s): 2 vectors × 3 dims in \(folder.path)/output.json (JSON key \"embeddings\").")
        #expect(try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("output.json")))["embeddings"]?.array?.count == 2)
        (text, _) = try render(
            ["results": [["index": 1, "relevance_score": 0.912], ["index": 0, "relevance_score": 0.1]]], operation: "rerank", input: ["documents": ["alpha", "beta"]])
        #expect(text.hasSuffix("\n1. document 1 — 0.912: beta\n2. document 0 — 0.1: alpha"))
        let detection: JSONObject = [
            "coordinate_system": "pixels", "detections": [["label": "person", "score": 0.93, "box": ["xmin": 12, "ymin": 40, "xmax": 300, "ymax": 620]]]
        ]
        (text, _) = try render(detection, operation: "detect")
        #expect(text.hasPrefix("Found 1 object with m (took 4 s; coordinates: pixels).") && text.hasSuffix("\n1. person 0.93 — box (12, 40)–(300, 620)"))
        (text, _) = try render(["coordinate_system": "pixels", "detections": [["label": "OK button", "point": ["x": 10.5, "y": 20]]]], operation: "ground_screen")
        #expect(text.hasPrefix("Located 1 target with m") && text.hasSuffix("1. OK button — point (10.5, 20)"))
        (text, _) = try render(["translations": [["text": "Dzień dobry", "detected_source_language": "en"]]], operation: "translate")
        #expect(text.hasPrefix("Translation from m (from English, took 4 s).") && text.hasSuffix("\nDzień dobry"))
        for placeholder in ["und", "unknown", ""] {
            (text, _) = try render(["translations": [["text": "Dzień dobry", "detected_source_language": .string(placeholder)]]], operation: "translate")
            #expect(text.hasPrefix("Translation from m (took 4 s).") && !text.contains("from und"), "\(text)")
        }
        #expect(MCPFormat.language("pl-PL") == "Polish" && MCPFormat.language("Polish") == "Polish" && MCPFormat.language(nil) == nil)
        (text, folder) = try render(["pages": [["index": 0, "markdown": "# Title", "images": [["id": "a", "image_base64": "AAAA"]]]]], operation: "read_document")
        #expect(text.hasPrefix("Read 1 page with m (took 4 s). 1 embedded image are in \(folder.path)/output.json.") && text.hasSuffix("\nPage 1:\n# Title"))
        (text, _) = try render(["data": ["name": "Ada", "age": 36], "finish_reason": "stop"], operation: "extract")
        #expect(text.hasSuffix("\n{\n  \"age\" : 36,\n  \"name\" : \"Ada\"\n}"))
        let answers: JSONObject = [
            "answers": [
                "bug": ["type": "noul", "noul": 0.19], "kind": ["type": "choice", "choice": "frontend", "probabilities": ["frontend": 0.623]],
                "priority": ["type": "score", "score": 1.0663, "confidence": 0.5691]
            ]
        ]
        (text, _) = try render(answers, operation: "decide")
        #expect(text.hasSuffix("\nbug: no 0.81\nkind: frontend 0.62\npriority: score 1.07 (confidence 0.57)"))
        (text, _) = try render(["results": [["index": 0, "ok": true, "answers": ["bug": ["noul": 0.9]]], ["index": 1, "ok": false, "error": "Text only."]]], operation: "judge")
        #expect(text.hasSuffix("\n1. bug yes 0.9\n2. error: Text only."))
        (text, _) = try render(["text": "Hi", "usage": ["completion_tokens": 2]], operation: "chat")
        #expect(text.hasPrefix("Reply from m (2 tokens, took 4 s).") && text.hasSuffix("\nHi"))
        (text, _) = try render(["tokens": 8], operation: "count_tokens")
        #expect(text == "8 tokens, counted by m.")
        (text, _) = try render(["session_id": "s1", "closed": true], operation: "segment_session")
        #expect(text.hasPrefix("Result of segment_session from m (took 4 s).") && text.hasSuffix("\n{\n  \"closed\" : true,\n  \"session_id\" : \"s1\"\n}"))
    }

    @Test func longTextIsNeverTrimmed() throws {
        let long = String(repeating: "word ", count: 12_000)
        let (text, folder) = try render(["text": .string(long)], operation: "chat")
        #expect(text.count > 60_000 && text.hasSuffix(long))
        #expect(try String(contentsOf: folder.appendingPathComponent("output.txt"), encoding: .utf8).hasSuffix(long))
    }

    @Test func filesComeBackAsPathsLinksAndThumbnails() async throws {
        let h = try await MCPHarness(rows: [ModelRow(id: "stub-images", name: "Stub Image", kind: "images", states: ["bf16": .init(state: .ready)])])
        defer { Task { await h.close() } }
        let result = try await h.call("run", ["model": "stub-images", "input": ["prompt": "fixture", "n": 2]])
        let text = mcpResultText(result)
        let lines = text.components(separatedBy: "\n")
        #expect(lines[0].hasPrefix("Created 2 images with stub-images (") && lines[0].contains(", seed 1:"), "\(text)")
        #expect(lines[1].contains("stub-image-0.png (1×1, PNG, ") && FileManager.default.fileExists(atPath: String(lines[1].prefix { $0 != " " })))
        #expect(
            result.content.filter {
                if case .resourceLink = $0 { return true }; return false
            }.count == 2)
        #expect(
            result.content.contains {
                if case .image = $0 { return true }; return false
            })
        let json = try await h.call("run", ["model": "stub-images", "input": ["prompt": "fixture"], "format": "json"])
        let object = try WireJSON.decode(JSONEncoder().encode(try #require(json.structuredContent)))
        #expect(object["files"]?.array?.count == 1 && object["model"] == "stub-images" && object["job"]?.string?.count == 10)
        #expect(mcpResultText(json) == String(decoding: try WireJSON.encode(object), as: UTF8.self))
    }

    @MainActor @Test func speechSaysWhetherItPlayedAndKeepsTheFileWhenPlaybackFails() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-mcp-speech-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let queue = PlaybackQueue(muted: false, outputEnabled: false, meter: { _ in throw WorkerFailure(code: "audio", message: "Test output unavailable.") })
        let service = StubModelsService(rows: [ModelRow(id: "voice", name: "Voice", kind: "voices", states: ["bf16": .init(state: .ready)])], supportDirectory: root)
        let router = APIRouter(service: service, support: root, playbackQueue: queue)
        let failed = mcpResultText(try await router.callMCPTool("\(MCPNames.prefix)run", arguments: ["model": "voice", "input": ["text": "Hi"]], requester: "Test", notify: nil))
        #expect(failed.hasSuffix("\nNot played: Test output unavailable. The file is kept."), "\(failed)")
        let path = String(failed.components(separatedBy: "\n")[1].prefix { $0 != " " })
        #expect(FileManager.default.fileExists(atPath: path))
        let muted = APIRouter(service: service, support: root, playbackQueue: PlaybackQueue(muted: true))
        #expect(
            mcpResultText(try await muted.callMCPTool("\(MCPNames.prefix)run", arguments: ["model": "voice", "input": ["text": "Hi"]], requester: "Test", notify: nil)).hasSuffix(
                "\nNot played: playback is muted."))
        let silent = mcpResultText(
            try await muted.callMCPTool("\(MCPNames.prefix)run", arguments: ["model": "voice", "input": ["text": "Hi"], "play": false], requester: "Test", notify: nil))
        #expect(silent.hasPrefix("Created 1 audio file with voice") && !silent.contains("played"))
    }

    @Test func pathInputsCloneRegularFilesAndRefuseEverythingElse() async throws {
        let h = try await MCPHarness(rows: [ModelRow(id: "voice", name: "Voice", kind: "voices", states: ["bf16": .init(state: .ready)])])
        defer { Task { await h.close() } }
        let outside = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-mcp-outside-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: outside) }
        let clip = outside.appendingPathComponent("clip.wav"); try Data("RIFF fixture".utf8).write(to: clip)
        let link = outside.appendingPathComponent("link.wav"); try FileManager.default.createSymbolicLink(at: link, withDestinationURL: clip)
        try await h.router.prepareMCP()
        let token = outside.appendingPathComponent("token.wav")
        try FileManager.default.createSymbolicLink(at: token, withDestinationURL: h.root.appendingPathComponent("mcp-token"))
        let inbox = mcpReal(h.root.appendingPathComponent("inbox"))
        for source in [clip, link] {
            let (imported, copies) = try await h.router.mcpPrepareInputs(["reference": ["path": .string(source.path)]], environment: h.environment, importing: true)
            let path = try #require(imported["reference"]?["path"]?.string)
            #expect(copies.isEmpty && path.hasPrefix(inbox + "/") && path.hasSuffix("/clip.wav"), "\(path)")
            #expect(try Data(contentsOf: URL(fileURLWithPath: path)) == Data("RIFF fixture".utf8))
            #expect((try FileManager.default.attributesOfItem(atPath: path)[.posixPermissions] as? NSNumber)?.intValue == 0o600)
            let again = try await h.router.mcpPrepareInputs(imported, environment: h.environment, importing: true)
            #expect(again.0 == imported && again.1.isEmpty, "an inbox file is used in place")
        }
        let refusals: [(String, String)] = [
            ("clip.wav", "[invalid_input] Not run: input.reference.path must be an absolute path to a file on this Mac (\"clip.wav\" is not)."),
            (outside.path, "[invalid_input] Not run: \(outside.path) is not a regular file (it is a folder). Pass a file."),
            (
                outside.appendingPathComponent("missing.wav").path,
                "[invalid_input] Not run: \(outside.appendingPathComponent("missing.wav").path) does not exist or Arsenal cannot see it."
            ),
            (token.path, "[invalid_input] Not run: input.reference.path points into Arsenal's own support folder, which is not an input location."),
            (
                h.root.appendingPathComponent("mcp-token").path,
                "[invalid_input] Not run: input.reference.path points into Arsenal's own support folder, which is not an input location."
            ),
            ("/dev/fd/0", "[invalid_input] Not run: input.reference.path points at a device (/dev), which is not an input file."),
            ("/dev/null", "[invalid_input] Not run: input.reference.path points at a device (/dev), which is not an input file.")
        ]
        for (path, line) in refusals {
            let result = try await h.call("run", ["model": "voice", "input": ["text": "Hi", "reference": ["path": .string(path)]]])
            #expect(result.isError == true && mcpResultText(result) == line, "\(mcpResultText(result))")
        }
        // Across volumes the clone fails: the copy is queued for the job, and the disk check comes first.
        let tight = try await MCPHarness(rows: [], freeDisk: 4, cloneFile: { _, _ in EXDEV })
        defer { Task { await tight.close() } }
        let (_, copies) = try await tight.router.mcpPrepareInputs(["audio": ["path": .string(clip.path)]], environment: tight.environment, importing: true)
        #expect(copies.count == 1 && copies[0].bytes == 12 && copies[0].source == mcpReal(clip))
        #expect(throws: MCPFailure(.disk, "Not run: copying the input files needs 12 bytes, 4 bytes free on the startup disk.")) {
            try tight.router.mcpRequireCopyDisk(copies, environment: tight.environment)
        }
        let private_ = try await MCPHarness(rows: [], cloneFile: { _, _ in EPERM })
        defer { Task { await private_.close() } }
        do {
            _ = try await private_.router.mcpPrepareInputs(["audio": ["path": .string(clip.path)]], environment: private_.environment, importing: true)
            Issue.record("EPERM was accepted")
        } catch let failure as MCPFailure {
            #expect(failure.category == .invalidInput && failure.text.contains("macOS privacy settings may keep Arsenal out of that folder"))
        }
        // The run uses the clone.
        let ran = try await h.call("run", ["model": "voice", "input": ["text": "Hi", "reference": ["path": .string(clip.path)]], "play": false])
        #expect(ran.isError != true, "\(mcpResultText(ran))")
    }

    @Test func uploadsTakeSmallBase64Files() async throws {
        let h = try await MCPHarness(rows: [])
        defer { Task { await h.close() } }
        let stored = try await h.text("upload", ["filename": "a.wav", "data": .string(Data("fixture".utf8).base64EncodedString())])
        let path = try #require(stored.components(separatedBy: ": ").dropFirst().first?.components(separatedBy: "\n").first)
        #expect(stored.hasPrefix("Stored a.wav (7 bytes) in the Arsenal inbox for 24 hours: ") && FileManager.default.fileExists(atPath: path))
        #expect(try await h.text("upload", ["filename": "../a.wav", "data": "AA=="]).hasPrefix("[invalid_input] Not stored: filename"))
        #expect(try await h.text("upload", ["filename": "big.bin", "data": .string(Data(count: 800 * 1024).base64EncodedString())]).hasPrefix("[limit] Not stored"))
        let stale = URL(fileURLWithPath: path).deletingLastPathComponent()
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(-90000)], ofItemAtPath: stale.path)
        await h.router.pruneMCPInbox()
        #expect(!FileManager.default.fileExists(atPath: path))
    }

    @Test func tableActionsRefuseBusyModelsAndSayWhatHappened() async throws {
        let h = try await MCPHarness(
            rows: [decisionRow(bytes: 600_000_000, state: .ready), decisionRow("other", bytes: 0, state: .loaded)], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        #expect(try await h.text("add", ["model": "laya"]) == "Laya Decisions (laya) is already on the user's models; nothing changed.")
        #expect(try await h.text("unload", ["model": "laya"]) == "Laya Decisions wasn't loaded; nothing changed.")
        #expect(try await h.text("unload", ["model": "all"]) == "Unloaded 1 model: other.")
        let job = try await h.router.startMCPJob(kind: .run, entry: try await h.router.mcpEntry("laya"), operation: "decide", dtype: "fp16", requester: "Test") { _ in
            try await h.environment.sleep(100_000); return mcpText("never")
        }
        let busy = try await h.call("delete", ["model": "laya"])
        #expect(
            busy.isError == true
                && mcpResultText(busy) == "[busy] Not deleted: laya is busy with the decide with Laya Decisions (job \(job.id)). Cancel it with \(MCPNames.prefix)cancel or wait.")
        #expect(try await h.text("remove", ["model": "laya"]).hasPrefix("[busy] Not removed"))
        await job.cancel()
        // Without confirm:true nothing is deleted: the answer says what would go.
        #expect(
            try await h.text("delete", ["model": "laya"])
                == "Not deleted: Laya Decisions (laya) has fp16 600 MB on disk. Ask the user. With their yes, call \(MCPNames.prefix)delete(model:\"laya\", confirm:true).")
        #expect(await h.service.rows().first { $0.id == "laya" }?.states["fp16"]?.state == .ready)
        #expect(
            try await h.text("delete", ["model": "laya", "confirm": true])
                == "Deleted the downloaded files of Laya Decisions (laya), about 600 MB freed. It is off the user's table; using it again downloads it with the user's yes.")
        #expect(try await h.text("delete", ["model": "laya"]) == "Laya Decisions has no downloaded files; nothing deleted.")
        // Without files it is off the table already (the table holds exactly the models with files on this Mac).
        #expect(try await h.text("remove", ["model": "laya"]) == "Laya Decisions (laya) is not on the user's models; nothing changed.")
        #expect(
            try await h.text("add", ["model": "laya"])
                == "Nothing was downloaded, so Laya Decisions (laya) is not on the user's table yet; it joins it when its download starts. "
                + "\(MCPNames.prefix)run downloads it once the user agrees.")
        #expect(try await h.text("delete", ["model": "other", "remove": true]).hasPrefix("Not deleted: Laya Decisions (other) has fp16 on disk."))
        #expect(
            try await h.text("delete", ["model": "other", "remove": true, "confirm": true])
                == "Deleted the downloaded files of Laya Decisions (other). It is off the user's table; using it again downloads it with the user's yes.")
        // Neither has files now, so neither is on the table; both stay in the catalog.
        #expect(await h.service.rows().isEmpty)
        #expect(await h.service.catalog(query: .init()).filter(\.added).isEmpty)
    }

    @Test func settingsMapReservedKeysAndReportWhatApplied() async throws {
        let declaration: JSONObject = ["id": "speed", "type": "number", "title": "Speed", "default": 1, "min": 0.5, "max": 2, "applies": "request", "maps_to": "request.speed"]
        var row = ModelRow(id: "voice", name: "Voice", kind: "voices", precisions: ["fp16", "int8"], states: ["fp16": .init(state: .ready)], selectedPrecision: "fp16")
        row.runner = "native"
        let h = try await MCPHarness(rows: [row], metadata: ["voice": ["setting": [.object(declaration)]]])
        defer { Task { await h.close() } }
        let read = try await h.text("settings", ["model": "voice"])
        #expect(read.contains("  dtype: fp16 (fp16|int8)\n  keep_hot: false") && read.contains("  speed: 1 (0.5–2, applies at request) — Speed"), "\(read)")
        let changed = try await h.text("settings", ["model": "voice", "set": ["speed": 1.5, "dtype": "int8", "keep_hot": true, "show_in_menu": false]])
        #expect(changed == "Changed show_in_menu, speed, dtype, keep_hot for Voice.", "\(changed)")
        let saved = try #require(await h.service.rows().first)
        let speed = try await h.service.settings(id: "voice").first?.value
        #expect(saved.selectedPrecision == "int8" && saved.keepHot && !saved.toolsShown && speed == 1.5)
        let refused = try await h.call("settings", ["model": "voice", "set": ["speed": 1.0, "dtype": "q3"]])
        #expect(refused.isError == true && mcpResultText(refused) == "[invalid_input] Not changed: dtype must be one of fp16, int8.")
        #expect(try await h.service.settings(id: "voice").first?.value == 1.5, "a refused set applies nothing")
        #expect(try await h.text("settings", ["model": "voice", "set": ["speed": 99]]).hasPrefix("[invalid_input] Not changed:"))
        #expect(try await h.text("settings", ["model": "voice", "set": ["fast": true]]).hasPrefix("[invalid_input] Not changed: fast is chosen at load"))
        let app = try await h.text("settings")
        #expect(app.hasPrefix("App settings:\n  idle_unload_minutes: 15") && app.contains("Launch at Login: set in the Arsenal menu"))
        #expect(try await h.text("settings", ["set": ["idle_unload_minutes": 30, "when_full": "queue"]]) == "Changed idle_unload_minutes, when_full.")
        let residency = await h.service.residencySettings(), memory = await h.service.memorySettings()
        #expect(residency.onDemandIdleMinutes == 30 && memory.whenFull == .queue)
        #expect(try await h.text("settings", ["set": ["launch_at_login": true]]).hasPrefix("[invalid_input] Not changed: launch_at_login is not an app setting over MCP"))
    }

    @Test func approvingLocalCodeNeedsTheCurrentDigest() async throws {
        let h = try await MCPHarness(rows: [StubModelsService.localFixture])
        defer { Task { await h.close() } }
        let model = try await h.text("model", ["model": "stub-local"])
        #expect(model.contains("Runs local code nobody has approved (digest aaaaaaaaaaaa…): approve only with the user's yes"))
        let run = try await h.call("run", ["model": "stub-local", "input": ["text": "Hi"]])
        #expect(
            mcpResultText(run)
                == "[approval] Not run: Local runner fixture runs local code nobody has approved yet. Only with the user's yes: \(MCPNames.prefix)settings(model:\"stub-local\", action:\"approve_code\")."
        )
        #expect(try await h.text("settings", ["model": "stub-local", "action": "approve_code", "content_digest": "stale"]).hasPrefix("[approval] Not approved"))
        #expect(try await h.text("settings", ["model": "stub-local", "action": "approve_code"]).hasPrefix("Approved Local runner fixture's local code (digest aaaaaaaaaaaa…)"))
        // Not downloaded, so off the table: the catalog shows who approved its code.
        #expect(await h.service.catalog(query: .init()).first { $0.id == "stub-local" }?.row.approval?.by == .mcp)
    }

    @MainActor @Test func playbackControlsTheSharedQueue() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-mcp-playback-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let queue = PlaybackQueue(preference: root.appendingPathComponent("muted.json"), outputEnabled: false)
        let router = APIRouter(service: StubModelsService(), support: root, playbackQueue: queue)
        func say(_ action: String) async throws -> String {
            mcpResultText(try await router.callMCPTool("\(MCPNames.prefix)playback", arguments: ["action": .string(action)], requester: "Test", notify: nil))
        }
        #expect(try await say("status") == "Playback: idle, 0 queued, not muted.")
        #expect(try await say("mute") == "Muted. Speech is still saved as files; nothing plays until unmute." && queue.muted)
        #expect(try await say("unmute") == "Unmuted. Speech plays as it is made." && !queue.muted)
        #expect(try await say("stop") == "Nothing was playing; the queue is empty.")
    }

    @Test func statusSaysWhatIsOnTheMac() async throws {
        let h = try await MCPHarness(rows: [decisionRow("other", bytes: 1, state: .loaded)])
        defer { Task { await h.close() } }
        let status = try await h.text("status")
        #expect(
            status == """
                Mac memory: 40 GB of the 100 GB limit in use (128 GB RAM).
                Loaded: other fp16, idle.
                Running: nothing.
                Playback: idle, 0 queued, muted.
                """, "\(status)")
        let json = try await h.call("status", ["format": "json"])
        #expect(try WireJSON.decode(JSONEncoder().encode(try #require(json.structuredContent)))["loaded"]?.array?.first?["model"] == "other")
    }

    @Test func httpHintsNameTheNewToolsAndKeepNotInstalled() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-mcp-http-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let service = StubModelsService(rows: [decisionRow()], supportDirectory: root)
        let router = APIRouter(service: service, support: root)
        #expect(await router.resolveFailureForTest("laya") == "404 Model laya is not installed; \(MCPNames.prefix)download(model: laya) downloads 2.5 GB.")
        let manifest = serviceHTTPFailure(ManifestError.invalid("weights changed"), modelID: "laya")
        #expect(manifest.message.hasSuffix("; call \(MCPNames.prefix)download(model: laya) to repair."))
        let admission = try await service.runRefusal(id: "laya", op: "decide")
        #expect(
            admission?.message.contains("not installed") == true && admission?.message.contains("\(MCPNames.prefix)download(model: laya)") == true,
            "\(String(describing: admission))")
        // Model folders too: agents read READMEs through the model tool, and runner messages reach them as errors.
        for source in ["Sources/ArsenalAPI", "Sources/ArsenalCore", "skill", "docs/MCP.md", "models"] {
            let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent(source)
            let files = FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil)?.compactMap { $0 as? URL } ?? [url]
            for file in files where ["swift", "md"].contains(file.pathExtension) {
                let text = try String(contentsOf: file, encoding: .utf8)
                for old in [
                    "arsenal_model_download", "arsenal_model_get", "arsenal_catalog_browse", "arsenal_job(", "arsenal_job_cancel", "arsenal_settings_set", "arsenal_operations",
                    "arsenal_decide(", "arsenal_video_session", "arsenal_segment_session"
                ] {
                    #expect(!text.contains(old), "\(file.path) still names \(old)")
                }
            }
        }
    }
}

extension APIRouter {
    func resolveFailureForTest(_ id: String) async -> String? {
        do { _ = try await resolve(id, kind: nil); return nil } catch let failure as HTTPFailure { return "\(failure.status) \(failure.message)" } catch { return "\(error)" }
    }
}
