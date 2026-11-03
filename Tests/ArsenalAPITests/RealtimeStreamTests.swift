// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore
@testable import ArsenalAPI

/// A model declaring one native session operation. `transcribe_session` answers each push with a stable delta, as
/// Confucius4-R2T2's adapter does; `diarize_session` answers with its whole growing snapshot, as Nemotron's does.
private actor SessionService: ModelsService {
    let id: String
    let operation: String
    let failPush: Int?
    var calls: [String] = []
    var pushedRates: [Int] = []
    var pushedSamples: [Int] = []
    private var pushed = 0
    private var succeeded = 0
    private var sessions = 0
    init(id: String, operation: String, failPush: Int? = nil) { self.id = id; self.operation = operation; self.failPush = failPush }
    private func op(_ name: String, _ properties: JSONObject) -> ModelOperation {
        ModelOperation(
            name: name, title: name, inputSchema: ["type": "object", "properties": .object(properties), "additionalProperties": false], output: "json", stream: false)
    }
    func rows() async -> [ModelRow] { [ModelRow(id: id, name: id, kind: "audio", states: ["fp16": .init(state: .loaded)])] }
    func kinds() async -> [KindSpec] { KindSpec.defaults }
    func operations(id: String) async -> [ModelOperation] {
        let audio: JSONValue = ["type": "object", "properties": ["path": ["type": "string"]], "required": ["path"], "additionalProperties": false]
        if operation == "diarize_session" {
            return [op("diarize", ["audio": audio]), op("diarize_session", ["action": ["type": "string"], "session_id": ["type": "string"], "audio": audio])]
        }
        return [
            op("transcribe", ["audio": audio, "language": ["type": "string"]]),
            op("transcribe_session", ["action": ["type": "string"], "session_id": ["type": "string"], "audio": audio, "language": ["type": "string"]])
        ]
    }
    func events() async -> AsyncStream<ServiceEvent> { AsyncStream { $0.finish() } }
    func get(id: String, precision: String) async throws {}
    func cancelGet(id: String) async {}
    func load(id: String, precision: String?, fast: Bool?) async throws {}
    func unload(id: String) async throws {}
    func setKeepHot(id: String, on: Bool) async throws {}
    func delete(id: String) async throws {}
    func menuActions() async -> [LoadedAction] { [] }
    func memory() async -> MemorySnapshot { .init() }
    func catalogStatus() async -> CatalogStatus { .init() }
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject {
        let action = params["action"]?.string ?? op
        calls.append(action)
        guard op == operation else { throw WorkerFailure(code: "unsupported_op", message: "Batch operations must not run for a streaming model.") }
        let session = JSONValue.string("s\(sessions)")
        switch action {
        case "create":
            sessions += 1
            if op == "diarize_session" { return ["session_id": .string("s\(sessions)"), "done": false, "segments": [], "speakers": []] }
            return ["session_id": .string("s\(sessions)"), "text": "", "partial_text": "", "delta": "", "duration": 0, "done": false]
        case "push":
            let wav = try Data(contentsOf: URL(fileURLWithPath: params["audio"]?["path"]?.string ?? ""))
            pushedRates.append(Int(wav[24]) | Int(wav[25]) << 8 | Int(wav[26]) << 16 | Int(wav[27]) << 24)
            pushedSamples.append((wav.count - 44) / 2)
            pushed += 1
            if pushed == failPush { throw WorkerFailure(code: "invalid_request", message: "Push failed.") }
            succeeded += 1
            if op == "diarize_session" {
                // Snapshots grow and change: the first speaker's segment extends and a second speaker appears.
                var segments: [JSONValue] = [["start": 0, "end": 0.2, "speaker": "speaker-0"]]
                if succeeded > 1 { segments = [["start": 0, "end": 0.3, "speaker": "speaker-0"], ["start": 0.25, "end": 0.4, "speaker": "speaker-1"]] }
                return ["session_id": session, "done": false, "segments": .array(segments), "speakers": ["speaker-0", "speaker-1"]]
            }
            let delta = succeeded == 1 ? "hello" : succeeded == 2 ? " there" : ""
            return ["session_id": session, "text": "", "partial_text": "", "delta": .string(delta), "duration": 0.2, "done": false]
        case "finish":
            if op == "diarize_session" {
                let segments: [JSONValue] = [["start": 0, "end": 0.35, "speaker": "speaker-0"], ["start": 0.3, "end": 0.4, "speaker": "speaker-1"]]
                return ["session_id": session, "done": true, "segments": .array(segments), "speakers": ["speaker-0", "speaker-1"]]
            }
            return ["session_id": session, "text": failPush == nil ? "hello there." : "there.", "partial_text": "", "delta": ".", "duration": 0.4, "done": true]
        case "close":
            if op == "diarize_session" { return ["session_id": session, "done": true, "segments": [], "speakers": []] }
            return ["session_id": session, "text": "", "partial_text": "", "delta": "", "duration": 0, "done": true]
        default: throw WorkerFailure(code: "invalid_request", message: "Unknown action.")
        }
    }
}

private struct ReceiveTimeout: Error {}

/// One Realtime socket against an in-process server. Every receive has a 10-second deadline, so a missing event fails
/// the test instead of hanging the suite.
private final class RealtimeClient: @unchecked Sendable {
    let server: ArsenalAPIServer
    let session = URLSession(configuration: .ephemeral)
    let task: URLSessionWebSocketTask
    let root = URL(fileURLWithPath: "/tmp/arsenal-realtime-stream/\(UUID().uuidString)")
    init(_ service: some ModelsService, model: String) async throws {
        server = ArsenalAPIServer(service: service, supportDirectory: root, port: 0)
        let port = try await server.start()
        task = session.webSocketTask(with: URL(string: "ws://127.0.0.1:\(port)/v1/realtime?intent=transcription&model=\(model)")!)
        task.resume()
    }
    func close() {
        task.cancel(with: .normalClosure, reason: nil); session.invalidateAndCancel(); server.stop()
        try? FileManager.default.removeItem(at: root)
    }
    func send(_ object: [String: Any]) async throws {
        try await task.send(.string(String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)))
    }
    func next() async throws -> [String: Any] {
        let task = task
        let text = try await withThrowingTaskGroup(of: String?.self) { group in
            group.addTask {
                guard case .string(let text) = try await task.receive() else { return "" }
                return text
            }
            group.addTask {
                try await Task.sleep(for: .seconds(10))
                task.cancel(with: .goingAway, reason: nil)
                return nil
            }
            defer { group.cancelAll() }
            return try await group.next() ?? nil
        }
        guard let text else { throw ReceiveTimeout() }
        return (try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]) ?? [:]
    }
    /// Events up to and including the first of `type`.
    func events(until type: String) async throws -> [[String: Any]] {
        var events: [[String: Any]] = []
        while events.last?["type"] as? String != type { events.append(try await next()) }
        return events
    }
}

private func deltas(_ events: [[String: Any]]) -> [String] {
    events.filter { $0["type"] as? String == "conversation.item.input_audio_transcription.delta" }.map { $0["delta"] as? String ?? "" }
}

struct RealtimeStreamTests {
    @Test(.timeLimit(.minutes(1))) func appendedAudioStreamsNativeSessionDeltasAndCommitFinishes() async throws {
        let service = SessionService(id: "streamer", operation: "transcribe_session")
        let client = try await RealtimeClient(service, model: "streamer"); defer { client.close() }
        #expect(try await client.next()["type"] as? String == "session.created")
        // Two appends of 200 ms each: each reaches the native session as a push and streams its delta at once.
        let chunk = Data(count: RealtimeStream.pushBytes).base64EncodedString()
        try await client.send(["type": "input_audio_buffer.append", "audio": chunk])
        try await client.send(["type": "input_audio_buffer.append", "audio": chunk])
        var events: [[String: Any]] = []
        while deltas(events).count < 2 { events.append(try await client.next()) }
        #expect(!events.contains { $0["type"] as? String == "error" }, "\(events)")
        #expect(deltas(events) == ["hello", " there"])
        try await client.send(["type": "input_audio_buffer.commit"])
        events += try await client.events(until: "conversation.item.input_audio_transcription.completed")
        #expect(!events.contains { $0["type"] as? String == "error" }, "\(events)")
        #expect(deltas(events) == ["hello", " there", "."])
        #expect(events.last?["transcript"] as? String == "hello there.")
        // Commit flushes the resampler's last samples as a final push before finish.
        #expect(await service.calls == ["create", "push", "push", "push", "finish", "close"])
        // One resampler per session: 16 kHz pushes whose sizes add up to exactly two thirds of the 24 kHz input.
        #expect(await service.pushedRates == [16000, 16000, 16000])
        #expect(await service.pushedSamples.reduce(0, +) == 6400)
    }

    @Test(.timeLimit(.minutes(1))) func aFailedPushStartsAFreshSessionAndKeepsEarlierText() async throws {
        let service = SessionService(id: "streamer", operation: "transcribe_session", failPush: 2)
        let client = try await RealtimeClient(service, model: "streamer"); defer { client.close() }
        #expect(try await client.next()["type"] as? String == "session.created")
        let chunk = Data(count: RealtimeStream.pushBytes).base64EncodedString()
        try await client.send(["type": "input_audio_buffer.append", "audio": chunk])
        var events: [[String: Any]] = []
        while deltas(events).isEmpty { events.append(try await client.next()) }
        try await client.send(["type": "input_audio_buffer.append", "audio": chunk])
        events += try await client.events(until: "error")
        // The failed session is closed; the next append opens a new one instead of failing every later push.
        try await client.send(["type": "input_audio_buffer.append", "audio": chunk])
        while deltas(events).count < 2 { events.append(try await client.next()) }
        try await client.send(["type": "input_audio_buffer.commit"])
        events += try await client.events(until: "conversation.item.input_audio_transcription.completed")
        #expect(events.filter { $0["type"] as? String == "error" }.count == 1, "\(events)")
        #expect(deltas(events) == ["hello", " there", "."])
        #expect(events.last?["transcript"] as? String == "hello there.")
        #expect(await service.calls == ["create", "push", "push", "close", "create", "push", "push", "finish", "close"])
    }

    @Test(.timeLimit(.minutes(1))) func diarizerSnapshotsReplaceEachOtherAndFinishSendsFinalSegments() async throws {
        let service = SessionService(id: "diarizer", operation: "diarize_session")
        let client = try await RealtimeClient(service, model: "diarizer"); defer { client.close() }
        #expect(try await client.next()["type"] as? String == "session.created")
        let chunk = Data(count: RealtimeStream.pushBytes).base64EncodedString()
        try await client.send(["type": "input_audio_buffer.append", "audio": chunk])
        try await client.send(["type": "input_audio_buffer.append", "audio": chunk])
        var events: [[String: Any]] = []
        while events.filter({ $0["type"] as? String == "arsenal.diarization.snapshot" }).count < 2 { events.append(try await client.next()) }
        try await client.send(["type": "input_audio_buffer.commit"])
        events += try await client.events(until: "conversation.item.input_audio_transcription.completed")
        #expect(!events.contains { $0["type"] as? String == "error" }, "\(events)")
        let snapshots = events.filter { $0["type"] as? String == "arsenal.diarization.snapshot" }
        // Each changed snapshot is sent whole; the flush push repeats the last one, which is not resent.
        #expect(snapshots.map { ($0["segments"] as? [Any])?.count ?? -1 } == [1, 2, 2])
        #expect(snapshots.map { $0["final"] as? Bool } == [false, false, true])
        let first = (snapshots[1]["segments"] as? [[String: Any]])?.first
        #expect(first?["end"] as? Double == 0.3)
        // Only finish's corrected segments become standard segment events, so none is ever stale.
        let segments = events.filter { $0["type"] as? String == "conversation.item.input_audio_transcription.segment" }
        #expect(segments.map { $0["id"] as? String } == ["seg_0", "seg_1"])
        #expect(segments.map { $0["end"] as? Double } == [0.35, 0.4])
        #expect(segments.map { $0["speaker"] as? String } == ["speaker-0", "speaker-1"])
        #expect(await service.calls == ["create", "push", "push", "push", "finish", "close"])
    }

    @Test func resamplingIsContinuousAcrossPushBoundaries() {
        let tone = (0..<12000).map { Int16((10000 * sin(2 * Double.pi * 1000 * Double($0) / 24000)).rounded()) }
        func pcm(_ samples: ArraySlice<Int16>) -> Data { samples.reduce(into: Data()) { data, sample in withUnsafeBytes(of: sample.littleEndian) { data.append(contentsOf: $0) } } }
        func values(_ data: Data) -> [Int16] { stride(from: 0, to: data.count, by: 2).map { Int16(bitPattern: UInt16(data[$0]) | UInt16(data[$0 + 1]) << 8) } }
        var whole = StreamingResampler()
        let once = values(whole.process(pcm(tone[...]), flush: true))
        var pieces = StreamingResampler(), joined = Data(), offset = 0
        for size in [4800, 1, 2399, 4800] {
            joined += pieces.process(pcm(tone[offset..<offset + size])); offset += size
        }
        joined += pieces.process(Data(), flush: true)
        #expect(once.count == 8000)
        #expect(values(joined) == once)
        // A 1 kHz tone keeps its level and phase: output n is input time n / 16000 s.
        func expected(_ n: Int) -> Double { 10000 * sin(2 * Double.pi * 1000 * Double(n) / 16000) }
        let error = (100..<7900).map { (n: Int) -> Double in abs(Double(once[n]) - expected(n)) }.max() ?? .infinity
        #expect(error < 5)
    }
}
