// SPDX-License-Identifier: MIT
import Foundation
import CryptoKit
import ArsenalCore

extension HTTPConnection {
    func upgradeWebSocket(_ request: HTTPRequest) async throws {
        guard request.method == "GET", request.headers["upgrade"]?.lowercased() == "websocket",
            request.headers["connection"]?.lowercased().split(separator: ",").contains(where: { $0.trimmingCharacters(in: .whitespaces) == "upgrade" }) == true,
            request.headers["sec-websocket-version"] == "13", let key = request.headers["sec-websocket-key"], Data(base64Encoded: key)?.count == 16
        else {
            throw HTTPFailure(400, "Use a version 13 WebSocket upgrade.")
        }
        let accept = Data(Insecure.SHA1.hash(data: Data((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").utf8))).base64EncodedString()
        guard markOpened() else { throw HTTPFailure(500, "Response already started.") }
        try await send(Data("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: \(accept)\r\n\r\n".utf8))
    }
    func webSocketSend(_ payload: Data, opcode: UInt8 = 1) async throws {
        var frame = Data([0x80 | opcode])
        if payload.count < 126 {
            frame.append(UInt8(payload.count))
        } else if payload.count <= 65535 {
            frame.append(126); var count = UInt16(payload.count).bigEndian; withUnsafeBytes(of: &count) { frame.append(contentsOf: $0) }
        } else {
            frame.append(127); var count = UInt64(payload.count).bigEndian; withUnsafeBytes(of: &count) { frame.append(contentsOf: $0) }
        }
        frame += payload; try await send(frame)
    }
    func webSocketEvent(_ object: JSONObject) async throws {
        var event = object; event["event_id"] = .string("event_" + UUID().uuidString.lowercased())
        try await webSocketSend(WireJSON.encode(event))
    }
}

struct WebSocketReader {
    var buffer = Data()
    var fragmented = Data()
    var assembling = false
    mutating func take(_ count: Int, from connection: HTTPConnection) async throws -> Data {
        while buffer.count < count { buffer += try await connection.receive() }
        let output = Data(buffer.prefix(count)); buffer = Data(buffer.dropFirst(count)); return output
    }
    mutating func message(from connection: HTTPConnection) async throws -> Data? {
        while true {
            let head = [UInt8](try await take(2, from: connection))
            let fin = head[0] & 0x80 != 0, opcode = head[0] & 0x0f
            guard head[0] & 0x70 == 0, head[1] & 0x80 != 0 else { throw HTTPFailure(400, "WebSocket frames must be masked without extensions.") }
            var count = UInt64(head[1] & 0x7f)
            if count == 126 {
                count = try await take(2, from: connection).reduce(0) { ($0 << 8) | UInt64($1) }
            } else if count == 127 {
                count = try await take(8, from: connection).reduce(0) { ($0 << 8) | UInt64($1) }
            }
            guard count <= 1024 * 1024, fragmented.count + Int(count) <= 1024 * 1024 else { throw HTTPFailure(413, "WebSocket message exceeds 1 MiB.") }
            if opcode >= 8, !fin || count > 125 { throw HTTPFailure(400, "Malformed WebSocket control frame.") }
            let mask = [UInt8](try await take(4, from: connection))
            let bytes = [UInt8](try await take(Int(count), from: connection))
            let payload = Data(bytes.enumerated().map { $0.element ^ mask[$0.offset % 4] })
            switch opcode {
            case 8: try await connection.webSocketSend(payload, opcode: 8); return nil
            case 9: try await connection.webSocketSend(payload, opcode: 10); continue
            case 10: continue
            case 1:
                guard !assembling else { throw HTTPFailure(400, "Interleaved WebSocket messages.") }
                if fin { return payload }; fragmented = payload; assembling = true
            case 0:
                guard assembling else { throw HTTPFailure(400, "Unexpected continuation.") }
                fragmented += payload
                if fin { let result = fragmented; fragmented = Data(); assembling = false; return result }
            default: throw HTTPFailure(400, "Only JSON text WebSocket messages are supported.")
            }
        }
    }
}

extension APIRouter {
    func realtime(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        guard request.query["intent"] == nil || request.query["intent"] == "transcription" else { throw HTTPFailure(400, "Only transcription Realtime sessions are supported.") }
        try await connection.upgradeWebSocket(request)
        let sessionID = "sess_" + UUID().uuidString.lowercased()
        try await connection.webSocketEvent(["type": "session.created", "session": ["id": .string(sessionID), "type": "transcription"]])
        // pcm is the utterance for batch transcription and turn detection; pcmBytes counts all of it for durations.
        var reader = WebSocketReader(), pcm = Data(), pcmBytes = 0
        var item = "item_" + UUID().uuidString.lowercased()
        var model = request.query["model"], turnModel: String?, language: String?, speaking = false
        // A model declaring a native session operation streams each appended chunk; others transcribe at commit.
        var stream: RealtimeStream?, streamChecked = false
        let directory = try newJob(); defer { removeTransientJob(directory) }
        defer {
            // Disconnect releases the worker's session state; nothing is left to report to a closed socket.
            if let open = stream, let session = open.session {
                Task { _ = try? await self.capabilityRun(["model": .string(open.model)], operation: open.operation, params: ["action": "close", "session_id": .string(session)]) }
            }
        }
        while let bytes = try await reader.message(from: connection) {
            do {
                let event = try WireJSON.decode(bytes)
                switch event["type"]?.string {
                case "session.update", "transcription_session.update":
                    guard let session = event["session"]?.object, session["type"] == nil || session["type"] == "transcription" else {
                        throw HTTPFailure(422, "session is required.")
                    }
                    let input = session["audio"]?["input"]
                    if let format = input?["format"], format["type"]?.string != "audio/pcm" || format["rate"]?.number != 24000 {
                        throw HTTPFailure(422, "Realtime accepts mono PCM16 at 24000 Hz.")
                    }
                    if let oldFormat = session["input_audio_format"], oldFormat != "pcm16" { throw HTTPFailure(422, "Realtime accepts pcm16 only.") }
                    let transcription = input?["transcription"] ?? session["input_audio_transcription"]
                    model = transcription?["model"]?.string ?? model
                    language = transcription?["language"]?.string
                    if let model { _ = try await realtimeModel(model) }
                    // A changed selection or language starts a fresh native session at the next append.
                    try await realtimeClose(&stream); streamChecked = false
                    let detection = input?["turn_detection"] ?? session["turn_detection"]
                    if let detection, detection != .null {
                        guard let options = detection.object, detection["type"]?.string == "semantic_vad",
                            Set(options.keys).isSubset(of: ["type", "model", "eagerness", "create_response", "interrupt_response"])
                        else { throw HTTPFailure(422, "Only semantic_vad turn detection is supported.") }
                        if let eagerness = detection["eagerness"], ![JSONValue.string("auto"), .string("medium")].contains(eagerness) {
                            throw HTTPFailure(422, "This local semantic_vad profile supports auto/medium eagerness only.")
                        }
                        for key in ["create_response", "interrupt_response"] {
                            if let value = detection[key], value != .bool(false) {
                                throw HTTPFailure(422, "Transcription-only sessions support \(key):false only.")
                            }
                        }
                        let selector = detection["model"].map { ["model": $0] } ?? [:]
                        turnModel = try await capabilityModel(selector, operation: "end_of_turn")
                    } else {
                        turnModel = nil
                    }
                    var reported = session
                    reported["id"] = .string(sessionID); reported["type"] = "transcription"
                    reported["arsenal_model"] = model.map(JSONValue.string)
                    // The pinned SDK restricts transcription.model to cloud model literals. Preserve the local
                    // selector in arsenal_model instead of returning an invalid literal or impersonating a cloud model.
                    if var audio = reported["audio"]?.object, var input = audio["input"]?.object, var transcription = input["transcription"]?.object {
                        transcription["model"] = nil; input["transcription"] = .object(transcription)
                        audio["input"] = .object(input); reported["audio"] = .object(audio)
                    }
                    try await connection.webSocketEvent(["type": "session.updated", "session": .object(reported)])
                case "input_audio_buffer.append":
                    guard let encoded = event["audio"]?.string, let chunk = Data(base64Encoded: encoded), chunk.count % 2 == 0, !chunk.isEmpty else {
                        throw HTTPFailure(422, "Audio must be bounded base64 PCM16.")
                    }
                    if stream == nil, !streamChecked {
                        let selected = try await realtimeModel(model)
                        if let operation = selected.stream { stream = RealtimeStream(model: selected.id, operation: operation) } else { streamChecked = true }
                    }
                    // Batch transcription holds the whole utterance (16 MiB). A native session holds its own audio, bounded by
                    // the worker, so the host keeps only the tail turn detection reads.
                    guard stream != nil || pcm.count + chunk.count <= 16 * 1024 * 1024 else { throw HTTPFailure(422, "Audio must be bounded base64 PCM16.") }
                    pcm += chunk; pcmBytes += chunk.count
                    if stream != nil, pcm.count > RealtimeStream.tailBytes { pcm = Data(pcm.suffix(RealtimeStream.tailBytes)) }
                    if !speaking {
                        speaking = true; try await connection.webSocketEvent(["type": "input_audio_buffer.speech_started", "audio_start_ms": 0, "item_id": .string(item)])
                    }
                    if stream != nil {
                        stream!.unsent += chunk
                        if stream!.unsent.count >= RealtimeStream.pushBytes {
                            try await realtimePush(&stream!, connection, language: language, directory: directory, item: item)
                        }
                    }
                    if let turnModel, pcm.count >= 4800 {
                        let path = directory.appendingPathComponent("turn.wav")
                        try PCM.wav(Data(pcm.suffix(RealtimeStream.tailBytes)), rate: 24000).write(to: path)
                        let result = try await capabilityRun(["model": .string(turnModel)], operation: "end_of_turn", params: ["audio": ["path": .string(path.path)]])
                        if result["prediction"]?.bool == true {
                            try await realtimeCommit(
                                connection, model: model, language: language, pcm: pcm, bytes: pcmBytes, directory: directory, item: item, stream: &stream)
                            pcm = Data(); pcmBytes = 0; speaking = false
                            item = "item_" + UUID().uuidString.lowercased()
                        }
                    }
                case "input_audio_buffer.commit":
                    try await realtimeCommit(connection, model: model, language: language, pcm: pcm, bytes: pcmBytes, directory: directory, item: item, stream: &stream)
                    pcm = Data(); pcmBytes = 0; speaking = false
                    item = "item_" + UUID().uuidString.lowercased()
                case "input_audio_buffer.clear":
                    try await realtimeClose(&stream)
                    pcm = Data(); pcmBytes = 0; speaking = false; item = "item_" + UUID().uuidString.lowercased()
                    try await connection.webSocketEvent(["type": "input_audio_buffer.cleared"])
                default: throw HTTPFailure(422, "Unsupported Realtime transcription event.")
                }
            } catch {
                let failure = serviceHTTPFailure(error)
                try await connection.webSocketEvent([
                    "type": "error", "error": ["type": "invalid_request_error", "code": .string(failure.code ?? "invalid_request"), "message": .string(failure.message)]
                ])
            }
        }
    }
    /// The model a Realtime session transcribes with, and its native session operation when it declares one
    /// (`transcribe_session` first; a diarizer's `diarize_session` when the model is chosen explicitly).
    func realtimeModel(_ model: String?) async throws -> (id: String, stream: String?) {
        if let model {
            let declared = Set(await service.operations(id: model).map(\.name))
            for operation in ["transcribe_session", "diarize_session"] where declared.contains(operation) {
                return (try await capabilityModel(["model": .string(model)], operation: operation), operation)
            }
        }
        let id = try await capabilityModel(model.map { ["model": .string($0)] } ?? [:], operation: "transcribe")
        return (id, await service.operations(id: id).contains { $0.name == "transcribe_session" } ? "transcribe_session" : nil)
    }
    /// Sends the unsent audio to the native session (creating it first) and forwards its stable deltas.
    /// The session's one resampler turns the 24 kHz stream into 16 kHz pushes continuously across push boundaries.
    func realtimePush(
        _ stream: inout RealtimeStream, _ connection: HTTPConnection, language: String?, directory: URL, item: String, flush: Bool = false
    ) async throws {
        let selector: JSONObject = ["model": .string(stream.model)]
        if stream.session == nil {
            var create: JSONObject = ["action": "create"]
            if let language, stream.operation == "transcribe_session" { create["language"] = .string(language) }
            guard let session = try await capabilityRun(selector, operation: stream.operation, params: create)["session_id"]?.string else {
                throw HTTPFailure(502, "The model did not open a streaming session.")
            }
            stream.session = session; stream.resampler = StreamingResampler(); stream.snapshot = nil
        }
        guard let session = stream.session else { return }
        let samples = stream.resampler.process(stream.unsent, flush: flush)
        stream.unsent = Data()
        guard !samples.isEmpty else { return }
        let path = directory.appendingPathComponent("push-" + UUID().uuidString + ".wav")
        try PCM.wav(samples, rate: StreamingResampler.outputRate).write(to: path)
        defer { try? FileManager.default.removeItem(at: path) }
        let push: JSONObject = ["action": "push", "session_id": .string(session), "audio": ["path": .string(path.path)]]
        let result: JSONObject
        do {
            result = try await capabilityRun(selector, operation: stream.operation, params: push)
        } catch {
            // The worker drops a session whose push failed. Start a fresh one at the next append instead of failing every
            // later push; the text already forwarded for this item is kept ahead of the new session's text.
            stream.carried = stream.transcript; stream.session = nil; stream.snapshot = nil
            _ = try? await capabilityRun(selector, operation: stream.operation, params: ["action": "close", "session_id": .string(session)])
            throw error
        }
        try await realtimeForward(result, &stream, connection, item: item)
    }
    /// Stable text deltas as transcription deltas. A diarizer answers every push with its whole snapshot (segments grow,
    /// merge and are renumbered), so each changed snapshot is sent in full as `arsenal.diarization.snapshot`, replacing the
    /// previous one; at finish the corrected final segments also arrive as standard segment events.
    func realtimeForward(_ result: JSONObject, _ stream: inout RealtimeStream, _ connection: HTTPConnection, item: String, final: Bool = false) async throws {
        if let delta = result["delta"]?.string, !delta.isEmpty {
            stream.transcript += delta
            try await connection.webSocketEvent([
                "type": "conversation.item.input_audio_transcription.delta", "item_id": .string(item), "content_index": 0, "delta": .string(delta)
            ])
        }
        guard let segments = result["segments"]?.array else { return }
        if final || segments != stream.snapshot {
            stream.snapshot = segments
            try await connection.webSocketEvent([
                "type": "arsenal.diarization.snapshot", "item_id": .string(item), "final": .bool(final), "segments": .array(segments),
                "speakers": result["speakers"] ?? .array([])
            ])
        }
        guard final else { return }
        for (index, segment) in segments.enumerated() {
            guard let value = segment.object else { continue }
            var event: JSONObject = ["type": "conversation.item.input_audio_transcription.segment", "item_id": .string(item), "content_index": 0]
            event["id"] = .string("seg_\(index)")
            for key in ["start", "end", "speaker", "text"] { if let field = value[key] { event[key] = field } }
            try await connection.webSocketEvent(event)
        }
    }
    /// Releases an open native session without a result (clear, re-selection, disconnect).
    func realtimeClose(_ stream: inout RealtimeStream?) async throws {
        defer { stream = nil }
        guard let open = stream, let session = open.session else { return }
        _ = try await capabilityRun(["model": .string(open.model)], operation: open.operation, params: ["action": "close", "session_id": .string(session)])
    }
    func realtimeCommit(
        _ connection: HTTPConnection, model: String?, language: String?, pcm: Data, bytes: Int, directory: URL, item: String,
        stream: inout RealtimeStream?
    ) async throws {
        guard bytes > 0 else { throw HTTPFailure(422, "Audio buffer is empty.") }
        var batchModel: String?
        if stream == nil { batchModel = try await capabilityModel(model.map { ["model": .string($0)] } ?? [:], operation: "transcribe") }
        try await connection.webSocketEvent(["type": "input_audio_buffer.speech_stopped", "audio_end_ms": .number(Double(bytes / 48)), "item_id": .string(item)])
        try await connection.webSocketEvent(["type": "input_audio_buffer.committed", "item_id": .string(item), "previous_item_id": .null])
        if var open = stream {
            // Streaming: the remaining audio, then finish (its final text wins), then close; the next item opens anew.
            stream = nil
            try await realtimePush(&open, connection, language: language, directory: directory, item: item, flush: true)
            let selector: JSONObject = ["model": .string(open.model)]
            if let session = open.session {
                let finished = try await capabilityRun(selector, operation: open.operation, params: ["action": "finish", "session_id": .string(session)])
                try await realtimeForward(finished, &open, connection, item: item, final: true)
                if let text = finished["text"]?.string { open.transcript = RealtimeStream.joined(open.carried, text) }
                _ = try? await capabilityRun(selector, operation: open.operation, params: ["action": "close", "session_id": .string(session)])
            }
            try await connection.webSocketEvent([
                "type": "conversation.item.input_audio_transcription.completed", "item_id": .string(item), "content_index": 0, "transcript": .string(open.transcript),
                "usage": ["type": "duration", "seconds": .number(Double(bytes) / 48000)]
            ])
            return
        }
        guard let id = batchModel else { return }
        let path = directory.appendingPathComponent("input.wav")
        try PCM.wav(pcm, rate: 24000).write(to: path)
        var params: JSONObject = ["audio": ["path": .string(path.path)]]
        if let language { params["language"] = .string(language) }
        let result = try await capabilityRun(["model": .string(id)], operation: "transcribe", params: params)
        guard let text = result["text"]?.string else { throw HTTPFailure(502, "No transcription text.") }
        try await connection.webSocketEvent(["type": "conversation.item.input_audio_transcription.delta", "item_id": .string(item), "content_index": 0, "delta": .string(text)])
        try await connection.webSocketEvent([
            "type": "conversation.item.input_audio_transcription.completed", "item_id": .string(item), "content_index": 0, "transcript": .string(text),
            "usage": ["type": "duration", "seconds": .number(Double(bytes) / 48000)]
        ])
    }
}

/// One native streaming session behind a Realtime socket (transcribe_session / diarize_session: create, push, finish, close).
struct RealtimeStream {
    /// About 200 ms of 24 kHz PCM16 per push: deltas arrive while the speaker talks without a request per tiny append.
    static let pushBytes = 9600
    /// The last eight seconds of 24 kHz PCM16, what turn detection reads; a streaming utterance keeps no more.
    static let tailBytes = 8 * 24000 * 2
    let model: String
    let operation: String
    var session: String?
    var unsent = Data()
    var resampler = StreamingResampler()
    /// Text forwarded for this item; `carried` is the part sent before a failed push restarted the session.
    var transcript = ""
    var carried = ""
    /// The diarizer's last forwarded snapshot.
    var snapshot: [JSONValue]?
    static func joined(_ carried: String, _ text: String) -> String {
        guard !carried.isEmpty, !text.isEmpty else { return carried + text }
        return carried.last!.isWhitespace || text.first!.isWhitespace ? carried + text : carried + " " + text
    }
}

/// Continuous 24 kHz to 16 kHz PCM16 resampling for one native session: a Blackman-windowed sinc low-pass applied
/// polyphase (up 2, down 3), whose input history spans pushes, so 200 ms pieces join without the edge transients of
/// converting each piece on its own. Output n is aligned with input time n * 3 / 2; `flush` ends the stream with zeros.
struct StreamingResampler {
    static let inputRate = 24000
    static let outputRate = 16000
    private let up: Int
    private let down: Int
    private let half: Int
    private let taps: [Double]
    private var input: [Double] = []
    private var base = 0
    private var total = 0
    private var produced = 0
    init(half: Int = 16) {
        var a = Self.inputRate, b = Self.outputRate
        while b != 0 { (a, b) = (b, a % b) }
        let upFactor = Self.outputRate / a
        up = upFactor; down = Self.inputRate / a; self.half = half
        let count = 2 * half * upFactor + 1, centre = Double(half * upFactor)
        // Pass band to 92% of the output Nyquist, in cycles per upsampled sample; the up gain restores zero-stuffed level.
        let cutoff = 0.92 * Double(min(Self.inputRate, Self.outputRate)) / 2 / Double(Self.inputRate * upFactor)
        taps = (0..<count).map { k in
            let x = 2 * cutoff * (Double(k) - centre)
            let sinc = x == 0 ? 1 : sin(Double.pi * x) / (Double.pi * x)
            let phase = 2 * Double.pi * Double(k) / Double(count - 1)
            let window = 0.42 - 0.5 * cos(phase) + 0.08 * cos(2 * phase)
            return Double(upFactor) * 2 * cutoff * sinc * window
        }
    }
    /// Little-endian mono PCM16 in, little-endian mono PCM16 out.
    mutating func process(_ data: Data, flush: Bool = false) -> Data {
        let count = data.count / 2
        data.withUnsafeBytes { raw in
            for index in 0..<count { input.append(Double(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: 2 * index, as: Int16.self)))) }
        }
        total += count
        var output = Data()
        while produced * down < total * up {
            let position = produced * down + half * up
            let newest = position / up
            if newest >= total, !flush { break }
            let oldest = max(0, position - taps.count + 1 + up - 1) / up
            var sum = 0.0
            if oldest <= min(newest, total - 1) {
                for index in oldest...min(newest, total - 1) { sum += input[index - base] * taps[position - index * up] }
            }
            var sample = Int16(max(-32768, min(32767, sum.rounded()))).littleEndian
            withUnsafeBytes(of: &sample) { output.append(contentsOf: $0) }
            produced += 1
        }
        let keep = max(0, produced * down - half * up + up - 1) / up
        if keep > base {
            input.removeFirst(min(input.count, keep - base)); base = keep
        }
        return output
    }
}
