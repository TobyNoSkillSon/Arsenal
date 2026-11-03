// SPDX-License-Identifier: MIT
import AVFoundation
import Foundation
import ArsenalCore

// Live transcription: the microphone (or any source of 24 kHz mono 16-bit PCM) streamed to Arsenal's Realtime
// transcription socket (docs/API.md, Live transcription sessions). A turn ends where an installed end-of-turn model
// says it does (semantic turn detection); without one, a pause of 0.7 s after speech ends it. Each finished turn is a
// line; the turn in progress shows its partial text when the service sends one.

/// 24 kHz mono signed 16-bit little-endian chunks, as the Realtime socket takes them.
@MainActor protocol LiveAudioSource: AnyObject {
    func start(_ chunk: @escaping @Sendable (Data) -> Void) async throws
    func stop()
}

/// The Mac's default input, converted to 24 kHz mono.
@MainActor final class MicrophoneSource: LiveAudioSource {
    private let engine = AVAudioEngine()

    /// The app declares why it uses the microphone (Info.plist); without it macOS refuses the input, so live mode is not
    /// offered.
    nonisolated static var declared: Bool { Bundle.main.object(forInfoDictionaryKey: "NSMicrophoneUsageDescription") != nil }

    func start(_ chunk: @escaping @Sendable (Data) -> Void) async throws {
        guard await AVCaptureDevice.requestAccess(for: .audio) else {
            throw ToolError("Arsenal may not use the microphone. Allow it in System Settings \u{203a} Privacy & Security \u{203a} Microphone.")
        }
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, let target = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 24_000, channels: 1, interleaved: true),
            let converter = AVAudioConverter(from: format, to: target)
        else { throw ToolError("The microphone has no input Arsenal can read.") }
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, _ in
            let capacity = AVAudioFrameCount(Double(buffer.frameLength) * 24_000 / format.sampleRate) + 32
            guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return }
            nonisolated(unsafe) var fed = false
            var error: NSError?
            converter.convert(to: out, error: &error) { _, status in
                if fed { status.pointee = .noDataNow; return nil }
                fed = true; status.pointee = .haveData; return buffer
            }
            guard error == nil, out.frameLength > 0, let samples = out.int16ChannelData else { return }
            chunk(Data(bytes: samples[0], count: Int(out.frameLength) * 2))
        }
        engine.prepare()
        try engine.start()
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }
}

/// The Realtime socket as the window uses it (tests give their own).
protocol LiveSocket: AnyObject, Sendable {
    func send(_ object: [String: Any]) async throws
    func receive() async throws -> [String: Any]
    func close()
}

final class WebSocketLive: LiveSocket, @unchecked Sendable {
    private let task: URLSessionWebSocketTask
    init(port: Int, model: String) throws {
        guard
            let url = URL(string: "ws://127.0.0.1:\(port)/v1/realtime?intent=transcription&model=\(model.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? model)")
        else { throw ToolError("Bad live transcription address.") }
        var request = URLRequest(url: url)
        request.setValue(ArsenalClient.clientName, forHTTPHeaderField: "X-Arsenal-Client")
        task = URLSession.shared.webSocketTask(with: request)
        task.resume()
    }
    func send(_ object: [String: Any]) async throws {
        let data = try JSONSerialization.data(withJSONObject: object)
        try await task.send(.string(String(decoding: data, as: UTF8.self)))
    }
    func receive() async throws -> [String: Any] {
        switch try await task.receive() {
        case .string(let s): return (try? JSONSerialization.jsonObject(with: Data(s.utf8))) as? [String: Any] ?? [:]
        case .data(let d): return (try? JSONSerialization.jsonObject(with: d)) as? [String: Any] ?? [:]
        @unknown default: return [:]
        }
    }
    func close() { task.cancel(with: .normalClosure, reason: nil) }
}

@MainActor final class LiveTranscription: ObservableObject {
    /// Finished turns, in order.
    @Published private(set) var lines: [String] = []
    /// The turn in progress, as far as the service has sent it.
    @Published private(set) var partial = ""
    /// Who spoke when so far, from a diarizer's live session: its latest whole timeline (`segments`, `speakers`).
    @Published private(set) var timeline: JSONObject?
    @Published private(set) var listening = false
    /// The input's loudness now, 0…1 (drawn only while listening).
    @Published private(set) var level: Float = 0
    @Published private(set) var started: Date?
    @Published var problem: String?

    private var socket: LiveSocket?
    private var source: LiveAudioSource?
    private var receiving: Task<Void, Never>?
    /// The source's chunks in the order it delivered them, appended one at a time.
    private var audio: AsyncStream<Data>.Continuation?
    private var feeding: Task<Void, Never>?
    /// Messages for the socket, sent strictly in order by one sender; a failed send ends the session with its reason.
    private var outbox: [Data] = []
    private var sending: Task<Void, Never>?
    /// Silence-based turn ends when no end-of-turn model decides them (transcription only: a diarizer keeps one
    /// recording, so its speaker numbers hold for the whole session).
    private var detectsTurns = false
    private var diarizing = false
    private var heard = false
    private var quiet: Double = 0
    /// Audio sent since this window last committed a turn.
    private var appendedSinceCommit = false
    /// The service accepted the session settings; refusals before that end the session, most later ones do not.
    private var configured = false
    /// Stopping: the final turn is committed and its transcript awaited before the socket closes (a new listen waits).
    @Published private(set) var stopping = false
    private var awaitingCommit = false
    private var latestItem: String?
    private var pendingItems: Set<String> = []
    private var onFinished: ((LiveTranscription) -> Void)?
    private var deadline: Task<Void, Never>?

    /// Sources and sockets (tests give their own).
    var makeSource: () -> LiveAudioSource = { MicrophoneSource() }
    var makeSocket: (_ port: Int, _ model: String) throws -> LiveSocket = { try WebSocketLive(port: $0, model: $1) }

    var text: String { (lines + (partial.isEmpty ? [] : [partial])).joined(separator: "\n") }

    /// Errors that mean the model cannot serve this session; any other refusal after setup (one rejected chunk, an empty
    /// commit, a turn that failed to transcribe) shows its words while listening goes on.
    static let fatalCodes: Set<String> = [
        "insufficient_memory", "memory", "not_installed", "not_found", "code_consent", "integrity", "quarantined", "shutting_down", "updating", "catalog_key",
        "precision"
    ]

    /// Starts listening: `prepare` loads the models first (with the same memory warning and Load Anyway as every other
    /// load), then the session is configured (the model, its language, the turn model when there is one) and audio
    /// flows until `stop`. `diarize` streams a diarizer's speaker timeline instead of text.
    func start(
        port: Int, model: String, language: String?, turnModel: String?, diarize: Bool = false, prepare: @escaping () async throws -> Void = {}
    ) {
        guard !listening, !stopping else { return }
        lines = []; partial = ""; timeline = nil; problem = nil; heard = false; quiet = 0; level = 0
        appendedSinceCommit = false; configured = false; awaitingCommit = false; latestItem = nil; pendingItems = []; outbox = []
        diarizing = diarize
        detectsTurns = turnModel != nil && !diarize
        listening = true; started = Date()
        Task { await begin(port: port, model: model, language: language, turnModel: diarize ? nil : turnModel, prepare: prepare) }
    }

    private func begin(port: Int, model: String, language: String?, turnModel: String?, prepare: () async throws -> Void) async {
        do {
            try await prepare()
        } catch {
            // Cancel in the memory warning ends the attempt quietly.
            if error is CancellationError { listening = false; started = nil } else { fail(OperationProblem.wording(error, model: model)) }
            return
        }
        guard listening else { complete(); return }
        do {
            let socket = try makeSocket(port, model)
            self.socket = socket
            receiving = Task { [weak self] in await self?.receive(socket) }
            var transcription: [String: Any] = ["model": model]
            if let language, !language.isEmpty, !diarizing { transcription["language"] = language }
            var input: [String: Any] = ["format": ["type": "audio/pcm", "rate": 24_000], "transcription": transcription]
            if let turnModel { input["turn_detection"] = ["type": "semantic_vad", "model": turnModel, "create_response": false, "interrupt_response": false] }
            enqueue(["type": "session.update", "session": ["type": "transcription", "audio": ["input": input]]])
            let source = makeSource()
            self.source = source
            let (stream, continuation) = AsyncStream<Data>.makeStream()
            audio = continuation
            feeding = Task { [weak self] in await self?.feed(stream) }
            try await source.start { chunk in continuation.yield(chunk) }
            // Stopped while the source was starting (a permission prompt): it must not keep the microphone open.
            if !listening { source.stop() }
        } catch {
            fail(error.localizedDescription)
        }
    }

    /// Stops listening. What was heard since the last turn is committed after its audio; `finished` receives this
    /// object once that turn's transcript has arrived (or the service closed, or 15 seconds passed), then the socket closes.
    func stop(_ finished: @escaping (LiveTranscription) -> Void = { _ in }) {
        guard listening else { return }
        onFinished = finished
        stopping = true; listening = false; level = 0
        source?.stop(); source = nil
        deadline = Task { [weak self] in
            try? await Task.sleep(for: .seconds(15))
            if !Task.isCancelled { self?.complete() }
        }
        // The feeder appends what the source already delivered, then commits; without one nothing was sent.
        if let audio { audio.finish(); self.audio = nil } else { complete() }
    }

    func close() {
        source?.stop(); source = nil
        audio?.finish(); audio = nil
        feeding?.cancel(); feeding = nil
        receiving?.cancel(); receiving = nil
        sending?.cancel(); sending = nil; outbox = []
        deadline?.cancel(); deadline = nil
        socket?.close(); socket = nil
        listening = false; stopping = false; onFinished = nil
    }

    private func fail(_ message: String) {
        problem = message
        source?.stop(); source = nil
        audio?.finish(); audio = nil
        listening = false; level = 0
        // A failure while the last turn was awaited still hands over what was heard.
        if stopping { complete() } else { socket?.close(); socket = nil; outbox = [] }
    }

    /// Ends a stop: the socket closes and the transcript goes to whoever stopped.
    private func complete() {
        deadline?.cancel(); deadline = nil
        receiving?.cancel(); receiving = nil
        sending?.cancel(); sending = nil; outbox = []
        socket?.close(); socket = nil
        stopping = false
        let finished = onFinished; onFinished = nil
        finished?(self)
    }

    /// The stop is complete once the final commit was acknowledged (or refused) and every committed turn has its text.
    private func settle() {
        if stopping, !awaitingCommit, pendingItems.isEmpty { complete() }
    }

    private func feed(_ stream: AsyncStream<Data>) async {
        for await chunk in stream { append(chunk) }
        guard stopping else { return }
        if appendedSinceCommit {
            enqueue(["type": "input_audio_buffer.commit"]); appendedSinceCommit = false; awaitingCommit = true
        }
        heard = false; quiet = 0
        settle()
    }

    private func append(_ chunk: Data) {
        guard listening || stopping, socket != nil else { return }
        // Loudness of this chunk (RMS of 16-bit samples), for the meter and the pause-based turn end.
        let rms: Float = chunk.withUnsafeBytes { raw in
            let samples = raw.bindMemory(to: Int16.self)
            guard !samples.isEmpty else { return 0 }
            let sum = samples.reduce(Float(0)) { $0 + Float($1) * Float($1) }
            return (sum / Float(samples.count)).squareRoot() / 32768
        }
        if listening {
            let shown = min(1, rms * 6)
            if abs(shown - level) > 0.02 { level = shown }
        }
        let seconds = Double(chunk.count / 2) / 24_000
        if rms > 0.02 { heard = true; quiet = 0 } else if heard { quiet += seconds }
        enqueue(["type": "input_audio_buffer.append", "audio": chunk.base64EncodedString()])
        appendedSinceCommit = true
        if !detectsTurns, !diarizing, heard, quiet > 0.7 {
            heard = false; quiet = 0
            enqueue(["type": "input_audio_buffer.commit"]); appendedSinceCommit = false
        }
    }

    private func enqueue(_ object: [String: Any]) {
        guard let socket, let data = try? JSONSerialization.data(withJSONObject: object) else { return }
        outbox.append(data)
        if sending == nil { sending = Task { [weak self] in await self?.drain(socket) } }
    }

    private func drain(_ socket: LiveSocket) async {
        while !outbox.isEmpty, !Task.isCancelled {
            let data = outbox.removeFirst()
            guard let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { continue }
            do {
                try await socket.send(object)
            } catch {
                // A cancelled sender belongs to a session already ended; a later session may own `sending` now.
                guard !Task.isCancelled else { return }
                sending = nil
                if self.socket === socket { fail("Live transcription lost its connection to Arsenal: \(error.localizedDescription)") }
                return
            }
        }
        if !Task.isCancelled { sending = nil }
    }

    private func receive(_ socket: LiveSocket) async {
        while !Task.isCancelled {
            guard let event = try? await socket.receive() else {
                // The service closed the socket: a stop has all it will get; a session in progress has ended.
                if self.socket === socket, !Task.isCancelled {
                    if stopping { complete() } else if listening { fail("Live transcription lost its connection to Arsenal.") }
                }
                return
            }
            let item = event["item_id"] as? String
            switch event["type"] as? String {
            case "session.updated":
                configured = true
            case "input_audio_buffer.speech_started":
                latestItem = item
            case "input_audio_buffer.committed":
                if let item {
                    pendingItems.insert(item)
                    if item == latestItem { awaitingCommit = false }
                }
                settle()
            case "conversation.item.input_audio_transcription.delta":
                partial += event["delta"] as? String ?? ""
            case "conversation.item.input_audio_transcription.completed":
                let line = (event["transcript"] as? String ?? partial).trimmingCharacters(in: .whitespacesAndNewlines)
                if !line.isEmpty { lines.append(line) }
                partial = ""
                // A turn that arrived clears the words of an earlier refused chunk or turn.
                if listening { problem = nil }
                if let item { pendingItems.remove(item) }
                settle()
            case "arsenal.diarization.snapshot":
                var snapshot: [String: Any] = [:]
                for key in ["segments", "speakers"] { snapshot[key] = event[key] }
                if let data = try? JSONSerialization.data(withJSONObject: snapshot), case .object(let object)? = try? JSONDecoder().decode(JSONValue.self, from: data) {
                    timeline = object
                }
            case "error":
                let error = event["error"] as? [String: Any]
                let message = error?["message"] as? String ?? "Live transcription stopped."
                if stopping {
                    // The final commit was refused (nothing left to commit) or its turn failed: nothing more is coming for it.
                    awaitingCommit = false; settle(); continue
                }
                if configured, !Self.fatalCodes.contains(error?["code"] as? String ?? "") { problem = message; continue }
                fail(message)
                return
            default: break
            }
        }
    }

    /// Render harness and tests: a state without a socket.
    func preview(lines: [String], partial: String = "", listening: Bool = false, level: Float = 0, timeline: JSONObject? = nil) {
        self.lines = lines; self.partial = partial; self.listening = listening; self.level = level; self.timeline = timeline
        started = listening ? Date().addingTimeInterval(-14) : nil
    }
}
