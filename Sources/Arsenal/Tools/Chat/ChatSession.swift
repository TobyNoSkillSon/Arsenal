// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ImageIO
import ArsenalCore

// The chat test window's state: a plain conversation with one loaded model through Arsenal's OpenAI chat route
// (streaming, usage, C4 `arsenal_timing` in the final chunk), the live figures of the latest reply, the model's
// reference (C5) and the speed presets. Every failure ends as a line in the conversation, never as a crash.

struct ChatMessage: Identifiable, Equatable {
    enum Role: String { case user, assistant }
    let id = UUID()
    var role: Role
    var text: String
    /// Pictures and clips sent with a user message (only to a model whose chat schema takes them).
    var attachments: [URL] = []
    /// The reply was asked to follow a JSON Schema: shown as JSON.
    var structured = false
    var reasoning = ""
    /// A preset's nominal size when the user message is one ("8K"); its text is not shown.
    var preset: PresetPrompt?
    /// What was actually sent for a user message (a preset's generated text plus anything typed).
    var sent: String?
    var error: String?
    var streaming = false
}

/// What one reply measured, only from the service: the activity stream (C3) while it runs, then the final chunk's
/// `usage` and `arsenal_timing` (C4). Nothing is estimated from the client's own clock.
struct ChatRun: Equatable {
    var sent = Date()
    /// The preset the prompt was (nil: typed), for the reference comparison.
    var preset: PresetPrompt?
    /// From C3 while the request runs: counts, and the request's timing so far (first token and prompt rate once the
    /// first token is out, the decode rate, peak memory, the longest pause).
    var livePromptTokens: Int?
    var liveGeneratedTokens: Int?
    var liveTiming: [String: Double] = [:]
    /// From the final chunk.
    var promptTokens: Int?
    var cachedTokens: Int?
    var completionTokens: Int?
    var timing: [String: Double] = [:]
    var done = false
    var stopped = false

    /// The service's figures from a final chunk (`usage`, `arsenal_timing`); absent values stay unknown.
    mutating func finish(usage: [String: Any]?, timing: [String: Any]?) {
        func int(_ v: Any?) -> Int? { (v as? NSNumber).map { $0.intValue } }
        if let usage {
            promptTokens = int(usage["prompt_tokens"]) ?? promptTokens
            completionTokens = int(usage["completion_tokens"]) ?? completionTokens
            cachedTokens = int((usage["prompt_tokens_details"] as? [String: Any])?["cached_tokens"]) ?? cachedTokens
        }
        for (key, value) in timing ?? [:] { if let n = (value as? NSNumber)?.doubleValue, n.isFinite { self.timing[key] = n } }
        done = true
    }
    var prompt: Int? { promptTokens ?? livePromptTokens }
    var generated: Int? { completionTokens ?? liveGeneratedTokens }
    /// A timing figure: the final chunk's, else the live one.
    func figure(_ key: String) -> Double? { timing[key] ?? liveTiming[key] }
}

/// What the model's chat operation declares beyond text (its input schema's content parts and response formats).
struct ChatCapabilities: Equatable {
    var images = false
    /// The audio formats an `input_audio` part takes ("wav", "mp3"); empty: no audio.
    var audioFormats: [String] = []
    var jsonSchema = false
    var attaches: Bool { images || !audioFormats.isEmpty }

    /// Read from the `chat` input schema: `messages[].content` array parts by their `type`, and `response_format`
    /// alternatives with `type: json_schema`.
    init(schema: JSONObject) {
        let refs = SchemaRefs(root: schema)
        func resolve(_ v: JSONValue?) -> JSONObject? { v?.object.map(refs.resolve) }
        func alternatives(_ o: JSONObject?) -> [JSONObject] {
            guard let o else { return [] }
            let list = (o["anyOf"] ?? o["oneOf"])?.array?.compactMap(resolve) ?? []
            return list.isEmpty ? [o] : list.flatMap(alternatives)
        }
        func types(_ o: JSONObject) -> [String] {
            let t = resolve(o["properties"]?["type"])
            return (t?["enum"]?.array?.compactMap(\.string) ?? []) + [t?["const"]?.string].compactMap { $0 }
        }
        let content = resolve(resolve(resolve(schema["properties"]?["messages"])?["items"])?["properties"]?["content"])
        let parts = alternatives(content).filter { $0["type"]?.string == "array" }.flatMap { alternatives(resolve($0["items"])) }
        images = parts.contains { types($0).contains("image_url") || types($0).contains("image") }
        audioFormats =
            parts.first { types($0).contains("input_audio") }.map { part in
                resolve(resolve(part["properties"]?["input_audio"])?["properties"]?["format"])?["enum"]?.array?.compactMap(\.string) ?? ["wav"]
            } ?? []
        jsonSchema = alternatives(resolve(schema["properties"]?["response_format"])).contains { types($0).contains("json_schema") }
    }
    init() {}
}

@MainActor final class ChatSession: ObservableObject {
    @Published private(set) var model: ToolModel
    /// What the model takes besides text; nothing until its operations are read.
    @Published private(set) var capabilities = ChatCapabilities()
    /// Pictures and clips for the next message.
    @Published private(set) var attachments: [URL] = []
    /// The answer must follow this JSON Schema (when on and the model takes one).
    @Published var structured = false
    @Published var responseSchema = ""
    /// The open panel (tests answer it).
    var chooseFiles: () -> [URL] = {
        let panel = NSOpenPanel(); panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [.image, .audio]
        return panel.runModal() == .OK ? panel.urls : []
    }
    @Published private(set) var messages: [ChatMessage] = []
    @Published var draft = ""
    @Published var preset: PresetPrompt?
    @Published private(set) var run: ChatRun?
    @Published private(set) var reference: ReferenceState = .loading
    @Published private(set) var generating = false
    /// The window's own problem (settings or reference could not be read), shown once under the presets.
    @Published private(set) var notice: String?

    /// Every chat model the window can switch to (the band's pop-up); one or none shows the name alone.
    @Published var choices: [ModelChoice] = []
    /// Switches the window to another chat model; each model keeps its own conversation (ToolCenter sets it).
    var switchModel: (String) -> Void = { _ in }

    let client: () -> ArsenalClient?
    private var task: Task<Void, Never>?
    private var activity: Task<Void, Never>?
    private var preparing: Task<Void, Never>?
    /// The request whose answers may still change the conversation; a stopped or replaced one writes nothing.
    private var current: UUID?
    /// The reply budget the preset prompts leave room for: the chat model's Answer limit default (qwen3.8-27b).
    nonisolated static let maxReplyTokens = 1024

    init(model: ToolModel, client: @escaping () -> ArsenalClient?) {
        self.model = model; self.client = client
    }

    /// Reads the loaded context limit and the reference; called when the window opens and after each reload.
    func prepare() {
        preparing?.cancel()
        preparing = Task {
            guard let client = client() else { reference = .failed(Self.noAPI); notice = Self.noAPI; return }
            // What the model's chat takes besides text (attachments, a JSON answer) comes from its declared schema.
            if let chat = try? await client.operations(model.id).first(where: { $0.name == "chat" }) {
                capabilities = ChatCapabilities(schema: chat.inputSchema)
                if !capabilities.attaches { attachments = [] }
                if !capabilities.jsonSchema { structured = false }
            }
            do {
                let settings = try await client.settings(model.id)
                model.contextLimit = settings.first { $0.id == "context_limit" }?.value.number.flatMap(\.roundedInt).flatMap { $0 > 0 ? $0 : nil }
            } catch { notice = "Could not read \(model.name)\u{2019}s settings: \(error.localizedDescription)" }
            do {
                reference = try await client.reference(model.id, dtype: model.dtype, mode: model.mode).map { .found($0) } ?? .none
            } catch let error as ToolError where error.status == 404 {
                reference = .none
            } catch {
                if !Task.isCancelled { reference = .failed(error.localizedDescription) }
            }
        }
    }
    /// The window closed: whatever runs for it stops (the reply, the activity stream, the settings read); the
    /// conversation stays for the next opening.
    func close() {
        stop()
        preparing?.cancel(); preparing = nil
        activity?.cancel(); activity = nil
    }
    /// Tests: work still running for this window.
    var hasRunningWork: Bool { task != nil || activity != nil || generating }
    static let noAPI = "Arsenal's API is not running, so the tool cannot reach the model."

    func update(model: ToolModel) {
        guard model.id == self.model.id else { return }
        let changed = model.dtype != self.model.dtype || model.mode != self.model.mode
        var next = model; next.contextLimit = self.model.contextLimit
        self.model = next
        if changed { reference = .loading; prepare() }
    }

    func unavailable(_ preset: PresetPrompt) -> String? { preset.unavailable(contextLimit: model.contextLimit, model: model.name) }
    var canSend: Bool {
        guard !generating, structured ? Self.schemaObject(responseSchema) != nil : true else { return false }
        return preset != nil || !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !attachments.isEmpty
    }

    // MARK: Attachments

    /// Files dropped, pasted or chosen: pictures when the model takes them, clips in a format it takes.
    func attach(_ urls: [URL]) {
        guard !generating else { return }
        for url in urls where !attachments.contains(url) {
            if capabilities.images, ToolFiles.isImageFile(url) {
                attachments.append(url)
            } else if capabilities.audioFormats.contains(url.pathExtension.lowercased() == "mp3" ? "mp3" : url.pathExtension.lowercased() == "wav" ? "wav" : "") {
                attachments.append(url)
            } else {
                notice =
                    "\(model.name) takes "
                    + [
                        capabilities.images ? "pictures" : nil,
                        capabilities.audioFormats.isEmpty ? nil : capabilities.audioFormats.map { $0.uppercased() }.joined(separator: " or ") + " clips"
                    ]
                    .compactMap { $0 }.joined(separator: " and ") + "; \(url.lastPathComponent) is neither."
            }
        }
    }
    func detach(_ url: URL) { attachments.removeAll { $0 == url } }
    func pickAttachments() { attach(chooseFiles()) }
    /// ⌘V in the window: copied files, else a copied picture (kept as a PNG). False when nothing fits.
    @discardableResult func paste(_ pasteboard: NSPasteboard) -> Bool {
        guard capabilities.attaches, !generating else { return false }
        let files = (pasteboard.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL]) ?? []
        if !files.isEmpty { attach(files); return true }
        guard capabilities.images, let data = pasteboard.data(forType: .png) ?? pasteboard.data(forType: .tiff),
            let source = CGImageSourceCreateWithData(data as CFData, nil)
        else { return false }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("Arsenal chat/\(UUID().uuidString.prefix(8)) Pasted image.png")
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard (try? ToolFiles.upright(source, to: url)) != nil else { return false }
        attach([url]); return true
    }

    /// A message's content as the chat route takes it: the text alone, or text and its pictures (data URLs) and clips.
    static func content(_ text: String, attachments: [URL]) throws -> Any {
        guard !attachments.isEmpty else { return text }
        var parts: [[String: Any]] = text.isEmpty ? [] : [["type": "text", "text": text]]
        for url in attachments {
            let data = try Data(contentsOf: url)
            if ToolFiles.isImageFile(url) {
                let mime = FileAccept.mime(url) ?? "image/png"
                parts.append(["type": "image_url", "image_url": ["url": "data:\(mime);base64,\(data.base64EncodedString())"]])
            } else {
                parts.append(["type": "input_audio", "input_audio": ["data": data.base64EncodedString(), "format": url.pathExtension.lowercased() == "mp3" ? "mp3" : "wav"]])
            }
        }
        return parts
    }
    /// The schema typed for the answer, when it is a JSON object.
    static func schemaObject(_ text: String) -> [String: Any]? {
        (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [String: Any]
    }

    func insert(_ preset: PresetPrompt) {
        guard unavailable(preset) == nil, !generating else { return }
        self.preset = self.preset == preset ? nil : preset
    }

    func newChat() {
        stop()
        messages = []; run = nil; draft = ""; preset = nil
    }

    func send() {
        guard canSend else { return }
        let typed = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        let chosen = preset
        let attached = attachments
        draft = ""; preset = nil; attachments = []
        // A speed preset always starts a fresh conversation; prior replies never change its workload.
        if chosen != nil { messages = []; run = nil }
        messages.append(ChatMessage(role: .user, text: typed, attachments: attached, preset: chosen))
        messages.append(ChatMessage(role: .assistant, text: "", structured: structured && capabilities.jsonSchema, streaming: true))
        generating = true
        let id = UUID()
        current = id
        task = Task { await self.stream(typed: typed, preset: chosen, id: id) }
    }

    /// Stops the reply: the HTTP request ends, which cancels the model's work; nothing that request still answers is
    /// written.
    func stop() {
        guard generating else { return }
        current = nil
        task?.cancel(); task = nil
        run?.stopped = true
        end(error: nil)
    }

    private func stream(typed: String, preset: PresetPrompt?, id: UUID) async {
        guard let client = client() else { end(error: Self.noAPI); return }
        var userText = typed
        var countedTokens: Int?
        if let preset {
            do {
                let sized = try await preset.sized(run: String(UUID().uuidString.prefix(8)).lowercased(), typed: typed, contextLimit: model.contextLimit) { text in
                    let body: [String: Any] = ["model": self.model.id, "messages": [["role": "user", "content": text]]]
                    let (data, _) = try await client.send("POST", "/providers/anthropic/v1/messages/count_tokens", json: body)
                    guard let n = (eventObject(data)?["input_tokens"] as? NSNumber)?.intValue else { throw ToolError("The model returned no prompt token count.") }
                    return n
                }
                guard !Task.isCancelled, current == id else { return }
                userText = sized.text; countedTokens = sized.tokens
            } catch {
                guard !Task.isCancelled, current == id else { return }
                end(error: error.localizedDescription); return
            }
        }
        var history: [[String: Any]] = []
        do {
            history = try messages.dropLast(2).compactMap { m -> [String: Any]? in
                guard m.error == nil, !(m.role == .assistant && m.text.isEmpty) else { return nil }
                return ["role": m.role.rawValue, "content": try Self.content(m.sent ?? m.text, attachments: m.attachments)]
            }
            let attached = messages.count >= 2 ? messages[messages.count - 2].attachments : []
            history.append(["role": "user", "content": try Self.content(userText, attachments: attached)])
        } catch {
            end(error: (error as? WordedError)?.line ?? FileProblem(error)?.line ?? "Couldn\u{2019}t read an attachment."); return
        }
        if messages.count >= 2 { messages[messages.count - 2].sent = userText }
        var run = ChatRun()
        run.preset = preset; run.promptTokens = countedTokens
        self.run = run
        watchActivity(client, id: id)
        var body: [String: Any] = [
            // No max_tokens: the model's own Answer limit setting applies (a caller's value would win over it).
            "model": model.id, "messages": history, "stream": true, "stream_options": ["include_usage": true]
        ]
        if messages.last?.structured == true, let schema = Self.schemaObject(responseSchema) {
            body["response_format"] = ["type": "json_schema", "json_schema": ["name": "answer", "schema": schema, "strict": true]]
        }
        do {
            var finished = false
            for try await data in client.events("POST", "/v1/chat/completions", json: body, requireDone: true) {
                guard !Task.isCancelled, current == id else { return }
                guard let chunk = eventObject(data) else { throw ToolError("Arsenal sent a reply chunk that is not JSON.") }
                // An error inside the stream (the model failed after it started) is the reply's error.
                if let error = chunk["error"] { throw ToolError.from(status: 500, body: (try? JSONSerialization.data(withJSONObject: ["error": error])) ?? Data()) }
                if let choices = chunk["choices"] {
                    guard let list = choices as? [[String: Any]] else { throw ToolError("Arsenal sent a reply chunk without its choices.") }
                    let first = list.first
                    let delta = first?["delta"] as? [String: Any] ?? [:]
                    let content = delta["content"] as? String ?? "", reasoning = delta["reasoning_content"] as? String ?? ""
                    if !content.isEmpty || !reasoning.isEmpty { appendToReply(content: content, reasoning: reasoning) }
                    if first?["finish_reason"] is String { finished = true }
                }
                if chunk["usage"] != nil || chunk["arsenal_timing"] != nil {
                    self.run?.finish(usage: chunk["usage"] as? [String: Any], timing: chunk["arsenal_timing"] as? [String: Any])
                }
            }
            guard !Task.isCancelled, current == id else { return }
            guard finished else { throw ToolError("The reply stopped before it finished.") }
            if self.run?.done != true { self.run?.done = true }
            end(error: nil)
        } catch is CancellationError {
            return
        } catch {
            guard !Task.isCancelled, current == id else { return }
            end(error: OperationProblem.wording(error, model: model.name))
        }
    }

    private func appendToReply(content: String, reasoning: String) {
        guard let i = messages.indices.last, messages[i].role == .assistant else { return }
        messages[i].text += content; messages[i].reasoning += reasoning
    }

    private func end(error: String?) {
        activity?.cancel(); activity = nil
        if let i = messages.indices.last, messages[i].role == .assistant {
            messages[i].streaming = false
            if let error { messages[i].error = error }
        }
        if run?.done == false, error == nil { run?.done = true }
        generating = false; task = nil; current = nil
    }

    /// C3 while the reply runs: the prompt's token count once the model has it, and the model's memory.
    private func watchActivity(_ client: ArsenalClient, id request: UUID) {
        activity?.cancel()
        let id = model.id
        activity = Task { [weak self] in
            do {
                for try await data in client.events("GET", "/arsenal/v1/activity/stream?machine=0") {
                    guard let snapshot = eventObject(data) else { continue }
                    guard let self, !Task.isCancelled, self.current == request else { return }
                    guard let m = (snapshot["models"] as? [[String: Any]])?.first(where: { $0["id"] as? String == id }) else { continue }
                    let mine = (m["requests"] as? [[String: Any]] ?? []).first { $0["requester"] as? String == ArsenalClient.clientName && $0["op"] as? String == "chat" }
                    if let n = (mine?["prompt_tokens"] as? NSNumber)?.intValue { self.run?.livePromptTokens = n }
                    if let n = (mine?["generated_tokens"] as? NSNumber)?.intValue { self.run?.liveGeneratedTokens = n }
                    for (key, value) in mine?["timing"] as? [String: Any] ?? [:] {
                        if let n = (value as? NSNumber)?.doubleValue, n.isFinite { self.run?.liveTiming[key] = n }
                    }
                }
            } catch {}
        }
    }

    /// The figures for the grid (chat), in the user's words, each with its definition as the title's tooltip.
    func figures() -> [Figure] {
        let r = run, ref = comparable
        let load = r?.figure("load_s").flatMap { FigureText.seconds($0) }
        let cached = r?.cachedTokens.flatMap { $0 > 0 ? " \($0.grouped) of them came from the cache and are not in Prompt." : nil } ?? ""
        return [
            Figure(
                title: "First token", value: FigureText.seconds(r?.figure("ttft_s")), reference: FigureText.seconds(ref?.value(["ttft_s", "first_token_s"])),
                help: "From the moment Arsenal accepted the prompt to the first token of the reply" + (load.map { ", including \($0) to load the model." } ?? "."),
                width: 90),
            Figure(
                title: "Prompt", value: FigureText.rate(r?.figure("prefill_tps")), reference: FigureText.rate(ref?.value(["prefill_tps", "prompt_tps"])),
                help: "How fast the model read the prompt: prompt tokens per second, counting only tokens it had to compute.", width: 112),
            Figure(
                title: "Decode", value: FigureText.rate(r?.figure("decode_tps")), reference: FigureText.rate(ref?.value(["decode_tps", "tokens_per_s", "generation_tps"])),
                help: "How fast the model wrote the reply: tokens per second after the first one.", width: 100),
            Figure(
                title: "Peak memory", value: FigureText.gigabytes(mb: r?.figure("peak_mb")),
                reference: FigureText.gigabytes(mb: ref?.value(["peak_mb"]) ?? ref?.value(["peak_ram_gb", "peak_ram"]).map { $0 * 1000 }),
                help: "The most memory the model used during this reply.", width: 84),
            Figure(
                title: "Tokens", value: FigureText.tokens(in: r?.prompt, out: r?.generated), reference: nil,
                help: "Tokens in the prompt and in the reply." + cached, width: 144),
            Figure(
                title: "Longest pause", value: FigureText.milliseconds(r?.figure("max_itl_ms")), reference: FigureText.milliseconds(ref?.value(["max_itl_ms"])),
                help: "The longest wait between two tokens of the reply (another request's prompt can cause one).", width: 100)
        ]
    }

    /// The reference when it measured the same kind of work as this run: its suite asks one-line questions, so a
    /// preset or any prompt over 256 tokens has nothing to compare with.
    private var comparable: ToolReference? {
        if case .compared(let r) = referenceLine { return r } else { return nil }
    }
    var referenceLine: ReferenceLine? {
        switch reference {
        case .loading: return nil
        case .failed: return .unavailable("Reference unavailable")
        // Nothing measured: no row at all, rather than a line saying so.
        case .none: return nil
        case .found(let r):
            guard r.workload == .shortPrompt else { return .unavailable("No reference for this prompt size") }
            if let run, run.preset != nil || (run.prompt ?? 0) > 256 { return .unavailable("No reference for this prompt size") }
            if run == nil, preset != nil { return .unavailable("No reference for this prompt size") }
            return .compared(r)
        }
    }

    /// Render harness and tests: what the model's chat takes, without reading it.
    func preview(capabilities: ChatCapabilities, attachments: [URL] = [], structured: Bool = false, schema: String = "") {
        self.capabilities = capabilities; self.attachments = attachments; self.structured = structured; responseSchema = schema
    }

    /// Render harness: a finished conversation with figures, without a model.
    func preview(messages: [ChatMessage], run: ChatRun?, reference: ReferenceState, generating: Bool = false, draft: String = "", preset: PresetPrompt? = nil) {
        self.messages = messages; self.run = run; self.reference = reference; self.generating = generating; self.draft = draft; self.preset = preset
    }
    func preview(contextLimit: Int?) { model.contextLimit = contextLimit }
}
