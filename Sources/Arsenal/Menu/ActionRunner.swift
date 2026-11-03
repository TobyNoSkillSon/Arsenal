// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore

/// Runs a loaded model's menu action through the service and turns its result into one line (and the visible effect:
/// speech plays, an image opens, an answer is copied). Routed by op, never a shell command. Speech goes to the shared
/// playback queue (C7), the same one agents speak through, so Mute and Stop cover it and two voices never overlap.
@MainActor final class ActionRunner {
    let service: any ModelsService
    let jobs: URL
    var readClipboard: () -> String? = { NSPasteboard.general.string(forType: .string) }
    var writeClipboard: (String) -> Void = {
        NSPasteboard.general.clearContents(); NSPasteboard.general.setString($0, forType: .string)
    }
    var openFile: (URL) -> Void = { NSWorkspace.shared.open($0) }
    /// Puts a speech file on the shared queue as Arsenal's; false when it was not played (muted or stopped).
    var play: (_ file: URL, _ model: String, _ voice: String?) async throws -> Bool = { file, model, voice in
        try await PlaybackQueue.shared.enqueue(path: file, requester: ArsenalClient.clientName, model: model, voice: voice) != nil
    }
    /// The last result line ("Spoke 3.2 s of audio"); shown under the actions.
    private(set) var lastResult: String?
    private(set) var running: String?
    /// The last action's failure (shown in the menu until the next action); nil after a success.
    private(set) var failure: String?
    /// The model of the running action (the voice section's "who is speaking").
    private(set) var runningModel: String?
    var onChange: () -> Void = {}
    /// Stops the queue when what plays is Arsenal's own speech (a preview), never an agent's.
    var stopOwnSpeech: () -> Void = {
        if PlaybackQueue.shared.snapshot().current?.requester == ArsenalClient.clientName { PlaybackQueue.shared.stop() }
    }
    private var task: Task<Void, Never>?
    private var runningActionID: String?
    private var stopped = false

    init(service: any ModelsService, jobs: URL) { self.service = service; self.jobs = jobs }

    /// The request for an action: its declared params plus the clipboard under the op's input name.
    static func params(_ action: LoadedAction, clipboard: String?, outDir: URL) throws -> JSONObject {
        var params = action.params
        if action.input == "clipboard" {
            guard let text = clipboard?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else {
                throw WorkerFailure(code: "empty_clipboard", message: "The clipboard has no text; copy some text first.")
            }
            switch action.op {
            case "speak": params["text"] = .string(text)
            case "generate": params["prompt"] = .string(text)
            case "chat": params["messages"] = [["role": "user", "content": .string(text)]]
            case "decide": params["state"] = .string(text)
            default: params["text"] = .string(text)
            }
        }
        switch action.op {
        case "speak": params["stream"] = false; params["out_dir"] = .string(outDir.path)
        case "chat": params["stream"] = false
        case "generate": params["out_dir"] = .string(outDir.path)
        default: break
        }
        return params
    }

    /// One line from a result.
    /// `played`: speech reached the speakers (false: muted or stopped, the file is kept).
    static func summary(_ action: LoadedAction, _ result: JSONObject, played: Bool = true) -> String {
        switch action.op {
        case "speak":
            let length = result["duration_s"]?.number.map { String(format: "%.1f s of audio", $0) }
            if !played { return "Muted: " + (length ?? "the audio") + " not played" }
            return length.map { "Spoke " + $0 } ?? "Spoke the clipboard"
        case "generate":
            let n = result["images"]?.array?.count ?? 0
            let s = result["seconds"]?.number.map { String(format: " in %.1f s", $0) } ?? ""
            return n == 1 ? "Made 1 image\(s); opened it" : "Made \(n) images\(s)"
        case "chat": return "Answer copied to the clipboard"
        case "decide":
            let answers = result["answers"].flatMap { try? WireJSON.encode(["a": $0]) }.flatMap { String(data: $0, encoding: .utf8) }
            return "Decided: " + (answers.map { String($0.dropFirst(5).dropLast()) } ?? "done")
        default: return "\(action.title): done"
        }
    }

    /// True only when the runner accepts this action; a busy or stopped runner keeps its current owner.
    @discardableResult func run(_ action: LoadedAction) -> Bool {
        guard running == nil, !stopped else { return false }
        let outDir = jobs.appendingPathComponent(UUID().uuidString)
        let params: JSONObject
        do { params = try Self.params(action, clipboard: readClipboard(), outDir: outDir) } catch {
            finish(TableController.message(error), failed: true); return false
        }
        runningActionID = action.id; running = action.title; runningModel = action.modelID; lastResult = "\(action.title)\u{2026}"; onChange()
        if ["generate", "speak"].contains(action.op) { try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true) }
        task = Task {
            do {
                let notice = await service.loadMemoryNotice(id: action.modelID, precision: nil)
                if let notice { lastResult = "\(action.title): \(notice)"; onChange() }
                let result = try await MemoryLoadPresenter.perform { force in
                    try await self.service.run(id: action.modelID, op: action.op, params: params, force: force)
                }
                try Task.checkCancellation()
                var played = true
                if action.op == "speak", let path = result["path"]?.string {
                    played = try await play(URL(fileURLWithPath: path), action.modelID, params["voice"]?.string)
                }
                try Task.checkCancellation()
                if action.op == "generate", let path = result["images"]?.array?.first?["path"]?.string { openFile(URL(fileURLWithPath: path)) }
                if action.op == "chat", let text = result["text"]?.string { writeClipboard(text) }
                finish(Self.summary(action, result, played: played) + (notice.map { " · " + $0 } ?? ""))
            } catch is CancellationError {
                if !stopped { finish("Cancelled") }
            } catch {
                if !stopped && !Task.isCancelled { finish("\(action.title) failed: \(TableController.message(error))", failed: true) }
            }
        }
        return true
    }
    /// A second click on a voice preview: the running action stops and so does what it was saying; new ones may start.
    func cancel() {
        task?.cancel(); task = nil
        stopOwnSpeech()
        if running != nil { finish("Stopped") }
    }
    /// A model's settings stop only its own accepted preview, never another model or a menu action.
    @discardableResult func cancel(actionID: String) -> Bool {
        guard runningActionID == actionID else { return false }
        cancel(); return true
    }
    /// Quit: the running action is cancelled, speech stops, nothing new starts.
    func stop() { stopped = true; task?.cancel(); task = nil; PlaybackQueue.shared.stop(); running = nil; runningModel = nil; runningActionID = nil }
    func previewResult(_ line: String?) { lastResult = line }
    private func finish(_ line: String, failed: Bool = false) {
        task = nil; running = nil; runningModel = nil; runningActionID = nil; lastResult = line; failure = failed ? line : nil; onChange()
    }
}
