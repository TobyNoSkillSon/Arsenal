// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore

// Make a Sound, as the tool kit composes it: one window for every model that declares the sound tool, the chosen model's
// operations that make audio (sound, speech, music, …) with their options from their schemas, and the takes of this
// window, each saved to ~/Music/Arsenal as it arrives. Voices are listed through the model's own `voices` operation once
// it is loaded (listing never loads a model).

struct SoundTake: Identifiable, Equatable {
    let id = UUID()
    var url: URL
    var model: String
    var modelName: String
    var operation: String
    /// The main text the take was made from.
    var text: String
    var seed: Int?
    var voice: String?
    var shape: AudioShape
    /// From Generate to the take, by this window's clock.
    var seconds: Double
    /// The operation's name and the options it was made with: choosing the take brings them back.
    var operationName = ""
    var values: [String: FieldValue] = [:]
}

@MainActor final class SoundSession: ToolSession {
    @Published private(set) var takes: [SoundTake] = []
    @Published var shown: UUID?

    static var musicDirectory: URL { ToolFiles.musicDirectory }

    /// The menu task this window serves (Speak, Make Music, Make a Sound); nil offers every audio operation.
    let task: SoundTask?

    init(
        models: [ModelChoice], selected: String, client: @escaping () -> ArsenalClient?, inbox: URL, outputDirectory: URL = SoundSession.musicDirectory, scratch: URL? = nil,
        task: SoundTask? = nil
    ) {
        self.task = task
        super.init(models: models, selected: selected, client: client, inbox: inbox, outputDirectory: outputDirectory, scratch: scratch)
    }

    /// An operation that answers with audio, or a sound task's operation that answers with files: songs from ACE-Step,
    /// YuE2 and MiniMax Music come back as files from a job (`output = "file"`), not as streamed audio.
    override func offers(_ op: OperationDescriptor) -> Bool {
        guard op.output == "audio" || (op.output == "file" && Self.task(for: op.name) != nil) else { return false }
        return task.map { $0.operations.contains(op.name) } ?? true
    }
    /// The task's own operations first (Speak's `speak` before cloning); without a task, sound before speech before music.
    override var preferredOperations: [String] { task?.operations ?? ["generate_sound", "speak", "generate_music"] }
    /// The task's title; a window without one names what its chosen operation's task is.
    override var taskTitle: String { (task ?? Self.task(for: operation))?.windowTitle ?? "Make a Sound" }
    /// The task's verb (Speak, Make Music, Make Sound), never "Generate".
    override var runVerb: String { (task ?? Self.task(for: operation))?.runTitle ?? (descriptor == nil ? SoundTask.sound.runTitle : super.runVerb) }
    /// Pills only for this window's own task (a window opened without one: its operation's task), so Make Music and
    /// Make a Sound look the same whichever menu entry opened them; another task is its own menu entry.
    override var pillOperations: [OperationDescriptor] {
        guard let own = task ?? Self.task(for: operation) else { return super.pillOperations }
        return super.pillOperations.filter { own.operations.contains($0.name) }
    }
    /// The task an operation belongs to (`clone_voice` is Speak's).
    static func task(for op: String?) -> SoundTask? { op.flatMap { op in SoundTask.allCases.first { $0.operations.contains(op) } } }
    override var resultNoun: String { "take" }
    /// Songs that may run ten minutes or more (YuE2, ACE-Step) come back from a durable job with progress and Stop.
    override func runsAsJob(_ op: OperationDescriptor) -> Bool { op.returnsFiles && CapabilitySession.isLong(op) }
    override func noOperations(_ name: String) -> String { "\(name) declares no operation that makes audio." }

    var current: SoundTake? { takes.first { $0.id == shown } ?? takes.last }
    override var shownSeed: Int? { current?.seed }
    override var shownFile: URL? { current?.url }

    /// The take's audio, saved as `<date> <time> <model> <seed or voice>.wav`, its length and envelope read once. A batch
    /// (ACE-Step, YuE2 and MiniMax Music take `batch_size`) gives one take per song, the last one shown.
    override func deliver(_ result: OperationResult, op: OperationDescriptor, model: ModelChoice, input: JSONObject) throws {
        let audio = result.files.filter(\.isAudio)
        let files = audio.isEmpty ? Array(result.files.prefix(1)) : audio
        guard !files.isEmpty else { throw Undelivered(.init(message: "The model answered without audio.", action: nil)) }
        let seed = input["seed"]?.number.flatMap(\.roundedInt), voice = input["voice"]?.string
        let text = form?.main.flatMap { input[$0.key]?.string } ?? ""
        for file in files {
            let url = try ToolFiles.save(file.url, into: outputDirectory, model: model.id, tag: seed.map(String.init) ?? voice ?? "", what: "sound")
            guard let shape = AudioShape.read(url) else {
                throw Undelivered(
                    .init(
                        message: "\(model.name) answered with audio Arsenal cannot read; it is saved in \(outputDirectory.lastPathComponent) as \(url.lastPathComponent).",
                        action: nil)
                )
            }
            takes.append(
                SoundTake(
                    url: url, model: model.id, modelName: model.name, operation: op.title, text: text, seed: seed, voice: voice, shape: shape,
                    seconds: result.seconds, operationName: op.name, values: values))
        }
        shown = takes.last?.id
        if model.id == selected, voices == nil { listVoices() }
    }

    /// A take chosen: shown, and (unless a take is being made) its words and options come back (a random seed stays
    /// random; Reuse last takes the take's).
    func show(_ id: UUID) {
        guard let take = takes.first(where: { $0.id == id }) else { return }
        shown = id
        guard !running, take.model == selected, !take.values.isEmpty else { return }
        if take.operationName != operation { select(take.operationName) }
        guard take.operationName == operation else { return }
        for (k, v) in take.values { values[k] = v }
        // A take made from a clip brings its clip back on show (New Voice…), never hidden and sent.
        newVoice = form?.fields.contains { isCloneField($0) && !value($0.key).files.isEmpty } ?? false
    }

    // MARK: Figures

    /// Time is this window's clock from Generate to the take; length from the file; speed is length over time ("vs real
    /// time", the table's × real time); memory from the activity stream (C3).
    override func figures() -> [Figure] {
        let r = run
        let take = takes.last.flatMap { t in r?.done == true && r?.model == t.model ? t : nil }
        let seconds = take?.seconds
        let length = take?.shape.seconds
        let speed = seconds.flatMap { s in length.flatMap { l in s > 0 ? l / s : nil } }
        return [
            Figure(title: "Time", value: FigureText.seconds(seconds), reference: nil, help: "From Generate to the saved file, as this window measured it.", width: 84),
            Figure(title: "Length", value: FigureText.seconds(length), reference: nil, help: "How long the take plays.", width: 84),
            Figure(
                title: "vs real time", value: speed.map { String(format: $0 < 10 ? "%.2f" : "%.1f", $0) + "\u{00d7}" }, reference: nil,
                help: "Length divided by time: above 1\u{00d7} the model makes audio faster than it plays.", width: 84),
            memoryFigure()
        ]
    }

    /// Render harness and tests: a state without the API.
    func preview(
        operations: [OperationDescriptor], selected op: String? = nil, values: [String: FieldValue] = [:], voices: [VoiceChoice]? = nil, run: ToolRun? = nil,
        takes: [SoundTake] = [], problem: OperationProblem? = nil, settings: [String: JSONValue] = [:]
    ) {
        previewOptions(operations: operations, selected: op, values: values, settings: settings)
        self.voices = voices; self.run = run; self.takes = takes; shown = takes.last?.id; self.problem = problem
    }
}
