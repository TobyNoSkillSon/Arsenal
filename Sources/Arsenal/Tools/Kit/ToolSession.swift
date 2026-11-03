// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore

// The tool kit's state, shared by every tool window: the models it offers (the table's rows that declare the tool, or
// the one model a window belongs to), the chosen model's operations as the API describes them, the chosen operation's
// options from its input schema (ToolForm) with each model's settings as their starting values, one run at a time
// through the universal operation route with JSON negotiation, its progress and memory from the activity stream (C3),
// and its failures as a line with the one action that fixes them. A tool subclasses it for what it offers and what it
// keeps of a result (takes, images, cut-outs, files).

/// A model a tool window can pick: the table's row as the window needs it.
struct ModelChoice: Equatable, Identifiable {
    var model: ToolModel
    /// Loaded now: its voices can be listed without loading it.
    var resident = false
    /// Why it cannot run here (not on this Mac, approval needed); nil when a request may run (it loads on demand).
    var unavailable: OperationProblem?
    var id: String { model.id }
    var name: String { model.name }
}

/// Why a tool cannot run now, worded for the window, with the one action that fixes it.
struct OperationProblem: Equatable {
    /// Open Memory opens the Arsenal menu, as Open Models does: its Memory item unloads models and sets what happens when
    /// one does not fit.
    enum Action: String, Equatable {
        /// `loadAnyway` runs past a memory refusal; its button names the window's own verb ("Speak Anyway (slower)"), never Load.
        case openModels = "Open Models", openMemory = "Open Memory", loadAnyway = "Run Anyway (slower)", retry = "Try Again"
        /// Fixed in the window itself, through the table's own Get, approval and Update flow (`ToolSession.prepareModel`).
        case download = "Download", approve = "Approve\u{2026}", update = "Update"
    }
    var message: String
    var action: Action?
    /// The action's button title when it says more than the action's name ("Download 342 MB").
    var actionTitle: String?
    /// The form field it concerns (shown under that field).
    var field: String?
    /// The full story behind a short line (the memory refusal's numbers, a file system's own words): the line's hover.
    var detail: String?
    /// The memory refusal behind a Run Anyway line: the window watches freed memory against it.
    var refusal: LoadMemoryRefusal?

    /// What failed and what to do: an unreachable API reads "Couldn't reach <model> — try again"; the service's own
    /// refusal keeps its words (they name the cause, e.g. not enough memory).
    @MainActor static func wording(_ error: Error, model: String) -> String {
        guard let e = error as? ToolError else {
            if let w = error as? WordedError { return w.line }
            if let file = FileProblem(error) { return file.line }
            return "\(model) failed: \(error.localizedDescription)"
        }
        if e.code == "insufficient_memory", let details = e.details, let refusal = LoadMemoryRefusal(details: details) { return Self.memoryLine(refusal) }
        if e.status == nil, e.message != ChatSession.noAPI, e.message.hasPrefix("Arsenal's API did not answer") { return "Couldn\u{2019}t reach \(model) \u{2014} try again." }
        return e.message
    }

    /// The service's refusals keep their words; the ones a click in Models fixes carry that action.
    /// `task` names what stopped when the model's process ended mid-run ("Transcription stopped unexpectedly.").
    @MainActor static func from(_ error: Error, model: String, task: String = "The run") -> OperationProblem {
        let message = wording(error, model: model)
        if let w = error as? WordedError { return .init(message: w.line, action: nil, detail: w.raw) }
        guard let e = error as? ToolError else { return .init(message: message, action: nil, detail: FileProblem(error)?.raw) }
        switch e.code {
        // Fixed in the window: the table's own approval and Get flows, started from here.
        case "code_consent": return .init(message: "\(model)\u{2019}s folder changed since it was approved.", action: .approve)
        case "not_installed": return .init(message: "\(model) isn\u{2019}t on this Mac yet.", action: .download)
        // The model's process ended mid-run: said by what stopped, never "the worker"; the run button becomes Try Again.
        case let code? where crashCodes.contains(code): return .init(message: "\(task) stopped unexpectedly.", action: .retry, detail: e.message)
        case "insufficient_memory":
            // One line with the numbers; the whole story (the Mac-wide line, swap) is its hover.
            let refusal = e.details.flatMap(LoadMemoryRefusal.init(details:))
            return .init(message: message, action: .loadAnyway, detail: refusal?.summary ?? e.message, refusal: refusal)
        case "not_found" where e.status == 404 && message.contains("not in your models"):
            return .init(message: "\(model) is not in your models. Add it in Models.", action: .openModels)
        default:
            if e.status == nil, message != ChatSession.noAPI { return .init(message: message, action: .retry) }
            return .init(message: message, action: nil)
        }
    }

    /// "Needs 6 GB of memory; 2.9 GB free." (the alert's whole story is the line's hover). One wording for the error,
    /// the declined line ("Not run: needs …") and every window.
    static func memoryLine(_ r: LoadMemoryRefusal) -> String {
        "Needs \(memoryFigures(r))."
    }
    /// "6 GB of memory; 2.9 GB free": what the memory lines share.
    static func memoryFigures(_ r: LoadMemoryRefusal) -> String {
        "\(gigabytes(r.neededGB)) of memory; \(gigabytes(r.freeGB)) free"
    }
    /// One decimal, a whole number without it: 5.96 → "6 GB", 2.89 → "2.9 GB" (as Activity Monitor counts a GB).
    static func gigabytes(_ decimalGB: Double) -> String {
        let tenths = (MemoryDisplay.gib(decimalGB: decimalGB) * 10).rounded() / 10
        return (tenths == tenths.rounded() ? String(format: "%.0f", tenths) : String(format: "%.1f", tenths)) + " GB"
    }
    /// The service's codes for a model process that ended mid-run.
    static let crashCodes: Set<String> = ["worker_exit", "wire", "protocol"]
    /// Running past a memory refusal, named by the window's verb and what it costs: "Make Sound Anyway (slower)".
    static func anywayTitle(_ verb: String) -> String { "\(verb) Anyway (slower)" }
    /// What running past a memory refusal costs: the Anyway button's hover.
    static let anywayHelp = "Uses disk as memory; the Mac may slow down."
    /// Cancel in the memory alert: the run did not happen, and the window says so (without the refusal's figures).
    static let declined = OperationProblem(message: "Not run: not enough free memory.", action: .loadAnyway)
    /// Cancel in the memory alert, keeping the alert's figures: "Not run: needs 6 GB of memory; 2.9 GB free."
    static func declinedRun(_ refusal: LoadMemoryRefusal?) -> OperationProblem {
        guard let refusal else { return declined }
        return OperationProblem(message: "Not run: needs \(memoryFigures(refusal)).", action: .loadAnyway, detail: refusal.summary, refusal: refusal)
    }
}

/// A failure already worded for a window ("Couldn't read beach.wav."), the system's own sentence kept for the hover.
struct WordedError: LocalizedError, Equatable {
    let line: String
    let raw: String
    var errorDescription: String? { line }
    /// A file failure worded ("Couldn't save sound.wav."), else `line` with the error's own text as the hover.
    static func file(_ error: Error, verb: String, name: String) -> WordedError {
        WordedError(line: "Couldn\u{2019}t \(verb) \(name).", raw: (error as? WordedError)?.raw ?? error.localizedDescription)
    }
}

/// A file system's failure (NSCocoaErrorDomain, POSIX), worded for a window: "Couldn't read beach.wav." The system's
/// own sentence is kept for the hover, never shown as the line.
struct FileProblem {
    let line: String
    let raw: String
    init?(_ error: Error, verb: String = "read") {
        let e = error as NSError
        // File errors only (Cocoa's file range, POSIX): a decoding or other Cocoa error is not "couldn't read the file".
        let file = e.domain == NSCocoaErrorDomain && (NSFileErrorMinimum...NSFileErrorMaximum).contains(e.code)
        guard file || e.domain == NSPOSIXErrorDomain || e.domain == NSURLErrorDomain && e.code == NSURLErrorFileDoesNotExist else { return nil }
        let path = (e.userInfo[NSFilePathErrorKey] as? String) ?? (e.userInfo[NSURLErrorKey] as? URL)?.path
        let name = path.map { ($0 as NSString).lastPathComponent }
        let writing = (NSFileWriteUnknownError...NSFileWriteVolumeReadOnlyError).contains(e.code)
        let v = writing && verb == "read" ? "save" : verb
        line = "Couldn\u{2019}t \(v) \(name ?? "the file")."
        raw = e.localizedDescription
    }
}

/// A durable job the service reports cancelled.
struct JobCanceled: Error {}

/// A run that answered without what the tool keeps (no image, unreadable audio): its problem, as the window words it.
struct Undelivered: Error {
    let problem: OperationProblem
    init(_ problem: OperationProblem) { self.problem = problem }
}

/// One run of a tool: what it asked, and its progress and memory as the activity stream reports them.
struct ToolRun: Equatable {
    var model: String
    var operation: String
    var started = Date()
    var finished: Date?
    /// Stopped or failed: the last step seen stays, no time is reported.
    var cancelled = false
    var phase: String?
    var label: String?
    var step: Int?
    var total: Int?
    var unit: String?
    var memoryMB: Double?
    /// Only an MLX peak from this request's C2 timing is comparable with reference peaks.
    var peakMB: Double?
    /// The request as sent (its size, steps and seed are the result's).
    var input: JSONObject = [:]

    func seconds(now: Date) -> Double { (finished ?? now).timeIntervalSince(started) }
    var done: Bool { finished != nil && !cancelled }
    var fraction: Double? { step.flatMap { s in total.flatMap { $0 > 0 ? min(1, Double(s) / Double($0)) : nil } } }
    /// "Loading Stable Audio 3 Medium", "Waiting", "Step 3 of 8", "Separating layers · 2 of 4 layers", or `idle`
    /// before the runner reports anything. A tool whose idle line says what it does ("Removing background") keeps it
    /// before the count; the plain "Starting" gives way to "Step".
    func line(model: String, idle: String = "Starting") -> String {
        if phase == "loading" { return "Loading \(model)" }
        if phase == "queued" { return "Waiting" }
        let count = step.map { s in total.map { "\(s) of \($0)" } ?? "\(s)" }.map { c in unit.map { "\(c) \($0)" } ?? c }
        guard let count, total != 1 else { return label ?? idle }
        if let label { return "\(label) \u{00b7} \(count)" }
        return idle == "Starting" ? "Step \(count)" : "\(idle) \u{00b7} \(count)"
    }
}

@MainActor class ToolSession: ObservableObject {
    @Published var models: [ModelChoice]
    @Published var selected: String
    /// The chosen model's operations this tool offers, in the tool's order.
    @Published var operations: [OperationDescriptor] = []
    @Published var operation: String?
    /// The chosen operation's options, from its input schema.
    @Published var form: ToolForm?
    @Published var values: [String: FieldValue] = [:]
    @Published var loading = true
    @Published var run: ToolRun?
    /// The run in progress refines the result on show: no progress card covers it (Select's clicks).
    @Published var quietRun = false
    @Published var problem: OperationProblem? {
        didSet { if problem?.refusal != oldValue?.refusal { watchMemory() } }
    }
    /// The field a drag is over (its well lights up).
    @Published var dropTarget: String?
    /// The chosen model's voices; nil until it is loaded and has listed them.
    @Published var voices: [VoiceChoice]?
    /// The Advanced disclosure under the options is open (closed in every new window).
    @Published var advancedOpen = false
    /// Speak shows the voice-to-clone well because New Voice… was chosen (a model whose `speak` takes the clip itself,
    /// with no clone operation to switch to).
    @Published var newVoice = false
    /// Save As… (tests answer it): the file's name and folder suggested.
    var chooseSaveURL: (_ suggested: URL) -> URL? = { file in
        let panel = NSSavePanel()
        panel.nameFieldStringValue = file.lastPathComponent
        panel.directoryURL = file.deletingLastPathComponent()
        panel.canCreateDirectories = true
        return panel.runModal() == .OK ? panel.url : nil
    }
    /// Save All… (tests answer it): a folder.
    var chooseSaveFolder: () -> URL? = {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.canCreateDirectories = true; panel.prompt = "Save"
        return panel.runModal() == .OK ? panel.url : nil
    }
    /// Plays audio inputs and results (one at a time).
    let player = SoundPlayer()

    let client: () -> ArsenalClient?
    /// Where input files are staged for the service (`<support>/inbox`).
    let inbox: URL
    /// Where results are kept (~/Pictures/Arsenal, ~/Music/Arsenal).
    var outputDirectory: URL
    /// This window's own files: fetched results before they are saved, pasted images (removed at quit).
    let scratch: URL
    var openModels: () -> Void = {}
    /// Gets, approves or updates a model from the window (ToolCenter routes it to the Models table's own flow, with its
    /// size confirmation and code approval).
    /// Unset (a window built without the menu's flows): Open Models, so the button is never dead.
    var prepareModel: ((String) -> Void)?
    /// Operations already read for a model (ToolCenter keeps them): the window shows its form on its first frame and
    /// the API's answer refreshes it.
    var cachedOperations: (String) -> [OperationDescriptor]? = { _ in nil }

    /// Each model's settings: a field of the same name starts from its model's setting.
    var settings: [String: [String: JSONValue]] = [:]
    /// Values per model and operation, so switching back finds them as left.
    private var stored: [String: [String: FieldValue]] = [:]
    /// Every operation the chosen model declares (the window offers only its own).
    private(set) var declared: Set<String> = []
    /// What the last finished run was made from, for `isFresh`.
    private var lastMade: (model: String, operation: String, values: [String: FieldValue])?
    /// The options the run in progress was made from.
    private var runValues: [String: FieldValue] = [:]
    private var task: Task<Void, Never>?
    /// The durable job the run in progress is (its cancel path), when the operation runs as one.
    private var job: (client: ArsenalClient, cancel: String)?
    private var activity: Task<Void, Never>?
    /// Watches the Mac's memory while a memory refusal is on show (`watchMemory`).
    private var memoryWatch: Task<Void, Never>?
    private var preparing: Task<Void, Never>?
    private var listing: Task<Void, Never>?

    init(models: [ModelChoice], selected: String, client: @escaping () -> ArsenalClient?, inbox: URL, outputDirectory: URL, scratch: URL?) {
        self.models = models; self.selected = selected; self.client = client; self.inbox = inbox; self.outputDirectory = outputDirectory
        self.scratch = scratch ?? FileManager.default.temporaryDirectory.appendingPathComponent("Arsenal tools/\(UUID().uuidString)", isDirectory: true)
    }

    // MARK: What a tool says about itself

    /// The operations this tool offers of a model's declared ones.
    func offers(_ op: OperationDescriptor) -> Bool { true }
    /// Operations never shown as pills in any window: plumbing the windows drive themselves and destructive ones that
    /// only agents call (`ToolCatalog.behindTheScenes`, plus Ming's `delete_saved_voice`).
    static let neverPills: Set<String> = ToolCatalog.behindTheScenes.union(["delete_voice", "delete_saved_voice", "voices"])
    /// What the window lists of a model's operations: what it offers, less what is never a pill.
    final func shows(_ op: OperationDescriptor) -> Bool { offers(op) && !Self.neverPills.contains(op.name) }
    /// The window's task, its band's title ("Speak", "Make an Image", "Transcribe").
    var taskTitle: String { "Run" }
    /// What stops when a run ends unexpectedly: "Transcription", "Speech" ("Transcription stopped unexpectedly.").
    var stoppedNoun: String { Self.stoppedNoun(taskTitle) }
    static func stoppedNoun(_ task: String) -> String {
        switch task {
        case "Transcribe": return "Transcription"
        case "Speak": return "Speech"
        case "Make Music": return "The music"
        case "Make a Sound": return "The sound"
        case "Make an Image": return "The image"
        case "Remove Background": return "Background removal"
        case "Separate Audio": return "Separation"
        case "Translate": return "Translation"
        case "Read Document": return "Reading"
        case "Extract": return "Extraction"
        case "Select Objects": return "Selection"
        case "Find Objects": return "Finding"
        case "Depth": return "Depth"
        case "Upscale": return "Upscaling"
        case "Smooth Video": return "Smoothing"
        case "Make 3D": return "The 3D model"
        case "Search": return "Ranking"
        case "Chat": return "The reply"
        default: return "The run"
        }
    }
    /// The run button's words: the task's verb.
    var runVerb: String { descriptor.map { OperationPills.title($0.title) } ?? taskTitle }
    /// The pills' order: these names first, then the rest by title.
    var preferredOperations: [String] { [] }
    /// An operation's pill: its title without "Generate" (the verb is the run button's); a window may name its own.
    func pillTitle(_ op: OperationDescriptor) -> String { OperationPills.title(op.title) }
    /// The progress line before the runner reports anything.
    var idleLine: String { "Starting" }
    /// "take", "image", "cut": the help on a locked model pop-up names it.
    var resultNoun: String { "run" }
    /// Why a model offers nothing here.
    func noOperations(_ name: String) -> String { "\(name) declares no operations." }
    /// The seed of the result on show (Reuse last).
    var shownSeed: Int? { nil }
    /// The file of the result on show (Show in Finder, Copy).
    var shownFile: URL? { nil }
    /// Where a delivered result is recorded for the menu's Copy Last item (ToolCenter sets the app's).
    var lastOutput: LastOutput?
    /// A delivered result becomes the menu's Copy Last item: by default the file on show.
    func recordLastOutput(model: ModelChoice) {
        if let url = shownFile { lastOutput?.record(file: url, model: model.name) }
    }
    /// Something else the run needs before it can start (an image to cut out).
    var readyToRun: Bool { true }
    /// The main text is drawn above the options (a capability window draws it inside its chooser when it is one of
    /// several alternatives).
    var showsMain: Bool { true }
    /// The main text is one line (a search query) rather than an editor.
    var mainSingleLine: Bool { false }
    /// The options drawn under the main text (Remove Background draws its image as the well instead).
    var optionFields: [ToolField] { form?.fields.filter { !$0.isMain } ?? [] }
    /// The values the request is built from (Remove Background adds its image).
    func requestValues() -> [String: FieldValue] { values }
    /// The request as the window sends it, after the form built it (Select swaps its image for its session).
    func prepareInput(_ input: JSONObject, op: OperationDescriptor) -> JSONObject { input }
    /// Keeps what a run made (saves it, adds it to the window's history); throws when it made nothing usable.
    func deliver(_ result: OperationResult, op: OperationDescriptor, model: ModelChoice, input: JSONObject) throws {}
    /// After a run ends, finished, stopped or failed.
    func didFinish() {}
    /// After the chosen model's operations were read (the image tool reads its reference figures).
    func didPrepare() {}
    /// The chosen operation's form (a capability window names its own main text and option titles).
    func makeForm(_ op: OperationDescriptor) -> ToolForm { ToolForm(schema: op.inputSchema, operation: op.name, output: op.output) }
    /// Runs the operation as a durable job (`Prefer: respond-async`): its progress is polled and Stop cancels the work
    /// itself, not only the wait. For long file operations (video, 3D, separation).
    func runsAsJob(_ op: OperationDescriptor) -> Bool { false }

    // MARK: State

    var model: ModelChoice? { models.first { $0.id == selected } }
    var descriptor: OperationDescriptor? { operations.first { $0.name == operation } }
    var running: Bool { run != nil && run?.finished == nil }
    var blocked: OperationProblem? { model?.unavailable }
    /// Running again would make what the last run made (same model, operation and options, no random seed): the run
    /// button steps back from the primary blue. One rule for every tool.
    var isFresh: Bool {
        guard !running, let last = lastMade, last.model == selected, last.operation == operation else { return false }
        if form?.takesSeed == true, value("seed").text.trimmingCharacters(in: .whitespaces).isEmpty { return false }
        return last.values == requestValues()
    }
    /// Why the run button is off, for its hover; nil while it can run (or runs).
    var notRunnable: String? {
        guard !running, !canRun else { return nil }
        if let blocked { return blocked.message }
        if loading || form == nil || descriptor == nil { return "Reading \(model?.name ?? "the model")\u{2019}s options\u{2026}" }
        if let unsupported = form?.unsupported { return unsupported }
        if !readyToRun { return notReadyReason }
        if let main = form?.main, value(main.key).text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return Self.emptyMainReason(operation) }
        return nil
    }
    /// What the window still needs besides its text (a picture, a clip).
    var notReadyReason: String { "Add what to work on first" }
    static func emptyMainReason(_ op: String?) -> String {
        switch op {
        case "speak", "clone_voice", "design_voice": return "Type something to speak"
        case "generate_music", "generate_sound", "generate": return "Describe what to make"
        default: return "Type something first"
        }
    }
    var canRun: Bool {
        guard !running, !loading, model != nil, blocked == nil, let form, descriptor != nil, form.unsupported == nil, readyToRun else { return false }
        guard let main = form.main else { return true }
        return !value(main.key).text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    // MARK: Models

    /// The models the window offers, as the table has them now. The chosen one stays chosen while it is in the table.
    func update(models: [ModelChoice]) {
        let before = model
        if self.models != models { self.models = models }
        guard let now = model else {
            if let first = models.first(where: { $0.unavailable == nil }) ?? models.first { choose(first.id) }
            return
        }
        // Loaded since the last look: its voices can be listed now.
        if now.resident, before?.resident != true || voices == nil { listVoices() }
        if !now.resident, before?.resident == true { voices = nil }
    }

    /// Picks a model: its operations and options are read again, the main text carries over.
    func choose(_ id: String) {
        // A model that cannot run yet (not downloaded, approval needed) can be chosen: the window then shows its reason with
        // the action that fixes it (Download, Approve…).
        guard !running, models.contains(where: { $0.id == id }) else { return }
        if id == selected, !operations.isEmpty { return }
        keep()
        let text = form?.main.map { value($0.key).text } ?? ""
        selected = id; operations = []; declared = []; operation = nil; form = nil; values = [:]; voices = nil; problem = nil; newVoice = false
        prepare(carry: text)
    }

    /// Reads the chosen model's operations and settings; `prefer` picks the operation shown first.
    func prepare(carry text: String = "", prefer: String? = nil) {
        preparing?.cancel()
        let id = selected
        var seeded = false
        if operations.isEmpty, let cached = cachedOperations(id) {
            let ops = ordered(cached.filter(shows))
            if !ops.isEmpty {
                operations = ops; declared = Set(cached.map(\.name))
                select(primary(ops, prefer: prefer), carry: text)
                seeded = true
            }
        }
        // Drawn from cached operations, the window is usable at once; otherwise it waits for the API's answer.
        loading = !seeded
        preparing = Task {
            defer { if selected == id { loading = false } }
            guard let client = client() else { problem = .init(message: ChatSession.noAPI, action: .retry); return }
            do {
                let all = try await client.operations(id)
                let ops = ordered(all.filter(shows))
                if let values = try? await client.settings(id) {
                    settings[id] = Dictionary(values.map { ($0.id, $0.value) }, uniquingKeysWith: { a, _ in a })
                }
                guard !Task.isCancelled, selected == id else { return }
                // A form drawn from cached operations takes the model's settings where it still shows the defaults.
                if seeded, let form, let modelSettings = settings[id] {
                    for f in form.fields where values[f.key] == form.initialValue(f, setting: nil) {
                        if let v = form.initialValue(f, settings: modelSettings) { values[f.key] = v }
                    }
                }
                operations = ops; declared = Set(all.map(\.name))
                guard !ops.isEmpty else { problem = .init(message: noOperations(model?.name ?? id), action: nil); return }
                problem = nil
                let keep = operation.flatMap { name in ops.contains { $0.name == name } ? name : nil }
                select(keep ?? primary(ops, prefer: prefer), carry: text)
                if model?.resident == true { listVoices() }
                didPrepare()
            } catch is CancellationError {
            } catch {
                guard selected == id else { return }
                problem = .init(message: OperationProblem.wording(error, model: model?.name ?? id), action: .retry)
            }
        }
    }

    /// The tool's preferred operations first, then the rest in the order the model declares them.
    func ordered(_ ops: [OperationDescriptor]) -> [OperationDescriptor] {
        let first = preferredOperations
        return ops.enumerated().sorted { (first.firstIndex(of: $0.element.name) ?? 99, $0.offset) < (first.firstIndex(of: $1.element.name) ?? 99, $1.offset) }
            .map(\.element)
    }

    /// The named one, else one annotated primary, else the tool's first preference, else the first.
    func primary(_ ops: [OperationDescriptor], prefer: String? = nil) -> String? {
        if let prefer, ops.contains(where: { $0.name == prefer }) { return prefer }
        if let annotated = ops.first(where: { $0.annotations["primary"]?.bool == true }) { return annotated.name }
        for name in preferredOperations where ops.contains(where: { $0.name == name }) { return name }
        return ops.first?.name
    }

    // MARK: Options

    func select(_ name: String?, carry text: String? = nil) {
        guard let name, let op = operations.first(where: { $0.name == name }), !running else { return }
        let carried = text ?? form?.main.map { value($0.key).text } ?? ""
        keep()
        // Another operation: the last one's failure (a save, a declined load) and New Voice… do not follow it.
        if name != operation { problem = nil }
        if name != "speak" { newVoice = false }
        operation = name
        let form = makeForm(op)
        self.form = form
        var values = stored[storeKey] ?? [:]
        let modelSettings = settings[selected] ?? [:]
        for f in form.fields where values[f.key] == nil {
            values[f.key] = form.initialValue(f, settings: modelSettings) ?? form.initialValue(f, setting: nil)
        }
        if let main = form.main, values[main.key]?.text.isEmpty ?? true, !carried.isEmpty { values[main.key] = .text(carried) }
        self.values = values
        if problem?.field != nil { problem = nil }
    }

    /// The model's settings changed: the options they name start from them again (a cleared seed is random again).
    func apply(settings list: [SettingState]) {
        let settings = Dictionary(list.map { ($0.id, $0.value) }, uniquingKeysWith: { a, _ in a })
        self.settings[selected, default: [:]].merge(settings) { _, b in b }
        guard let form else { return }
        for f in form.fields { if let v = form.initialValue(f, settings: settings) { values[f.key] = v } }
    }

    private var storeKey: String { "\(selected)/\(operation ?? "")" }
    private func keep() { if operation != nil { stored[storeKey] = values } }

    func value(_ key: String) -> FieldValue { values[key] ?? .text("") }
    func set(_ key: String, _ value: FieldValue) {
        values[key] = value
        if problem?.field == key { problem = nil }
    }

    /// Files dropped or chosen for a field: only those it accepts, up to its limit; the first replaces a single file.
    func addFiles(_ urls: [URL], to key: String) {
        guard let field = form?.fields.first(where: { $0.key == key }) else { return }
        let accept: FileAccept, multiple: Bool, most: Int
        switch field.control {
        case .generic(.file(let a, let m, let n, _)): accept = a; multiple = m; most = n
        case .reference: accept = FileAccept(mimes: ["audio/*"]); multiple = false; most = 1
        case .images(let n): accept = FileAccept(mimes: ["image/*"]); multiple = true; most = n
        default: return
        }
        let accepted = urls.filter { accept.accepts($0) || (accept.images && ToolFiles.isImageFile($0)) }
        guard !accepted.isEmpty else {
            var message = "\(field.title) takes \(accept.title.lowercased()) files."
            if case .reference = field.control { message = "A voice to clone takes a WAV, FLAC, MP3 or M4A clip." }
            problem = .init(message: message, action: nil, field: key); return
        }
        let current = value(key).files
        set(key, .files(multiple ? Array((current + accepted.filter { !current.contains($0) }).prefix(most)) : [accepted[0]]))
    }
    func removeFile(_ url: URL, from key: String) { set(key, .files(value(key).files.filter { $0 != url })) }

    /// Lists the chosen model's voices through its own `voices` operation, once it is loaded (listing never loads it).
    func listVoices() {
        guard declared.contains("voices"), form?.fields.contains(where: { $0.control == .voice }) ?? false else { return }
        listing?.cancel()
        let id = selected
        listing = Task {
            guard let client = client() else { return }
            guard let (data, _) = try? await client.send("POST", "/arsenal/v1/models/\(id)/ops/voices", json: [String: Any](), timeout: 60),
                case .object(let object)? = try? JSONDecoder().decode(JSONValue.self, from: data)
            else { return }
            guard !Task.isCancelled, selected == id else { return }
            voices = VoiceChoice.list(object)
        }
    }

    // MARK: Running

    func start() {
        guard canRun, let op = descriptor, let form, let model else { return }
        var all = requestValues()
        // A clip to clone that is not on show (Speak without New Voice…) is never sent.
        for f in form.fields where !showsField(f) {
            all[f.key] = .files([])
            all[ToolForm.transcriptKey(f.key)] = nil
        }
        let seed = form.takesSeed && (all["seed"]?.text ?? "").trimmingCharacters(in: .whitespaces).isEmpty ? Int.random(in: 0...Int(UInt32.max)) : nil
        // Each file is staged as its own field accepts it.
        var accepts: [URL: FileAccept] = [:]
        for f in form.fields {
            switch f.control {
            case .generic(.file(let accept, _, _, _)): for url in all[f.key]?.files ?? [] where accepts[url] == nil { accepts[url] = accept }
            case .reference: for url in all[f.key]?.files ?? [] { accepts[url] = FileAccept(mimes: ["audio/*"]) }
            default: break
            }
        }
        var staged: [URL] = []
        var input: JSONObject
        do {
            input = try form.input(
                all, seed: seed,
                stage: { url in
                    let (folder, path) = try ToolFiles.stage(url, accept: accepts[url] ?? FileAccept(mimes: []), inbox: self.inbox)
                    staged.append(folder)
                    return path
                })
            input = prepareInput(input, op: op)
            try OperationSchemaValidator.validate(.object(input), schema: .object(op.inputSchema))
        } catch let p as OperationForm.Problem {
            for folder in staged { try? FileManager.default.removeItem(at: folder) }
            problem = .init(message: p.message, action: nil, field: fieldProblemKey(p.key)); return
        } catch {
            for folder in staged { try? FileManager.default.removeItem(at: folder) }
            if let w = error as? WordedError { problem = .init(message: w.line, action: nil, detail: w.raw); return }
            if let file = FileProblem(error) { problem = .init(message: file.line, action: nil, detail: file.raw); return }
            let raw = TableController.message(error)
            // The schema's alternatives (a saved voice or a clip): said as a person would, the validator's words in the hover.
            if raw.contains("no schema alternative matched") {
                let speech = Self.speechOperations.contains(op.name) && !newVoiceChoices.isEmpty
                problem = .init(message: speech ? "Choose a voice, or make one with New Voice\u{2026}" : "Fill in one of the inputs this model needs.", action: nil, detail: raw)
                return
            }
            problem = .init(message: TableController.message(error), action: nil); return
        }
        problem = nil
        player.stop()
        run = ToolRun(model: model.id, operation: op.name, input: input)
        lastMade = nil; runValues = all
        let force = LoadMemoryContext.force
        let declines = MemoryLoadPresenter.DeclineRecord()
        task = Task { await MemoryLoadPresenter.$declines.withValue(declines) { await perform(op, model: model, input: input, staged: staged, force: force, declines: declines) } }
    }
    /// The field a problem is shown under (a tool may show some elsewhere).
    func fieldProblemKey(_ key: String) -> String? { key }

    func stop() {
        guard running else { return }
        if let job {
            // The job keeps running on the service unless it is cancelled there.
            self.job = nil
            Task { _ = try? await job.client.send("POST", job.cancel, json: [String: Any]()) }
        }
        task?.cancel(); task = nil
        run?.cancelled = true
        finish(nil)
    }

    /// The window closed: the run, the activity stream and reads stop; the results stay for the next opening.
    func close() {
        stop()
        player.stop()
        preparing?.cancel(); preparing = nil
        listing?.cancel(); listing = nil
        activity?.cancel(); activity = nil
        memoryWatch?.cancel(); memoryWatch = nil
    }
    /// Quit: the window's own files go too (saved results stay in their folder).
    func discard() {
        close()
        try? FileManager.default.removeItem(at: scratch)
    }
    var hasRunningWork: Bool { task != nil || activity != nil }

    private func perform(_ op: OperationDescriptor, model: ModelChoice, input: JSONObject, staged: [URL], force: Bool, declines: MemoryLoadPresenter.DeclineRecord) async {
        defer { for folder in staged { try? FileManager.default.removeItem(at: folder) } }
        guard let client = client() else { finish(.init(message: ChatSession.noAPI, action: .retry)); return }
        watchActivity(client, model: model.id, op: op.name)
        do {
            let (data, response) =
                runsAsJob(op)
                ? try await runJob(client, op: op, model: model, input: input, force: force)
                : try await MemoryLoadPresenter.perform(force: force) { approved in
                    try await LoadMemoryContext.$force.withValue(approved) {
                        try await client.send(
                            "POST", "/arsenal/v1/models/\(model.id)/ops/\(op.name)", json: input,
                            timeout: (op.timeout ?? 3600) + 30, accept: "application/json")
                    }
                }
            if Task.isCancelled { return }
            var result = try await ToolFiles.result(
                op: op.name, data: data, mime: response.value(forHTTPHeaderField: "Content-Type") ?? response.mimeType, client: client,
                into: scratch.appendingPathComponent("results", isDirectory: true))
            if Task.isCancelled { return }
            run?.finished = Date()
            result.seconds = run?.seconds(now: Date()) ?? 0
            try deliver(result, op: op, model: model, input: input)
            lastMade = (model.id, op.name, runValues)
            recordLastOutput(model: model)
            finish(nil)
        } catch is CancellationError {
            run?.cancelled = true
            // Not Stop (which cancels this task): Cancel in the memory alert. The window says the run did not happen.
            finish(Task.isCancelled ? nil : .declinedRun(declines.refusal)); return
        } catch is JobCanceled {
            run?.cancelled = true; finish(nil); return
        } catch let undelivered as Undelivered {
            finish(undelivered.problem)
        } catch {
            if Task.isCancelled { return }
            finish(OperationProblem.from(error, model: model.name, task: stoppedNoun))
        }
    }

    /// A durable job: the model is loaded first when it is not (so a shortage asks before anything starts, as a direct
    /// run does), then the job is created and polled until it ends; its reply is the job's output.
    private func runJob(_ client: ArsenalClient, op: OperationDescriptor, model: ModelChoice, input: JSONObject, force: Bool) async throws -> (Data, HTTPURLResponse) {
        if !model.resident {
            _ = try await MemoryLoadPresenter.perform(force: force) { approved in
                try await LoadMemoryContext.$force.withValue(approved) {
                    try await client.send("POST", "/arsenal/v1/load", json: ["id": model.id], timeout: 3600)
                }
            }
        }
        try Task.checkCancellation()
        let (created, _) = try await client.send(
            "POST", "/arsenal/v1/models/\(model.id)/ops/\(op.name)", json: input, timeout: 120, accept: "application/json", headers: ["Prefer": "respond-async"])
        guard let first = try? JSONDecoder().decode(JSONValue.self, from: created).object, let id = first["id"]?.string else {
            throw ToolError("Arsenal did not start the job.")
        }
        job = (client, "/arsenal/v1/jobs/\(id)/cancel")
        defer { job = nil }
        var wait: Duration = .milliseconds(250)
        while true {
            try await Task.sleep(for: wait)
            wait = min(wait * 2, .seconds(2))
            let (data, response) = try await client.send("GET", "/arsenal/v1/jobs/\(id)", timeout: 60)
            guard let record = try? JSONDecoder().decode(JSONValue.self, from: data).object else { continue }
            if let progress = record["progress"]?.object, var run, run.finished == nil {
                if let s = progress["step"]?.number.flatMap(\.roundedInt) { run.step = s }
                if let t = progress["total"]?.number.flatMap(\.roundedInt) { run.total = t }
                if let label = progress["label"]?.string, !label.isEmpty { run.label = String(label.prefix(80)) }
                if let phase = progress["phase"]?.string { run.phase = phase }
                if run != self.run { self.run = run }
            }
            switch record["status"]?.string {
            case "succeeded":
                let output = record["output"] ?? .object([:])
                return (try JSONEncoder().encode(output), response)
            case "failed":
                // The same error object a direct request answers with: code and details (memory figures) survive.
                if let failure = record["failure"]?.object, let body = try? WireJSON.encode(failure) {
                    throw ToolError.from(status: failure["status"]?.number.flatMap(\.roundedInt) ?? 500, body: body)
                }
                throw ToolError(record["error"]?.string ?? "\(model.name) failed.")
            // Cancelled on the service (by another client or Stop): ends quietly, as Stop does.
            case "canceled": throw JobCanceled()
            default: continue
            }
        }
    }

    func finish(_ problem: OperationProblem?) {
        quietRun = false
        activity?.cancel(); activity = nil
        job = nil
        task = nil
        if run?.finished == nil { run?.finished = Date() }
        // A failed run has no figures, as a stopped one has none.
        if problem != nil { run?.cancelled = true }
        self.problem = problem
        didFinish()
    }

    private func watchActivity(_ client: ArsenalClient, model id: String, op: String) {
        activity?.cancel()
        activity = Task { [weak self] in
            do {
                for try await data in client.events("GET", "/arsenal/v1/activity/stream?machine=0") {
                    guard let snapshot = eventObject(data) else { continue }
                    guard let self, !Task.isCancelled, var run = self.run, run.finished == nil else { return }
                    guard let m = (snapshot["models"] as? [[String: Any]])?.first(where: { $0["id"] as? String == id }) else { continue }
                    if let mb = (m["memory_mb"] as? NSNumber)?.doubleValue { run.memoryMB = max(run.memoryMB ?? 0, mb) }
                    if let mine = (m["requests"] as? [[String: Any]] ?? []).first(where: {
                        $0["requester"] as? String == ArsenalClient.clientName && $0["op"] as? String == op
                    }) {
                        run.phase = mine["phase"] as? String
                        run.label = (mine["label"] as? String).flatMap { $0.isEmpty ? nil : String($0.prefix(80)) }
                        run.unit = (mine["unit"] as? String).flatMap { $0.isEmpty ? nil : String($0.prefix(24)) }
                        if let s = (mine["step"] as? NSNumber)?.intValue { run.step = s }
                        if let t = (mine["total"] as? NSNumber)?.intValue { run.total = t }
                        if let peak = ((mine["timing"] as? [String: Any])?["peak_mb"] as? NSNumber)?.doubleValue, peak.isFinite, peak >= 0 { run.peakMB = peak }
                    }
                    if run != self.run { self.run = run }
                }
            } catch {}
        }
    }

    /// A memory refusal on show: the Mac's memory in use is watched, and once enough is freed for the model (another
    /// app quit, idle models unloaded) the refusal goes and the button is the window's own verb again, not "Anyway".
    private func watchMemory() {
        memoryWatch?.cancel(); memoryWatch = nil
        guard let refusal = problem?.refusal, problem?.action == .loadAnyway, let client = client() else { return }
        memoryWatch = Task { [weak self] in
            // Measured from the Mac's memory in use when the load was refused, else from the first sample after it.
            var baseline = refusal.usedMB
            do {
                for try await data in client.events("GET", "/arsenal/v1/activity/stream") {
                    guard let machine = eventObject(data)?["machine"] as? [String: Any], let used = (machine["memory_used_mb"] as? NSNumber)?.doubleValue else { continue }
                    guard let self, !Task.isCancelled else { return }
                    let start = baseline ?? used
                    baseline = start
                    if self.memoryFreed(refusal, baselineMB: start, usedMB: used) { return }
                }
            } catch {}
        }
    }
    /// Memory in use fell from `baselineMB` (when the refusal was shown) to `usedMB`: if the model fits now, the refusal
    /// goes. Returns whether it went.
    @discardableResult func memoryFreed(_ refusal: LoadMemoryRefusal, baselineMB: Double, usedMB: Double) -> Bool {
        guard problem?.refusal == refusal, refusal.fits(afterFreeingMB: baselineMB - usedMB) else { return false }
        problem = nil
        return true
    }

    func performAction(_ action: OperationProblem.Action) {
        switch action {
        case .openModels, .openMemory: openModels()
        case .loadAnyway: problem = nil; LoadMemoryContext.$force.withValue(true) { start() }
        case .retry: problem = nil; if operations.isEmpty { prepare() } else { start() }
        case .download, .approve, .update:
            if let prepareModel { prepareModel(selected) } else { openModels() }
        }
    }

    // MARK: Advanced options

    /// Sampler knobs, folded under Advanced in every window: a person tunes them rarely, and they read as risky.
    static let samplerKeys: Set<String> = [
        "max_tokens", "max_new_tokens", "max_length", "repetition_penalty", "temperature", "top_k", "top_p", "min_p", "typical_p", "presence_penalty",
        "frequency_penalty"
    ]
    /// Diffusion knobs: a detail in every window (images, sound, speech, Separate Audio), folded with the samplers.
    static let diffusionKeys: Set<String> = [
        "steps", "num_steps", "inference_steps", "num_inference_steps", "guidance", "guidance_scale", "cfg", "cfg_scale", "cfg_strength"
    ]
    static let speechOperations: Set<String> = ["speak", "clone_voice", "design_voice"]
    /// The option keys folded under Advanced for an operation, by name (never per model): samplers, diffusion knobs
    /// (steps, guidance) and the seed, in every window.
    func advancedKeys(_ op: String?) -> Set<String> {
        // The seed means nothing until a result is worth repeating: folded everywhere, Reuse last beside it.
        Self.samplerKeys.union(Self.diffusionKeys).union(["seed"])
    }
    /// Raw JSON inputs (Remove Background's visual exemplars) are expert options too.
    func isAdvanced(_ f: ToolField) -> Bool {
        if case .generic(.json) = f.control { return true }
        return advancedKeys(operation).contains(f.key)
    }

    // MARK: Voices

    /// Operations that make a voice rather than speak with one; a window that speaks reaches them from its Voice picker.
    static let voiceMakers = ["clone_voice", "design_voice"]
    /// Keys of a clip to clone inside `speak` itself.
    static let cloneKeys: Set<String> = ["reference", "reference_audio", "ref_audio", "speaker_wav", "voice_clip"]
    var speaks: Bool { operations.contains { $0.name == "speak" } }
    /// The pills: in a window that speaks, cloning and designing are Voice picker items, not modes.
    var pillOperations: [OperationDescriptor] { speaks ? operations.filter { !Self.voiceMakers.contains($0.name) } : operations }
    /// The Voice picker's ways to a new voice: (title, operation; nil = show the clip well inside Speak).
    var newVoiceChoices: [(title: String, operation: String?)] {
        guard speaks else { return [] }
        let makers = operations.filter { Self.voiceMakers.contains($0.name) }
        if makers.count > 1 { return makers.map { ($0.name == "clone_voice" ? "Clone a Voice\u{2026}" : "Design a Voice\u{2026}", $0.name) } }
        if let one = makers.first { return [("New Voice\u{2026}", one.name)] }
        let speakForm = operation == "speak" ? form : nil
        if speakForm?.fields.contains(where: isCloneField) == true { return [("New Voice\u{2026}", nil)] }
        return []
    }
    /// A voice is being made (cloned or designed), not chosen.
    var makingVoice: Bool { newVoice && operation == "speak" || Self.voiceMakers.contains(operation ?? "") && speaks }
    func isCloneField(_ f: ToolField) -> Bool {
        if case .reference = f.control { return true }
        return Self.cloneKeys.contains(f.key)
    }
    /// Drawn in the options: the clip to clone shows in Speak only after New Voice….
    func showsField(_ f: ToolField) -> Bool { !(operation == "speak" && !newVoice && isCloneField(f)) }
    func chooseNewVoice(_ op: String?) {
        guard !running else { return }
        if let op { select(op) } else { newVoice = true }
    }
    /// A voice picked: back to Speak with it (the clip to clone, unused now, is cleared).
    func chooseVoice(_ id: String) {
        guard !running else { return }
        if operation != "speak", speaks { select("speak") }
        newVoice = false
        for f in form?.fields ?? [] where isCloneField(f) { values[f.key] = .files([]) }
        if let key = form?.fields.first(where: { $0.control == .voice })?.key { set(key, .text(id)) }
    }

    // MARK: Results

    func revealInFinder() { ToolFiles.reveal(shownFile, folder: outputDirectory) }

    /// Copy what is shown; a failure is a line.
    func copy(_ url: URL, to pasteboard: NSPasteboard = .general) {
        do { try ToolFiles.copy(url, to: pasteboard) } catch {
            problem = .init(message: "Couldn\u{2019}t copy \(url.lastPathComponent).", action: nil, detail: error.localizedDescription)
        }
    }

    /// Save As…: a copy where the user chooses, the file's own name and folder suggested.
    func saveAs(_ url: URL) {
        guard let target = chooseSaveURL(url), target.standardizedFileURL != url.standardizedFileURL else { return }
        do { try Self.place(url, at: target) } catch {
            problem = .init(message: "Couldn\u{2019}t save \(target.lastPathComponent).", action: nil, detail: error.localizedDescription)
        }
    }
    /// Save All…: every file into one folder, numbered in their order ("1 Vocals.wav", "2 Drums.wav").
    func saveAll(_ files: [(url: URL, name: String)]) {
        guard !files.isEmpty, let folder = chooseSaveFolder() else { return }
        do {
            for (i, file) in files.enumerated() { try Self.place(file.url, at: folder.appendingPathComponent("\(i + 1) \(file.name)")) }
        } catch {
            problem = .init(message: "Couldn\u{2019}t save the files.", action: nil, detail: error.localizedDescription)
        }
    }
    /// A copy at `target`; a file already there is replaced only once the copy exists (a failed copy loses nothing).
    static func place(_ source: URL, at target: URL) throws {
        let fm = FileManager.default
        guard fm.fileExists(atPath: target.path) else { try fm.copyItem(at: source, to: target); return }
        let temp = target.deletingLastPathComponent().appendingPathComponent(".\(UUID().uuidString)-\(target.lastPathComponent)")
        try fm.copyItem(at: source, to: temp)
        do { _ = try fm.replaceItemAt(target, withItemAt: temp) } catch {
            try? fm.removeItem(at: temp)
            throw error
        }
    }
    /// "Music › Arsenal": the folder Show in Finder opens (the shown result's, else the window's).
    var revealFolderName: String {
        let folder = shownFile?.deletingLastPathComponent() ?? outputDirectory
        return [folder.deletingLastPathComponent().lastPathComponent, folder.lastPathComponent].filter { !$0.isEmpty && $0 != "/" }.joined(separator: " \u{203a} ")
    }
    /// A model that is not ready yet (not downloaded, approval or an update needed): the run button becomes that action.
    var preparesInstead: OperationProblem? {
        guard !running, let b = blocked, let a = b.action, [.download, .approve, .update].contains(a) else { return nil }
        return b
    }
    /// A failure on show whose fix is running again: the run button becomes it ("Try Again" after a crash, "Speak Anyway
    /// (slower)" after a memory refusal), so no second bright button retries the run that just failed.
    var rerunInstead: OperationProblem.Action? {
        guard !running, canRun, let p = problem, p.field == nil, let a = p.action, a == .retry || a == .loadAnyway else { return nil }
        return a
    }
    /// The run button's words now: the window's verb, or the rerun it became.
    func runButtonTitle(_ verb: String) -> String {
        switch rerunInstead {
        case .loadAnyway: return OperationProblem.anywayTitle(verb)
        case .retry: return OperationProblem.Action.retry.rawValue
        default: return verb
        }
    }
    /// The run button's click: a run, or the rerun it became (past the memory refusal).
    func runButtonAction() {
        if let a = rerunInstead { performAction(a) } else { start() }
    }
    /// A failure is on show whose action is the run button's own (Try Again beside Speak).
    func duplicatesRun(_ p: OperationProblem) -> Bool { p == problem && rerunInstead != nil }
    /// A problem as the window shows it: running past a memory refusal is named by the window's verb.
    func worded(_ p: OperationProblem) -> OperationProblem {
        guard p.action == .loadAnyway, p.actionTitle == nil else { return p }
        var q = p; q.actionTitle = OperationProblem.anywayTitle(runVerb); return q
    }

    // MARK: Figures

    /// Memory of this run, named as the Models table names it: the request's MLX peak once the runner reports it, else
    /// the worker's largest sampled footprint; after a run that reported neither, it says so rather than a bare dash.
    func memoryFigure(reference: String? = nil) -> Figure {
        let r = run
        let value = FigureText.gigabytes(mb: r?.peakMB ?? r?.memoryMB) ?? (r?.finished != nil ? "not reported" : nil)
        let help =
            r?.peakMB != nil
            ? "The most memory the model used while making this \(resultNoun)."
            : r?.memoryMB != nil
                ? "The most memory the model\u{2019}s process used while making this \(resultNoun), the loaded model included."
                : r?.finished != nil
                    ? "The service reported no memory for this \(resultNoun): it ended before the first reading."
                    : "The memory this \(resultNoun) peaks at, once one runs."
        return Figure(title: "Peak memory", value: value, reference: r?.peakMB == nil ? nil : reference, help: help, width: 104)
    }

    /// Time by this window's clock, from the run to its saved result, and memory.
    func figures() -> [Figure] {
        [
            Figure(
                title: "Time", value: FigureText.seconds(run?.done == true ? run?.seconds(now: Date()) : nil), reference: nil,
                help: "From the start of this \(resultNoun) to its result, as this window measured it.", width: 84),
            memoryFigure()
        ]
    }

    /// Render harness and tests: a state without the API.
    func previewOptions(operations: [OperationDescriptor], selected op: String?, values: [String: FieldValue], settings: [String: JSONValue] = [:]) {
        self.operations = ordered(operations.filter(shows)); loading = false; declared = Set(operations.map(\.name))
        self.settings[selected] = settings
        select(op ?? primary(self.operations))
        for (k, v) in values { self.values[k] = v }
    }
}
