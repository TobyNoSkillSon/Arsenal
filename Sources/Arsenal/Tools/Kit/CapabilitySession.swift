// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ImageIO
import ArsenalCore
import UniformTypeIdentifiers

// A capability window's state, the same for every one of them (ToolCatalog): the models that declare its operations,
// the chosen operation's options from its schema, what the run works on (the subject: an image, a clip, a video, a
// document, or several, dropped, chosen or pasted into one well that routes each file to the field that takes it),
// one run at a time, and this window's results. Every file a run returns is saved as it arrives (images to Pictures,
// sound to Music, video to Movies, the rest to Documents); a window shows a result through its own view.

/// One result of a capability window.
struct Outcome: Identifiable, Equatable {
    let id = UUID()
    var model: String
    var modelName: String
    var operation: String
    var operationTitle: String
    /// What the run worked on: the subject's files, as the user gave them.
    var inputs: [URL]
    /// The main text it was given (a query, the text to translate).
    var text: String
    /// The options as they were (Run steps back from blue while nothing differs).
    var values: [String: FieldValue]
    /// The request as sent (paths staged, seed chosen).
    var input: JSONObject
    /// The reply without its file list.
    var reply: JSONObject
    /// The returned files, saved where results of their type are kept.
    var files: [ResultFile]
    var seconds: Double

    /// The first returned file of this type ("image/", "audio/", "video/", "model/").
    func file(_ prefix: String) -> ResultFile? { files.first { $0.mime.hasPrefix(prefix) } }
}

@MainActor class CapabilitySession: ToolSession {
    let tool: ToolKind
    @Published private(set) var outcomes: [Outcome] = []
    @Published var shown: UUID?
    /// A file is dragged over the window's drop places (the well and the result light up).
    @Published var dropTargeted = false
    /// What a click on the picture places (a point to include or exclude, or a box).
    @Published var promptMode: PromptMode = .include
    /// The option chosen in each chooser, by operation and group.
    @Published var chosen: [String: Int] = [:]
    /// The options the run in progress was made from and the subject it worked on.
    private var runInputs: [URL] = []

    /// The open panel (tests answer it).
    var chooseFiles: (_ accept: [FileAccept], _ multiple: Bool) -> [URL] = { accept, multiple in
        let panel = NSOpenPanel()
        let types = accept.flatMap(\.mimes).compactMap { m -> UTType? in
            UTType(mimeType: m) ?? (m.hasPrefix("image/") ? .image : m.hasPrefix("audio/") ? .audio : m.hasPrefix("video/") ? .movie : nil)
        }
        if !types.isEmpty { panel.allowedContentTypes = types }
        panel.allowsMultipleSelection = multiple
        return panel.runModal() == .OK ? panel.urls : []
    }
    /// Where Copy writes (tests use their own pasteboard).
    var copyPasteboard: NSPasteboard = .general

    init(tool: ToolKind, models: [ModelChoice], selected: String, client: @escaping () -> ArsenalClient?, inbox: URL, outputDirectory: URL? = nil, scratch: URL? = nil) {
        self.tool = tool
        super.init(models: models, selected: selected, client: client, inbox: inbox, outputDirectory: outputDirectory ?? ToolCatalog.folder(tool), scratch: scratch)
    }

    // MARK: What the window says about itself

    override func offers(_ op: OperationDescriptor) -> Bool { ToolCatalog.operations(tool).contains(op.name) }
    override var preferredOperations: [String] { ToolCatalog.operations(tool) }
    override var resultNoun: String { "result" }
    override func noOperations(_ name: String) -> String { "\(name) declares nothing \(tool.windowTitle) can run." }
    override var shownFile: URL? { current?.files.first?.url }
    override var shownSeed: Int? { current?.input["seed"]?.number.flatMap(\.roundedInt) }

    /// The keys a window takes as its main text, with their placeholders (none: every text is an option).
    func mainKeys(_ op: OperationDescriptor) -> [String: String] { [:] }
    /// The window's own names for its options.
    func titles(_ op: OperationDescriptor) -> [String: String] { [:] }
    /// Inputs the window manages itself (a session id, frame prompt updates): never shown, never typed.
    var managedKeys: Set<String> { [] }
    /// The run button's word for an operation.
    func runTitle(_ op: OperationDescriptor?) -> String { op.map { OperationPills.title($0.title) } ?? tool.windowTitle }
    override var taskTitle: String { tool.windowTitle }
    override var runVerb: String { runTitle(descriptor) }
    override var notReadyReason: String { subject.isEmpty ? "Add \(subjectNoun) first" : "Fill in what the run needs first" }

    /// The window's own hovers for its options, in plain words (what a range is written as, which way a slider goes).
    func helps(_ op: OperationDescriptor) -> [String: String] { [:] }

    override func makeForm(_ op: OperationDescriptor) -> ToolForm {
        var form = ToolForm(schema: op.inputSchema, operation: op.name, output: op.output, main: mainKeys(op), titles: titles(op))
        let helps = helps(op)
        for i in form.fields.indices { if let help = helps[form.fields[i].key] { form.fields[i].help = help } }
        return form
    }
    /// Files come back from a durable job (progress polled, Stop cancels it) when the operation streams its progress or
    /// may run for ten minutes or more (TRELLIS, SeedVR2, RIFE: 3600 s), rather than one request held open that long.
    override func runsAsJob(_ op: OperationDescriptor) -> Bool { op.returnsFiles && (op.stream || Self.isLong(op)) }
    /// The service allows the operation ten minutes or more.
    static func isLong(_ op: OperationDescriptor) -> Bool { (op.timeout ?? 0) >= 600 }

    // MARK: The subject

    /// The operation's file fields: what the run works on, shown as the window's one well.
    var subjectFields: [ToolField] { fileFields.filter { !hidden($0.key) } }
    /// Every file field of the operation, chosen or not.
    var fileFields: [ToolField] {
        form?.fields.filter { f in
            if case .generic(.file) = f.control, !managedKeys.contains(f.key) { return true }; return false
        } ?? []
    }
    /// Alternatives the window offers as one chooser (OptionGroups); none by default.
    func groups(_ op: OperationDescriptor) -> [OptionGroup] { [] }
    /// A key the window hides outside its choosers (Smooth Video's positions without two frames).
    func hiddenOutsideGroups(_ key: String) -> Bool { false }
    override var showsMain: Bool {
        guard let main = form?.main?.key else { return true }
        return !hidden(main) && !groupedKeys.contains(main)
    }
    /// Values of options not chosen are not sent (nor compared for Run's state).
    override func requestValues() -> [String: FieldValue] {
        var all = values
        for (key, value) in values where hidden(key) {
            switch value {
            case .files: all[key] = .files([])
            case .text: all[key] = .text("")
            case .choice: all[key] = .choice(nil)
            case .toggle: all[key] = .toggle(false)
            }
        }
        return all
    }
    /// What every file field takes, chosen or not (a paste or drop may choose the option that takes it).
    var allAccepts: [FileAccept] {
        fileFields.compactMap { f in
            if case .generic(.file(let accept, _, _, _)) = f.control { return accept }; return nil
        }
    }
    var subjectAccepts: [FileAccept] {
        subjectFields.compactMap { f in
            if case .generic(.file(let accept, _, _, _)) = f.control { return accept }; return nil
        }
    }
    /// The files in the well, in the order given.
    var subject: [URL] { subjectFields.flatMap { value($0.key).files } }
    /// Points and boxes drawn over the subject.
    var spatialFields: [ToolField] { form?.fields.filter { $0.isSpatial && !managedKeys.contains($0.key) } ?? [] }
    override var optionFields: [ToolField] {
        let subjects = Set(fileFields.map(\.key)), grouped = groupedKeys
        return form?.fields.filter {
            !$0.isMain && !$0.isSpatial && !subjects.contains($0.key) && !managedKeys.contains($0.key) && !grouped.contains($0.key) && !hidden($0.key)
                && !Self.jsonOnly($0)
        } ?? []
    }

    /// An optional input only JSON could set (Select's visual exemplars): a window is not a JSON editor, so it is left
    /// to the API. A required one stays, as the run needs it.
    static func jsonOnly(_ f: ToolField) -> Bool {
        if case .generic(.json) = f.control { return !f.required }; return false
    }

    /// Files dropped, chosen or pasted into the well: each goes to the field that takes it. Several files of one type go
    /// to a field that takes several (Depth's views, two frames); one file to the field that takes exactly its type (a
    /// PNG goes to an image field before a document field that also takes PNG). The alternatives the operation offers
    /// are exclusive: giving one clears the others.
    func setSubject(_ urls: [URL]) {
        guard !running, !urls.isEmpty else { return }
        keepDraft()
        let fields = fileFields
        func accept(_ f: ToolField) -> (FileAccept, Bool, Int)? {
            if case .generic(.file(let a, let multiple, let most, _)) = f.control { return (a, multiple, most) }; return nil
        }
        func takes(_ f: ToolField, _ url: URL) -> Bool {
            guard let (a, _, _) = accept(f) else { return false }
            return a.accepts(url) || (a.images && ToolFiles.isImageFile(url)) || (a.mimes.contains { $0.hasPrefix("image/") } && ToolFiles.isImageFile(url) && !a.document)
        }
        // Narrowest first: an images-only field before a document field that also takes images.
        func breadth(_ f: ToolField) -> Int { accept(f).map { $0.0.document ? 2 : $0.0.mimes.isEmpty ? 3 : 1 } ?? 9 }
        let ordered = fields.sorted { breadth($0) < breadth($1) }
        var target: ToolField?
        var files = urls
        if urls.count > 1 {
            target = ordered.first { f in accept(f)?.1 == true && urls.allSatisfy { takes(f, $0) } }
            if let t = target, let most = accept(t)?.2 { files = Array(urls.prefix(most)) }
        }
        if target == nil {
            let first = urls.first { url in ordered.contains { takes($0, url) } }
            if let first {
                files = [first]
                target = ordered.first { accept($0)?.1 == false && takes($0, first) } ?? ordered.first { takes($0, first) }
            }
        }
        guard let target else {
            let kinds = fileFields.compactMap { f -> FileAccept? in
                if case .generic(.file(let a, _, _, _)) = f.control { return a }; return nil
            }.map(\.title).joined(separator: " or ")
            problem = .init(message: "\(urls[0].lastPathComponent) is not something \(model?.name ?? "this model") takes here\(kinds.isEmpty ? "" : " (\(kinds))").", action: nil)
            return
        }
        for f in fields where f.key != target.key { values[f.key] = .files([]) }
        set(target.key, .files(files))
        // A file given chooses the option that takes it (Extract's File rather than Text).
        for group in optionGroups {
            if let i = group.options.firstIndex(where: { $0.keys.contains(target.key) }) { choose(group, i) }
        }
        shown = nil
        if problem?.field == nil { problem = nil }
        didSetSubject()
    }
    /// After the subject changed (Select releases its image session; spatial prompts are cleared).
    func didSetSubject() {
        for f in spatialFields { values[f.key] = .text("") }
    }
    func clearSubject() {
        guard !running else { return }
        for f in subjectFields { values[f.key] = .files([]) }
        shown = nil
        didSetSubject()
    }
    func pickSubject() {
        guard !running else { return }
        let multiple = subjectFields.contains { f in
            if case .generic(.file(_, true, _, _)) = f.control { return true }; return false
        }
        let urls = chooseFiles(subjectAccepts, multiple)
        if !urls.isEmpty { setSubject(urls) }
    }

    /// ⌘V: files copied in Finder, else image data (a copied picture or screenshot). False when nothing fits.
    @discardableResult func paste(_ pasteboard: NSPasteboard) -> Bool {
        guard !running, !fileFields.isEmpty else { return false }
        let files = (pasteboard.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL]) ?? []
        if !files.isEmpty { setSubject(files); return true }
        guard allAccepts.contains(where: { $0.images || $0.document || $0.mimes.contains { $0.hasPrefix("image/") } }) else { return false }
        for type in [NSPasteboard.PasteboardType.png, .tiff, NSPasteboard.PasteboardType(UTType.jpeg.identifier), NSPasteboard.PasteboardType(UTType.heic.identifier)] {
            if let data = pasteboard.data(forType: type) { addImage(data); return true }
        }
        return false
    }
    /// Image bytes without a file (pasted or dragged in): kept in the window's folder as an upright PNG.
    func addImage(_ data: Data, name: String? = nil) {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil), CGImageSourceGetCount(source) > 0 else {
            problem = .init(message: "That is not an image Arsenal can read.", action: nil); return
        }
        do {
            let folder = scratch.appendingPathComponent("inputs", isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let stem = ToolFiles.safe(URL(fileURLWithPath: name ?? "Pasted image").deletingPathExtension().lastPathComponent)
            var url = folder.appendingPathComponent((stem.isEmpty ? "Pasted image" : stem) + ".png")
            var n = 2
            while FileManager.default.fileExists(atPath: url.path) { url = folder.appendingPathComponent("\(stem) \(n).png"); n += 1 }
            try ToolFiles.upright(source, to: url)
            setSubject([url])
        } catch {
            problem = .init(message: "Couldn\u{2019}t keep the pasted image.", action: nil, detail: error.localizedDescription)
        }
    }

    /// Files or image data dropped on the well or the result.
    @discardableResult func receive(_ providers: [NSItemProvider]) -> Bool {
        guard !running else { return false }
        let urls = providers.filter { $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) }
        if !urls.isEmpty {
            let collected = DropCollector(count: urls.count) { [weak self] in self?.setSubject($0) }
            for (i, provider) in urls.enumerated() {
                _ = provider.loadObject(ofClass: URL.self) { url, _ in Task { @MainActor in collected.add(url, at: i) } }
            }
            return true
        }
        if let provider = providers.first(where: { $0.hasItemConformingToTypeIdentifier(UTType.image.identifier) }) {
            let name = provider.suggestedName
            _ = provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in
                guard let data else { return }
                Task { @MainActor [weak self] in self?.addImage(data, name: name) }
            }
            return true
        }
        return false
    }

    // MARK: Readiness

    /// The main text when the operation needs it, and something to work on: one of the operation's alternatives
    /// ("an image or images") or every required file field.
    override var canRun: Bool {
        guard !running, !loading, model != nil, blocked == nil, let form, descriptor != nil, form.unsupported == nil, readyToRun else { return false }
        if let main = form.main, main.required, value(main.key).text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return false }
        return true
    }
    override var readyToRun: Bool {
        guard let op = descriptor else { return false }
        let alternatives = ToolForm.alternatives(op.inputSchema)
        if !alternatives.isEmpty { return alternatives.contains { $0.allSatisfy(filled) } }
        return subjectFields.filter(\.required).allSatisfy { filled($0.key) }
    }
    /// A field holds something (files, text, a switch that is on).
    func filled(_ key: String) -> Bool {
        if managedKeys.contains(key) { return managedFilled(key) }
        if hidden(key) { return false }
        switch values[key] {
        case .files(let f)?: return !f.isEmpty
        case .text(let t)?: return !t.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        case .toggle(let on)?: return on
        case .choice(let i)?: return i != nil
        case nil: return false
        }
    }
    /// A managed input is in place (Select's session).
    func managedFilled(_ key: String) -> Bool { false }

    // MARK: Running

    override func start() {
        let inputs = subject
        super.start()
        if running {
            runInputs = inputs
            if !quietRun { shown = nil }
        }
    }
    /// A run that refines what is shown (Select after a click): the result stays on screen without a progress card
    /// until the new one replaces it, and nothing is saved until the result is kept.
    func startQuiet() {
        guard canRun else { return }
        quietRun = true
        start()
        if !running { quietRun = false }
    }

    /// The result of quiet runs, shown but not yet kept: its files stay in the window's own folder.
    @Published private(set) var draft: Outcome?
    /// Keeps the draft: its files saved where results of their type are kept, and it joins the history. Called when
    /// it is copied, dragged out, run for (Select), or its picture is replaced.
    @discardableResult func keepDraft() -> Outcome? {
        guard var outcome = draft else { return nil }
        let tag = outcome.inputs.first?.deletingPathExtension().lastPathComponent ?? ""
        do {
            outcome.files = try outcome.files.map { file in
                var file = file
                let folder = outputDirectory == ToolCatalog.folder(tool) ? ToolFiles.folder(mime: file.mime) : outputDirectory
                let label = outcome.files.count > 1 ? [tag, file.label].filter { !$0.isEmpty }.joined(separator: " ") : tag
                file.url = try ToolFiles.save(file.url, into: folder, model: outcome.model, tag: label, what: file.label.lowercased())
                return file
            }
        } catch {
            problem = OperationProblem.from(error, model: outcome.modelName, task: stoppedNoun)
            return nil
        }
        draft = nil
        outcomes.append(outcome)
        if shown == outcome.id || shown == nil { shown = outcome.id }
        record(outcome)
        return outcome
    }
    private func dropDraft() {
        for file in draft?.files ?? [] { try? FileManager.default.removeItem(at: file.url) }
        draft = nil
    }

    /// Every returned file saved where results of its type are kept, as `<date> <time> <model> <label>.<ext>`.
    override func deliver(_ result: OperationResult, op: OperationDescriptor, model: ModelChoice, input: JSONObject) throws {
        if quietRun { try deliverDraft(result, op: op, model: model, input: input); return }
        var files: [ResultFile] = []
        let tag = runInputs.first?.deletingPathExtension().lastPathComponent ?? ""
        for var file in result.files {
            let folder = outputDirectory == ToolCatalog.folder(tool) ? ToolFiles.folder(mime: file.mime) : outputDirectory
            let label = result.files.count > 1 ? [tag, file.label].filter { !$0.isEmpty }.joined(separator: " ") : tag
            file.url = try ToolFiles.save(file.url, into: folder, model: model.id, tag: label, what: file.label.lowercased())
            files.append(file)
        }
        let text = form?.main.flatMap { f in values[f.key]?.text } ?? ""
        let outcome = Outcome(
            model: model.id, modelName: model.name, operation: op.name, operationTitle: op.title, inputs: runInputs, text: text, values: requestValues(),
            input: input, reply: result.reply, files: files, seconds: result.seconds)
        try check(outcome)
        dropDraft()
        outcomes.append(outcome)
        shown = outcome.id
    }
    /// A quiet run's result replaces the draft in place; its files stay where they were fetched.
    private func deliverDraft(_ result: OperationResult, op: OperationDescriptor, model: ModelChoice, input: JSONObject) throws {
        let text = form?.main.flatMap { f in values[f.key]?.text } ?? ""
        let outcome = Outcome(
            model: model.id, modelName: model.name, operation: op.name, operationTitle: op.title, inputs: runInputs, text: text, values: requestValues(),
            input: input, reply: result.reply, files: result.files, seconds: result.seconds)
        try check(outcome)
        dropDraft()
        draft = outcome
        shown = outcome.id
    }
    /// The points or boxes on the picture changed (Select runs again at once).
    func promptsChanged() {}
    /// Masks drawn over the subject picture where prompts are placed (Select's current selection).
    var canvasMasks: [URL] { [] }
    /// The video frame prompts are placed on (Select's frame), in seconds.
    var promptTime: Double { 0 }
    /// A window refuses a reply it cannot show (no mask, no text): the run is reported, not shown blank.
    func check(_ outcome: Outcome) throws {}

    /// The result on show: the last run's, or one picked from the history; none once the subject changes.
    var current: Outcome? {
        if let draft, draft.id == shown { return draft }
        return outcomes.first { $0.id == shown }
    }
    /// An earlier result: shown, and (unless a run is going) its subject and options come back.
    func show(_ id: UUID) {
        guard let outcome = outcomes.first(where: { $0.id == id }) else { return }
        shown = id
        guard !running, outcome.model == selected, outcome.operation == operation else { return }
        for (k, v) in outcome.values { values[k] = v }
    }

    /// The shown result is what running again would make (same model, operation, options and subject).
    override var isFresh: Bool {
        // A draft on show: the run button keeps it, so it stays the primary action.
        guard draft == nil, !running, let c = current, c.model == selected, c.operation == operation else { return false }
        if form?.takesSeed == true, value("seed").text.trimmingCharacters(in: .whitespaces).isEmpty { return false }
        return c.values == requestValues()
    }

    func copy(_ file: ResultFile) { copy(file.url, to: copyPasteboard) }
    func copy(text: String) {
        copyPasteboard.clearContents()
        copyPasteboard.setString(text, forType: .string)
    }
    /// Text a window made (a transcript, a page's markdown) saved beside its other results.
    func save(text: String, name: String, ext: String) -> URL? {
        do {
            let folder = outputDirectory == ToolCatalog.folder(tool) ? ToolFiles.folder(mime: "text/plain") : outputDirectory
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let temp = scratch.appendingPathComponent("\(UUID().uuidString).\(ext)")
            try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
            try text.write(to: temp, atomically: true, encoding: .utf8)
            let url = try ToolFiles.save(temp, into: folder, model: current?.model ?? selected, tag: name, ext: ext, what: "text")
            ToolFiles.reveal(url, folder: folder)
            return url
        } catch {
            problem = OperationProblem.from(error, model: model?.name ?? selected, task: stoppedNoun)
            return nil
        }
    }

    // MARK: Figures

    override func figures() -> [Figure] {
        let r = run
        let done = current.flatMap { c in r?.done == true && r?.model == c.model ? c : nil }
        return [
            Figure(
                title: "Time", value: FigureText.seconds(done?.seconds), reference: nil, help: "From the start of this run to its result, as this window measured it.", width: 84),
            memoryFigure()
        ] + extraFigures(done)
    }
    /// The picture a result shows as in the history: its own image, else what it was made from.
    func historyPicture(_ o: Outcome) -> URL? { OutcomeHistory.picture(o) }
    /// The file Copy takes when the window copies a file (the first returned one).
    func copyFile(_ o: Outcome) -> URL? { o.files.first?.url }
    /// The text the window's Copy takes, for windows whose result is text (a transcript, a translation); nil copies the
    /// file. `outputNoun` names it in the menu's Copy Last item.
    func outputText(_ o: Outcome) -> String? { nil }
    var outputNoun: String { "Text" }
    override func recordLastOutput(model: ModelChoice) {
        // A quiet run's draft is not kept yet; it is recorded when kept.
        guard !quietRun, let o = current else { return }
        record(o)
    }
    private func record(_ o: Outcome) {
        if let text = outputText(o) {
            lastOutput?.record(.text(text), noun: outputNoun, model: o.modelName)
        } else if let url = copyFile(o) {
            lastOutput?.record(file: url, mime: o.files.first { $0.url == url }?.mime, model: o.modelName)
        }
    }
    /// A window's own figures of the shown result (audio length, pages, detections).
    func extraFigures(_ outcome: Outcome?) -> [Figure] { [] }

    /// Render harness and tests: a state without the API.
    func preview(
        operations: [OperationDescriptor], selected op: String? = nil, values: [String: FieldValue] = [:], run: ToolRun? = nil, outcomes: [Outcome] = [],
        shown: UUID? = nil, problem: OperationProblem? = nil
    ) {
        previewOptions(operations: operations, selected: op, values: values)
        self.run = run; self.outcomes = outcomes; self.shown = shown ?? outcomes.last?.id; self.problem = problem
    }
    /// Tests: a result arrived (as `deliver` keeps one) without a run.
    func append(_ outcome: Outcome) { outcomes.append(outcome); shown = outcome.id }
}

/// Collects the URLs of a multi-file drop (each provider answers on its own) and hands them over once, in order.
@MainActor final class DropCollector {
    private var urls: [URL?]
    private var answered = 0
    private let done: ([URL]) -> Void
    init(count: Int, done: @escaping ([URL]) -> Void) { urls = Array(repeating: nil, count: count); self.done = done }
    func add(_ url: URL?, at index: Int) {
        urls[index] = url; answered += 1
        if answered == urls.count { done(urls.compactMap { $0 }) }
    }
}
