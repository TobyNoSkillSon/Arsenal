// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore

// Model tools (C1): a model folder declares `tools = ["chat"|"image"|"voice"|"sound"|"background"]`; when its
// "Show in menu" switch is on, its tool appears as a section of the main menu under Models / Keep Hot / Memory. The
// tools are shared: one chat window, one image window, one voice section, each working with any model that declares it.
// Sound and Remove Background are one window each for every model that declares them, with the model chosen inside it.
// A tool talks to Arsenal only through the API (ArsenalClient); its failure is a line in its own place, never a crash.

enum ToolKind: String, CaseIterable {
    case chat, image, voice
    /// One window to make sound or speech with any sound or voice model that declares it.
    case sound
    /// The generic tool: any model's declared operations as a form and their result (schema-3 folders).
    case operation
    /// One window to cut an image's subject out of its background with any model that declares it.
    case background
    // The capability windows (ToolCatalog): one window each, for every model that declares one of its operations.
    case transcribe, separate, search, translate, document, extract, select, find, depth, upscale, smooth, make3d

    /// The menu item that opens the tool (voice has controls in the menu instead of a window).
    var title: String {
        switch self {
        case .chat: return "Chat\u{2026}"
        case .image: return "Make an Image\u{2026}"
        case .voice: return "Voice"
        case .sound: return "Make a Sound\u{2026}"
        case .operation: return "Run\u{2026}"
        case .background: return "Remove Background\u{2026}"
        default: return windowTitle + "\u{2026}"
        }
    }
    /// The window's own title (the menu item without its ellipsis).
    var windowTitle: String {
        switch self {
        case .chat: return "Chat"
        case .image: return "Make an Image"
        case .voice: return "Voice"
        case .sound: return "Make a Sound"
        case .operation: return "Run"
        case .background: return "Remove Background"
        case .transcribe: return "Transcribe"
        case .separate: return "Separate Audio"
        case .search: return "Rank Passages"
        case .translate: return "Translate"
        case .document: return "Read Document"
        case .extract: return "Extract"
        case .select: return "Select Objects"
        case .find: return "Find Objects"
        case .depth: return "Depth"
        case .upscale: return "Upscale"
        case .smooth: return "Smooth Video"
        case .make3d: return "Make 3D"
        }
    }
    /// "Show Chat in menu".
    var noun: String {
        switch self {
        case .chat: return "Chat"
        case .image: return "Image"
        case .voice: return "Voice"
        case .sound: return "Sound"
        case .operation: return "Tool"
        default: return windowTitle
        }
    }
    var symbol: String {
        switch self {
        case .chat: return "text.bubble"
        case .image: return "photo"
        case .voice: return "waveform"
        case .sound: return "music.note"
        case .operation: return "square.stack.3d.up"
        case .background: return "person.and.background.dotted"
        case .transcribe: return "quote.bubble"
        case .separate: return "slider.vertical.3"
        case .search: return "text.magnifyingglass"
        case .translate: return "translate"
        case .document: return "doc.text.viewfinder"
        case .extract: return "curlybraces"
        case .select: return "lasso"
        case .find: return "viewfinder"
        case .depth: return "square.3.layers.3d.down.right"
        case .upscale: return "arrow.up.left.and.arrow.down.right"
        case .smooth: return "film.stack"
        case .make3d: return "cube"
        }
    }
    /// "\(model) cannot make images on this Mac."
    var cannot: String {
        switch self {
        case .image: return "make images"
        case .sound: return "make sound"
        case .background: return "remove backgrounds"
        case .transcribe: return "transcribe"
        case .separate: return "separate audio"
        case .search: return "rank text"
        case .translate: return "translate"
        case .document: return "read documents"
        case .extract: return "extract data"
        case .select: return "select objects"
        case .find: return "find objects"
        case .depth: return "estimate depth"
        case .upscale: return "upscale"
        case .smooth: return "smooth video"
        case .make3d: return "make 3D models"
        default: return "run here"
        }
    }
    var help: String {
        switch self {
        case .chat: return "Opens a plain chat with this model, with its speed figures."
        case .image: return "Opens a window to make images with this model."
        case .voice: return "The voice agents and menu actions speak with."
        case .sound: return "Opens a window to make sounds or speech with this model and play them back."
        case .operation: return "Opens a window to run this model\u{2019}s operations."
        case .background: return "Opens a window to cut an image\u{2019}s subject out of its background with this model."
        default: return "Opens a window to \(cannot) with this model."
        }
    }
}

/// What plays speech aloud (C7's queue): mute, stop, and who is speaking. The app binds the voice section to it.
@MainActor protocol PlaybackControlling: AnyObject {
    var muted: Bool { get }
    var playing: Bool { get }
    /// The item playing: who asked, which model, which voice.
    var current: (requester: String, model: String, voice: String?)? { get }
    func setMuted(_ muted: Bool) throws
    func stop()
    /// Called on any change (an item starts or ends, mute flips).
    var onChange: () -> Void { get set }
}

@MainActor final class ToolCenter {
    let client: () -> ArsenalClient?
    var playback: PlaybackControlling
    private(set) var chats: [String: (window: ToolWindowController, session: ChatSession)] = [:]
    private(set) var operations: [String: (window: ToolWindowController, session: OperationSession)] = [:]
    /// The one Make an Image window, for every model that declares the image tool.
    private(set) var image: (window: ToolWindowController, session: ImageSession)?
    /// The one sound window, for every model that declares the sound tool.
    private(set) var sound: (window: ToolWindowController, session: SoundSession)?
    /// The main menu's audio task windows (Speak, Make Music, Make a Sound), one each.
    private(set) var soundTasks: [SoundTask: (window: ToolWindowController, session: SoundSession)] = [:]
    /// The one Remove Background window, for every model that declares the background tool.
    private(set) var background: (window: ToolWindowController, session: BackgroundSession)?
    /// The capability windows (ToolCatalog), one each, for every model whose operations place it there.
    private(set) var capability: [ToolKind: (window: ToolWindowController, session: CapabilitySession)] = [:]
    /// Where windows record their newest result for the menu's Copy Last item (the app's; none in tests).
    var lastOutput: LastOutput?
    /// Each model's declared operation names, read from its manifest (keyed by what would change them).
    private(set) var declared: [String: (key: String, names: Set<String>)] = [:]
    /// Each model's operations as the service resolved them, so a window opens complete on its first frame.
    private(set) var descriptors: [String: [OperationDescriptor]] = [:]
    /// The table as last seen (the picking windows list their models from it).
    private var rows: [ModelRow] = []
    /// The table's kinds: a model's kind names its primary operation and symbol.
    var kinds: () -> [KindSpec] = { [] }
    /// Where the generic tool stages dropped files (`<support>/inbox`, where the service accepts input files).
    var inbox = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-inbox", isDirectory: true)
    /// Opens the Models table (where Get, Load and approval happen).
    var openModels: () -> Void = {}
    /// Gets, approves or updates a model through the Models table's own flow (its size confirmation, code approval).
    var prepareModel: (ModelRow) -> Void = { _ in }
    /// Voice presets per voice model, and the one each speaks with (its `voice` setting).
    private(set) var voices: [String: [VoiceChoice]] = [:]
    private(set) var currentVoice: [String: String] = [:]
    private var voiceInventories: [String: String] = [:]
    private var voiceRequests: [String: (id: UUID, inventory: String, task: Task<[VoiceChoice], Error>)] = [:]
    private var voiceGeneration = 0
    /// A tool's last failure for a model, shown under its section until the next try.
    private(set) var errors: [String: String] = [:]
    var onChange: () -> Void = {}
    /// Brings a tool window front (tests keep it off screen).
    var present: (ToolWindowController) -> Void = { $0.show() }
    /// Where images are saved (tests use their own folder).
    var imageDirectory = ToolFiles.picturesDirectory
    /// Where sounds are saved (tests use their own folder).
    var soundDirectory = ToolFiles.musicDirectory
    /// Where cut-outs are saved (tests use their own folder).
    var backgroundDirectory = ToolFiles.picturesDirectory
    /// Where capability windows save; nil: each file where its type is kept (tests use their own folder).
    var capabilityDirectory: URL?

    init(client: @escaping () -> ArsenalClient?, playback: PlaybackControlling) {
        self.client = client; self.playback = playback
    }

    /// The switch controls presence; requests load installed models on demand through the API.
    nonisolated static func tools(_ row: ModelRow) -> [ToolKind] {
        guard row.toolsShown else { return [] }
        return row.tools.compactMap(ToolKind.init(rawValue:))
    }
    /// The tools a model's menu section offers: its folder's tools, the generic tool replaced by the capability windows
    /// its declared operations place it in (ToolCatalog.resolve).
    func resolvedTools(_ row: ModelRow) -> [ToolKind] {
        ToolCatalog.resolve(tools: Self.tools(row), declared: declared[row.id]?.names)
    }
    /// The same, whether or not the model shows its tools in the menu (a window lists every model it can run).
    func windowTools(_ row: ModelRow) -> [ToolKind] {
        ToolCatalog.resolve(tools: row.tools.compactMap(ToolKind.init(rawValue:)), declared: declared[row.id]?.names)
    }
    /// A model that only serves agents and other models (decisions, embeddings, rankings): never in the main menu.
    func backgroundOnly(_ row: ModelRow) -> Bool {
        let own = row.tools.compactMap(ToolKind.init(rawValue:)).filter { $0 != .operation }
        guard own.isEmpty else { return false }
        return ToolCatalog.backgroundOnly(kind: row.kind, declared: declared[row.id]?.names)
    }
    /// What would change a model's declared operations: its folder's digest and its active revision.
    nonisolated static func declarationKey(_ row: ModelRow) -> String { "\(row.contentDigest ?? "")|\(row.revision?.active ?? "")|\(row.approval?.digest ?? "")" }

    func open(_ tool: ToolKind, _ row: ModelRow) {
        errors[row.id] = nil
        switch tool {
        case .chat: openChat(row)
        case .image: openImage(row)
        case .voice: break
        case .sound:
            openTask(
                .sound(
                    tasks(row).compactMap { t -> SoundTask? in
                        if case .sound(let s) = t { return s }; return nil
                    }.first ?? .sound), row)
        case .operation:
            // A model whose operations all belong to capability windows opens the first of them.
            let tools = windowTools(row)
            if !tools.contains(.operation), let first = tools.first(where: { ToolCatalog.windows.contains($0) }) { openCapability(first, row) } else { openOperation(row) }
        case .background: openBackground(row)
        default: openCapability(tool, row)
        }
    }

    /// The kind's primary operation: declared `primary_operation`, else its default operation, else its first.
    nonisolated static func primaryOperation(_ kind: KindSpec?) -> OpDeclaration? {
        guard let kind else { return nil }
        return kind.operations.first { $0.name == kind.primaryOperation } ?? kind.operations.first(where: \.isDefault) ?? kind.operations.first
    }
    /// The menu item for the generic tool: its primary operation's title ("Separate Layers…"), else "Run…".
    func menuTitle(_ tool: ToolKind, _ row: ModelRow) -> String {
        guard tool == .operation else { return tool.title }
        // A generic entry beside capability windows names what is left for it.
        if let names = declared[row.id]?.names {
            let windows = ToolCatalog.windows(declared: names)
            let left = names.subtracting(windows.flatMap(ToolCatalog.operations)).subtracting(ToolCatalog.behindTheScenes)
            if !windows.isEmpty, left.count == 1, let op = kinds().first(where: { $0.id == row.kind })?.operations.first(where: { $0.name == left.first }) {
                return op.title + "\u{2026}"
            }
        }
        guard let op = Self.primaryOperation(kinds().first { $0.id == row.kind }) else { return tool.title }
        return op.title + "\u{2026}"
    }
    func symbol(_ tool: ToolKind, _ row: ModelRow) -> String {
        guard tool == .operation, let icon = kinds().first(where: { $0.id == row.kind })?.icon, NSImage(systemSymbolName: icon, accessibilityDescription: nil) != nil else {
            return tool.symbol
        }
        return icon
    }

    struct Admission {
        let operation: String
        let refusal: WorkerFailure?
        init(operation: String, refusal: WorkerFailure?) {
            self.operation = operation
            self.refusal = refusal?.code == "insufficient_memory" ? nil : refusal
        }
    }
    private(set) var admissions: [String: [ToolKind: Admission]] = [:]

    /// Resolve the operation exactly as the tool does; all readiness rules stay in the service.
    func requestAdmissions(rows: [ModelRow], service: any ModelsService) async -> [String: [ToolKind: Admission]] {
        // Routing reads the folder's declared operation names from its manifest (no schema read, no admission), again
        // only when its folder or revision changes; they place the model in its capability windows whether or not it can
        // run yet. The admission resolves the operations first, so a schema failure reaches the menu as its refusal; the
        // service keeps that failure until the folder changes, so the descriptor read below neither reads the failing
        // folder a second time nor reports it again. Descriptors open each window complete on its first frame.
        var changed: Set<String> = []
        for row in rows {
            let generic = row.tools.contains(ToolKind.operation.rawValue), background = row.tools.contains(ToolKind.background.rawValue)
            // Sound and voice models too: their declared operations place them in Speak, Make Music or Make a Sound.
            let audio = row.tools.contains(ToolKind.sound.rawValue) || row.tools.contains(ToolKind.voice.rawValue)
            guard generic || background || audio else { continue }
            let key = Self.declarationKey(row)
            guard declared[row.id]?.key != key else { continue }
            let names = await service.declaredOperationNames(id: row.id)
            declared[row.id] = names.isEmpty ? nil : (key, Set(names))
            descriptors[row.id] = nil
            changed.insert(row.id)
        }
        // Every question for every row, asked of the service in one call (it verifies what they share once, rather than
        // once per row: 48 rows made the menu's refresh 100 ms).
        // `fallback`: the operation the menu names when the service resolves none (the kind's primary one).
        var asked: [(id: String, tool: ToolKind, query: AdmissionQuery, fallback: String)] = []
        for row in rows {
            let generic = row.tools.contains(ToolKind.operation.rawValue), background = row.tools.contains(ToolKind.background.rawValue)
            if generic {
                let named = Self.primaryOperation(kinds().first { $0.id == row.kind })?.name
                let preferred = named.flatMap { n in declared[row.id]?.names.contains(n) == true ? n : nil }
                asked.append((row.id, .operation, AdmissionQuery(id: row.id, primary: true, preferred: preferred), named ?? ""))
            } else if background {
                asked.append((row.id, .background, AdmissionQuery(id: row.id, op: BackgroundSession.operation), ""))
            }
        }
        for row in rows {
            // Hiding a menu item must not block a tool window that is already open.
            for tool in windowTools(row) where tool != .voice && !asked.contains(where: { $0.id == row.id && $0.tool == tool }) {
                let query: AdmissionQuery
                switch tool {
                case .chat: query = AdmissionQuery(id: row.id, op: "chat")
                case .image: query = AdmissionQuery(id: row.id, op: "generate")
                case .background: query = AdmissionQuery(id: row.id, op: BackgroundSession.operation)
                case .sound:
                    query = AdmissionQuery(
                        id: row.id,
                        op: Self.soundOperation(kinds().first { $0.id == row.kind }, declared: declared[row.id]?.names)
                            ?? (row.kind == "voices" ? "speak" : "generate_sound"))
                case .operation:
                    query = AdmissionQuery(id: row.id, primary: true, preferred: Self.primaryOperation(kinds().first { $0.id == row.kind })?.name)
                case .voice: continue
                default:
                    let names = declared[row.id]?.names ?? []
                    guard let first = ToolCatalog.operations(tool).first(where: names.contains) else { continue }
                    query = AdmissionQuery(id: row.id, op: first)
                }
                asked.append((row.id, tool, query, ""))
            }
        }
        let replies = await service.admissions(asked.map(\.query))
        var answers: [String: [ToolKind: Admission]] = [:]
        for (question, reply) in zip(asked, replies) {
            let operation = reply.operation.isEmpty ? question.fallback : reply.operation
            answers[question.id, default: [:]][question.tool] = Admission(operation: operation, refusal: reply.refusal)
        }
        // Descriptors only for the windows that build from them (generic and background); a sound or voice model's names
        // place it in its task, and reading its schemas here would report a broken folder as an event (no admission saw it).
        let schemaWindows = [ToolKind.operation.rawValue, ToolKind.background.rawValue]
        for row in rows where changed.contains(row.id) && declared[row.id] != nil && row.tools.contains(where: schemaWindows.contains) {
            let operations = await service.operations(id: row.id)
            if !operations.isEmpty { descriptors[row.id] = operations.map { OperationDescriptor($0) } }
        }
        return answers
    }
    func applyAdmissions(_ answers: [String: [ToolKind: Admission]]) { admissions = answers }

    /// The generic window's state line: Download, Try Again or Approve… as the task windows offer them, else the
    /// service's refusal; "Checking…" until the service has answered.
    func runBlocked(_ row: ModelRow) -> OperationProblem? {
        if let ready = readiness(row, task: .run("")) { return ready.problem }
        return admissions[row.id]?[.operation] == nil ? blocked(row) : nil
    }
    func blocked(_ row: ModelRow, tool: ToolKind = .operation) -> OperationProblem? {
        guard let answer = admissions[row.id]?[tool] else { return .init(message: "Checking whether this model can run.", action: nil) }
        return answer.refusal.map { .init(message: $0.message, action: .openModels) }
    }

    struct MenuRecovery {
        let reason: String
        let action: String?
    }
    /// Visible, actionable readiness for every tool kind. Actual execution admission remains in
    /// the service; the menu never interprets a cached quality result as authorization to run.
    func menuRecovery(_ row: ModelRow, tool: ToolKind) -> MenuRecovery? {
        // A resident or verified recovery snapshot can run even if the current folder/selection
        // cannot cold-load. The service's affirmative admission outranks those row hints.
        if tool != .voice, let admission = admissions[row.id]?[tool], admission.refusal == nil { return nil }
        if let refusal = admissions[row.id]?[tool]?.refusal,
            ["unsupported_op", "unsupported_format", "not_found"].contains(refusal.code)
        {
            return .init(reason: "unavailable", action: nil)
        }
        if row.residentDType == nil {
            switch row.state {
            case .updateAvailable: return .init(reason: "update available", action: "Update")
            case .notInstalled: return .init(reason: "not installed", action: "Get")
            case .downloading: return .init(reason: "getting model", action: nil)
            case .deriving: return .init(reason: "preparing precision", action: nil)
            case .loading: return .init(reason: "loading", action: nil)
            case .failed where row.states[row.selectedPrecision]?.gate != nil: return .init(reason: "waiting for access", action: "Get")
            case .failed: return .init(reason: "setup failed", action: "Get")
            case .ready, .loaded: break
            }
        }
        if row.needsCodeConsent { return .init(reason: "approval needed", action: (row.state == .notInstalled) ? "Get" : "Load") }
        if admissions[row.id]?[tool]?.refusal != nil {
            // Unknown refusals have no proven repair. Omit the entry; tool windows retain
            // the service's full reason when opened through another authorized surface.
            return .init(reason: "unavailable", action: nil)
        }
        if tool == .voice, row.residentDType == nil { return .init(reason: "not loaded", action: "Load") }
        if tool == .voice, row.residentDType != nil { return nil }
        if admissions[row.id]?[tool] == nil { return .init(reason: "checking availability", action: nil) }
        return nil
    }

    private func openOperation(_ row: ModelRow) {
        if let open = operations[row.id] {
            open.session.update(model: ToolModel(row), blocked: runBlocked(row), revision: row.revision?.active, approvedDescriptor: row.approval?.digest)
            open.session.reopen()
            present(open.window); return
        }
        let kind = kinds().first { $0.id == row.kind }
        let session = OperationSession(model: ToolModel(row), client: client, inbox: inbox)
        session.lastOutput = lastOutput
        session.update(model: ToolModel(row), blocked: runBlocked(row), revision: row.revision?.active, approvedDescriptor: row.approval?.digest)
        session.openModels = { [weak self] in self?.openModels() }
        session.prepareModel = { [weak self] id in
            guard let self, let row = self.rows.first(where: { $0.id == id }) else { return }
            self.prepareModel(row)
        }
        let window = ToolWindowController(
            title: "\(kind?.titleSingular ?? kind?.title ?? "Tool") \u{00b7} \(row.name)", autosave: "ArsenalOperation", size: OperationView.size,
            minSize: OperationView.minSize, content: OperationView(session: session, symbol: symbol(.operation, row)))
        operations[row.id] = (window, session)
        window.onClose = { [weak session] in session?.close() }
        session.prepare(primary: admissions[row.id]?[.operation]?.operation ?? Self.primaryOperation(kind)?.name)
        present(window)
    }

    /// The operation a sound model's menu entry is admitted for: its kind's default audio operation, else its first.
    /// The audio operation a sound window's admission asks about: the kind's default, unless the model declares only
    /// another one (ACE-Step makes music, not sound effects: asking it for generate_sound left Make Music "Unavailable").
    nonisolated static func soundOperation(_ kind: KindSpec?, declared: Set<String>? = nil) -> String? {
        let audio = kind?.operations.filter { $0.output == "audio" } ?? []
        let ordered = audio.filter(\.isDefault) + audio.filter { !$0.isDefault }
        guard let declared else { return ordered.first?.name }
        return ordered.first { declared.contains($0.name) }?.name ?? ordered.first?.name
    }

    /// The models a picking tool window offers (Make an Image, Make a Sound, Remove Background): every table model that
    /// declares the tool (Show in menu only places the menu entry), with the reason one cannot run here.
    func models(_ rows: [ModelRow], for tool: ToolKind) -> [ModelChoice] {
        rows.filter { $0.tools.contains(tool.rawValue) || (ToolCatalog.windows.contains(tool) && windowTools($0).contains(tool)) }.map { row in
            ModelChoice(model: ToolModel(row), resident: row.residentDType != nil, unavailable: readiness(row, task: .tool(tool))?.problem)
        }
    }
    /// The models a main-menu task's window offers: every table model that serves the task, shown in the menu or not.
    func models(_ rows: [ModelRow], for task: MenuTask) -> [ModelChoice] {
        rows.filter { tasks($0).contains(task) }.map { row in
            ModelChoice(model: ToolModel(row), resident: row.residentDType != nil, unavailable: readiness(row, task: task)?.problem)
        }
    }

    /// Opens a main-menu task's window with this model chosen.
    func openTask(_ task: MenuTask, _ row: ModelRow) {
        errors[row.id] = nil
        switch task {
        case .sound(let s): openSoundTask(s, row)
        case .tool(let t): open(t, row)
        case .run: open(.operation, row)
        }
    }
    /// The model chosen in a task's open window, if it is open.
    func openModel(_ task: MenuTask) -> String? {
        switch task {
        case .sound(let s): return soundTasks[s].flatMap { $0.window.window?.isVisible == true ? $0.session.selected : nil }
        case .tool(.chat): return chats.first { $0.value.window.window?.isVisible == true }?.key
        case .tool(.image): return image.flatMap { $0.window.window?.isVisible == true ? $0.session.selected : nil }
        case .tool(.background): return background.flatMap { $0.window.window?.isVisible == true ? $0.session.selected : nil }
        case .tool(let t): return capability[t].flatMap { $0.window.window?.isVisible == true ? $0.session.selected : nil }
        case .run: return nil
        }
    }
    /// The one window of a picking tool, opened (or brought front) with this model chosen.
    private func openPicking<S: ToolSession>(
        _ tool: ToolKind, _ row: ModelRow, open: (window: ToolWindowController, session: S)?, choices given: [ModelChoice]? = nil,
        make: ([ModelChoice]) -> (ToolWindowController, S)
    ) -> (window: ToolWindowController, session: S) {
        var all = rows
        if !all.contains(where: { $0.id == row.id }) { all.append(row) }
        let choices = given ?? models(all, for: tool)
        if let open {
            open.session.update(models: choices)
            open.session.choose(row.id)
            present(open.window); return open
        }
        let (window, session) = make(choices)
        session.openModels = { [weak self] in self?.openModels() }
        session.prepareModel = { [weak self] id in
            guard let self, let row = self.rows.first(where: { $0.id == id }) else { return }
            self.prepareModel(row)
        }
        session.lastOutput = lastOutput
        window.onClose = { [weak session] in session?.close() }
        session.prepare()
        present(window)
        return (window, session)
    }

    private func openImage(_ row: ModelRow) {
        image = openPicking(.image, row, open: image) { models in
            let session = ImageSession(models: models, selected: row.id, client: client, inbox: inbox, outputDirectory: imageDirectory)
            return (
                ToolWindowController(title: "Make an Image", autosave: "ArsenalImage", size: ImageView.size, minSize: ImageView.minSize, content: ImageView(session: session)),
                session
            )
        }
    }

    private func openSoundTask(_ task: SoundTask, _ row: ModelRow) {
        var all = rows
        if !all.contains(where: { $0.id == row.id }) { all.append(row) }
        soundTasks[task] = openPicking(.sound, row, open: soundTasks[task], choices: models(all, for: .sound(task))) { models in
            let session = SoundSession(models: models, selected: row.id, client: client, inbox: inbox, outputDirectory: soundDirectory, task: task)
            let autosave = task == .sound ? "ArsenalSound" : "ArsenalSound" + task.rawValue.capitalized
            return (
                ToolWindowController(title: task.windowTitle, autosave: autosave, size: SoundView.size, minSize: SoundView.minSize, content: SoundView(session: session)),
                session
            )
        }
    }

    private func openBackground(_ row: ModelRow) {
        background = openPicking(.background, row, open: background) { models in
            let session = BackgroundSession(models: models, selected: row.id, client: client, inbox: inbox, outputDirectory: backgroundDirectory)
            let window = ToolWindowController(
                title: "Remove Background", autosave: "ArsenalBackground", size: BackgroundView.size, minSize: BackgroundView.minSize,
                content: BackgroundView(session: session))
            window.onPaste = { [weak session] pasteboard in session?.paste(pasteboard) ?? false }
            return (window, session)
        }
    }

    /// The one window of a capability tool, opened (or brought front) with this model chosen.
    private func openCapability(_ tool: ToolKind, _ row: ModelRow) {
        capability[tool] = openPicking(tool, row, open: capability[tool]) { models in
            let (session, view) = CapabilityWindows.make(tool, models: models, selected: row.id, client: client, inbox: inbox, output: capabilityDirectory)
            session.cachedOperations = { [weak self] id in self?.descriptors[id] }
            let window = ToolWindowController(
                title: tool.windowTitle, autosave: "Arsenal" + tool.rawValue.capitalized, size: ToolLayout.size, minSize: ToolLayout.minSize, content: view)
            window.onPaste = { [weak session] pasteboard in session?.paste(pasteboard) ?? false }
            return (window, session)
        }
        configure(capability[tool]?.session)
    }
    /// What a window needs from the table beyond its models (Transcribe's end-of-turn models for live mode).
    private func configure(_ session: CapabilitySession?) {
        guard let transcribe = session as? TranscribeSession else { return }
        transcribe.turnModels = rows.filter { row in (declared[row.id]?.names.contains("end_of_turn") ?? false) && row.state != .notInstalled && !row.needsCodeConsent }.map(\.id)
    }

    private func openChat(_ row: ModelRow, frame: NSRect? = nil) {
        if let open = chats[row.id] {
            open.session.update(model: ToolModel(row)); open.session.choices = chatChoices()
            if let frame { open.window.window?.setFrame(frame, display: false) }
            present(open.window); return
        }
        let session = ChatSession(model: ToolModel(row), client: client)
        session.choices = chatChoices()
        session.switchModel = { [weak self, weak session] id in self?.switchChat(from: session?.model.id, to: id) }
        let window = ToolWindowController(
            title: "Chat \u{00b7} \(row.name)", autosave: "ArsenalChat", size: ChatView.size, minSize: ChatView.minSize, content: ChatView(session: session))
        if let frame { window.window?.setFrame(frame, display: false) }
        chats[row.id] = (window, session)
        window.onPaste = { [weak session] pasteboard in session?.paste(pasteboard) ?? false }
        window.onClose = { [weak session] in session?.close() }
        session.prepare()
        present(window)
    }
    /// The chat models the band's pop-up offers: every table model with the chat tool, as Chat's menu entry serves them.
    private func chatChoices() -> [ModelChoice] { models(rows, for: .tool(.chat)) }
    /// The pop-up chose another model: its window takes this one's place (same frame); each keeps its conversation.
    private func switchChat(from current: String?, to id: String) {
        guard id != current, let row = rows.first(where: { $0.id == id }) else { return }
        let frame = current.flatMap { chats[$0]?.window.window?.frame }
        openChat(row, frame: frame)
        if let current, let old = chats[current]?.window { old.close() }
    }

    /// Keeps open windows in step with their rows (a Reload with another dtype changes the figures' reference).
    func update(rows: [ModelRow]) {
        self.rows = rows
        image?.session.update(models: models(rows, for: .image))
        sound?.session.update(models: models(rows, for: .sound))
        for (task, open) in soundTasks { open.session.update(models: models(rows, for: .sound(task))) }
        background?.session.update(models: models(rows, for: .background))
        for (tool, open) in capability {
            open.session.update(models: models(rows, for: tool))
            configure(open.session)
        }
        for row in rows {
            chats[row.id]?.session.update(model: ToolModel(row))
            chats[row.id]?.session.choices = chatChoices()
            operations[row.id]?.session.update(model: ToolModel(row), blocked: runBlocked(row), revision: row.revision?.active, approvedDescriptor: row.approval?.digest)
        }
        // Every loaded voice model keeps its voices: Settings builds its agent voice whether or not it shows in the menu.
        for (id, _) in voices where !rows.contains(where: { $0.id == id && $0.tools.contains(ToolKind.voice.rawValue) && $0.residentDType != nil }) {
            voices[id] = nil; currentVoice[id] = nil; voiceInventories[id] = nil
        }
    }

    /// Event-free service reads; settings need no worker. Re-fetch presets after an inventory change or a real
    /// refresh retries a failed read, and share in-flight reads from overlapping explicit actions.
    func refreshVoices(_ rows: [ModelRow], service: any ModelsService) async {
        voiceGeneration += 1
        let generation = voiceGeneration
        // Every table refresh ends here (several a second while models run): the menu rebuilds only when voices changed.
        let before = (voices, currentVoice, errors)
        defer { if generation == voiceGeneration, before != (voices, currentVoice, errors) { onChange() } }
        for row in rows where row.tools.contains(ToolKind.voice.rawValue) && row.residentDType != nil {
            do {
                let setting = try await service.settings(id: row.id).first(where: { $0.id == "voice" })
                guard generation == voiceGeneration else { return }
                currentVoice[row.id] = setting?.value.string
                let inventory = setting?.declaration["saved_voices_sha256"]?.string ?? ""
                if voices[row.id] == nil || voiceInventories[row.id] != inventory {
                    let request: (id: UUID, inventory: String, task: Task<[VoiceChoice], Error>)
                    if let running = voiceRequests[row.id], running.inventory == inventory {
                        request = running
                    } else {
                        request = (
                            UUID(), inventory,
                            Task {
                                return VoiceChoice.list(try await service.voiceInventory(id: row.id))
                            }
                        )
                        voiceRequests[row.id] = request
                    }
                    defer { if voiceRequests[row.id]?.id == request.id { voiceRequests[row.id] = nil } }
                    let choices = try await request.task.value
                    guard generation == voiceGeneration else { return }
                    voices[row.id] = choices; voiceInventories[row.id] = inventory
                }
                errors[row.id] = nil
            } catch {
                guard generation == voiceGeneration else { return }
                errors[row.id] = "Voices unavailable: \(error.localizedDescription)"
            }
        }
    }

    func chooseVoice(_ model: String, _ voice: String) {
        let previous = currentVoice[model]
        currentVoice[model] = voice; onChange()
        Task {
            guard let client = client() else { errors[model] = ChatSession.noAPI; currentVoice[model] = previous; onChange(); return }
            do { try await client.setSettings(model, ["voice": voice]); errors[model] = nil } catch {
                currentVoice[model] = previous; errors[model] = "Could not change the voice: \(error.localizedDescription)"
            }
            onChange()
        }
    }

    func toggleMute() {
        do { try playback.setMuted(!playback.muted) } catch { errors["playback"] = error.localizedDescription }
        onChange()
    }
    func stopSpeaking() { playback.stop(); onChange() }

    /// Render harness and tests: voices without a worker.
    func preview(voices: [String: [VoiceChoice]], current: [String: String], errors: [String: String] = [:]) {
        self.voices = voices; currentVoice = current; self.errors = errors
    }

    /// Quit: tool windows close with the app.
    func closeAll() {
        for (_, open) in chats { open.session.stop(); open.window.close() }
        for (_, open) in operations { open.session.discard(); open.window.close() }
        for open in [image?.window, sound?.window, background?.window].compactMap({ $0 }) + capability.values.map(\.window) + soundTasks.values.map(\.window) {
            open.close()
        }
        image?.session.discard(); sound?.session.discard(); background?.session.discard()
        for open in soundTasks.values { open.session.discard() }
        for open in capability.values { open.session.discard() }
    }
}

/// JSON object decoding for small API answers.
enum JSONWireValue {
    static func object(_ data: Data) throws -> JSONObject {
        guard case .object(let object) = try JSONDecoder().decode(JSONValue.self, from: data) else { throw ToolError("Arsenal answered with something other than an object.") }
        return object
    }
}

/// The shared playback queue (C7) as the voice section sees it: its state follows the queue's own events (no polling;
/// the stream waits while nothing changes).
@MainActor final class QueuePlayback: PlaybackControlling {
    let queue: PlaybackQueue
    var onChange: () -> Void = {}
    private var state: PlaybackSnapshot
    private var watching: Task<Void, Never>?
    init(queue: PlaybackQueue = .shared) {
        self.queue = queue
        state = queue.snapshot()
        watching = Task { [weak self] in
            for await snapshot in queue.events() {
                guard let self else { return }
                self.state = snapshot; self.onChange()
            }
        }
    }
    var muted: Bool { state.muted }
    /// Something is playing, or waits paused behind Mute: Stop has something to stop.
    var playing: Bool { state.playing || state.current != nil }
    var current: (requester: String, model: String, voice: String?)? { state.current.map { ($0.requester, $0.model, $0.voice) } }
    func setMuted(_ muted: Bool) throws { try queue.setMuted(muted); state = queue.snapshot() }
    func stop() { queue.stop(); state = queue.snapshot() }
}
