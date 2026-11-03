// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore

// The main menu by task, not by model: one entry per thing a person does (Speak, Make Music, Make a Sound, Make an Image,
// Chat, Transcribe, …), each opening its window in every state with the model as a choice inside it. Every model whose
// Show in menu switch is on adds itself to the tasks it serves; background-only models (decisions, embeddings, rankings)
// serve none. An entry whose models cannot run yet still opens, with a quiet trailing note ("16 GB download", "42 %").

/// What a main-menu entry opens.
enum MenuTask: Hashable {
    /// Speak, Make Music, Make a Sound: the sound kit, narrowed to the task.
    case sound(SoundTask)
    /// A window of its own (Make an Image, Chat, Transcribe, Remove Background, a capability window).
    case tool(ToolKind)
    /// A model's own operations no window offers (the generic Run window), keyed by the entry's title.
    case run(String)
}

struct MenuTaskEntry: Equatable {
    var task: MenuTask
    var title: String
    var symbol: String
    /// The shown models that serve it, in the table's order (every serving model when none is shown).
    var rows: [ModelRow]
    /// Every model on the table that serves it, shown in the menu or not: a menu builds every entry and hides the ones
    /// no shown model serves, so a Show in menu switch changes no structure while the menu is open.
    var all: [ModelRow] = []
    /// A model whose Show in menu is on serves it.
    var shown = true
    /// The model the window opens with: one that runs now, else the first.
    var lead: ModelRow
    /// The lead's state when no model of the task can run yet; nil when one can.
    var note: String?
    var help: String
}

extension ToolCenter {
    /// The menu order of the tasks (a person's everyday ones first).
    static let taskOrder: [MenuTask] =
        [.sound(.speak), .sound(.music), .sound(.sound), .tool(.image), .tool(.chat), .tool(.transcribe)]
        + ToolCatalog.windows.filter { $0 != .transcribe && $0 != .search }.map { .tool($0) } + [.tool(.background)]

    /// Every operation a model declares: its folder's declared names once read, else its kind's operations.
    func operationNames(_ row: ModelRow) -> Set<String> {
        if let names = declared[row.id]?.names { return names }
        return Set(kinds().first { $0.id == row.kind }?.operations.map(\.name) ?? [])
    }

    /// The tasks a model serves, whether or not it shows in the menu (a task's window lists every model that serves it).
    func tasks(_ row: ModelRow) -> [MenuTask] {
        guard !backgroundOnly(row) else { return [] }
        let names = operationNames(row)
        var tasks: [MenuTask] = []
        let audio = row.tools.contains(ToolKind.sound.rawValue) || row.tools.contains(ToolKind.voice.rawValue)
        for task in SoundTask.allCases where names.contains(task.placing) {
            // A sound or voice model, or a generic model whose speech or music the sound kit can make.
            if audio || row.tools.contains(ToolKind.operation.rawValue) { tasks.append(.sound(task)) }
        }
        if audio, tasks.isEmpty { tasks.append(.sound(row.kind == "voices" ? .speak : .sound)) }
        let placed = Set(
            tasks.compactMap { t -> [String]? in
                if case .sound(let s) = t { return s.operations }; return nil
            }.flatMap { $0 })
        for tool in windowTools(row) {
            switch tool {
            case .voice, .sound, .search: continue
            case .operation:
                // The generic window only for what no task window offers.
                let windows = ToolCatalog.windows(declared: names).flatMap(ToolCatalog.operations)
                let left = names.subtracting(windows).subtracting(placed).subtracting(ToolCatalog.behindTheScenes).subtracting(ToolCatalog.backgroundOperations)
                if !left.isEmpty || names.isEmpty { tasks.append(.run(runTitle(row, left: left))) }
            default: tasks.append(.tool(tool))
            }
        }
        return tasks
    }

    /// The generic entry's title: the one operation it is left with, else the kind's primary operation, else Run….
    private func runTitle(_ row: ModelRow, left: Set<String>) -> String {
        let kind = kinds().first { $0.id == row.kind }
        if left.count == 1, let op = kind?.operations.first(where: { $0.name == left.first }) { return op.title + "\u{2026}" }
        return menuTitle(.operation, row)
    }

    /// The main menu's task entries, in task order; generic entries follow, by title. `includeHidden`: also the tasks
    /// only models with Show in menu off serve, marked not shown.
    func taskEntries(_ rows: [ModelRow], includeHidden: Bool = false) -> [MenuTaskEntry] {
        var serving: [MenuTask: [ModelRow]] = [:]
        for row in rows where includeHidden || row.toolsShown {
            for task in tasks(row) { serving[task, default: []].append(row) }
        }
        let generic = serving.keys.compactMap { t -> String? in
            if case .run(let title) = t { return title }; return nil
        }.sorted()
        return (Self.taskOrder + generic.map { MenuTask.run($0) }).compactMap { task in
            guard let all = serving[task], !all.isEmpty else { return nil }
            let shownRows = all.filter(\.toolsShown)
            let rows = shownRows.isEmpty ? all : shownRows
            let lead = leadModel(rows, task: task)
            let waiting = rows.contains { readiness($0, task: task) == nil } ? nil : readiness(lead, task: task)
            return MenuTaskEntry(
                task: task, title: Self.title(task), symbol: symbol(task, lead), rows: rows, all: all, shown: !shownRows.isEmpty, lead: lead,
                note: waiting?.note, help: waiting?.hover ?? Self.help(task))
        }
    }

    /// The model a task's window opens with: the one in use, else one that runs now, else the first.
    func leadModel(_ rows: [ModelRow], task: MenuTask) -> ModelRow {
        if let open = openModel(task), let row = rows.first(where: { $0.id == open }) { return row }
        return rows.first { $0.residentDType != nil && readiness($0, task: task) == nil } ?? rows.first { readiness($0, task: task) == nil } ?? rows[0]
    }

    static func title(_ task: MenuTask) -> String {
        switch task {
        case .sound(let s): return s.menuTitle
        case .tool(let t): return t.title
        case .run(let title): return title
        }
    }
    func symbol(_ task: MenuTask, _ row: ModelRow) -> String {
        switch task {
        case .sound(let s): return s.symbol
        case .tool(let t): return t.symbol
        case .run: return symbol(.operation, row)
        }
    }
    static func help(_ task: MenuTask) -> String {
        switch task {
        case .sound(.speak): return "Type text and hear it in any voice."
        case .sound(.music): return "Make a song or a piece of music from a description."
        case .sound(.sound): return "Make a sound effect from a description."
        case .tool(.chat): return "Chat with a model."
        case .tool(let t): return "Opens a window to \(t.cannot)."
        case .run: return "Opens a window to run this model\u{2019}s operations."
        }
    }

    /// The tool whose admission answers for a task.
    func admissionTool(_ row: ModelRow, task: MenuTask) -> ToolKind {
        switch task {
        case .sound: return windowTools(row).contains(.sound) ? .sound : .operation
        case .tool(let t): return t
        case .run: return .operation
        }
    }

    struct Readiness: Equatable {
        /// The menu's trailing note: only for what changes or needs a person (a download's percent, a failure, an approval);
        /// a model not downloaded yet adds nothing to the row, its window offers Download with the size.
        var note: String?
        /// The row's hover, when the model is the task's only one.
        var hover: String?
        /// The window's line and its one action.
        var problem: OperationProblem
    }

    /// Why a model cannot run a task yet, with the note the menu shows and the problem its window shows; nil when a
    /// request may run (installed models load on demand).
    func readiness(_ row: ModelRow, task: MenuTask) -> Readiness? {
        let tool = admissionTool(row, task: task)
        let dtype = row.selectedPrecision
        let state = row.states[dtype]
        let size = (row.downloadBytes[dtype] ?? row.sizeBytes).map(Self.shortSize)
        switch row.residentDType == nil ? row.state : .loaded {
        case .notInstalled:
            return .init(
                note: nil, hover: "Not on this Mac yet" + (size.map { " (\($0))" } ?? "") + ". The window downloads it.",
                problem: .init(message: "\(row.name) isn\u{2019}t on this Mac yet.", action: .download, actionTitle: size.map { "Download \($0)" } ?? "Download"))
        case .downloading:
            let pct = state?.progress.map { "\(Int(($0 * 100).rounded())) %" }
            return .init(note: pct ?? "Downloading", problem: .init(message: "Downloading \(row.name)" + (pct.map { " \u{00b7} " + $0 } ?? "") + "\u{2026}", action: nil))
        case .deriving: return .init(note: "Preparing", problem: .init(message: "Preparing \(row.name)\u{2026}", action: nil))
        case .failed where state?.gate != nil:
            // A gated model: the person's step on Hugging Face, in a few words; Try Again retries now.
            let gate = state!.gate!, line = RowPresentation.gateLine(gate)
            let message =
                gate.gaveUp
                ? "No access to \(row.name) on Hugging Face yet."
                : gate.hasToken
                    ? "Waiting for access to \(row.name) on Hugging Face."
                    : gate.tokenRejected == true
                        ? "Hugging Face refused the saved token: replace it in Settings." : "\(row.name) needs a Hugging Face token: add one in Settings."
            return .init(
                note: gate.gaveUp ? "No access" : "Waiting", hover: line.help,
                problem: .init(message: message, action: .download, actionTitle: "Try Again", detail: line.help))
        case .failed:
            let why = state?.error.map(Self.plainReason).map { " " + $0 + "." } ?? ""
            return .init(
                note: "Failed", hover: "The download stopped. The window tries again.",
                problem: .init(message: "\(row.name) didn\u{2019}t finish downloading.\(why)", action: .download, actionTitle: "Try Again"))
        case .updateAvailable:
            return .init(note: "Update", problem: .init(message: "\(row.name) needs an update before it can run.", action: .update))
        case .loading, .ready, .loaded: break
        }
        if row.needsCodeConsent {
            return .init(
                note: "Approve", hover: "Its folder changed since you approved it. The window asks again.",
                problem: .init(message: "\(row.name)\u{2019}s folder changed since you approved it.", action: .approve))
        }
        if let refusal = admissions[row.id]?[tool]?.refusal {
            return .init(note: "Unavailable", hover: refusal.message, problem: .init(message: refusal.message, action: nil))
        }
        return nil
    }

    /// A download's size as the menu's note says it: "342 MB", "3.4 GB", "16 GB".
    static func shortSize(_ bytes: Int64) -> String {
        let gb = Double(bytes) / 1e9
        if gb < 1 { return "\(max(1, Int((Double(bytes) / 1e6).rounded()))) MB" }
        return gb < 10 ? String(format: "%.1f GB", gb) : "\(Int(gb.rounded())) GB"
    }

    /// A download failure's reason in a person's words: its first sentence, with file-system jargon cut.
    static func plainReason(_ error: String) -> String {
        let first = error.split(separator: "\n").first.map(String.init) ?? error
        if first.contains("couldn\u{2019}t be opened") || first.contains("couldn't be opened") || first.contains("NSCocoaErrorDomain") { return "A file could not be read" }
        let sentence = (first.split(separator: ".", maxSplits: 1).first.map(String.init) ?? first).trimmingCharacters(in: .whitespaces)
        return sentence.isEmpty ? "The download stopped" : sentence
    }
}
