// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore

// The generic operation tool, as the tool kit composes it: one window per model, every operation the model declares
// with its options from its schema, one run at a time through the universal route, and the result: ordered files
// fetched into this window's folder (saved where the user chooses), or the reply's fields.

@MainActor final class OperationSession: ToolSession {
    @Published private(set) var result: OperationResult?
    /// The save panels (tests answer them).
    var chooseFile: (_ suggested: String) -> URL? = { name in
        let panel = NSSavePanel(); panel.nameFieldStringValue = name; panel.canCreateDirectories = true
        return panel.runModal() == .OK ? panel.url : nil
    }
    var chooseFolder: () -> URL? = {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.canCreateDirectories = true; panel.prompt = "Save"
        return panel.runModal() == .OK ? panel.url : nil
    }
    /// The operation's descriptors changed (another revision) while a run went: they are read again when it ends.
    private var needsPreparation = false
    /// The revision whose operations the form shows: a swap to another one re-reads them (its schemas may differ).
    private var revision: String?
    private var approvedDescriptor: String?
    private var descriptorInitialized = false

    init(model: ToolModel, client: @escaping () -> ArsenalClient?, inbox: URL, results: URL? = nil) {
        let folder = results ?? FileManager.default.temporaryDirectory.appendingPathComponent("Arsenal results/\(UUID().uuidString)", isDirectory: true)
        super.init(models: [ModelChoice(model: model)], selected: model.id, client: client, inbox: inbox, outputDirectory: folder, scratch: folder)
    }

    /// The kind's declared primary operation, else one annotated primary, else the first.
    func prepare(primary: String?) { prepare(prefer: primary) }

    /// The window's task: its model's everyday operation ("Speak"), steady while the pills change the operation.
    override var taskTitle: String {
        guard let name = primary(operations), let op = operations.first(where: { $0.name == name }) else { return "Run" }
        return OperationPills.title(op.title)
    }

    // MARK: Inputs

    /// A required file input still empty (the picture to separate): the run waits for it, its reason on the button's hover.
    var missingFile: ToolField? {
        form?.fields.first { f in
            guard f.required, showsField(f) else { return false }
            switch f.control {
            case .generic(.file), .images, .reference: return value(f.key).files.isEmpty
            default: return false
            }
        }
    }
    override var readyToRun: Bool { missingFile == nil }
    override var notReadyReason: String {
        guard let f = missingFile else { return super.notReadyReason }
        let noun = f.title.lowercased()
        let article = noun.first.map { "aeiou".contains($0) } == true ? "an" : "a"
        return "Add \(article) \(noun) first"
    }

    // MARK: Show in Finder

    /// The result's files (fetched into this window's own folder until saved), else where files of their type are kept.
    override func revealInFinder() { ToolFiles.reveal(result?.files.first?.url, folder: ToolFiles.folder(mime: "text/plain")) }
    override var revealFolderName: String { result?.files.isEmpty == false ? "this result" : "Documents \u{203a} Arsenal" }

    func update(model: ToolModel, blocked: OperationProblem?, revision: String? = nil, approvedDescriptor: String? = nil) {
        guard model.id == selected else { return }
        var choice = self.model ?? ModelChoice(model: model)
        choice.model = model; choice.unavailable = blocked
        update(models: [choice])
        if revision != self.revision || approvedDescriptor != self.approvedDescriptor {
            let first = !descriptorInitialized
            self.revision = revision; self.approvedDescriptor = approvedDescriptor
            if !first {
                needsPreparation = true
                if !running { needsPreparation = false; prepare(prefer: operation) }
            }
        }
        descriptorInitialized = true
    }

    func reopen() { if needsPreparation || operations.isEmpty { needsPreparation = false; prepare(prefer: operation) } }

    override func close() {
        super.close()
        loading = false; needsPreparation = true
    }

    override func didFinish() {
        if needsPreparation { needsPreparation = false; prepare(prefer: operation) }
    }

    override func deliver(_ result: OperationResult, op: OperationDescriptor, model: ModelChoice, input: JSONObject) throws {
        self.result = result
    }

    // MARK: Save

    func save(_ file: ResultFile) {
        guard let target = chooseFile(file.suggestedName) else { return }
        do { try Self.copy(file.url, to: target) } catch {
            problem = .init(message: "Couldn\u{2019}t save \(file.label).", action: nil, detail: error.localizedDescription)
        }
    }

    /// Every file into one folder, numbered in their order: "1 Near.png", "2 Middle.png", ….
    func saveAll() {
        guard let files = result?.files, !files.isEmpty, let folder = chooseFolder() else { return }
        do {
            for (i, file) in files.enumerated() { try Self.copy(file.url, to: folder.appendingPathComponent("\(i + 1) \(file.suggestedName)")) }
        } catch {
            problem = .init(message: "Couldn\u{2019}t save the files.", action: nil, detail: error.localizedDescription)
        }
    }

    private static func copy(_ source: URL, to target: URL) throws {
        if FileManager.default.fileExists(atPath: target.path) { try FileManager.default.removeItem(at: target) }
        try FileManager.default.copyItem(at: source, to: target)
    }

    /// Render harness and tests: a state without the API.
    func preview(
        operations: [OperationDescriptor], selected op: String? = nil, values: [String: FieldValue] = [:], run: ToolRun? = nil, result: OperationResult? = nil,
        problem: OperationProblem? = nil, blocked: OperationProblem? = nil
    ) {
        if let model { var m = model; m.unavailable = blocked; models = [m] }
        previewOptions(operations: operations, selected: op, values: values)
        self.run = run; self.result = result; self.problem = problem
    }
}
