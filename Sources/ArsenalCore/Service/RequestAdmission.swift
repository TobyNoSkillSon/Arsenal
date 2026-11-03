// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire

/// The kinds directory a batch of admissions verified once, visible only to that batch's task and its children.
enum KindsBatch {
    @TaskLocal static var verified: URL?
}

extension RealModelsService {
    static func localCodeRequiresConsent(_ local: LocalModelFolder) throws -> Bool {
        // A bundled Swift runner can still compile unsigned, folder-delivered Metal source.
        if local.requiresCodeConsent { return true }
        return try !KernelManifest.loadIfPresent(folder: local.directory).kernels.isEmpty
    }

    public func runRefusal(id: String, op: String) async -> WorkerFailure? {
        do {
            _ = try await admitRequest(id: id, op: op, notifyChanges: false)
            return nil
        } catch let failure as WorkerFailure { return failure } catch { return WorkerFailure(code: "readiness", message: error.localizedDescription) }
    }

    public func primaryOperationAdmission(id: String, preferred: String?) async -> RunAdmission {
        var op = preferred ?? ""
        do {
            try admitRun(id)
            if localModels[id] != nil { try await discoverLocalModels(notifyChanges: false) }
            let operations = try await resolvedOperations(id: id)
            op = RunAdmission.primaryOperation(operations, preferred: preferred)
            guard !op.isEmpty else { throw WorkerFailure(code: "unsupported_op", message: "Model \(id) has no runnable operations.") }
            _ = try await admitRequest(id: id, op: op, notifyChanges: false)
            return RunAdmission(operation: op, refusal: nil)
        } catch let failure as WorkerFailure { return RunAdmission(operation: op, refusal: failure) } catch {
            return RunAdmission(operation: op, refusal: WorkerFailure(code: "readiness", message: error.localizedDescription))
        }
    }

    /// The menu's questions for every model at once: the shared kinds directory is verified once for the batch (a stat
    /// of each kind file), not once per question, which for 48 models was most of a 100 ms menu refresh.
    /// The verification belongs to this batch's task (`KindsBatch.verified`), never to the actor: another caller that
    /// interleaves at one of the batch's awaits still verifies the directory itself.
    public func admissions(_ queries: [AdmissionQuery]) async -> [RunAdmission] {
        guard let verified = try? await installer.verifiedKindsDirectory() else { return await answer(queries) }
        kindsDirectory = verified
        return await KindsBatch.$verified.withValue(verified) { await answer(queries) }
    }
    private func answer(_ queries: [AdmissionQuery]) async -> [RunAdmission] {
        var answers: [RunAdmission] = []
        for q in queries {
            if q.primary {
                answers.append(await primaryOperationAdmission(id: q.id, preferred: q.preferred))
            } else {
                answers.append(RunAdmission(operation: q.op, refusal: await runRefusal(id: q.id, op: q.op)))
            }
        }
        return answers
    }

    /// The API calls this same admission query before loading. A true answer needs an on-demand load;
    /// a false answer can run its resident (or an app-owned operation) immediately. Refusals always throw.
    func admitRequest(id: String, op: String, operationValidated: Bool = false, notifyChanges: Bool = true) async throws -> Bool {
        try admitRun(id)
        if !operationValidated { _ = try await requestOperation(id: id, op: op, notifyChanges: notifyChanges) }
        if op == "delete_voice", usesVoicePersistence(id) { return false }
        if residents[id] != nil, !(await supervisor.hasExited(id: id)) { return false }
        if loadTasks[id] != nil { return true } // The request joins that load; it does not allocate a second model.
        // Recovery restores its verified launch and weights before inspecting the current selection.
        if let old = recoveryRevision(id) { try checkLoadMemory(old.manifest, precision: old.preference.precision, requirement: revisionMemoryRequirement(old)); return true }
        let (model, precision, _) = try await loadSelection(id: id, precision: nil, notifyChanges: notifyChanges)
        try requireLoadAssets(model, precision: precision)
        try checkLoadMemory(model, precision: precision)
        return true
    }

    func admitRun(_ id: String) throws {
        try admitNewWork(); try admitWork(allowRefresh: residents[id] != nil || reloading.contains(id))
    }

    /// Shared by request validation and the menu's admission query. Input validation remains request-specific.
    func requestOperation(id: String, op: String, rawDecision: Bool = false, notifyChanges: Bool = true) async throws -> (ModelManifest, ModelOperation?) {
        if !(op == "delete_voice" && usesVoicePersistence(id)), residents[id] == nil, recoveryRevision(id) == nil {
            try await checkLocalCode(id, allowBrokenResident: true, notifyChanges: notifyChanges)
        }
        let model = try residentManifests[id] ?? recoveryRevision(id)?.manifest ?? manifest(id)
        let ops = try await resolvedOperations(id: id)
        let operation = ops.first { $0.name == op }
        let legacyDecision = rawDecision && model.schema < 2 && model.kind == "decisions" && ["decide", "judge"].contains(op)
        if operation == nil, !legacyDecision, model.schema >= 2 || !ops.isEmpty {
            throw WorkerFailure(code: "unsupported_op", message: "Model \(id) does not support \(op).")
        }
        return (model, operation)
    }

    /// Agents and the API use a downloaded model without it being on the table; only Add (or the user) puts it there.
    func loadable(_ id: String) -> Bool {
        if config.table.contains(id) || residents[id] != nil { return true }
        guard let m = try? manifest(id) else { return false }
        return m.precisions.contains { FileManager.default.fileExists(atPath: StoredNames.snapshotURL(in: snapshot(m, $0)).path) }
    }
    static func notOnTable(_ id: String) -> WorkerFailure {
        WorkerFailure(code: "not_found", message: "Model \(id) is not in your models; add it before Load or run.")
    }
    /// A saved, recommended or default precision as it resolves now: `runnablePrecision`, except that a retired int8 falls back to
    /// the model's 16-bit format (or another runnable format that is not retired).
    func servablePrecision(_ m: ModelManifest, _ dtype: String) -> String {
        let resolved = m.runnablePrecision(dtype)
        guard RetiredDtype.isRetired(resolved) else { return resolved }
        let runnable = m.formats.filter { !$0.runners.isEmpty && !RetiredDtype.isRetired($0.id) }.map(\.id)
        return runnable.first { ["fp16", "bf16"].contains($0) } ?? runnable.first ?? m.basePrecision
    }
    static func retiredRefusal(_ dtype: String) -> WorkerFailure {
        WorkerFailure(code: "unsupported_format", message: "\(dtype) is no longer offered; use the 16-bit or int4 precision.")
    }
    /// The same cold-load selection guards used by ensureLoaded; resident/recovery paths bypass them.
    func loadSelection(id: String, precision: String?, notifyChanges: Bool = true) async throws -> (ModelManifest, String, RunnerSpec) {
        try admitNewWork()
        guard loadable(id) else { throw Self.notOnTable(id) }
        try admitWork()
        try await checkLocalCode(id, notifyChanges: notifyChanges)
        if let requested = precision, RetiredDtype.isRetired(requested) { throw Self.retiredRefusal(requested) }
        let m = try manifest(id); let p = precision ?? servablePrecision(m, modelRows[id]?.selectedPrecision ?? m.defaultPrecision)
        guard m.precisions.contains(p) else { throw WorkerFailure(code: "precision", message: "That dtype is not offered.") }
        guard let selected = selectedRunner(m), m.supports(dtype: p, runner: selected.name) else {
            throw WorkerFailure(code: "unsupported_format", message: modelRows[id]?.dtypeUnavailableReason(p) ?? "No runner in this Arsenal yet.")
        }
        if let reason = modelRows[id]?.dtypeUnavailableReason(p) { throw WorkerFailure(code: "unsupported_format", message: reason) }
        guard gets[id] == nil else { throw WorkerFailure(code: "busy", message: "Wait for Get to finish before loading \(id).") }
        try requireCurrentInstallation(id, precision: p)
        return (m, p, selected)
    }

    func requireLoadAssets(_ model: ModelManifest, precision: String) throws {
        guard snapshotPresent(model, precision) else {
            throw WorkerFailure(
                code: "not_installed", message: "Model \(model.id) (\(precision)) is not installed; use \(MCPNames.prefix)download(model: \(model.id), dtype: \(precision)) first.")
        }
        if let runner = selectedRunner(model) { try requireNativeRunner(runner) }
    }

    /// Both cold admission and performLoad (including revision swaps) use this refusal.
    func requireNativeRunner(_ selected: RunnerSpec) throws {
        guard selected.language != "python" else {
            throw WorkerFailure(code: "unsupported_runner", message: "Python model runners are unsupported; choose a native runner.")
        }
    }
}
