// SPDX-License-Identifier: MIT
import Foundation

public struct ModelRevision: Codable, Sendable, Equatable {
    public var active: String?
    public var pending: String?
    public var previous: String?
    public var swapping: Bool
    public var error: String?
    public init(active: String? = nil, pending: String? = nil, previous: String? = nil, swapping: Bool = false, error: String? = nil) {
        self.active = active; self.pending = pending; self.previous = previous; self.swapping = swapping; self.error = error
    }
}
struct ResidentRevision: Sendable {
    var manifest: ModelManifest
    var operations: [ModelOperation]
    var settings: ModelSettings
    var launch: WorkerLaunch
    var load: JSONObject
    var preference: ModelPreferences
    var resident: ResidentModel
    var digest: String
    var menus: [LoadedMenuSection]
    var memoryRequirement: LoadMemoryRequirement?
    // Only an interrupted resident's in-process recovery inherits this; never serialize it as a future permission.
    var memoryForceApproved = false
}
/// A recovery file (`state/revisions/<id>.json`); internal so tests can read the files older builds wrote.
struct SavedRevision: Codable {
    var digest: String
    var settings: ModelSettings
    var load: JSONObject
    var preference: ModelPreferences
    var resident: ResidentModel
    var runner: String?
    var menus: [LoadedMenuSection]
    var memoryRequirement: LoadMemoryRequirement?
}
extension RealModelsService {
    func saveRecoveryRevision(_ id: String, _ revision: ResidentRevision) throws {
        let file = paths.support.appendingPathComponent("state/revisions/\(id).json")
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        let value = SavedRevision(
            digest: revision.digest, settings: revision.settings, load: revision.load, preference: revision.preference, resident: revision.resident, runner: revision.launch.runner,
            menus: revision.menus, memoryRequirement: revision.memoryRequirement)
        try JSONEncoder().encode(value).write(to: file, options: .atomic)
    }
    func readRecoveryRevision(_ id: String) async throws {
        guard dormantRevisions[id] == nil, let local = localModels[id], config.rejectedRevisions[id] == local.contentDigest else { return }
        let file = paths.support.appendingPathComponent("state/revisions/\(id).json")
        let value = try JSONDecoder().decode(SavedRevision.self, from: Data(contentsOf: file))
        let folder = paths.support.appendingPathComponent("state/local-folders/\(value.digest)/\(id)")
        let old = try LocalModelFolders.discoverSnapshot(folder)
        guard old.contentDigest == value.digest, let runner = old.manifest.runners.first(where: { $0.name == value.runner }) else {
            throw WorkerFailure(code: "integrity", message: "The previous approved revision snapshot changed.")
        }
        let launch = try await artifacts.launch(model: old.manifest, runner: runner, directory: folder, paths: paths)
        let operations = try await resolvedOperations(id: id, candidate: true, modelOverride: old.manifest, folderOverride: folder)
        let oldRevision = ResidentRevision(
            manifest: old.manifest, operations: operations, settings: value.settings, launch: launch, load: value.load, preference: value.preference, resident: value.resident,
            digest: value.digest, menus: value.menus, memoryRequirement: value.memoryRequirement)
        dormantRevisions[id] = oldRevision
    }

    func protectedRevisionDigests() -> Set<String> {
        Set(residentLocalDigests.values).union(previousRevisions.values.map(\.digest)).union(dormantRevisions.values.map(\.digest)).union(localModels.values.map(\.contentDigest))
    }
    func clearResidentDescriptor(_ id: String) {
        forceLoadedResidents.remove(id)
        residentMemoryRequirements[id] = nil
        guard !reloading.contains(id) else { return }
        residentMenus[id] = nil; residentOperations[id] = nil; residentManifests[id] = nil; residentSettings[id] = nil
        residentLaunches[id] = nil; residentLoads[id] = nil; residentLocalDigests[id] = nil
    }
    func waitForRevisionSwap(_ id: String) async throws {
        guard reloading.contains(id) else { return }
        waitingRevisionRuns[id, default: 0] += 1
        defer { waitingRevisionRuns[id, default: 0] -= 1; if Task.isCancelled { scheduleRevisionSwap(id) } }
        while reloading.contains(id) { try await waitForMemoryChange(); try Task.checkCancellation() }
    }
    func recoveryRevision(_ id: String) -> ResidentRevision? {
        guard residents[id] == nil, config.table.contains(id), let local = localModels[id], rejectedRevisionDigests[id] == local.contentDigest else { return nil }
        return dormantRevisions[id]
    }
    func restoreDormantRevisionIfNeeded(_ id: String, residency: ResidencyClass) async throws -> Bool {
        guard var previous = recoveryRevision(id) else { return false }
        if let task = loadTasks[id] { try await task.value; return true }
        previous.resident.residency = residency
        let restored = previous
        // Registered: Refresh is refused from here (`refreshRefusal`), so say so before the recovery starts its worker.
        let task = Task { try await self.restoreRevision(id, restored) }; loadTasks[id] = task; changed()
        defer { loadTasks[id] = nil; signalMemoryChange(); changed() }
        try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
        return true
    }
    func readResults(folder: URL) -> [ResultsFile] { resultCache[folder.lastPathComponent] ?? [] }
    func currentResults(_ model: ModelManifest) -> [ResultsFile] {
        let results = resultCache[model.id] ?? []
        guard model.schema == 3, let local = localModels[model.id] else { return results }
        var assessed = local
        if !sameImplementation(model.id), let digest = residentLocalDigests[model.id] ?? recoveryRevision(model.id)?.digest,
            let active = try? LocalModelFolders.discoverSnapshot(paths.support.appendingPathComponent("state/local-folders/\(digest)/\(model.id)"))
        {
            assessed = active
        }
        return results.filter { $0.checks.map { ImplementationEvidence.compatible($0, local: assessed) } == true }
    }
    func captureRevision(_ id: String) -> ResidentRevision? {
        guard let manifest = residentManifests[id], let operations = residentOperations[id], let settings = residentSettings[id],
            let launch = residentLaunches[id], let load = residentLoads[id], let preference = loadedPreferences[id], let resident = residents[id],
            let digest = residentLocalDigests[id]
        else { return nil }
        return .init(
            manifest: manifest, operations: operations, settings: settings, launch: launch, load: load, preference: preference, resident: resident, digest: digest,
            menus: residentMenus[id] ?? [], memoryRequirement: residentMemoryRequirements[id], memoryForceApproved: forceLoadedResidents.contains(id))
    }
    func sameImplementation(_ id: String) -> Bool {
        guard let local = localModels[id], let digest = residentLocalDigests[id] else { return localModels[id] == nil }
        if local.contentDigest == digest { return true }
        if let oldFingerprint = fingerprintCache[digest] { return oldFingerprint == fingerprintCache[local.contentDigest] }
        let folder = paths.support.appendingPathComponent("state/local-folders/\(digest)/\(id)")
        guard let old = try? LocalModelFolders.discoverSnapshot(folder) else { return false }
        fingerprintCache[digest] = try? ImplementationEvidence.fingerprint(old)
        fingerprintCache[local.contentDigest] = try? ImplementationEvidence.fingerprint(local)
        return fingerprintCache[digest] == fingerprintCache[local.contentDigest]
    }
    func updateRevisionRow(_ id: String) {
        guard let local = localModels[id] else { return }
        let digest = residentLocalDigests[id] ?? recoveryRevision(id)?.digest
        let error = modelRows[id]?.revision?.error
        let pending =
            digest != nil && config.localCodeConsent[id] == local.contentDigest
                && rejectedRevisionDigests[id] != local.contentDigest && !sameImplementation(id) ? local.contentDigest : nil
        modelRows[id]?.revision = .init(
            active: digest, pending: pending,
            previous: previousRevisions[id]?.digest, swapping: reloading.contains(id), error: error)
    }
    func updateChecks(_ model: ModelManifest) {
        guard let local = localModels[model.id] else { return }
        let results = (resultCache[model.id] ?? []).sorted(by: ResultsFile.newestFirst)
        let result = newestCompatibleResult(local) ?? results.first { $0.checks != nil }
        var checks = result?.checks
        fingerprintCache[local.contentDigest] = try? ImplementationEvidence.fingerprint(local)
        if let current = checks { checks?.stale = result?.recordedAt == nil || !ImplementationEvidence.compatible(current, local: local) }
        modelRows[model.id]?.checks = checks
    }
    private func newestCompatibleResult(_ local: LocalModelFolder) -> ResultsFile? {
        (resultCache[local.manifest.id] ?? []).sorted(by: ResultsFile.newestFirst).first {
            $0.checks.map { ImplementationEvidence.compatible($0, local: local) } == true
        }
    }
    func requireSwapEvidence(_ local: LocalModelFolder) throws {
        let id = local.manifest.id
        guard let digest = residentLocalDigests[id] ?? config.localCodeConsent[id], digest != local.contentDigest else { return }
        let previous = paths.support.appendingPathComponent("state/local-folders/\(digest)/\(id)")
        guard let old = try? LocalModelFolders.discoverSnapshot(previous) else {
            throw WorkerFailure(code: "revision_gate", message: "The previous approved snapshot is unreadable; repair it before approving a revision.")
        }
        if try ImplementationEvidence.fingerprint(old) == ImplementationEvidence.fingerprint(local) { return }
        guard let result = newestCompatibleResult(local), result.recordedAt != nil, result.quickCheck, result.gate_passed, !result.metrics.isEmpty,
            let checks = result.checks, [checks.protocolStatus, checks.parity, checks.task, checks.speed].allSatisfy({ $0 == .pass }),
            checks.reference?["independent"]?.bool == true
        else {
            throw WorkerFailure(
                code: "revision_gate",
                message: "Staged revisions need current protocol/parity/task evidence against an independent reference and a passed quick speed check before approval.")
        }
    }
    func scheduleRevisionSwap(_ id: String) {
        guard !shuttingDown, !updating, !reloading.contains(id), gets[id] == nil, loadTasks[id] == nil, pendingRuns[id, default: 0] == 0, waitingRevisionRuns[id, default: 0] == 0,
            residents[id]?.busy == 0, let local = localModels[id], config.localCodeConsent[id] == local.contentDigest,
            rejectedRevisionDigests[id] != local.contentDigest, !sameImplementation(id),
            config.localCodeApprovals[id]?.digest == local.contentDigest || (try? requireSwapEvidence(local)) != nil
        else { return }
        let precision = modelRows[id]?.selectedPrecision ?? local.manifest.defaultPrecision
        guard snapshotPresent(local.manifest, precision), selectedRunner(local.manifest) != nil
        else { return }
        reloading.insert(id); updateRevisionRow(id); changed()
        swapTasks[id] = Task { await LoadMemoryContext.$force.withValue(false) { await self.swapRevision(id) } }
    }
    func swapRevision(_ id: String) async {
        let attemptedDigest = localModels[id]?.contentDigest
        defer {
            reloading.remove(id); swapTasks[id] = nil
            if residents[id] == nil { clearResidentDescriptor(id) }
            updateRevisionRow(id); signalMemoryChange(); changed()
            if localModels[id]?.contentDigest != attemptedDigest { scheduleRevisionSwap(id) }
        }
        guard let old = captureRevision(id), let candidate = localModels[id] else { return }
        do {
            try Task.checkCancellation()
            operationCache[id] = nil; operationFailures[id] = nil
            try await performLoad(
                candidate.manifest, precision: modelRows[id]?.selectedPrecision ?? candidate.manifest.defaultPrecision, fast: old.preference.fast,
                residency: old.resident.residency,
                waitForMemory: true)
            armTimer(id)
            previousRevisions[id] = old; rejectedRevisionDigests[id] = nil; config.rejectedRevisions[id] = nil; try persist(); modelRows[id]?.revision?.error = nil
        } catch {
            if Task.isCancelled || shuttingDown { return }
            let transient = ["busy", "revision", "not_installed", "insufficient_memory", "updating"].contains((error as? WorkerFailure)?.code ?? "")
            if !transient {
                try? saveRecoveryRevision(id, old); rejectedRevisionDigests[id] = candidate.contentDigest; config.rejectedRevisions[id] = candidate.contentDigest; try? persist()
            }
            let failure = "Revision swap failed: \(error.localizedDescription)"
            let exited = await supervisor.hasExited(id: id)
            if !shuttingDown, residentLocalDigests[id] != old.digest || exited {
                do { try await restoreInterruptedRevision(id, old) } catch {
                    emit(.error(id: id, message: "\(failure); previous revision restore failed: \(error.localizedDescription)"))
                }
            }
            modelRows[id]?.revision?.error = failure; emit(.error(id: id, message: failure))
        }
    }
    /// A failed replacement may put back the resident the user already approved. A candidate, explicit rollback or
    /// later cold Load still goes through fresh admission; saved revision files carry no force permission.
    func restoreInterruptedRevision(_ id: String, _ old: ResidentRevision) async throws {
        try await LoadMemoryContext.$force.withValue(LoadMemoryContext.force || old.memoryForceApproved) { try await restoreRevision(id, old) }
    }
    func restoreRevision(_ id: String, _ old: ResidentRevision) async throws {
        let releasing = residents[id] != nil ? !(await supervisor.hasExited(id: id)) : false
        try checkLoadMemory(old.manifest, precision: old.preference.precision, reserve: true, requirement: revisionMemoryRequirement(old), releasingResident: releasing)
        defer { loadReservations[id] = nil; loadReservationReleases.remove(id); signalMemoryChange() }
        if residents[id] != nil { try await unloadResident(id: id) }
        loadReservationReleases.remove(id)
        let worker = try await supervisor.worker(
            id: id, launch: old.launch, manual: true,
            status: { [weak self] memory in await self?.status(id, memory) }, statusDetails: { [weak self] details in await self?.kernelStatus(id, details) })
        do {
            try await worker.verifyOperations(old.operations.filter { $0.name != "delete_voice" || old.manifest.kind != "voices" }.map(\.name))
            _ = try await worker.request(
                op: "load", params: old.load, timeout: 180,
                memoryCeiling: workerMemoryCeiling(
                    id, requirement: revisionMemoryRequirement(old), precision: old.preference.precision, forced: LoadMemoryContext.force))
            residentManifests[id] = old.manifest; residentOperations[id] = old.operations; residentSettings[id] = old.settings
            residentMemoryRequirements[id] = revisionMemoryRequirement(old)
            residentLaunches[id] = old.launch; residentLoads[id] = old.load; residentLocalDigests[id] = old.digest; residentMenus[id] = old.menus
            loadedPreferences[id] = old.preference; loadedRunners[id] = old.launch.runner
            var resident = old.resident; resident.lastUsed = Date().timeIntervalSince1970; resident.memoryMB = footprints[id]?.footprintMB ?? old.resident.memoryMB
            residents[id] = resident; recordResidentMemoryApproval(id)
            modelSettings.applied[id] = old.settings.applied[id]; modelRows[id]?.residentRunner = old.launch.runner
            modelRows[id]?.states[old.preference.precision] = .init(state: .loaded)
            await fetcher.store.retain(model: id); updatePending(id); updateMetrics(old.manifest); armTimer(id)
        } catch { await supervisor.unload(id: id); throw error }
    }
    public func rollbackRevision(id: String) async throws {
        guard let previous = previousRevisions[id], !reloading.contains(id) else {
            throw WorkerFailure(code: "revision", message: "No previous revision is available, or a swap is already in progress.")
        }
        reloading.insert(id); updateRevisionRow(id); changed()
        defer { reloading.remove(id); updateRevisionRow(id); signalMemoryChange(); changed() }
        while pendingRuns[id, default: 0] > 0 || (residents[id]?.busy ?? 0) > 0 { try await waitForMemoryChange(); try Task.checkCancellation() }
        let current = captureRevision(id)
        do {
            try saveRecoveryRevision(id, previous)
            try await restoreRevision(id, previous)
            previousRevisions[id] = current; rejectedRevisionDigests[id] = localModels[id]?.contentDigest
            config.rejectedRevisions[id] = localModels[id]?.contentDigest; try persist()
        } catch {
            if let current, residentLocalDigests[id] != current.digest || residents[id] == nil { try? await restoreInterruptedRevision(id, current) }; throw error
        }
    }
}
