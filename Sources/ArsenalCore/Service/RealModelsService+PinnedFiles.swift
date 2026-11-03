// SPDX-License-Identifier: MIT
import Foundation

/// A local folder's pinned weights files are checked against its Hugging Face repository at the pinned revision, once per
/// folder content digest: the base weights in the background when the folder is discovered and awaited at approval, and
/// what a precision's Get downloads before that Get. A file the repository lacks makes the folder an error row naming it, so it is never approved or downloaded.
/// When Hugging Face cannot be reached the pins stay unchecked and approval goes ahead, saying so (`pinnedFilesNote`); a
/// gated listing leaves the gate to the Get's own wait. Folders whose weights are already on the Mac are not checked.
struct PinnedFilesState {
    /// Lists a folder's Hugging Face repository; the app sets it, nil in tests and tools (no check, no network call).
    var lister: PinnedFilesLister?
    /// Results by folder content digest, and the checks in flight.
    var checks: [String: PinnedFilesCheck] = [:]
    var tasks: [String: Task<PinnedFilesCheck, Never>] = [:]
}

extension RealModelsService {
    /// The app passes Hugging Face (`HuggingFaceClient.listing`); nil turns the check off.
    public func setPinnedFilesLister(_ lister: PinnedFilesLister?) { pinnedFiles = PinnedFilesState(lister: lister) }
    /// Approval: a broken folder (also one shown only as an error row) is refused with its reason, then the pins are
    /// checked against Hugging Face before the yes is recorded.
    func requireApprovableFolder(_ id: String) async throws {
        if let problem = brokenLocalFolders[id] ?? lastLocalDiscovery?.problems.first(where: { $0.id == id })?.message {
            throw WorkerFailure(code: "local_folder", message: problem)
        }
        try await requirePinnedFiles(id)
    }
    public func pinnedFilesNote(id: String) async -> String? {
        localModels[id].flatMap { pinnedFiles.checks[Self.pinnedKey($0.contentDigest, PinnedFilesCheck.baseSources($0.manifest))]?.note }
    }
    /// Discovery: a folder whose last check found a missing pin becomes a problem; an unchecked one starts its check.
    func applyPinnedFilesChecks(_ report: LocalFolderDiscovery) -> LocalFolderDiscovery {
        guard pinnedFiles.lister != nil else { return report }
        var result = report
        result.folders = []
        for local in report.folders {
            guard !weightsOnMac(local) else { result.folders.append(local); continue }
            let sources = PinnedFilesCheck.baseSources(local.manifest), key = Self.pinnedKey(local.contentDigest, sources)
            if let problem = pinnedFiles.checks[key]?.problem {
                result.problems.append(.init(id: local.manifest.id, message: problem)); continue
            }
            if pinnedFiles.checks[key] == nil { startPinnedFilesCheck(key, sources: sources) }
            result.folders.append(local)
        }
        result.problems.sort { $0.id < $1.id }
        return result
    }
    /// Approval: waits for this digest's base check (repeating one that could not reach Hugging Face) and refuses a folder
    /// whose base pins the repository lacks, turning it into its error row.
    func requirePinnedFiles(_ id: String) async throws {
        guard pinnedFiles.lister != nil, let local = localModels[id], !weightsOnMac(local) else { return }
        guard let problem = await pinnedFilesResult(local.contentDigest, sources: PinnedFilesCheck.baseSources(local.manifest))?.problem else { return }
        try? await discoverLocalModels(); changed()
        throw WorkerFailure(code: "local_folder", message: problem)
    }
    /// Get: checks what this precision's Get downloads (its vendor release, or the base weights, and the assets they need),
    /// skipped when that precision is already on disk. A base pin the repository lacks also makes the folder its error row;
    /// a bad vendor pin refuses only that precision.
    func requirePinnedFiles(_ id: String, precision: String) async throws {
        guard pinnedFiles.lister != nil, let local = localModels[id] else { return }
        let m = local.manifest, sources = downloadSources(m, precision: precision)
        guard !snapshotPresent(m, precision), !sources.isEmpty else { return }
        let base = PinnedFilesCheck.baseSources(m)
        guard let problem = await pinnedFilesResult(local.contentDigest, sources: sources)?.problem else { return }
        if sources == base || pinnedFiles.checks[Self.pinnedKey(local.contentDigest, base)]?.problem != nil {
            try? await discoverLocalModels(); changed()
            throw WorkerFailure(code: "local_folder", message: problem)
        }
        // The row says why this precision cannot be got (an MCP agent following it is told not to retry).
        modelRows[id]?.states[precision] = .init(
            state: loadedPreferences[id]?.precision == precision ? .loaded : .failed, error: problem, phase: Self.pinnedFileMissing)
        changed()
        throw WorkerFailure(code: Self.pinnedFileMissing, message: problem)
    }
    /// The row phase of a Get that failed on a pinned file the repository lacks: MCP following that Get says not to retry.
    public static let pinnedFileMissing = "pinned_file_missing"
    static func pinnedFileMissingPhase(_ error: any Error) -> String? {
        error is HFMissingFileError || (error as? WorkerFailure)?.code == pinnedFileMissing ? pinnedFileMissing : nil
    }
    /// A token was saved: checks that saw no working token are dropped and the folders scanned again.
    func pinnedFilesTokenChanged() async {
        guard pinnedFiles.lister != nil else { return }
        pinnedFiles.checks = pinnedFiles.checks.filter { _, result in
            if case .repoMissing = result { return false }
            if case .gated = result { return false }
            return result.isFinal
        }
        try? await discoverLocalModels(); changed()
    }
    /// The cached result for these sources of this digest, listing now (or joining the listing in flight) when there is none
    /// or the last one is worth repeating.
    private func pinnedFilesResult(_ digest: String, sources: [WeightSpec]) async -> PinnedFilesCheck? {
        let key = Self.pinnedKey(digest, sources)
        if pinnedFiles.checks[key]?.isFinal == false { pinnedFiles.checks[key] = nil }
        if let cached = pinnedFiles.checks[key] { return cached }
        guard let task = startPinnedFilesCheck(key, sources: sources) else { return nil }
        let result = await task.value
        pinnedFiles.checks[key] = result
        return result
    }
    /// Results are kept per folder digest and per set of sources (the base, or one precision's Get).
    static func pinnedKey(_ digest: String, _ sources: [WeightSpec]) -> String {
        digest + "|" + FileDigest.sha256(Data(sources.map { "\($0.repoType ?? "model"):\($0.repo)@\($0.revision):\($0.files.joined(separator: ","))" }.joined(separator: ";").utf8))
    }
    @discardableResult
    private func startPinnedFilesCheck(_ key: String, sources: [WeightSpec]) -> Task<PinnedFilesCheck, Never>? {
        guard let lister = pinnedFiles.lister else { return nil }
        if let running = pinnedFiles.tasks[key] { return running }
        let task = Task { await PinnedFilesCheck.run(sources: sources, lister: lister) }
        pinnedFiles.tasks[key] = task
        Task { [weak self] in await self?.finishPinnedFilesCheck(key, await task.value) }
        return task
    }
    private func finishPinnedFilesCheck(_ key: String, _ result: PinnedFilesCheck) async {
        pinnedFiles.tasks[key] = nil
        guard pinnedFiles.lister != nil else { return }
        // Unreachable is kept too, so discovery does not list again on every scan; approval and Get repeat it.
        pinnedFiles.checks[key] = result
        if pinnedFiles.checks.count > 256 { pinnedFiles.checks = [key: result] }
        guard result.problem != nil, localModels.values.contains(where: { Self.pinnedKey($0.contentDigest, PinnedFilesCheck.baseSources($0.manifest)) == key }) else { return }
        try? await discoverLocalModels(); changed()
    }
    /// A precision already on the Mac needs no download: discovery and approval leave such a folder alone, and its Get checks
    /// only what it still downloads.
    private func weightsOnMac(_ local: LocalModelFolder) -> Bool {
        local.manifest.precisions.contains { snapshotPresent(local.manifest, $0) }
    }
}
