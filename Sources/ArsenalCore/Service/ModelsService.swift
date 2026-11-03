// SPDX-License-Identifier: MIT
import Foundation
@_exported import ArsenalWire

public enum ModelState: String, Codable, Sendable { case notInstalled, updateAvailable, downloading, deriving, ready, loading, loaded, failed }
public enum ModelBadge: String, Codable, Sendable { case measured = "Measured", otherChip = "Measured on another chip", unverified = "Unverified" }
public struct PrecisionState: Codable, Sendable, Equatable {
    public var state: ModelState; public var progress: Double?; public var bytes: Int64?; public var totalBytes: Int64?; public var error: String?
    public var bytesPerSecond: Double?; public var phase: String?
    /// A Get waiting for a person to accept a gated model's terms on Hugging Face (phase `gated`).
    public var gate: GateWait?
    /// `PrecisionState.phase` of a Get refused by a gated repository.
    public static let gatedPhase = "gated"
    public init(
        state: ModelState = .notInstalled, progress: Double? = nil, bytes: Int64? = nil, totalBytes: Int64? = nil, error: String? = nil, bytesPerSecond: Double? = nil,
        phase: String? = nil, gate: GateWait? = nil
    ) {
        self.state = state; self.progress = progress; self.bytes = bytes; self.totalBytes = totalBytes; self.error = error; self.bytesPerSecond = bytesPerSecond; self.phase = phase
        self.gate = gate
    }
}
public enum ApprovalSource: String, Codable, Sendable { case app, cli, api, mcp }
public struct CodeApproval: Codable, Sendable, Equatable {
    public var by: ApprovalSource; public var at: String; public var digest: String
    public init(by: ApprovalSource, at: String = ISO8601DateFormatter().string(from: Date()), digest: String) { self.by = by; self.at = at; self.digest = digest }
}
public struct ModelRow: Codable, Sendable, Equatable, Identifiable {
    public var id: String; public var name: String; public var kind: String; public var params: String; public var licence: String
    public var precisions: [String]; public var states: [String: PrecisionState]; public var selectedPrecision: String; public var fast: Bool; public var runner: String
    public var badge: ModelBadge; public var metrics: JSONObject; public var peakRAMMB: Double?; public var keepHot: Bool; public var sizeBytes: Int64?; public var summary: String
    public var downloadBytes: [String: Int64]; public var verification: String; public var publisher: String; public var released: Int?; public var commercial: Bool
    public var runnerLanguage: String; public var fastExact: Bool; public var quickMetrics: [String: Bool]; public var retentionNote: String?
    public var localCopyBytes: [String: Int64]
    /// What the model does in a few plain words (the folder's `purpose`, else derived): the line under its name.
    public var purpose: String = ""
    public var tools: [String] = []
    public var toolsShown: Bool = true
    public var activity: String = "default"
    public var formats: [ModelFormat] = []
    public var recommendedPrecision: String?
    public var recommendationPending: Bool = false
    public var recommendationReason: String = ""
    public var precisionDtypes: [String: String] = [:]
    public var precisionNotes: [String: String] = [:]
    public var precisionStats: [String: PrecisionStatSet] = [:]
    /// A local folder whose kind declares no table figures: its quick-check figures by precision, from Arsenal's own
    /// Resources or the folder's quick-check records, never from a file added to the approved folder (FolderReferenceFigures).
    public var folderFigures: [String: FolderReferenceFigures.Entry] = [:]
    public var precisionQualityWarnings: [String: String] {
        guard precisions.contains("int4"), precisionDtypes["int4"] == nil else { return [:] }
        let stats = precisionStats["int4"]
        let passed =
            stats?.runner == runner && stats?.dtype == "int4" && stats?.kind.rawValue == kind
            && stats?.qualityCheckPassed == true && stats?.availability.state != .unavailable
        let failed =
            stats?.runner == runner && stats?.dtype == "int4" && stats?.kind.rawValue == kind
            && stats?.applyingInt4Policy.qualityCheckPassed == false
        return passed ? [:] : ["int4": failed ? PrecisionStatSet.int4QualityWarning : PrecisionStatSet.int4UncheckedWarning]
    }
    public func dtypeUnavailableReason(_ dtype: String) -> String? {
        if let format = formats.first(where: { $0.id == dtype }), !format.runners.contains(runner) {
            return format.runners.isEmpty ? "No runner in this Arsenal yet." : "The selected runner does not support this dtype."
        }
        if let stats = precisionStats[dtype], stats.runner == runner, stats.dtype == dtype, stats.kind.rawValue == kind {
            return stats.selectionUnavailableReason
        }
        return nil
    }
    public var runnerOverride: String?
    public var defaultRunner: String?
    public var residentRunner: String?
    public var approval: CodeApproval?
    public var settingsWarning: String?
    public var local: Bool = false
    public var needsCodeConsent: Bool = false
    public var codeConsentRequired: Bool {
        get { needsCodeConsent }
        set { needsCodeConsent = newValue }
    }
    public var contentDigest: String?
    public var fastAvailable: Bool { fastExact || hasFastKernels }
    public var hasFastKernels: Bool = false
    public var kernels: [JSONObject] = []
    public var revision: ModelRevision?
    public var checks: ModelChecksStatus?
    public var needsReload: Bool = false
    public var loadMemoryNote: String?
    /// Precisions still on disk that this model's entry no longer offers (int8/int4 left after it became 16-bit only),
    /// with their logical bytes. Arsenal never deletes them on its own; `deleteWithdrawnTiers` removes them, logged.
    public var withdrawnTiers: [String: Int64] = [:]
    /// In `rows()`, false for a running model the user has not put on the Models table (one an agent loaded): the table,
    /// the menu and tool windows show only table rows, while agents and the API use every downloaded model.
    public var onTable: Bool = true
    /// The Models table's rule: a model is on the table exactly while some precision has weights on this Mac or a Get of
    /// it is under way (a failed Get stays until it is retried or the app restarts). Everything else lives only in Get more
    /// models, and no weights on disk are ever hidden there, where they could not be deleted.
    public var holdsWeightsOrGet: Bool {
        !withdrawnTiers.isEmpty || states.values.contains(where: Self.holdsWeightsOrGet)
    }
    static func holdsWeightsOrGet(_ s: PrecisionState) -> Bool {
        (tableStates.contains(s.state) && !noWeightsFailures.contains(s.phase ?? "")) || s.phase == "localFolderErrorLoaded"
    }
    /// Failures of a local folder or its engine: no Get and no weights, so they keep no row on the table.
    public static let noWeightsFailures: Set<String> = ["localEngineError", "localFolderError"]
    public static let tableStates: Set<ModelState> = [.ready, .loaded, .loading, .updateAvailable, .downloading, .deriving, .failed]
    public var catalogRemoved: Bool { states.values.contains { $0.phase == "catalogRemoved" } }
    public var state: ModelState { states[selectedPrecision]?.state ?? .notInstalled }
    /// A broken source folder does not terminate its already-verified resident snapshot.
    public var residentDType: String? { precisions.first { states[$0]?.state == .loaded || states[$0]?.phase == "localFolderErrorLoaded" } }
    public init(
        id: String, name: String, kind: String, params: String = "", licence: String = "", precisions: [String] = ["bf16"], states: [String: PrecisionState] = [:],
        selectedPrecision: String = "bf16", fast: Bool = true, runner: String = "native", badge: ModelBadge = .unverified, metrics: JSONObject = [:], peakRAMMB: Double? = nil,
        keepHot: Bool = false, sizeBytes: Int64? = nil, summary: String = "", downloadBytes: [String: Int64] = [:], verification: String = "Not tried on this Mac yet",
        publisher: String = "", released: Int? = nil, commercial: Bool = true, runnerLanguage: String = "swift", fastExact: Bool = true, quickMetrics: [String: Bool] = [:],
        retentionNote: String? = nil, localCopyBytes: [String: Int64] = [:]
    ) {
        self.id = id; self.name = name; self.kind = kind; self.params = params; self.licence = licence; self.precisions = precisions; self.states = states
        self.selectedPrecision = selectedPrecision; self.fast = fast; self.runner = runner; self.badge = badge; self.metrics = metrics; self.peakRAMMB = peakRAMMB
        self.keepHot = keepHot; self.sizeBytes = sizeBytes; self.summary = summary; self.downloadBytes = downloadBytes; self.verification = verification
        self.publisher = publisher; self.released = released; self.commercial = commercial; self.runnerLanguage = runnerLanguage; self.fastExact = fastExact
        self.quickMetrics = quickMetrics; self.retentionNote = retentionNote; self.localCopyBytes = localCopyBytes
    }
}
extension KindSpec {
    public static let defaults = [
        KindSpec(id: "sound", title: "Sound", order: 4, columns: [.init(id: "rtf", title: "Speed", unit: "×")], icon: "waveform"),
        KindSpec(id: "enhancement", title: "Enhancement", order: 5, columns: [.init(id: "rtf", title: "Speed", unit: "×")], icon: "waveform.path"),
        KindSpec(
            id: "voices", title: "Voices", order: 0,
            columns: [
                .init(id: "rtf", title: "Speed", unit: "×"), .init(id: "first_audio_s", title: "First audio", unit: "s"), .init(id: "wer", title: "Round-trip WER", unit: "%")
            ]),
        KindSpec(
            id: "decisions", title: "Decisions", order: 1, columns: [.init(id: "accuracy", title: "Accuracy", unit: "%"), .init(id: "latency_ms", title: "Latency", unit: "ms")]),
        KindSpec(id: "images", title: "Images", order: 2, columns: [.init(id: "seconds", title: "Seconds per image", unit: "s")]),
        KindSpec(
            id: "chat", title: "Chat", order: 3,
            columns: [.init(id: "context", title: "Context"), .init(id: "tokens_per_s", title: "Tokens/s"), .init(id: "ttft_s", title: "First token", unit: "s")])
    ]
}
public struct LoadedAction: Codable, Sendable, Equatable, Identifiable {
    public var id: String; public var modelID: String; public var title: String; public var op: String; public var input: String; public var params: JSONObject
    public init(id: String, modelID: String, title: String, op: String, input: String = "none", params: JSONObject = [:]) {
        self.id = id; self.modelID = modelID; self.title = title; self.op = op; self.input = input; self.params = params
    }
}
public struct MemorySnapshot: Codable, Sendable, Equatable {
    public var totalMB: Double; public var availableMB: Double; public var workersMB: Double; public var models: [String: WorkerMemory]
    public init(totalMB: Double = Double(ProcessInfo.processInfo.physicalMemory) / 1e6, availableMB: Double = 0, workersMB: Double = 0, models: [String: WorkerMemory] = [:]) {
        self.totalMB = totalMB; self.availableMB = availableMB; self.workersMB = workersMB; self.models = models
    }
}
public struct CatalogStatus: Codable, Sendable, Equatable {
    public var source: String; public var pinnedCommit: String?; public var verified: Bool; public var error: String?
    public init(source: String = "local", pinnedCommit: String? = nil, verified: Bool = false, error: String? = nil) {
        self.source = source; self.pinnedCommit = pinnedCommit; self.verified = verified; self.error = error
    }
}
public enum ServiceEvent: Sendable {
    /// A getter's diagnostic notification, not an independent state change. Consumers may display it, but
    /// a refresh must not retry itself because of this feedback. Unsolicited worker/command events stay unwrapped.
    indirect case readFeedback(ServiceEvent)
    case rowsChanged, progress(id: String, precision: String, progress: Double), memory(MemorySnapshot), error(id: String?, message: String), catalog(CatalogStatus)
}
public typealias RunSink = @Sendable (WireFrame) async throws -> Void
/// A primary operation and the same readiness refusal a well-formed API request would receive.
public struct RunAdmission: Sendable {
    public let operation: String
    public let refusal: WorkerFailure?
    public init(operation: String, refusal: WorkerFailure?) { self.operation = operation; self.refusal = refusal }
    public static func primaryOperation(_ operations: [ModelOperation], preferred: String?) -> String {
        preferred.flatMap { name in operations.first { $0.name == name }?.name }
            ?? operations.first { $0.annotations?["primary"]?.bool == true }?.name ?? operations.first?.name ?? ""
    }
}
/// One admission question: a request's (`op`), or the primary operation's (`primary`, `preferred` first when it has it).
public struct AdmissionQuery: Sendable, Equatable {
    public var id: String
    public var op: String
    public var primary: Bool
    public var preferred: String?
    public init(id: String, op: String = "", primary: Bool = false, preferred: String? = nil) {
        self.id = id; self.op = op; self.primary = primary; self.preferred = preferred
    }
}
/// One precision of a model on disk: its logical bytes (every revision summed) and whether a loaded worker is using it.
public struct OnDiskPrecision: Sendable, Equatable {
    public let dtype: String
    public let bytes: Int64
    public let inUse: Bool
    /// Other copies on disk built from this one: deleting this copy would leave them unverifiable, so it is refused until they go.
    public let dependents: [String]
    public init(dtype: String, bytes: Int64, inUse: Bool, dependents: [String] = []) {
        self.dtype = dtype; self.bytes = bytes; self.inUse = inUse; self.dependents = dependents
    }
}
/// A model with weights or caches on disk, catalog or not. `cacheBytes` is its separate-asset snapshots and partial downloads.
public struct OnDiskModel: Sendable, Equatable, Identifiable {
    public let id: String
    public let precisions: [OnDiskPrecision]
    public let cacheBytes: Int64
    /// Still offered: in the signed catalog, or a local model folder on this Mac.
    public let offered: Bool
    public var bytes: Int64 { precisions.reduce(0) { $0 + $1.bytes } + cacheBytes }
    public init(id: String, precisions: [OnDiskPrecision], cacheBytes: Int64, offered: Bool) {
        self.id = id; self.precisions = precisions; self.cacheBytes = cacheBytes; self.offered = offered
    }
}
/// Since 21 Oct, only 16-bit is offered and int8 is dropped; int4 stays. A saved, recommended or default int8 falls back to the
/// model's 16-bit format, and a request for int8 (get, load, plan, precision choice) is refused.
public enum RetiredDtype {
    public static let names: Set<String> = ["int8"]
    public static func isRetired(_ dtype: String) -> Bool { names.contains(dtype) }
}
public protocol ModelsService: Sendable {
    func setToolsShown(id: String, on: Bool) async throws
    func reference(id: String, dtype: String?, mode: String?) async throws -> ModelReference?
    func memorySettings() async -> MemorySettings
    func setMemorySettings(_ settings: MemorySettings) async throws
    func beginUpdate() async throws -> [String]
    func cancelUpdate() async
    func restoreAfterUpdate(ids: [String]) async
    func activity() async -> ActivitySnapshot
    /// `machine: false`: requests only, without starting the machine readings (a tool window following its own run).
    func activityStream(machine: Bool) async -> AsyncStream<ActivitySnapshot>
    func setRunner(id: String, runner: String?) async throws
    func shutdown() async
    func approveLocalCode(id: String, contentDigest: String, approvedBy: ApprovalSource) async throws
    func approveLocalCode(id: String, contentDigest: String) async throws
    func approveLocalCode(id: String) async throws
    /// Forgets the code approval of a local model; the next Get or Load asks the owner again.
    func revokeLocalCode(id: String) async throws
    /// What approval could not check about a local folder's pinned weights files (Hugging Face unreachable, a gated
    /// listing), said once with the approval; nil when the pins were checked or no check applies.
    func pinnedFilesNote(id: String) async -> String?
    func operations(id: String) async -> [ModelOperation]
    /// The operation names a model declares, read from its manifest without resolving schemas, admitting or reporting
    /// anything (tool-window routing). Empty for an unknown model.
    func declaredOperationNames(id: String) async -> [String]
    func catalog(query: CatalogQuery) async -> [CatalogEntry]
    func add(id: String) async throws
    func remove(id: String, deleteWeights: Bool) async throws
    /// Lists a resident model’s voices without loading, emitting service events or recording an inference job.
    func voiceInventory(id: String) async throws -> JSONObject
    func settings(id: String) async throws -> [SettingState]
    /// The folder's settings panel (`settings.toml`), validated against its declarations; nil without one.
    func settingsPanel(id: String) async -> SettingsPanel?
    func saveDesignedVoice(id: String, previewID: String, name: String) async throws -> JSONObject
    func deleteSavedVoice(id: String, voiceID: String) async throws
    /// Atomically set model values and optional __tools (boolean menu placement), validating all before committing.
    func setSettings(id: String, values: JSONObject) async throws
    func setSetting(id: String, key: String, value: JSONValue) async throws
    func applySettings(id: String) async throws
    func rows() async -> [ModelRow]
    func kinds() async -> [KindSpec]
    /// Command/worker state changes use ordinary events. Getter diagnostics and subscription snapshots use
    /// `readFeedback`, so an observer can display them without mistaking a read for a new state change.
    func events() async -> AsyncStream<ServiceEvent>
    func get(id: String, precision: String) async throws
    func cancelGet(id: String) async
    func loadMemoryNotice(id: String, precision: String?) async -> String?
    func load(id: String, precision: String?, fast: Bool?) async throws
    func unload(id: String) async throws
    /// Unload Idle Models: unloads only those of `ids` still idle at the moment (a run that started, or Keep loaded turned on,
    /// while the memory pop-up was open keeps its model).
    func unloadIdle(ids: [String]) async
    /// Saves a supported precision preference without downloading or replacing a resident model.
    func setPrecision(id: String, precision: String) async throws
    func setKeepHot(id: String, on: Bool) async throws
    func delete(id: String) async throws
    /// Removes the snapshots of `ModelRow.withdrawnTiers` through the store's logged deletion, keeping offered tiers.
    func deleteWithdrawnTiers(id: String) async throws
    /// Every model id with any weights or caches on disk, catalog or not.
    func onDiskModels() async throws -> [OnDiskModel]
    /// Deletes one precision across every revision. Returns the bytes freed. Refuses while that precision is loaded or
    /// while the model has a Get or Load running; a repeated call finds nothing and returns 0.
    func deletePrecision(id: String, precision: String) async throws -> Int64
    /// Deletes every precision, revision, separate-asset snapshot and partial download of a model. Returns the bytes freed.
    func deleteAllWeights(id: String) async throws -> Int64
    /// Bytes of weights no model refers to (orphan blobs, unowned asset snapshots, stale import staging).
    func orphanBytes() async throws -> Int64
    /// Deletes the orphans `orphanBytes` reports. Returns the bytes freed.
    func deleteOrphans() async throws -> Int64
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject
    /// Admission for a well-formed request, without loading a worker or running the operation.
    /// Uses the same readiness refusals as `run`; nil permits on-demand loading or queueing.
    func runRefusal(id: String, op: String) async -> WorkerFailure?
    /// Resolves the primary operation and its admission without emitting events, even on lookup failure.
    func primaryOperationAdmission(id: String, preferred: String?) async -> RunAdmission
    /// Several admission questions answered in one call, in order (the menu asks for every model's tools at once).
    func admissions(_ queries: [AdmissionQuery]) async -> [RunAdmission]
    func runRaw(id: String, op: String, params: JSONObject) async throws -> Data
    func modelREADME(id: String) async throws -> String
    func rollbackRevision(id: String) async throws
    func refresh() async throws
    /// Why `refresh()` would be refused right now (a Get or Load is registered), or nil. A service emits `rowsChanged`
    /// whenever this changes, so a view can show Refresh as off for exactly as long as the service would refuse it.
    func refreshRefusal() async -> String?
    func get(id: String, precision: String, approvedPlan: GetPlan) async throws
    func getPlan(id: String, dtype: String) async throws -> GetPlan
    func menuSections() async -> [LoadedMenuSection]
    func menuActions() async -> [LoadedAction]
    func memory() async -> MemorySnapshot
    func catalogStatus() async -> CatalogStatus
    /// The saved IDs, including kinds absent from the current inventory.
    func persistedKindOrder() async -> [String]
    /// Replaces remembered positions; absent IDs may be included or deliberately omitted.
    func setKindOrder(_ order: [String]) async throws
    func residencySettings() async -> ResidencySettings
    func setResidencySettings(_ settings: ResidencySettings) async throws
    /// Retry now every Get waiting on a gated model (a Hugging Face token was just saved).
    func retryGatedGets() async
}
extension ModelsService {
    public func unloadIdle(ids: [String]) async { for id in ids { try? await unload(id: id) } }
    public func pinnedFilesNote(id: String) async -> String? { nil }
    public func retryGatedGets() async {}
    public func deleteWithdrawnTiers(id: String) async throws {
        throw WorkerFailure(code: "unsupported", message: "This service cannot remove withdrawn precisions.")
    }
    public func onDiskModels() async throws -> [OnDiskModel] { throw WorkerFailure(code: "unsupported", message: "This service cannot list weights on disk.") }
    public func deletePrecision(id: String, precision: String) async throws -> Int64 {
        throw WorkerFailure(code: "unsupported", message: "This service cannot delete a precision's weights.")
    }
    public func deleteAllWeights(id: String) async throws -> Int64 {
        throw WorkerFailure(code: "unsupported", message: "This service cannot delete a model's weights.")
    }
    public func orphanBytes() async throws -> Int64 { throw WorkerFailure(code: "unsupported", message: "This service cannot measure orphaned weights.") }
    public func deleteOrphans() async throws -> Int64 { throw WorkerFailure(code: "unsupported", message: "This service cannot delete orphaned weights.") }
    public func setPrecision(id: String, precision: String) async throws {
        throw WorkerFailure(code: "unsupported", message: "This service cannot save precision preferences.")
    }
    public func setToolsShown(id: String, on: Bool) async throws {}
    public func reference(id: String, dtype: String?, mode: String?) async throws -> ModelReference? { nil }
    public func memorySettings() async -> MemorySettings { MemorySettings() }
    public func setMemorySettings(_ settings: MemorySettings) async throws {}
    public func beginUpdate() async throws -> [String] { await rows().filter { $0.residentDType != nil }.map(\.id) }
    public func cancelUpdate() async {}
    public func restoreAfterUpdate(ids: [String]) async { for id in ids { try? await load(id: id) } }
    public func shutdown() async {}
    public func modelREADME(id: String) async throws -> String { throw WorkerFailure(code: "not_found", message: "Model README unavailable.") }
    public func rollbackRevision(id: String) async throws { throw WorkerFailure(code: "unsupported", message: "No previous model revision.") }
    public func refreshRefusal() async -> String? { nil }
    public func voiceInventory(id: String) async throws -> JSONObject {
        throw WorkerFailure(code: "unsupported", message: "This service cannot read voice inventory.")
    }
    public func runRefusal(id: String, op: String) async -> WorkerFailure? { nil }
    public func declaredOperationNames(id: String) async -> [String] { await operations(id: id).map(\.name) }
    public func primaryOperationAdmission(id: String, preferred: String?) async -> RunAdmission {
        RunAdmission(operation: preferred ?? "", refusal: WorkerFailure(code: "unsupported", message: "This service cannot check operation readiness."))
    }
    public func admissions(_ queries: [AdmissionQuery]) async -> [RunAdmission] {
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
    public func persistedKindOrder() async -> [String] { await kinds().map(\.id) }
    public func setKindOrder(_ order: [String]) async throws {}
    public func residencySettings() async -> ResidencySettings { ResidencySettings() }
    public func setResidencySettings(_ settings: ResidencySettings) async throws {}
    public func load(id: String) async throws { try await load(id: id, precision: nil, fast: nil) }
    public func run(id: String, op: String, params: JSONObject = [:]) async throws -> JSONObject { try await run(id: id, op: op, params: params, sink: nil) }
}
