// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

/// The stub service with the fixture kinds (kinds/*.toml) and a working group order: each `setKindOrder` is recorded
/// and `kinds()` comes back in it, as core's service does. Used by the stub app, the renders, the smoke and tests.
actor FixtureService: ModelsService {
    let stub: StubModelsService
    private(set) var saved: [[String]] = []
    /// The table's v2 calls in order ("load kokoro-82m int8", "apply qwen", …), for tests.
    private(set) var calls: [String] = []
    let kindSpecs: [KindSpec]
    /// `rows`: the table (all added); `catalog`: further models only the browser lists; `metadata`: folder fields
    /// (settings, facets, suggested, popularity) per id.
    init(
        rows: [ModelRow], catalog: [ModelRow] = [], metadata: [String: JSONObject] = [:], kinds: [KindSpec] = Fixtures.kinds, supportDirectory: URL? = nil,
        operations: [String: [ModelOperation]] = [:]
    ) {
        let stub = StubModelsService(
            rows: rows + catalog, metadata: metadata, operations: operations.merging([LayersFixture.id: [LayersFixture.operation]]) { a, _ in a },
            supportDirectory: supportDirectory)
        self.stub = stub; kindSpecs = kinds
        removed = Set(catalog.map(\.id))
        withdrawn = Dictionary(rows.filter { !$0.withdrawnTiers.isEmpty }.map { ($0.id, $0.withdrawnTiers) }) { first, _ in first }
    }
    /// Old precisions the fixture rows report on this Mac (`ModelRow.withdrawnTiers`); `deleteWithdrawnTiers` removes them.
    private var withdrawn: [String: [String: Int64]]
    /// The stub adds every row it is given to its table; catalog-only rows start removed.
    private var removed: Set<String>
    private func membership() async {
        for id in removed { try? await stub.remove(id: id, deleteWeights: false) }
        removed = []
    }
    func catalog(query: CatalogQuery) async -> [CatalogEntry] {
        await membership()
        return await stub.catalog(query: query).map { entry in
            var entry = entry
            if let sizes = catalogSizes[entry.id] { entry.sizesByDtype = sizes }
            return entry
        }
    }
    /// Renders and tests: each model's size per precision as the real catalog lists it (`ModelManifest.sizesByDtype`);
    /// without one, the stub's entry carries the row's download bytes.
    private(set) var catalogSizes: [String: [String: Int64]] = [:]
    func setCatalogSizes(_ sizes: [String: [String: Int64]]) { catalogSizes = sizes }
    func add(id: String) async throws { await membership(); try await stub.add(id: id) }
    func remove(id: String, deleteWeights: Bool) async throws {
        await membership(); calls.append("remove \(id) \(deleteWeights ? "delete" : "keep")"); try await stub.remove(id: id, deleteWeights: deleteWeights)
    }
    func setRunner(id: String, runner: String?) async throws { calls.append("runner \(id) \(runner ?? "default")"); try await stub.setRunner(id: id, runner: runner) }
    private(set) var voiceReads: [String] = []
    func voiceInventory(id: String) async throws -> JSONObject {
        voiceReads.append(id)
        if let delay = delays["voices"] { try await Task.sleep(for: delay) }
        return try await stub.voiceInventory(id: id)
    }
    func settings(id: String) async throws -> [SettingState] { try await stub.settings(id: id) }
    func settingsPanel(id: String) async -> SettingsPanel? { await stub.settingsPanel(id: id) }
    func setSetting(id: String, key: String, value: JSONValue) async throws {
        if let failure = failures["setSetting"] { throw failure }
        try await stub.setSetting(id: id, key: key, value: value)
    }
    func applySettings(id: String) async throws {
        calls.append("apply \(id)")
        if let failure = failures["apply"] { throw failure }
        try await stub.applySettings(id: id)
    }
    /// Tests make a call refuse ("setSetting", "apply"), as the service refuses a value or a Reload fails.
    private var failures: [String: WorkerFailure] = [:]
    func fail(_ call: String, with failure: WorkerFailure?) { failures[call] = failure }
    func approveLocalCode(id: String, contentDigest: String) async throws { try await approveLocalCode(id: id, contentDigest: contentDigest, approvedBy: .cli) }
    func revokeLocalCode(id: String) async throws {
        calls.append("revoke \(id)")
        try await stub.revokeLocalCode(id: id)
    }
    func approveLocalCode(id: String, contentDigest: String, approvedBy: ApprovalSource) async throws {
        calls.append("approve \(id) \(contentDigest.prefix(4)) \(approvedBy.rawValue)")
        try await stub.approveLocalCode(id: id, contentDigest: contentDigest, approvedBy: approvedBy)
    }
    func setSettings(id: String, values: JSONObject) async throws { try await stub.setSettings(id: id, values: values) }
    func deleteSavedVoice(id: String, voiceID: String) async throws { try await stub.deleteSavedVoice(id: id, voiceID: voiceID) }
    func operations(id: String) async -> [ModelOperation] {
        id == LayersFixture.id ? [LayersFixture.operation] : await stub.operations(id: id)
    }
    func runRefusal(id: String, op: String) async -> WorkerFailure? { await stub.runRefusal(id: id, op: op) }
    func primaryOperationAdmission(id: String, preferred: String?) async -> RunAdmission { await stub.primaryOperationAdmission(id: id, preferred: preferred) }
    func rows() async -> [ModelRow] {
        await membership()
        var rows = await stub.rows()
        for i in rows.indices { rows[i].withdrawnTiers = withdrawn[rows[i].id] ?? [:] }
        return rows
    }
    /// Removes the fixture's old precisions for this model, as the service does; the offered precisions stay.
    func deleteWithdrawnTiers(id: String) async throws {
        calls.append("delete withdrawn \(id)")
        if let failure = failures["deleteWithdrawnTiers"] { throw failure }
        withdrawn[id] = nil
    }
    func persistedKindOrder() async -> [String] { saved.last ?? kindSpecs.map(\.id) }
    func kinds() async -> [KindSpec] {
        let order = await persistedKindOrder()
        return kindSpecs.sorted { a, b in
            let ia = order.firstIndex(of: a.id) ?? Int.max, ib = order.firstIndex(of: b.id) ?? Int.max
            return ia != ib ? ia < ib : (a.order, a.id) < (b.order, b.id)
        }
    }
    private var listeners: [AsyncStream<ServiceEvent>.Continuation] = []
    /// The stub's events plus any a test sends with `emit` (the stub itself only reports row changes).
    func events() async -> AsyncStream<ServiceEvent> {
        let upstream = await stub.events()
        let (stream, continuation) = AsyncStream<ServiceEvent>.makeStream()
        listeners.append(continuation)
        let forward = Task {
            for await event in upstream { continuation.yield(event) }; continuation.finish()
        }
        continuation.onTermination = { _ in forward.cancel() }
        return stream
    }
    func emit(_ event: ServiceEvent) { for listener in listeners { listener.yield(event) } }
    func get(id: String, precision: String) async throws { try await stub.get(id: id, precision: precision) }
    /// Tests make the next Get of a model find it needs more than its approved plan (a local copy failed its hash): it
    /// stops before downloading and returns this plan, as core does. `approvedGets`: each plan a Get ran under.
    private var revisions: [String: GetPlan] = [:]
    private(set) var approvedGets: [GetPlan] = []
    func setRevision(_ plan: GetPlan) { revisions[plan.id] = plan }
    func get(id: String, precision: String, approvedPlan: GetPlan) async throws {
        approvedGets.append(approvedPlan)
        if let revised = revisions.removeValue(forKey: id) { throw GetPlanChanged(revisedPlan: revised) }
        try await stub.get(id: id, precision: precision)
    }
    func cancelGet(id: String) async { await stub.cancelGet(id: id) }
    func load(id: String, precision: String?, fast: Bool?) async throws {
        calls.append("load \(id) \(precision ?? "-")"); try await stub.load(id: id, precision: precision, fast: fast)
    }
    func unload(id: String) async throws { try await stub.unload(id: id) }
    func setPrecision(id: String, precision: String) async throws { try await stub.setPrecision(id: id, precision: precision) }
    func setKeepHot(id: String, on: Bool) async throws { try await stub.setKeepHot(id: id, on: on) }
    func delete(id: String) async throws { try await stub.delete(id: id) }
    /// Tests: an operation that takes this long (cancellable), and a chat that fails after its first tokens.
    private var delays: [String: Duration] = [:]
    /// Every operation run, in order ("qwen3.8-27b chat").
    private(set) var runs: [String] = []
    /// API inputs reaching the fixture, before the stub merges model settings.
    private(set) var runInputs: [JSONObject] = []
    private var chatFailure: WorkerFailure?
    func setDelay(_ op: String, _ delay: Duration?) { delays[op] = delay }

    // MARK: Weights on disk
    /// Weights on disk that no catalog row lists (the retired models, or any model the catalog dropped): set by tests and
    /// renders. The stub lists only its own rows; these are added to its listing and deleted like its own.
    private(set) var unlisted: [OnDiskModel] = []
    func setUnlisted(_ models: [OnDiskModel]) { unlisted = models }
    /// Bytes of weights no model refers to (`orphanBytes`); `deleteOrphans` frees them all.
    private(set) var orphans: Int64 = 0
    func setOrphanBytes(_ bytes: Int64) { orphans = bytes }
    /// Every store call that deletes weights, in order ("precision id dtype", "all id", "orphans"), for tests.
    private(set) var deletes: [String] = []
    func onDiskModels() async throws -> [OnDiskModel] {
        let own = try await stub.onDiskModels().filter { onDiskOverride[$0.id] == nil }
        return own + onDiskOverride.values.sorted { $0.id < $1.id } + unlisted
    }
    /// Renders: a listed model's on-disk precisions as a real store reports them (bytes per precision, a withdrawn tier
    /// still on disk), in place of the stub's listing from the row's states.
    private(set) var onDiskOverride: [String: OnDiskModel] = [:]
    func setOnDisk(_ models: [OnDiskModel]) { onDiskOverride = Dictionary(models.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first }) }
    func deletePrecision(id: String, precision: String) async throws -> Int64 {
        deletes.append("precision \(id) \(precision)")
        if let delay = delays["deletePrecision"] { try await Task.sleep(for: delay) }
        if let failure = failures["deletePrecision"] { throw failure }
        guard let i = unlisted.firstIndex(where: { $0.id == id }) else { return try await stub.deletePrecision(id: id, precision: precision) }
        let model = unlisted[i]
        guard let copy = model.precisions.first(where: { $0.dtype == precision }) else { return 0 }
        unlisted[i] = OnDiskModel(id: id, precisions: model.precisions.filter { $0.dtype != precision }, cacheBytes: model.cacheBytes, offered: false)
        return copy.bytes
    }
    func deleteAllWeights(id: String) async throws -> Int64 {
        deletes.append("all \(id)")
        if let delay = delays["deleteAllWeights"] { try await Task.sleep(for: delay) }
        if let failure = failures["deleteAllWeights"] { throw failure }
        guard let i = unlisted.firstIndex(where: { $0.id == id }) else { return try await stub.deleteAllWeights(id: id) }
        let model = unlisted.remove(at: i)
        return model.bytes
    }
    func orphanBytes() async throws -> Int64 { orphans }
    func deleteOrphans() async throws -> Int64 {
        deletes.append("orphans")
        if let delay = delays["deleteOrphans"] { try await Task.sleep(for: delay) }
        if let failure = failures["deleteOrphans"] { throw failure }
        let freed = orphans
        orphans = 0
        return freed
    }
    /// Tests and renders answer an operation with their own fixture runner (its params include `out_dir` for files).
    private var runners: [String: @Sendable (JSONObject) throws -> JSONObject] = [:]
    func setRunner(_ op: String, _ runner: (@Sendable (JSONObject) throws -> JSONObject)?) { runners[op] = runner }
    func setChatFailure(_ failure: WorkerFailure?) { chatFailure = failure }
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject {
        if let refusal = await runRefusal(id: id, op: op) { throw refusal }
        runs.append("\(id) \(op)")
        runInputs.append(params)
        let activityID = activityAggregator.begin(model: id, op: op, requester: ActivityRequestContext.requester)
        defer { activityAggregator.finish(activityID) }
        activityAggregator.phase(activityID, op == "chat" ? "prefill" : "step")
        // As a runner does: it reports starting the request.
        activityAggregator.receive(activityID, .json(["event": "progress", "phase": .string(op == "chat" ? "prefill" : "step")]))
        if let runner = runners[op] {
            // Progress first, as a runner reports its first step before the work that takes time.
            let event: JSONObject = ["event": "progress", "phase": "step", "step": 1, "total": 2]
            activityAggregator.receive(activityID, .json(event))
            try await sink?(.json(event))
            if let delay = delays[op] { try await Task.sleep(for: delay) }
            return try runner(params)
        }
        if let delay = delays[op] { try await Task.sleep(for: delay) }
        if id == LayersFixture.id, op == "separate" { return try await separate(activityID, params: params, sink: sink) }
        if op == CutoutFixture.operation.name, await operations(id: id).contains(where: { $0.name == op }) {
            guard let dir = params["out_dir"]?.string else { throw WorkerFailure(code: "invalid_request", message: "out_dir is managed by Arsenal and was missing.") }
            let event: JSONObject = ["event": "progress", "phase": "step", "label": "Removing background"]
            activityAggregator.receive(activityID, .json(event))
            try await sink?(.json(event))
            return try CutoutFixture.run(params: params, directory: URL(fileURLWithPath: dir))
        }
        if op == "chat", let chatFailure {
            try await sink?(.json(["event": "token", "text": "Hello "]))
            throw chatFailure
        }
        return try await stub.run(id: id, op: op, params: params, sink: sink)
    }

    /// The layers fixture: one progress event per layer (label, step, total, unit), then that many PNGs in the job.
    private func separate(_ activityID: UUID, params: JSONObject, sink: RunSink?) async throws -> JSONObject {
        guard let dir = params["out_dir"]?.string else { throw WorkerFailure(code: "invalid_request", message: "out_dir is managed by Arsenal and was missing.") }
        let count = Int(params["layers"]?.number ?? 4)
        for step in 0...count {
            let event: JSONObject = [
                "event": "progress", "phase": "step", "label": "Separating layers", "unit": "layers", "step": .number(Double(step)), "total": .number(Double(count))
            ]
            activityAggregator.receive(activityID, .json(event))
            try await sink?(.json(event))
            if let delay = delays["separate"], step < count { try await Task.sleep(for: delay) }
        }
        return try LayersFixture.run(params: params, directory: URL(fileURLWithPath: dir))
    }

    /// C3 over the fixture rows, sampled by a fixed machine reading; tests count its subscribers.
    let activityAggregator = ActivityAggregator(sample: { MachineActivity(gpuBusyPct: 12, thermal: "nominal") })
    func activity() async -> ActivitySnapshot { await activityAggregator.snapshot(rows: rows(), memory: memory()) }
    func activityStream(machine: Bool) async -> AsyncStream<ActivitySnapshot> {
        await activityAggregator.stream(machine: machine, snapshot: { [weak self] in await self?.activity() ?? .init() })
    }
    func runLog() -> [String] { runs }
    func activitySubscribers() async -> Int { await activityAggregator.subscriberCount }
    func activitySamples() async -> Int { await activityAggregator.sampleCount }
    func menuActions() async -> [LoadedAction] { await stub.menuActions() }
    /// Tests make Refresh refuse (as core does while a Get or Load runs); `refreshes` counts the accepted ones.
    private(set) var refreshError: WorkerFailure?, refreshes = 0
    func setRefreshError(_ error: WorkerFailure?) { refreshError = error }
    func refresh() async throws {
        if let refreshError { throw refreshError }
        refreshes += 1
    }
    /// Tests set a model's Get plan (dependencies beyond the weights); otherwise the stub's weights-only plan.
    private(set) var plans: [String: GetPlan] = [:]
    func setPlan(_ plan: GetPlan) { plans[plan.id] = plan }
    /// Tests make a model's plan fail (the service could not work out its dependencies).
    private var planErrors: [String: WorkerFailure] = [:]
    func setPlanError(_ id: String, _ error: WorkerFailure?) { planErrors[id] = error }
    func getPlan(id: String, dtype: String) async throws -> GetPlan {
        if let error = planErrors[id] { throw error }
        if var plan = plans[id] { plan.dtype = dtype; return plan }
        var plan = try await stub.getPlan(id: id, dtype: dtype)
        // Fixture rows carry `sizeBytes` rather than per-dtype download figures: the table's own figure stands in. A
        // row with no size at all downloads nothing in the stub (an unknown size is a test's explicit `setPlan`).
        if let row = await stub.rows().first(where: { $0.id == id }) {
            for i in plan.items.indices where plan.items[i].downloadBytes == nil { plan.items[i].downloadBytes = RowPresentation.downloadBytes(row, dtype) ?? 0 }
        }
        return plan
    }
    func memory() async -> MemorySnapshot { await stub.memory() }
    func catalogStatus() async -> CatalogStatus { await stub.catalogStatus() }
    private var residency = ResidencySettings()
    func residencySettings() async -> ResidencySettings { residency }
    func setResidencySettings(_ settings: ResidencySettings) async throws { residency = settings }
    func memorySettings() async -> MemorySettings { await stub.memorySettings() }
    func setMemorySettings(_ settings: MemorySettings) async throws { calls.append("memory \(settings.whenFull.rawValue)"); try await stub.setMemorySettings(settings) }
    func setToolsShown(id: String, on: Bool) async throws { calls.append("tools \(id) \(on)"); try await stub.setToolsShown(id: id, on: on) }
    func reference(id: String, dtype: String?, mode: String?) async throws -> ModelReference? { try await stub.reference(id: id, dtype: dtype, mode: mode) }
    func beginUpdate() async throws -> [String] { try await stub.beginUpdate() }
    func cancelUpdate() async { await stub.cancelUpdate() }
    func setKindOrder(_ order: [String]) async throws {
        guard Set(order).count == order.count, order.allSatisfy(ModelManifest.validID) else {
            throw WorkerFailure(code: "kind_order", message: "Group order must contain unique kind IDs.")
        }
        saved.append(order)
    }
}
