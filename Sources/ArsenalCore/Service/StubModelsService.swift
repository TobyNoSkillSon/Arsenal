// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire

public actor StubModelsService: ModelsService {
    public static var localFixture: ModelRow {
        var row = ModelRow(id: "stub-local", name: "Local runner fixture", kind: "voices", runnerLanguage: "rust")
        row.local = true; row.codeConsentRequired = true; row.contentDigest = String(repeating: "a", count: 64)
        return row
    }
    private var loadedRunners: [String: String] = [:]
    private var loadedPreferences: [String: ModelPreferences] = [:]
    private let support: URL
    private var table: [String]
    private var modelSettings = ModelSettings()
    private var metadata: [String: JSONObject]
    private var declaredOps: [String: [ModelOperation]]
    private var models: [ModelRow]
    private var memoryValue = MemorySettings()
    private var settings = ResidencySettings()
    private var kindOrder: [String] = []
    private var listeners: [UUID: AsyncStream<ServiceEvent>.Continuation] = [:]
    public init(rows: [ModelRow]? = nil, metadata: [String: JSONObject] = [:], operations: [String: [ModelOperation]] = [:], supportDirectory: URL? = nil) {
        support = supportDirectory ?? URL(fileURLWithPath: "/tmp/arsenal-stub/" + UUID().uuidString)
        table = rows?.map(\.id) ?? []; self.metadata = metadata; declaredOps = operations
        for (id, fields) in metadata { modelSettings.declare(id, metadata: fields) }
        models =
            rows ?? [
                ModelRow(id: "stub-voices", name: "Stub Voice", kind: "voices", params: "82M", licence: "Apache-2.0"),
                ModelRow(id: "stub-decisions", name: "Stub Decision", kind: "decisions", params: "0.6B", licence: "Apache-2.0"),
                ModelRow(id: "stub-images", name: "Stub Image", kind: "images", params: "4B", licence: "Apache-2.0"), ModelRow(id: "stub-sound", name: "Stub Sound", kind: "sound"),
                ModelRow(id: "stub-enhancement", name: "Stub Enhancement", kind: "enhancement"),
                ModelRow(id: "stub-chat", name: "Qwen3.8-27B (stub)", kind: "chat", params: "27B", licence: "Apache-2.0")
            ]
        let saved = (try? ArsenalConfiguration.read(support.appendingPathComponent("config.json"))) ?? ArsenalConfiguration()
        memoryValue = saved.memory
        for i in models.indices {
            var row = models[i]
            if row.precisionStats.isEmpty {
                row.precisionStats = PrecisionStatsReader.sets(
                    kind: row.kind, baseDtype: row.precisions.first ?? "bf16", runner: row.runner, supported: row.precisions, results: [])
            }
            let recommendation = PrecisionRecommendation.choose(row.precisionStats, baseDtype: row.precisions.first ?? "bf16", chip: HostChip.current)
            row.recommendedPrecision = recommendation.dtype; row.recommendationPending = recommendation.pending; row.recommendationReason = recommendation.reason
            row.tools =
                metadata[row.id]?["tools"]?.array?.compactMap(\.string)
                ?? (row.kind == "voices" ? ["voice"] : row.kind == "chat" ? ["chat"] : row.kind == "images" ? ["image"] : [])
            if let precision = saved.selections[row.id]?.precision, row.precisions.contains(precision) { row.selectedPrecision = precision }
            row.toolsShown = saved.selections[row.id]?.toolsShown ?? true
            row.activity = metadata[row.id]?["activity"]?.string ?? row.tools.first ?? "default"
            row.defaultRunner = row.defaultRunner ?? metadata[row.id]?["selection"]?["default"]?.string ?? row.runner
            if let name = saved.runnerOverrides[row.id], let declared = metadata[row.id]?["runner"]?.array?.first(where: { $0["name"]?.string == name }) {
                let chips = declared["chips"]?.array?.compactMap(\.string) ?? ["*"]
                if chips.contains("*") || chips.contains(RunnerSelection.family(HostChip.current)) {
                    row.runner = name; row.runnerOverride = name; row.runnerLanguage = declared["language"]?.string ?? row.runnerLanguage
                }
            }
            if row.residentDType != nil { row.residentRunner = row.residentRunner ?? row.runner }
            if row.purpose.isEmpty {
                row.purpose =
                    metadata[row.id]?["purpose"]?.string
                    ?? ModelPurpose.derive(kind: row.kind, operations: operations[row.id]?.map(\.name) ?? Self.kindOperations(row.kind))
            }
            models[i] = row
        }
        for row in models {
            if let dtype = row.precisions.first(where: { row.states[$0]?.state == .loaded }) {
                loadedPreferences[row.id] = ModelPreferences(precision: dtype, fast: row.fast); loadedRunners[row.id] = row.residentRunner ?? row.runner
            }
        }
    }
    private func updatePending(_ id: String) throws {
        let i = try index(id), loaded = loadedPreferences[id]
        let selectable = [.ready, .loaded].contains(models[i].state)
        models[i].needsReload =
            (loaded != nil && (loaded?.precision != models[i].selectedPrecision || (selectable && (loaded?.fast != models[i].fast || loadedRunners[id] != models[i].runner))))
            || modelSettings.states(id, loaded: loaded != nil).contains { $0.pending }
    }
    public func getPlan(id: String, dtype: String) async throws -> GetPlan {
        let row = models[try index(id)]
        guard row.precisions.contains(dtype) else { throw WorkerFailure(code: "precision", message: "That dtype is not offered by \(id).") }
        if let reason = row.dtypeUnavailableReason(dtype) { throw WorkerFailure(code: "unsupported_format", message: reason) }
        // The stub never downloads. Explicit per-dtype or whole-row test sizes simulate a plan.
        return GetPlan(
            id: id, dtype: dtype, runner: row.runner,
            items: [
                .init(
                    id: "weights", name: row.name + " weights", kind: "weights", downloadBytes: row.downloadBytes[dtype] ?? row.sizeBytes ?? 0,
                    localCopyBytes: row.localCopyBytes[dtype] ?? 0, localCloneBytes: cloneBytes[id + "/" + dtype])
            ] + (extraPlanItems[id] ?? []))
    }
    /// Tests: a scripted Get. It reports each state (bytes, phase) as the row's state and a progress event, and may throw.
    public typealias GetScript = @Sendable (_ id: String, _ dtype: String, _ report: @escaping @Sendable (PrecisionState) async -> Void) async throws -> Void
    private var getScript: GetScript?
    private var extraPlanItems: [String: [GetPlanItem]] = [:]
    /// Every Get started, as "id dtype", in order.
    public private(set) var getCalls: [String] = []
    /// Every cancelGet, by model id, in order.
    public private(set) var cancelGetCalls: [String] = []
    private var cloneBytes: [String: Int64] = [:]
    /// Tests: the local-copy bytes of a plan that clone (keyed "id/dtype").
    public func setCloneBytes(_ id: String, _ dtype: String, _ bytes: Int64) { cloneBytes[id + "/" + dtype] = bytes }
    private var addSelects: [String: String] = [:]
    /// Tests: like the real service, adding a model may switch it to another dtype.
    public func setAddSelects(_ id: String, _ dtype: String) { addSelects[id] = dtype }
    /// Tests: runs at the start of every operation (a delay, or a thrown failure such as a memory refusal).
    public typealias RunHook = @Sendable (_ id: String, _ op: String) async throws -> Void
    private var runHook: RunHook?
    public func setRunHook(_ hook: RunHook?) { runHook = hook }
    public func setGetScript(_ script: GetScript?) { getScript = script }
    /// Tests: plan items beyond the weights (a dependency), per model.
    public func setExtraPlanItems(_ id: String, _ items: [GetPlanItem]) { extraPlanItems[id] = items }
    /// Tests: replaces a row's figures (sizes, states) while keeping the table.
    public func updateRow(_ id: String, _ change: @Sendable (inout ModelRow) -> Void) throws { change(&models[try index(id)]); changed() }
    private func report(_ id: String, _ dtype: String, _ state: PrecisionState) {
        guard let i = try? index(id) else { return }
        models[i].states[dtype] = state
        for c in listeners.values { c.yield(.progress(id: id, precision: dtype, progress: state.progress ?? 0)) }
    }
    /// As the real service: the models with weights or a Get under way (in table order), then models running off it
    /// (an agent's), marked `onTable: false`.
    public func rows() -> [ModelRow] {
        // A broken local folder stays where the user put it, as in the real service.
        let present = Set(models.filter { $0.holdsWeightsOrGet || (table.contains($0.id) && $0.states.values.contains { $0.phase == "localFolderError" }) }.map(\.id))
        let order = table.filter(present.contains) + models.map(\.id).filter { present.contains($0) && !table.contains($0) }
        return (order + loadedPreferences.keys.filter { !present.contains($0) }.sorted()).compactMap { id in
            models.first { $0.id == id }.map { row in
                var row = row; row.onTable = present.contains(id); return row
            }
        }
    }
    public func catalog(query: CatalogQuery = .init()) async -> [CatalogEntry] {
        CatalogEntry.query(
            models.map { row in
                var entry = CatalogEntry(row: row, metadata: metadata[row.id] ?? [:], added: row.holdsWeightsOrGet)
                entry.runners = runnerDetails(row); entry.settingsSummary = modelSettings.states(row.id, loaded: loadedPreferences[row.id] != nil)
                return entry
            }, query)
    }
    private func runnerDetails(_ row: ModelRow) -> [CatalogRunner] {
        let declarations = metadata[row.id]?["runner"]?.array?.compactMap(\.object) ?? [["name": .string(row.defaultRunner ?? row.runner), "language": .string(row.runnerLanguage)]]
        return declarations.compactMap { value in
            guard let name = value["name"]?.string else { return nil }
            let chips = value["chips"]?.array?.compactMap(\.string) ?? ["*"]
            let eligible = chips.contains("*") || chips.contains(RunnerSelection.family(HostChip.current))
            let language = value["language"]?.string ?? "swift"
            var detail = CatalogRunner(
                name: name, language: language, engine: value["worker"]?.string ?? value["entry"]?.string ?? "stub", acceptedDefault: name == row.defaultRunner, eligible: eligible,
                selected: name == row.runner,
                ineligibleReason: eligible ? nil : "Built for " + chips.map { $0.uppercased() }.joined(separator: ", "))
            detail.stats = row.precisionStats.filter { $0.value.runner == name }
            detail.features = value["features"]?.array?.compactMap { $0.string.flatMap(RunnerFeature.init(rawValue:)) } ?? []
            return detail
        }
    }
    public func setRunner(id: String, runner: String?) async throws {
        let i = try index(id), name = runner ?? models[i].defaultRunner
        guard let choice = runnerDetails(models[i]).first(where: { $0.name == name && $0.eligible }) else {
            throw WorkerFailure(code: "invalid_request", message: "That runner does not support this Mac.")
        }
        var configuration = try ArsenalConfiguration.read(support.appendingPathComponent("config.json")); configuration.runnerOverrides[id] = runner
        try configuration.write(support.appendingPathComponent("config.json"))
        models[i].runner = choice.name; models[i].runnerLanguage = choice.language; models[i].runnerOverride = runner
        try updatePending(id); changed()
    }
    public func add(id: String) async throws {
        let i = try index(id)
        if !table.contains(id) {
            table.append(id)
            if let dtype = addSelects[id] { models[i].selectedPrecision = dtype }
            // Added models stay out of the main menu until Show in menu is turned on.
            models[i].toolsShown = false
            let file = support.appendingPathComponent("config.json")
            var config = (try? ArsenalConfiguration.read(file)) ?? ArsenalConfiguration()
            var preference = config.selections[id] ?? .init(precision: models[i].selectedPrecision)
            preference.toolsShown = false; config.selections[id] = preference; try? config.write(file)
            changed()
        }
    }
    public func remove(id: String, deleteWeights: Bool = false) async throws {
        // As the real service: a loaded model is unloaded first, and a failed Get (no weights) leaves the table with its row;
        // so does a Get waiting at a gated repository (the real Remove cancels the wait).
        try unload(id: id)
        if deleteWeights {
            try delete(id: id)
        } else {
            let i = try index(id)
            for (p, s) in models[i].states where s.state == .failed && [nil, PrecisionState.gatedPhase].contains(s.phase) { models[i].states[p] = .init() }
        }
        table.removeAll { $0 == id }; changed()
    }
    public func approveLocalCode(id: String, contentDigest: String) async throws { try await approveLocalCode(id: id, contentDigest: contentDigest, approvedBy: .cli) }
    public func approveLocalCode(id: String, contentDigest: String, approvedBy: ApprovalSource) async throws {
        let i = try index(id)
        guard models[i].local, models[i].contentDigest == contentDigest else {
            throw WorkerFailure(code: "code_consent", message: "The local folder changed; approve its new contents.")
        }
        models[i].needsCodeConsent = false; models[i].approval = CodeApproval(by: approvedBy, digest: contentDigest); changed()
    }
    public func revokeLocalCode(id: String) async throws {
        let i = try index(id)
        guard models[i].local else { throw WorkerFailure(code: "not_found", message: "\(id) is not a local model; only local models have a code approval.") }
        guard models[i].approval != nil else { return }
        try unload(id: id)
        models[i].approval = nil; models[i].needsCodeConsent = true; changed()
    }
    public func approveLocalCode(id: String) async throws {
        let i = try index(id)
        guard let digest = models[i].contentDigest else { throw WorkerFailure(code: "not_found", message: "Unknown local model \(id).") }
        try await approveLocalCode(id: id, contentDigest: digest)
    }
    private func checkCode(_ index: Int) throws {
        if models[index].needsCodeConsent {
            throw WorkerFailure(code: "code_consent", message: "This local model runs unreviewed code on your Mac; approve it before Get or Load.")
        }
    }
    /// Fixtures carry a panel as `metadata[id]["settings_toml"]` (the file's text), read against `metadata[id]["setting"]`.
    public func settingsPanel(id: String) async -> SettingsPanel? {
        guard let text = metadata[id]?["settings_toml"]?.string else { return nil }
        let declarations = (metadata[id]?["setting"]?.array ?? []).compactMap { value -> SettingDeclaration? in
            guard let object = value.object, let data = try? WireJSON.encode(object) else { return nil }
            return try? JSONDecoder().decode(SettingDeclaration.self, from: data)
        }
        return try? SettingsPanel.parse(text, settings: declarations)
    }
    public func settings(id: String) async throws -> [SettingState] { _ = try index(id); return modelSettings.states(id, loaded: loadedPreferences[id] != nil) }
    public func setSetting(id: String, key: String, value: JSONValue) async throws { try await setSettings(id: id, values: [key: value]) }
    public func setSettings(id: String, values: JSONObject) async throws {
        let i = try index(id), write = try ModelSettings.splitMenuPlacement(values)
        var candidate = modelSettings
        for key in write.settings.keys.sorted() { try candidate.set(id, key: key, value: write.settings[key]!) }
        let file = support.appendingPathComponent("config.json")
        var config = try ArsenalConfiguration.read(file)
        config.settings = candidate.values
        if let shown = write.shown {
            var preference = config.selections[id] ?? .init(precision: models[i].selectedPrecision)
            preference.toolsShown = shown; config.selections[id] = preference
        }
        try config.write(file)
        modelSettings = candidate
        if let shown = write.shown { models[i].toolsShown = shown }
        try updatePending(id); changed()
    }
    public func saveDesignedVoice(id: String, previewID: String, name: String) async throws -> JSONObject {
        _ = try index(id); return try DesignedVoicePreviews.save(model: id, previewID: previewID, name: name, support: support)
    }
    public func deleteSavedVoice(id: String, voiceID: String) async throws { _ = try index(id); try SavedVoiceStore.delete(model: id, voiceID: voiceID, support: support) }
    public func applySettings(id: String) async throws { try unload(id: id); try load(id: id, precision: nil, fast: nil) }
    public func operations(id: String) async -> [ModelOperation] {
        if var ops = declaredOps[id] {
            if ops.contains(where: { ["design_voice", "clone_voice"].contains($0.name) }), !ops.contains(where: { $0.name == "delete_voice" }) {
                ops.append(
                    ModelOperation(
                        name: "delete_voice", title: "Delete saved voice",
                        inputSchema: ["type": "object", "required": ["voice_id"], "properties": ["voice_id": ["type": "string", "minLength": 1]], "additionalProperties": false],
                        output: "json", stream: false))
            }
            return ops
        }
        guard let i = try? index(id) else { return [] }
        return Self.kindOperations(models[i].kind).map {
            ModelOperation(
                name: $0, title: $0.capitalized, inputSchema: ["type": "object"],
                output: ["speak", "generate_sound", "generate_music", "enhance"].contains($0) ? "audio" : $0 == "generate" ? "image" : $0 == "chat" ? "text-stream" : "json",
                stream: ["speak", "chat"].contains($0))
        }
    }
    /// The operations a stub row of a kind answers when its fixture declares none.
    public static func kindOperations(_ kind: String) -> [String] {
        switch kind {
        case "voices": return ["speak", "voices"]
        case "sound": return ["generate_sound", "generate_music"]
        case "enhancement": return ["enhance"]
        case "decisions": return ["decide", "judge"]
        case "images": return ["generate"]
        default: return ["chat", "count_tokens"]
        }
    }
    public func kinds() -> [KindSpec] {
        KindSpec.defaults.sorted { a, b in
            let ia = kindOrder.firstIndex(of: a.id) ?? Int.max, ib = kindOrder.firstIndex(of: b.id) ?? Int.max
            return ia != ib ? ia < ib : (a.order, a.id) < (b.order, b.id)
        }
    }
    public func persistedKindOrder() async -> [String] { kindOrder }
    public func setKindOrder(_ order: [String]) async { kindOrder = order; changed() }
    public func residencySettings() async -> ResidencySettings { settings }
    public func setResidencySettings(_ value: ResidencySettings) async { settings = value; changed() }
    public func events() -> AsyncStream<ServiceEvent> {
        let id = UUID(); let (s, c) = AsyncStream<ServiceEvent>.makeStream(bufferingPolicy: .bufferingNewest(64)); listeners[id] = c
        c.onTermination = { [weak self] _ in Task { await self?.removeListener(id) } }; return s
    }
    private func removeListener(_ id: UUID) { listeners[id] = nil }
    private func changed() { for c in listeners.values { c.yield(.rowsChanged) } }
    /// Tests and renders: what `refreshRefusal` answers (core's service: a Get or Load is registered); emits a change.
    private var refusal: String?
    public func setRefreshRefusal(_ value: String?) { refusal = value; changed() }
    public func refreshRefusal() async -> String? { refusal }
    private func index(_ id: String) throws -> Int {
        guard let i = models.firstIndex(where: { $0.id == id }) else { throw WorkerFailure(code: "not_found", message: "Unknown model \(id).") }; return i
    }
    public func get(id: String, precision: String) async throws {
        let i = try index(id); try checkCode(i)
        if !table.contains(id) { try await add(id: id) } // As the real service: a Get records the model as Add would.
        models[i].selectedPrecision = precision; models[i].states[precision] = .init(state: .downloading, progress: 0)
        getCalls.append(id + " " + precision)
        try updatePending(id); changed()
        if let getScript {
            do {
                try await getScript(id, precision) { [weak self] state in await self?.report(id, precision, state) }
            } catch {
                models[i].states[precision] = error is CancellationError ? .init() : .init(state: .failed, error: error.localizedDescription)
                changed(); throw error
            }
        } else {
            try await Task.sleep(for: .milliseconds(30))
        }
        models[i].states[precision] = .init(state: .ready, progress: 1)
        models[i].selectedPrecision = precision; if loadedPreferences[id]?.precision == precision { models[i].states[precision] = .init(state: .loaded) }; try updatePending(id)
        changed()
    }
    public func cancelGet(id: String) {
        cancelGetCalls.append(id)
        if let i = try? index(id) { for p in models[i].precisions where models[i].states[p]?.state == .downloading { models[i].states[p] = .init() }; changed() }
    }
    private func requireLoad(_ i: Int, precision p: String) throws {
        try checkCode(i)
        let row = models[i]
        if let reason = row.dtypeUnavailableReason(p) { throw WorkerFailure(code: "unsupported_format", message: reason) }
        guard row.states[p]?.state == .ready || row.states[p]?.state == .loaded else {
            throw WorkerFailure(code: "not_installed", message: "Model \(row.id) is not installed; use \(MCPNames.prefix)download(model: \(row.id)) first.")
        }
    }
    private func requestOperation(id: String, op: String) async throws -> ModelOperation {
        if op != "delete_voice" { try checkCode(index(id)) }
        guard let operation = await operations(id: id).first(where: { $0.name == op }) else {
            throw WorkerFailure(code: "unsupported_op", message: "Model \(id) does not support \(op).")
        }
        return operation
    }
    public func runRefusal(id: String, op: String) async -> WorkerFailure? {
        do {
            _ = try await requestOperation(id: id, op: op)
            if op != "delete_voice", loadedPreferences[id] == nil { let i = try index(id); try requireLoad(i, precision: models[i].selectedPrecision) }
            return nil
        } catch let failure as WorkerFailure { return failure } catch { return WorkerFailure(code: "readiness", message: error.localizedDescription) }
    }
    public func primaryOperationAdmission(id: String, preferred: String?) async -> RunAdmission {
        let op = RunAdmission.primaryOperation(await operations(id: id), preferred: preferred)
        return RunAdmission(operation: op, refusal: await runRefusal(id: id, op: op))
    }
    public func load(id: String, precision: String?, fast: Bool?) throws {
        let i = try index(id), p = precision ?? models[i].selectedPrecision
        try requireLoad(i, precision: p)
        models[i].selectedPrecision = p; models[i].fast = models[i].fastAvailable
        for old in models[i].precisions where old != p && models[i].states[old]?.state == .loaded { models[i].states[old] = .init(state: .ready) }
        models[i].states[p] = .init(state: .loaded); loadedPreferences[id] = ModelPreferences(precision: p, fast: models[i].fast); loadedRunners[id] = models[i].runner
        models[i].residentRunner = models[i].runner; modelSettings.applied[id] = modelSettings.loadValues(id); models[i].needsReload = false; changed()
    }
    public func unload(id: String) throws {
        let i = try index(id); for p in models[i].precisions where models[i].states[p]?.state == .loaded { models[i].states[p] = .init(state: .ready) }
        modelSettings.applied[id] = nil; loadedPreferences[id] = nil; loadedRunners[id] = nil; models[i].residentRunner = nil; models[i].needsReload = false; changed()
    }
    public func setPrecision(id: String, precision: String) async throws {
        let i = try index(id), row = models[i]
        guard !row.states.values.contains(where: { [.downloading, .deriving, .loading].contains($0.state) }) else {
            throw WorkerFailure(code: "busy", message: "Wait for Get or Load to finish before changing precision.")
        }
        guard row.runner != "Unavailable", runnerDetails(row).contains(where: { $0.name == row.runner && $0.eligible }) else {
            throw WorkerFailure(code: "unsupported_format", message: "The selected runner does not support this Mac.")
        }
        guard row.precisions.contains(precision) else { throw WorkerFailure(code: "precision", message: "That dtype is not offered by \(id).") }
        if let reason = row.dtypeUnavailableReason(precision) { throw WorkerFailure(code: "unsupported_format", message: reason) }

        let file = support.appendingPathComponent("config.json")
        var config = try ArsenalConfiguration.read(file), preference = config.selections[id] ?? .init(precision: row.selectedPrecision)
        preference.precision = precision; preference.explicitSelection = true
        config.selections[id] = preference; try config.write(file)
        models[i].selectedPrecision = precision; try updatePending(id); changed()
    }
    public func setKeepHot(id: String, on: Bool) throws { models[try index(id)].keepHot = on; changed() }
    public func delete(id: String) throws { let i = try index(id); models[i].states = [:]; changed() }
    /// The stub keeps no files: a precision is on disk when its row says ready or loaded, with the row's figure for its bytes.
    public func onDiskModels() async throws -> [OnDiskModel] {
        models.compactMap { row in
            let onDisk = row.precisions.sorted().filter { [.ready, .loaded].contains(row.states[$0]?.state ?? .notInstalled) }
            let precisions = onDisk.map { OnDiskPrecision(dtype: $0, bytes: row.downloadBytes[$0] ?? row.sizeBytes ?? 0, inUse: row.states[$0]?.state == .loaded) }
            return precisions.isEmpty ? nil : OnDiskModel(id: row.id, precisions: precisions, cacheBytes: 0, offered: true)
        }
    }
    public func deletePrecision(id: String, precision: String) async throws -> Int64 {
        let i = try index(id), row = models[i]
        if loadedPreferences[id]?.precision == precision {
            throw WorkerFailure(code: "busy", message: "Unload \(row.name) before removing its \(precision.uppercased()) files.")
        }
        guard ![.downloading, .deriving, .loading].contains(row.states[precision]?.state ?? .notInstalled) else {
            throw WorkerFailure(code: "busy", message: "Finish Get or Load before removing files.")
        }
        guard [.ready, .loaded].contains(row.states[precision]?.state ?? .notInstalled) else { return 0 }
        models[i].states[precision] = PrecisionState()
        // The chosen precision's copy is gone: switch to a copy still on disk, as the real service does.
        if row.selectedPrecision == precision,
            let next = row.precisions.first(where: { $0 != precision && [.ready, .loaded].contains(row.states[$0]?.state ?? .notInstalled) })
        {
            models[i].selectedPrecision = next
            let file = support.appendingPathComponent("config.json")
            if var config = try? ArsenalConfiguration.read(file) {
                var preference = config.selections[id] ?? .init(precision: next); preference.precision = next
                config.selections[id] = preference; try? config.write(file)
            }
        }
        try updatePending(id); changed()
        return row.downloadBytes[precision] ?? row.sizeBytes ?? 0
    }
    public func deleteAllWeights(id: String) async throws -> Int64 {
        let i = try index(id), row = models[i]
        guard !row.states.values.contains(where: { [.downloading, .deriving, .loading].contains($0.state) }) else {
            throw WorkerFailure(code: "busy", message: "Finish Get or Load before deleting its weights.")
        }
        if loadedPreferences[id] != nil { try unload(id: id) }
        let freed = row.precisions.filter { [.ready, .loaded].contains(row.states[$0]?.state ?? .notInstalled) }.reduce(Int64(0)) {
            $0 + (row.downloadBytes[$1] ?? row.sizeBytes ?? 0)
        }
        models[i].states = [:]; changed()
        return freed
    }
    public func orphanBytes() async throws -> Int64 { 0 }
    public func deleteOrphans() async throws -> Int64 { 0 }
    public func voiceInventory(id: String) async throws -> JSONObject {
        _ = try index(id)
        guard loadedPreferences[id] != nil else { throw WorkerFailure(code: "not_running", message: "Load the model to list its voices.") }
        return ["ok": true, "voices": .array([["id": "af_heart", "name": "Heart", "language": "en", "gender": "female"]] + (try SavedVoiceStore.list(model: id, support: support)))]
    }
    /// Tests: whether the last run carried Load Anyway (`LoadMemoryContext.force`).
    public private(set) var lastRunForce: Bool?
    public func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject {
        lastRunForce = LoadMemoryContext.force
        if let runHook { try await runHook(id, op) }
        var params = params
        if let body = params["body"]?.string, let parsed = try? WireJSON.decode(Data(body.utf8)) { params = parsed }
        let operation = try await requestOperation(id: id, op: op)
        let (objects, ordered) = try OrderedJSONMembers.split(params, schema: operation.inputSchema)
        params = modelSettings.request(id, params: objects, schema: operation.inputSchema)
        try OperationSchemaValidator.validate(.object(params), schema: .object(operation.inputSchema))
        params = OrderedJSONMembers.attach(ordered, to: params)
        if op == "delete_voice", let voice = params["voice_id"]?.string { try await deleteSavedVoice(id: id, voiceID: voice); return ["deleted": .string(voice)] }
        if loadedPreferences[id] == nil { try load(id: id, precision: nil, fast: nil) }
        switch op {
        case "design_voice":
            let dir = params["out_dir"]?.string.map { URL(fileURLWithPath: $0) } ?? support.appendingPathComponent("jobs/" + UUID().uuidString)
            try TransientModelJobs.prepare(dir, support: support)
            let file = dir.appendingPathComponent("preview.wav"), pcm = Data(repeating: 0, count: 96)
            try (Self.wavHeader(bytes: pcm.count) + pcm).write(to: file)
            let result: JSONObject = [
                "voice_id": .string("preview-" + UUID().uuidString.lowercased()), "preview": ["path": .string(file.path)], "saved": false,
                "design": [
                    "seed": params["seed"] ?? .number(Double(UInt32.random(in: 0...UInt32.max))), "model_revision": "stub",
                    "preview_text": params["preview_text"] ?? "Hello. This is a preview of your new voice.", "language": params["language"] ?? "english",
                    "dtype": .string(models[try index(id)].selectedPrecision), "fast": .bool(models[try index(id)].fast)
                ]
            ]
            try DesignedVoicePreviews.record(model: id, request: params, result: result, support: support)
            if let name = params["name"]?.string { return try await saveDesignedVoice(id: id, previewID: result["voice_id"]!.string!, name: name) }
            return result
        case "voices": return try await voiceInventory(id: id)
        case "speak", "generate_sound", "generate_music", "enhance":
            guard params["voice"]?.string == nil || params["voice"]?.string == "af_heart" else {
                throw WorkerFailure(code: "invalid_request", message: "Unknown stub voice; use af_heart.")
            }
            let pcm = Data(repeating: 0, count: 9600)
            try await sink?(.json(["event": "audio_begin", "stream": 1, "sample_rate": 24000, "channels": 1, "encoding": "pcm_s16le"]))
            for i in stride(from: 0, to: pcm.count, by: 1024) { try await sink?(.binary(stream: 1, bytes: Data(pcm[i..<min(i + 1024, pcm.count)]))) }
            var result: JSONObject = ["ok": true, "duration_s": 0.2, "gen_s": 0.001, "first_audio_s": 0.001, "rtf": 0.005, "sample_rate": 24000, "channels": 1]
            if let path = params["out_dir"]?.string {
                let directory = URL(fileURLWithPath: path)
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                let file = directory.appendingPathComponent("stub-audio.wav")
                var wav = Data("RIFF".utf8)
                func u32(_ n: Int) { var n = UInt32(n).littleEndian; withUnsafeBytes(of: &n) { wav.append(contentsOf: $0) } }
                func u16(_ n: Int) { var n = UInt16(n).littleEndian; withUnsafeBytes(of: &n) { wav.append(contentsOf: $0) } }
                u32(pcm.count + 36); wav += Data("WAVEfmt ".utf8); u32(16); u16(1); u16(1); u32(24000); u32(48000); u16(2); u16(16); wav += Data("data".utf8); u32(pcm.count)
                wav += pcm
                try wav.write(to: file)
                result["path"] = .string(file.path); result["bytes"] = .number(Double(wav.count))
            }
            return result
        case "generate":
            guard let path = params["out_dir"]?.string else { throw WorkerFailure(code: "invalid_request", message: "Stub images need out_dir.") }
            let directory = URL(fileURLWithPath: path)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let png = Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j6ioAAAAASUVORK5CYII=")!
            let count = Int(params["n"]?.number ?? 1)
            guard (1...4).contains(count) else { throw WorkerFailure(code: "invalid_request", message: "Stub n must be 1 through 4.") }
            var images: [JSONValue] = []
            for i in 0..<count {
                let file = directory.appendingPathComponent("stub-image-\(i).png")
                try png.write(to: file)
                images.append(["path": .string(file.path), "bytes": .number(Double(png.count)), "width": 1, "height": 1])
            }
            return ["ok": true, "images": .array(images), "seed": params["seed"] ?? 1, "seconds": 0.001]
        case "decide", "judge":
            let questions = params["questions"]?.object ?? [:]
            guard !questions.isEmpty else { throw WorkerFailure(code: "invalid_request", message: "questions must be a nonempty object.") }
            let answers = try stubAnswers(questions, legacy: op == "judge")
            if op == "decide" { return ["ok": true, "answers": .object(answers), "model": .string(id), "usage": ["input_tokens": 218, "output_tokens": 0]] }
            let results: [JSONValue] = (params["items"]?.array ?? []).enumerated().map { index, item in
                if ["image", "images", "audio", "video", "videos"].contains(where: { item[$0] != nil }) {
                    return ["index": .number(Double(index)), "ok": false, "error": "Decision models accept text only."]
                }
                return ["index": .number(Double(index)), "ok": true, "answers": .object(answers), "model": .string(id), "ms": 1]
            }
            return ["ok": true, "results": .array(results)]
        case "count_tokens": return ["ok": true, "tokens": 8, "input_tokens": 8]
        case "chat":
            for text in ["Hello ", "from ", "Arsenal."] { try await sink?(.json(["event": "token", "text": .string(text)])) }
            return ["ok": true, "text": "Hello from Arsenal.", "finish_reason": "stop", "model": .string(id), "usage": ["prompt_tokens": 1, "completion_tokens": 4]]
        default: throw WorkerFailure(code: "unsupported_op", message: "Unsupported stub operation \(op).")
        }
    }
    private static func wavHeader(bytes: Int) -> Data {
        var data = Data("RIFF".utf8)
        func u32(_ n: Int) { var value = UInt32(n).littleEndian; withUnsafeBytes(of: &value) { data.append(contentsOf: $0) } }
        func u16(_ n: Int) { var value = UInt16(n).littleEndian; withUnsafeBytes(of: &value) { data.append(contentsOf: $0) } }
        u32(bytes + 36); data += Data("WAVEfmt ".utf8); u32(16); u16(1); u16(1); u32(24000); u32(48000); u16(2); u16(16); data += Data("data".utf8); u32(bytes)
        return data
    }
    /// Fixed probabilities from Verdict docs/API.md; these are transport fixtures, not classifications.
    private func stubAnswers(_ questions: JSONObject, legacy: Bool) throws -> JSONObject {
        try questions.mapValues { question in
            let type = question["type"]?.string ?? "noul"
            var answer: JSONObject
            switch type {
            case "noul": answer = ["noul": 0.7696]; if legacy { answer["confidence"] = 0.7696 }
            case "choice":
                let keys = question["criteria"]?.object?.keys.sorted() ?? question["criteria"]?.array?.compactMap(\.string) ?? []
                guard !keys.isEmpty else { throw WorkerFailure(code: "invalid_request", message: "Choice criteria cannot be empty.") }
                if Set(keys) == Set(["frontend", "infra", "payments"]) {
                    answer = ["choice": "frontend", "confidence": 0.1628, "probabilities": ["frontend": 0.623, "infra": 0.1607, "payments": 0.2163]]
                } else {
                    let probabilities = Dictionary(
                        uniqueKeysWithValues: keys.enumerated().map { i, key in (key, JSONValue.number(keys.count == 1 ? 1 : i == 0 ? 0.75 : 0.25 / Double(keys.count - 1))) })
                    answer = ["choice": .string(keys[0]), "confidence": 0.5, "probabilities": .object(probabilities)]
                }
            case "score":
                guard let levels = question["criteria"]?.array, !levels.isEmpty else { throw WorkerFailure(code: "invalid_request", message: "Score criteria cannot be empty.") }
                if levels.count == 3 {
                    answer = ["score": 1.0663, "confidence": 0.5691, "probabilities": ["0": 0.0346, "1": 0.8646, "2": 0.1008]]
                } else {
                    answer = [
                        "score": .number(Double(levels.count - 1) / 2), "confidence": 0,
                        "probabilities": .object(Dictionary(uniqueKeysWithValues: levels.indices.map { (String($0), .number(1.0 / Double(levels.count))) }))
                    ]
                }
                if !legacy { answer["legend"] = .object(Dictionary(uniqueKeysWithValues: levels.enumerated().map { (String($0.offset), $0.element) })) }
            default: throw WorkerFailure(code: "invalid_request", message: "Unknown question type \(type).")
            }
            if !legacy { answer["type"] = .string(type) }; return .object(answer)
        }
    }
    public func menuActions() -> [LoadedAction] {
        models.filter { loadedPreferences[$0.id] != nil && $0.kind == "voices" }.map {
            LoadedAction(id: "\($0.id)-speak", modelID: $0.id, title: "Speak Clipboard", op: "speak", input: "clipboard")
        }
    }
    public func memory() -> MemorySnapshot { MemorySnapshot(workersMB: Double(loadedPreferences.count) * 16) }
    public func catalogStatus() -> CatalogStatus { CatalogStatus(source: "stub", verified: true) }
}

extension StubModelsService {
    public func memorySettings() async -> MemorySettings { memoryValue }
    public func setMemorySettings(_ settings: MemorySettings) async throws {
        let file = support.appendingPathComponent("config.json")
        var config = try ArsenalConfiguration.read(file); config.memory = settings; try config.write(file)
        memoryValue = settings; changed()
    }
    public func setToolsShown(id: String, on: Bool) async throws {
        let i = try index(id), file = support.appendingPathComponent("config.json")
        var config = try ArsenalConfiguration.read(file), preference = config.selections[id] ?? .init(precision: models[i].selectedPrecision)
        preference.toolsShown = on; config.selections[id] = preference; try config.write(file)
        models[i].toolsShown = on; changed()
    }
}
