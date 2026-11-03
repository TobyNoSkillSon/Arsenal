// SPDX-License-Identifier: MIT
import Foundation
import os

/// Scoped to one explicit request, inherited by its load/run tasks; never a saved preference.
public enum LoadMemoryContext {
    @TaskLocal public static var force = false
}

public struct LoadMemoryRefusal: Codable, Sendable, Equatable {
    public let model: String
    public let name: String
    public let precision: String
    public let neededGB: Double
    /// Room under the Mac-wide line now (`MacMemoryGuard.safeInUseMB`), not raw free RAM.
    public let freeGB: Double
    public let releasedGB: Double?
    /// The Mac-wide line and the Mac's RAM, in decimal GB.
    public let limitGB: Double
    public let totalGB: Double
    /// Arsenal's own loaded models that are idle now (no request running or waiting, not Keep loaded), largest first.
    public let idle: [IdleModel]
    /// The Mac's memory in use when this was refused, as the activity stream counts it (`memory_used_mb`): what a window
    /// measures freed memory from.
    public let usedMB: Double?
    public struct IdleModel: Codable, Sendable, Equatable {
        public let id: String
        public let name: String
        public let gb: Double
        public init(id: String, name: String, gb: Double) { self.id = id; self.name = name; self.gb = gb }
    }
    public var idleGB: Double { idle.reduce(0) { $0 + max(0, $1.gb) } }
    /// Unloading the idle models makes room for this one: the refusal offers that besides Cancel and Load Anyway.
    public var unloadingIdleFits: Bool { !idle.isEmpty && freeGB + idleGB >= neededGB }
    /// "Kokoro 82M and FLUX.2 Klein 4B": the idle models, named.
    public var idleNames: String {
        let names = idle.map(\.name)
        return names.count < 2 ? names.joined() : names.dropLast().joined(separator: ", ") + " and " + names[names.count - 1]
    }
    /// Whether the model fits once `megabytes` more are free than when it was refused (memory other apps or idle models gave back).
    public func fits(afterFreeingMB megabytes: Double) -> Bool { megabytes.isFinite && freeGB + megabytes / 1000 >= neededGB }
    public var title: String { "Not enough free memory to load \(name)" }
    public var explanation: String {
        let limit = MemoryDisplay.text(decimalGB: limitGB), total = MemoryDisplay.text(decimalGB: totalGB)
        let needed = ModelTableFigure.memoryText(neededGB)
        let free = MemoryDisplay.text(decimalGB: freeGB)
        let capacity = (releasedGB ?? 0) > 0 ? "\(free) fits under that limit after unloading the current model" : "\(free) fits under that limit now"
        let unload =
            unloadingIdleFits ? " Unloading Arsenal's idle \(idle.count == 1 ? "model" : "models"), \(idleNames) (\(ModelTableFigure.memoryText(idleGB))), frees enough." : ""
        return
            "Arsenal keeps the Mac at or under \(limit) of \(total) in use, which keeps it out of swap. "
            + "This model needs about \(needed) at \(precision.uppercased()); \(capacity).\(unload) Loading anyway lets macOS use swap, which slows the whole Mac."
    }
    public var summary: String { title + ". " + explanation }
    public var failure: WorkerFailure {
        let unload =
            unloadingIdleFits
            ? " Or unload the idle \(idle.count == 1 ? "model" : "models") (POST /arsenal/v1/unload with \(idle.map { "{\"id\":\"\($0.id)\"}" }.joined(separator: ", "))) and retry."
            : ""
        return WorkerFailure(code: "insufficient_memory", message: summary + " Retry with force: true to load anyway." + unload, details: details)
    }
    public var details: JSONObject {
        [
            "model": .string(model), "name": .string(name), "precision": .string(precision),
            "needed_gb": .number(neededGB), "free_gb": .number(freeGB), "released_gb": .number(releasedGB ?? 0),
            "limit_gb": .number(limitGB), "total_gb": .number(totalGB), "status": 409,
            "force": true, "force_hint": "Retry the same request with force: true (macOS will use swap).",
            "idle_models": .array(idle.map { .object(["id": .string($0.id), "name": .string($0.name), "gb": .number($0.gb)]) }),
            "idle_gb": .number(idleGB), "unloading_idle_fits": .bool(unloadingIdleFits),
            "mac_used_mb": usedMB.map(JSONValue.number) ?? .null
        ]
    }
    public init(
        model: String, name: String, precision: String, neededGB: Double, freeGB: Double, releasedGB: Double = 0,
        limitGB: Double = MacMemoryGuard.macLimitGB, totalGB: Double = MacMemoryGuard.macTotalGB, idle: [IdleModel] = [],
        usedMB: Double? = nil
    ) {
        self.model = model; self.name = name; self.precision = precision; self.neededGB = neededGB; self.freeGB = freeGB; self.releasedGB = releasedGB
        self.limitGB = limitGB; self.totalGB = totalGB
        self.usedMB = usedMB.flatMap { $0.isFinite && $0 > 0 ? $0 : nil }
        self.idle = idle.filter { $0.id != model && $0.gb.isFinite && $0.gb > 0 }.sorted { $0.gb > $1.gb }
    }
    public init?(details: JSONObject) {
        guard let model = details["model"]?.string, let name = details["name"]?.string,
            let precision = details["precision"]?.string, let needed = details["needed_gb"]?.number,
            let free = details["free_gb"]?.number, needed.isFinite, needed > 0, free.isFinite, free >= 0
        else { return nil }
        self.init(
            model: model, name: name, precision: precision, neededGB: needed, freeGB: free, releasedGB: details["released_gb"]?.number ?? 0,
            limitGB: details["limit_gb"]?.number ?? MacMemoryGuard.macLimitGB, totalGB: details["total_gb"]?.number ?? MacMemoryGuard.macTotalGB,
            idle: (details["idle_models"]?.array ?? []).compactMap { value in
                guard let o = value.object, let id = o["id"]?.string, let gb = o["gb"]?.number else { return nil }
                return IdleModel(id: id, name: o["name"]?.string ?? id, gb: gb)
            }, usedMB: details["mac_used_mb"]?.number)
    }
}

/// Captured with a resident revision so recovery never borrows a candidate's figures.
public struct LoadMemoryRequirement: Codable, Sendable, Equatable {
    /// `peak`: the table's request peak, which arms the runaway stop. `unmeasuredPeak`: an operation kind's quick-check or
    /// estimated peak, taken on one short input; Load admission uses it as it uses `peak`, but the runaway stop does not,
    /// since a long recording or video legitimately grows far past it.
    public enum Basis: String, Codable, Sendable { case peak, unmeasuredPeak, loadPeak, forecast, resident, unknown }
    public var neededGB: Double?
    public var basis: Basis
    /// Whether the table peak was a controlled measurement (the runaway stop's wording); nil when not from a table peak.
    public var peakMeasured: Bool?
    public init(row: ModelRow?, precision: String, forecast: PrecisionMemoryRecipe?, minimumMB: Double = 0) {
        var known: [(Double, Basis)] = []
        if let row, let stats = row.precisionStats[precision], stats.dtype == precision, stats.runner == row.runner, stats.kind.rawValue == row.kind {
            if let peak = stats.peakMemory.value, peak > 0 {
                let measured = stats.peakMemory.stat.state == .measured
                known.append((peak, measured || !stats.kind.isOperation ? .peak : .unmeasuredPeak))
                peakMeasured = measured
            }
            let load = ModelTableFigure(metric: .loadPeakMemoryGB, stat: stats.value(.loadPeakMemoryGB)).value
            if known.isEmpty, let load, load > 0 { known.append((load, .loadPeak)) }
        }
        // A kit-kind model's figures come from its folder quick check, as the table shows them (VM run, 2 Nov: Chronos
        // showed 0.24 GB in the table and "Memory requirement unknown" in the menu and its window).
        if known.isEmpty, let peak = row?.folderFigures[precision]?.peakMemoryGB.value, peak > 0 { known.append((peak, .unmeasuredPeak)) }
        if !known.contains(where: { [.peak, .unmeasuredPeak].contains($0.1) }) {
            if let ram = row?.formats.first(where: { $0.id == precision })?.ramMB, ram.isFinite, ram > 0 {
                known.append((ram / 1000, .forecast))
            } else if let bytes = forecast?.residentBytes[precision], bytes > 0 {
                known.append((Double(bytes) / 1e9, .forecast))
            }
        }
        if minimumMB.isFinite, minimumMB > 0 { known.append((minimumMB / 1000, .resident)) }
        let largest = known.max { $0.0 < $1.0 }
        neededGB = largest?.0; basis = largest?.1 ?? .unknown
    }
    public func atLeast(_ megabytes: Double) -> Self {
        guard megabytes.isFinite, megabytes > (neededGB ?? 0) * 1000 else { return self }
        var result = self; result.neededGB = megabytes / 1000; result.basis = .resident; return result
    }
    public static let unknownNote = "Memory requirement unknown; fit could not be checked."
}

public struct LoadMemoryAssessment: Sendable, Equatable {
    public let requirement: LoadMemoryRequirement
    public var neededGB: Double? { requirement.neededGB }
    /// Room under the Mac-wide line for the sequential replacement: reclaimable RAM above the line's outside, plus the outgoing worker.
    public let freeGB: Double
    public let releasedGB: Double
    public var basis: LoadMemoryRequirement.Basis { requirement.basis }
    public var fits: Bool { neededGB.map { $0 <= freeGB } ?? true }
    public init(row: ModelRow?, precision: String, forecast: PrecisionMemoryRecipe?, freeMB: Double) {
        self.init(requirement: .init(row: row, precision: precision, forecast: forecast), freeMB: freeMB)
    }
    public init(requirement: LoadMemoryRequirement, freeMB: Double, releasedMB: Double = 0) {
        self.requirement = requirement; freeGB = max(0, freeMB) / 1000; releasedGB = max(0, releasedMB) / 1000
    }
}

extension ModelsService {
    public func loadMemoryNotice(id: String, precision: String?) async -> String? { nil }
    public func load(id: String, precision: String?, fast: Bool?, force: Bool) async throws {
        try await LoadMemoryContext.$force.withValue(force) { try await load(id: id, precision: precision, fast: fast) }
    }
    public func run(id: String, op: String, params: JSONObject, sink: RunSink? = nil, force: Bool) async throws -> JSONObject {
        try await LoadMemoryContext.$force.withValue(force) { try await run(id: id, op: op, params: params, sink: sink) }
    }
}

extension RealModelsService {
    /// Live OS counters already exclude resident workers. Only subtract pending, not-yet-materialized reservations.
    /// The fixture probe starts from a budget, so it subtracts the workers itself, once.
    func loadMemoryRequirement(_ model: ModelManifest, precision: String) -> LoadMemoryRequirement {
        LoadMemoryRequirement(row: modelRows[model.id], precision: precision, forecast: model.memory)
    }
    func revisionMemoryRequirement(_ revision: ResidentRevision) -> LoadMemoryRequirement {
        // The fallback row is built exclusively from the captured manifest, never the current candidate row.
        var row = ModelRow(id: revision.manifest.id, name: revision.manifest.name, kind: revision.manifest.kind, runner: revision.launch.runner ?? "Unavailable")
        row.precisionDtypes = revision.manifest.precisionDtypes
        row.precisionNotes = revision.manifest.precisionNotes
        row.formats = revision.manifest.formats
        return (revision.memoryRequirement ?? LoadMemoryRequirement(row: row, precision: revision.preference.precision, forecast: revision.manifest.memory))
            .atLeast(revision.resident.memoryMB)
    }
    func loadMemoryAssessment(_ model: ModelManifest, precision: String, requirement: LoadMemoryRequirement? = nil, releasingResident: Bool = false) -> LoadMemoryAssessment {
        let inFlight = loadReservations.reduce(0.0) { sum, entry in
            sum + (entry.key == model.id ? 0 : loadReservationReleases.contains(entry.key) ? entry.value : max(0, entry.value - (footprints[entry.key]?.footprintMB ?? 0)))
        }
        // Only this outgoing worker is released. Other residents are never credited or evicted.
        let released = !releasingResident ? 0 : max(0, footprints[model.id]?.footprintMB ?? residents[model.id]?.memoryMB ?? 0)
        // The Mac-wide guard stops any worker that takes the Mac past the line, so admission refuses it first.
        let room = MacMemoryGuard.roomUnderLineMB(totalMB: probe.totalMB, noSwapAvailableMB: probe.noSwapAvailableMB(loadedMB: memory().workersMB))
        return LoadMemoryAssessment(
            requirement: requirement ?? loadMemoryRequirement(model, precision: precision),
            freeMB: room + released - inFlight, releasedMB: released)
    }
    public func loadMemoryNotice(id: String, precision: String?) async -> String? {
        guard let model = try? manifest(id) else { return nil }
        let requirement =
            recoveryRevision(id).map(revisionMemoryRequirement) ?? loadMemoryRequirement(model, precision: precision ?? modelRows[id]?.selectedPrecision ?? model.defaultPrecision)
        return requirement.neededGB == nil ? LoadMemoryRequirement.unknownNote : nil
    }
    /// The worker memory ceiling for one Load or request, from the requirement admission used (see `WorkerMemoryCeiling`).
    func workerMemoryCeiling(_ id: String, requirement: LoadMemoryRequirement?, precision: String?, forced: Bool) -> WorkerMemoryCeiling {
        WorkerMemoryCeiling(name: modelRows[id]?.name ?? id, requirement: requirement, precision: precision, forced: forced)
    }
    func recordResidentMemoryApproval(_ id: String) {
        if LoadMemoryContext.force { forceLoadedResidents.insert(id) } else { forceLoadedResidents.remove(id) }
    }
    /// Arsenal's own loaded models that unloading would free now: nothing running or waiting, no lifecycle work, not Keep loaded.
    func idleResidents(except id: String) -> [LoadMemoryRefusal.IdleModel] {
        residents.values.compactMap { r in
            guard r.id != id, r.busy == 0, pendingRuns[r.id, default: 0] == 0, !r.keepHot, loadTasks[r.id] == nil, swapTasks[r.id] == nil,
                !reloading.contains(r.id), unloading[r.id, default: 0] == 0
            else { return nil }
            let mb = footprints[r.id]?.footprintMB ?? r.memoryMB
            return .init(id: r.id, name: modelRows[r.id]?.name ?? r.id, gb: mb / 1000)
        }
    }
    /// The memory pop-up's idle list was taken when the load was refused; each model is checked again right before it goes.
    public func unloadIdle(ids: [String]) async {
        for id in ids where idleResidents(except: "").contains(where: { $0.id == id }) { try? await unload(id: id) }
    }
    /// Every worker Load, including recovery, comes through here. No worker is evicted to suppress this warning.
    @discardableResult func checkLoadMemory(
        _ model: ModelManifest, precision: String, reserve: Bool = false, requirement: LoadMemoryRequirement? = nil, releasingResident: Bool = false
    ) throws -> LoadMemoryAssessment {
        try admitWork()
        if swapTasks[model.id] == nil { try admitNewWork() }
        let check = loadMemoryAssessment(model, precision: precision, requirement: requirement, releasingResident: releasingResident)
        if !LoadMemoryContext.force, !check.fits, let needed = check.neededGB {
            throw LoadMemoryRefusal(
                model: model.id, name: modelRows[model.id]?.name ?? model.name, precision: precision,
                neededGB: needed, freeGB: check.freeGB, releasedGB: check.releasedGB,
                limitGB: MacMemoryGuard.safeInUseMB(totalMB: probe.totalMB) / 1000, totalGB: probe.totalMB / 1000,
                idle: idleResidents(except: model.id), usedMB: MachineSampler.memory()?.used
            ).failure
        }
        if reserve {
            modelRows[model.id]?.loadMemoryNote = check.basis == .unknown ? LoadMemoryRequirement.unknownNote : nil
            if check.basis == .unknown {
                Logger(subsystem: "com.arsenal.app", category: "memory").notice(
                    "Loading \(model.id, privacy: .public) (\(precision, privacy: .public)): peak memory unknown; no memory fit guarantee.")
            }
            loadReservations[model.id] = (check.neededGB ?? 0) * 1000
            if releasingResident { loadReservationReleases.insert(model.id) }
        }
        return check
    }
}
