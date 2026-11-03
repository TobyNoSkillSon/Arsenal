// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import MCP

/// What a model needs before it can run: the facts every consent line, `find` and `model` text are built from.
struct MCPAvailability: Sendable {
    var entry: CatalogEntry
    var dtype: String
    var state: ModelState
    /// A Get is needed (with consent unless it is a local clone); `plan` says how much.
    var needsGet: Bool
    /// A Get or conversion of this dtype is running now, in the app or over MCP.
    var inProgress: Bool
    var update: Bool
    var plan: GetPlan?
    var networkBytes: Int64? { plan?.totalDownloadBytes }
    var localBytes: Int64 { plan?.items.reduce(0) { $0 + $1.localCopyBytes } ?? 0 }
    /// Local imports the core plan found on the store's own volume: APFS clones that use no disk.
    var cloneBytes: Int64 { plan?.items.reduce(0) { $0 + min($1.localCopyBytes, $1.localCloneBytes ?? 0) } ?? 0 }
    /// Local imports that are real copies (another volume, or not known): disk and the user's yes.
    var copyBytes: Int64 { localBytes - cloneBytes }
    /// A quantized tier made on this Mac from the installed weights (int8/int4 not yet installed): it writes about this
    /// much and runs a converter. Nil when no conversion is needed.
    var deriveBytes: Int64?
    var dependencies: [GetPlanItem] { plan?.items.filter { $0.kind != "weights" && ($0.downloadBytes ?? 1) > 0 } ?? [] }
}

/// Which call a consent line answers.
enum MCPConsentVerb: Sendable {
    case run, load, download
    var refused: String { self == .run ? "Not run" : self == .load ? "Not loaded" : "Not downloaded" }
    var tool: String { self == .run ? "\(MCPNames.prefix)run" : self == .load ? "\(MCPNames.prefix)load" : "\(MCPNames.prefix)download" }
}

extension APIRouter {
    func mcpCatalog() async -> [CatalogEntry] { await service.catalog(query: .init()) }
    /// The catalog entry for `id`, or `[unknown_model]` with the closest id.
    func mcpEntry(_ id: String) async throws -> CatalogEntry {
        let entries = await mcpCatalog()
        if let entry = entries.first(where: { $0.id == id }) { return entry }
        let lower = id.lowercased()
        let guess =
            entries.first { $0.id.lowercased().hasPrefix(lower) || lower.hasPrefix($0.id.lowercased()) }
            ?? entries.first { $0.id.lowercased().contains(lower) || $0.name.lowercased().contains(lower) }
            ?? entries.map { ($0, MCPSearch.distance($0.id.lowercased(), lower)) }.filter { $0.1 <= max(3, lower.count / 3) }.min { $0.1 < $1.1 }?.0
        let hint = guess.map { " Did you mean \($0.id)?" } ?? ""
        throw MCPFailure(.unknownModel, "No model \"\(id)\".\(hint) \(MCPNames.prefix)find(query:\"\(id)\") searches the catalog.")
    }
    /// RAM the model needs at `dtype`, decimal GB: the table peak, else the Load peak, else the declared forecast.
    nonisolated func mcpNeedGB(_ entry: CatalogEntry, dtype: String) -> Double? {
        if let needed = LoadMemoryRequirement(row: entry.row, precision: dtype, forecast: nil).neededGB, needed > 0 { return needed }
        return entry.ramMBByDtype[dtype].map { $0 / 1000 }
    }
    func mcpAvailability(_ entry: CatalogEntry, dtype: String, forDownload: Bool = false) async throws -> MCPAvailability {
        let state = entry.row.states[dtype]?.state ?? .notInstalled
        let installed = [ModelState.ready, .loading, .loaded].contains(state)
        var value = MCPAvailability(
            entry: entry, dtype: dtype, state: state, needsGet: !installed, inProgress: state == .downloading || state == .deriving, update: state == .updateAvailable)
        if !installed || forDownload {
            do { value.plan = try await service.getPlan(id: entry.id, dtype: dtype) } catch { throw await mcpFailure(error, entry: entry, verb: "Not downloaded") }
        }
        if installed, forDownload, let plan = value.plan {
            value.needsGet = plan.requiresDownload || plan.items.contains { $0.localCopyBytes > 0 } || entry.row.revision?.pending != nil
        }
        if value.inProgress { value.needsGet = true }
        if value.needsGet, !installed, entry.derivedDtypes.contains(dtype), !["fp16", "bf16", "fp32", entry.sourceDtype].contains(dtype) {
            let listed: Int64 = entry.sizesByDtype[dtype] ?? 0
            let format: Int64 = entry.formats.first(where: { $0.id == dtype })?.sizeBytes ?? 0
            value.deriveBytes = listed > 0 ? listed : max(0, format)
        }
        return value
    }
    /// Refuses a model that can never fit under the Mac-wide line, whatever is unloaded and whatever `force` says.
    func mcpRequireFits(_ entry: CatalogEntry, dtype: String) throws {
        let memory = mcpEnvironment.macMemory()
        guard let need = mcpNeedGB(entry, dtype: dtype), need > memory.limitGB else { return }
        throw MCPFailure(
            .memory,
            "Won't run on this Mac: \(entry.name) needs ~\(MCPFormat.memory(decimalGB: need)) RAM at \(dtype); the limit is \(MCPFormat.memory(decimalGB: memory.limitGB)).")
    }
    /// Disk the Get writes: network bytes, local copies that are not clones, and a tier made on this Mac.
    func mcpRequireDisk(_ availability: MCPAvailability) throws {
        let needed = (availability.networkBytes ?? 0) + availability.copyBytes + (availability.deriveBytes ?? 0)
        guard needed > 0, let free = mcpEnvironment.freeDiskBytes(support), free < needed else { return }
        throw MCPFailure(.disk, "Not started: \(availability.entry.name) needs \(MCPFormat.size(needed)), \(MCPFormat.size(free)) free on the startup disk.")
    }
    /// A Get that downloads, copies and converts nothing (installed blobs, or APFS clones): no consent needed.
    func mcpIsFreeImport(_ availability: MCPAvailability) -> Bool {
        (availability.networkBytes ?? 1) == 0 && availability.dependencies.isEmpty && availability.copyBytes == 0 && availability.deriveBytes == nil
    }
    nonisolated func mcpHost(_ entry: CatalogEntry) -> String { entry.hfRepo.isEmpty ? "other" : "huggingface.co" }
    /// "2.5 GB, about 4 min at this Mac's last measured 10 MB/s" and its variants.
    func mcpSizeAndTime(_ availability: MCPAvailability) -> String {
        let network = availability.networkBytes, local = availability.copyBytes
        var parts: [String] = []
        if let network, network == 0, local == 0, let derive = availability.deriveBytes {
            let size = derive > 0 ? "about \(MCPFormat.size(derive))" : "a size not declared yet"
            parts.append("nothing to download: Arsenal makes \(availability.dtype) from the installed weights on this Mac, writing \(size) (time not measured yet)")
        } else if let network, network == 0, local > 0 {
            parts.append("no download needed: copies \(MCPFormat.size(local)) from files already on this Mac")
        } else if let network {
            // A download larger than the copy it leaves (an FP32 source cast to FP16) names both; arsenal_model's size is the copy.
            var size = MCPFormat.size(network)
            if let disk = availability.entry.sizesByDtype[availability.dtype], disk > 0, availability.deriveBytes == nil,
                let weights = availability.plan?.items.first(where: { $0.kind == "weights" })?.downloadBytes, Double(weights) > Double(disk) * 1.05
            {
                size = "a \(size) download (\(MCPFormat.size(disk)) on disk at \(availability.dtype))"
            }
            if let rate = mcpRates.median(host: mcpHost(availability.entry)), rate > 0 {
                parts.append("\(size), \(MCPFormat.remaining(Double(network) / rate)) at this Mac's last measured \(MCPFormat.rate(rate))")
            } else {
                parts.append("\(size), time unknown until the download starts")
            }
            if local > 0 { parts[0] += " (plus \(MCPFormat.size(local)) copied from files already on this Mac)" }
            if let derive = availability.deriveBytes {
                parts[0] += ", then making \(availability.dtype) on this Mac (\(derive > 0 ? "about " + MCPFormat.size(derive) : "size not declared"), time not measured yet)"
            }
        } else {
            parts.append("size unknown until the download starts")
        }
        let dependencies = availability.dependencies.map { item in "\(item.name) (\(item.downloadBytes.map(MCPFormat.size) ?? "size unknown"))" }
        if !dependencies.isEmpty { parts[0] += ", including " + dependencies.joined(separator: " and ") }
        return parts.joined()
    }
    /// "Once downloaded it needs ~4 GB RAM (61 GB free now)." or what must be unloaded first.
    func mcpMemoryNote(_ entry: CatalogEntry, dtype: String, prefix: String = "Once downloaded it needs") async -> String {
        guard let need = mcpNeedGB(entry, dtype: dtype) else { return "Its RAM need is not declared." }
        let memory = mcpEnvironment.macMemory()
        if need <= memory.roomGB { return "\(prefix) ~\(MCPFormat.memory(decimalGB: need)) RAM (\(MCPFormat.memory(decimalGB: memory.roomGB)) free now)." }
        let candidates = await mcpUnloadCandidates(except: entry.id)
        let unload = candidates.isEmpty ? "close other apps first" : "unload \(candidates) with \(MCPNames.prefix)unload first"
        return
            "\(prefix) ~\(MCPFormat.memory(decimalGB: need)) RAM; \(MCPFormat.memory(decimalGB: memory.roomGB)) is free under the \(MCPFormat.memory(decimalGB: memory.limitGB)) limit now, so \(unload)."
    }
    /// Loaded models other than `except`, idle first, then largest: "qwen-image (18 GB), kokoro-82m (0.4 GB)".
    func mcpUnloadCandidates(except: String) async -> String {
        let memory = await service.memory(), activity = await service.activity()
        let entries = await mcpCatalog()
        let loaded = await service.rows().filter { $0.residentDType != nil && $0.id != except }
        let ranked = loaded.map { row -> (String, Double, Bool) in
            let entry = entries.first { $0.id == row.id }
            let size = (memory.models[row.id]?.footprintMB).map { $0 / 1000 } ?? entry.flatMap { mcpNeedGB($0, dtype: row.residentDType ?? row.selectedPrecision) } ?? 0
            let busy = activity.models.first { $0.id == row.id }.map { $0.running + $0.queued > 0 } ?? false
            return (row.id, size, busy)
        }.sorted { a, b in a.2 != b.2 ? !a.2 : a.1 > b.1 }
        return ranked.prefix(3).map { "\($0.0) (\(MCPFormat.memory(decimalGB: $0.1)))" }.joined(separator: ", ")
    }
    /// The not-downloaded answer: one line of facts, one of what to do. Not an error: nothing failed.
    func mcpConsentText(_ availability: MCPAvailability, verb: MCPConsentVerb, prefix: String? = nil) async -> String {
        let entry = availability.entry
        let what =
            availability.update
            ? "has an update that must be downloaded before it runs"
            : (availability.networkBytes ?? 1) == 0 && availability.copyBytes == 0 && availability.deriveBytes != nil
                ? "isn't made at \(availability.dtype) yet" : "isn't downloaded yet"
        let memory = await mcpMemoryNote(entry, dtype: availability.dtype)
        let outcome: String
        switch verb {
        case .run: outcome = "call \(MCPNames.prefix)run again with the same input and allow_download:true; it downloads and runs"
        case .load: outcome = "call \(MCPNames.prefix)load again with allow_download:true; it downloads and loads it"
        case .download: outcome = "call \(MCPNames.prefix)download again with allow_download:true; it downloads"
        }
        let first = "\(verb.refused): \(entry.name) \(what) — \(mcpSizeAndTime(availability)). \(memory)"
        return (prefix.map { $0 + "\n" } ?? "") + first + "\nAsk the user. With their yes, " + outcome + "."
    }
    /// Rule 4: a plan that grew more than 10% over the size last reported for this model and dtype (within 24 h) asks again.
    func mcpPlanChanged(_ availability: MCPAvailability) -> (old: Int64, new: Int64)? {
        let key = availability.entry.id + "/" + availability.dtype
        guard let reported = mcpReportedSizes[key], mcpEnvironment.now().timeIntervalSince(reported.at) < 86400 else { return nil }
        let now = (availability.networkBytes ?? 0) + availability.copyBytes + (availability.deriveBytes ?? 0)
        return Double(now) > Double(reported.bytes) * 1.1 ? (reported.bytes, now) : nil
    }
    func mcpRemember(_ availability: MCPAvailability) {
        mcpReportedSizes[availability.entry.id + "/" + availability.dtype] = (
            bytes: (availability.networkBytes ?? 0) + availability.copyBytes + (availability.deriveBytes ?? 0), at: mcpEnvironment.now()
        )
    }
}
