// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import MCP

/// The user's table, memory, settings and playback: the hand actions of the menu, over MCP.
extension APIRouter {
    func mcpAdd(_ body: JSONObject) async throws -> CallTool.Result {
        let entry = try await mcpEntry(mcpModelID(body))
        if entry.added { return mcpText("\(entry.name) (\(entry.id)) is already on the user's models; nothing changed.") }
        try await service.add(id: entry.id)
        // The table holds exactly the models with files on this Mac (or downloading): one without them joins it with its download.
        guard entry.installed else {
            return mcpText(
                "Nothing was downloaded, so \(entry.name) (\(entry.id)) is not on the user's table yet; it joins it when its download starts. "
                    + "\(MCPNames.prefix)run downloads it once the user agrees.")
        }
        return mcpText("Added \(entry.name) (\(entry.id)) to the user's models. It is already downloaded.")
    }
    func mcpBusy(_ entry: CatalogEntry, verb: String) async throws {
        guard let job = await mcpRunningJob(model: entry.id) else { return }
        throw MCPFailure(.busy, "\(verb): \(entry.id) is busy with the \(await job.title()) (job \(job.id)). Cancel it with \(MCPNames.prefix)cancel or wait.")
    }
    func mcpRemove(_ body: JSONObject) async throws -> CallTool.Result {
        let entry = try await mcpEntry(mcpModelID(body))
        guard entry.added else { return mcpText("\(entry.name) (\(entry.id)) is not on the user's models; nothing changed.") }
        try await mcpBusy(entry, verb: "Not removed")
        // On the table means its files are on this Mac or a download runs: taking it off deletes them, which needs the user's yes.
        let states = entry.row.states.values
        if states.contains(where: { [.downloading, .deriving].contains($0.state) }) {
            return mcpText("Not removed: \(entry.name) (\(entry.id)) is downloading. Cancel it with \(MCPNames.prefix)cancel first, only if the user asks.")
        }
        let updating = states.contains { $0.state == .updateAvailable || $0.phase == "localFolderErrorLoaded" }
        if entry.installed || !entry.row.withdrawnTiers.isEmpty || updating {
            return mcpText(
                "Not removed: \(entry.name) (\(entry.id)) is on the user's table because its files are on this Mac, and removing it deletes them. "
                    + "Ask the user. With their yes, call \(MCPNames.prefix)delete(model:\"\(entry.id)\", confirm:true).")
        }
        do { try await service.remove(id: entry.id, deleteWeights: false) } catch { throw await mcpFailure(error, entry: entry, verb: "Not removed") }
        let size = entry.installed ? entry.sizesByDtype[entry.row.selectedPrecision].flatMap { $0 > 0 ? " (\(MCPFormat.size($0)))" : nil } ?? "" : ""
        let files = entry.installed ? " Its downloaded files stay on disk\(size) and agents can still use it; \(MCPNames.prefix)delete removes them, only if the user asks." : ""
        return mcpText("Removed \(entry.name) (\(entry.id)) from the user's models\(entry.loaded ? " and unloaded it" : "").\(files)")
    }
    func mcpDelete(_ body: JSONObject) async throws -> CallTool.Result {
        let entry = try await mcpEntry(mcpModelID(body))
        try await mcpBusy(entry, verb: "Not deleted")
        // Every precision with files: offered ones by state, old ones the entry no longer offers from the store's listing.
        let offered = entry.dtypes.filter { [ModelState.ready, .loaded, .loading, .updateAvailable, .failed].contains(entry.row.states[$0]?.state ?? .notInstalled) }
        let installed = offered + entry.row.withdrawnTiers.keys.sorted().filter { !entry.dtypes.contains($0) }
        if let dtype = body["dtype"]?.string {
            guard body["remove"]?.bool != true else {
                throw MCPFailure(.invalidInput, "Not deleted: dtype deletes one precision and keeps the model on the table; remove:true goes with deleting every precision.")
            }
            return try await mcpDeletePrecision(entry, dtype: dtype, installed: installed, confirm: body["confirm"]?.bool == true)
        }
        let remove = body["remove"]?.bool == true
        if installed.isEmpty && !(remove && entry.added) { return mcpText("\(entry.name) has no downloaded files; nothing deleted.") }
        // Deleting weights is never one step: without confirm:true the answer is what would go and how much.
        if !installed.isEmpty && body["confirm"]?.bool != true {
            let parts = installed.map { dtype in entry.sizesByDtype[dtype].flatMap { $0 > 0 ? "\(dtype) \(MCPFormat.size($0))" : nil } ?? dtype }
            let total = installed.compactMap { entry.sizesByDtype[$0] }.reduce(0, +)
            let call = "\(MCPNames.prefix)delete(model:\"\(entry.id)\"\(remove ? ", remove:true" : ""), confirm:true)"
            return mcpText(
                "Not deleted: \(entry.name) (\(entry.id)) has \(parts.joined(separator: ", "))\(total > 0 && installed.count > 1 ? ", about \(MCPFormat.size(total)) in all" : "") "
                    + "on disk. Ask the user. With their yes, call \(call).")
        }
        if !installed.isEmpty {
            do { try await service.delete(id: entry.id) } catch { throw await mcpFailure(error, entry: entry, verb: "Not deleted") }
        }
        if remove && entry.added {
            do { try await service.remove(id: entry.id, deleteWeights: false) } catch { throw await mcpFailure(error, entry: entry, verb: "Deleted files, but not removed") }
        }
        let freed = installed.compactMap { entry.sizesByDtype[$0] }.reduce(0, +)
        var text = installed.isEmpty ? "Removed \(entry.name) (\(entry.id)) from the user's models." : "Deleted the downloaded files of \(entry.name) (\(entry.id))"
        if !installed.isEmpty {
            text += freed > 0 ? ", about \(MCPFormat.size(freed)) freed." : "."
            text +=
                " It is off the user's table; using it again downloads it with the user's yes."
        }
        return mcpText(text)
    }
    /// One precision's files (bug note 25 Oct 19:51: an agent could only delete every precision). When it was the precision the
    /// model uses, the model switches to a copy still on disk and the answer says which.
    func mcpDeletePrecision(_ entry: CatalogEntry, dtype: String, installed: [String], confirm: Bool) async throws -> CallTool.Result {
        guard entry.dtypes.contains(dtype) else { throw MCPFailure(.invalidInput, "\(entry.name) offers \(entry.dtypes.joined(separator: ", ")), not \(dtype).") }
        guard installed.contains(dtype) else { return mcpText("\(entry.name) has no downloaded \(dtype) files; nothing deleted.") }
        let size = entry.sizesByDtype[dtype].flatMap { $0 > 0 ? MCPFormat.size($0) : nil }
        guard confirm else {
            let call = "\(MCPNames.prefix)delete(model:\"\(entry.id)\", dtype:\"\(dtype)\", confirm:true)"
            return mcpText("Not deleted: \(entry.name) (\(entry.id)) has \(dtype)\(size.map { " \($0)" } ?? "") on disk. Ask the user. With their yes, call \(call).")
        }
        do { _ = try await service.deletePrecision(id: entry.id, precision: dtype) } catch { throw await mcpFailure(error, entry: entry, verb: "Not deleted") }
        var text = "Deleted the \(dtype) files of \(entry.name) (\(entry.id))\(size.map { ", about \($0) freed" } ?? "")."
        if dtype == entry.row.selectedPrecision {
            let now = await service.rows().first { $0.id == entry.id }?.selectedPrecision
            if let now, now != dtype {
                text += " It now uses \(now), the copy still on disk."
            } else {
                text += " No other usable copy is on disk; using it again downloads \(dtype) with the user's yes."
            }
        }
        return mcpText(text)
    }
    func mcpUnload(_ body: JSONObject) async throws -> CallTool.Result {
        let id = try mcpModelID(body)
        if id == "all" {
            let memory = await service.memory()
            var unloaded: [String] = [], kept: [String] = [], freed = 0.0
            for row in await service.rows() where row.residentDType != nil {
                if let job = await mcpRunningJob(model: row.id) { kept.append("\(row.id) (busy with job \(job.id))"); continue }
                do {
                    try await service.unload(id: row.id); unloaded.append(row.id); freed += (memory.models[row.id]?.footprintMB ?? 0) / 1000
                } catch { kept.append("\(row.id) (\((await mcpFailure(error, entry: nil, verb: "busy")).text))") }
            }
            if unloaded.isEmpty && kept.isEmpty { return mcpText("No models were loaded; nothing changed.") }
            var text = unloaded.isEmpty ? "Unloaded nothing." : "Unloaded \(MCPFormat.plural(unloaded.count, "model")): \(unloaded.joined(separator: ", "))"
            if !unloaded.isEmpty { text += freed > 0 ? "; about \(MCPFormat.memory(decimalGB: freed)) freed." : "." }
            if !kept.isEmpty { text += " Kept loaded: \(kept.joined(separator: ", "))." }
            return mcpText(text)
        }
        let entry = try await mcpEntry(id)
        guard entry.loaded else { return mcpText("\(entry.name) wasn't loaded; nothing changed.") }
        try await mcpBusy(entry, verb: "Not unloaded")
        let footprint = await service.memory().models[entry.id]?.footprintMB
        do { try await service.unload(id: entry.id) } catch { throw await mcpFailure(error, entry: entry, verb: "Not unloaded") }
        return mcpText("Unloaded \(entry.name)" + (footprint.map { "; about \(MCPFormat.memory(decimalGB: $0 / 1000)) freed." } ?? "."))
    }

    // MARK: Settings

    /// Reserved keys map onto their own service calls (`ModelSettingKeys`); everything else is the model's declared
    /// settings (one atomic set).
    func mcpSettings(_ body: JSONObject) async throws -> CallTool.Result {
        if let action = body["action"]?.string { return try await mcpSettingsAction(action, body: body) }
        guard body["model"]?.string != nil else { return try await mcpAppSettings(body["set"]?.object) }
        let entry = try await mcpEntry(mcpModelID(body))
        guard let values = body["set"]?.object else { return try await mcpModelSettingsText(entry) }
        guard !values.isEmpty else { throw MCPFailure(.invalidInput, "Not changed: set is empty.") }
        // Validate every reserved key before anything is applied.
        let split: (declared: JSONObject, reserved: [(String, JSONValue)])
        do { split = try ModelSettingKeys.split(values, entry: entry) } catch {
            throw MCPFailure(error.unsupported ? .unsupported : .invalidInput, "Not changed: \(error.text)")
        }
        var declared = split.declared
        let reserved = split.reserved
        if declared["fast"] != nil { throw MCPFailure(.invalidInput, "Not changed: fast is chosen at load; call \(MCPNames.prefix)load(model:\"\(entry.id)\", fast:true|false).") }
        var applied: [String] = []
        if !declared.isEmpty {
            declared = try await mcpImportSettingPaths(id: entry.id, values: declared)
            do { try await service.setSettings(id: entry.id, values: declared) } catch { throw await mcpFailure(error, entry: entry, verb: "Not changed") }
            applied += declared.keys.sorted().map { $0 == "__tools" ? "show_in_menu" : $0 }
        }
        for (key, value) in reserved {
            do {
                try await ModelSettingKeys.apply(key, value, id: entry.id, service: service)
                applied.append(key)
            } catch {
                let failure = await mcpFailure(error, entry: entry, verb: "Not changed")
                let done = applied.isEmpty ? "Nothing was applied" : "Applied \(applied.joined(separator: ", "))"
                throw MCPFailure(failure.category, "\(done); \(key) was not: \(failure.text)")
            }
        }
        var text = "Changed \(applied.joined(separator: ", ")) for \(entry.name)."
        if body["apply"]?.bool == true {
            do { try await service.applySettings(id: entry.id) } catch { throw await mcpFailure(error, entry: entry, verb: "Changed, but not applied") }
            text += " Applied now (reloaded)."
        } else if (try? await service.settings(id: entry.id))?.contains(where: \.pending) == true {
            text += " Load settings take effect at the next load; apply:true reloads now."
        }
        return mcpText(text)
    }
    func mcpModelSettingsText(_ entry: CatalogEntry) async throws -> CallTool.Result {
        let row = entry.row
        var lines = ["Settings for \(entry.name) (\(entry.id)):"]
        lines.append("  dtype: \(row.selectedPrecision) (\(entry.dtypes.joined(separator: "|")))")
        lines.append("  keep_hot: \(row.keepHot) — stays loaded instead of unloading when idle")
        lines.append("  show_in_menu: \(row.toolsShown) — its tools appear in the Arsenal menu")
        let runners = entry.runners.filter(\.eligible).map(\.name)
        if !runners.isEmpty { lines.append("  runner: \(row.runner) (\(runners.joined(separator: "|")))") }
        if row.fastAvailable { lines.append("  fast: \(row.fast) — set with \(MCPNames.prefix)load(fast:…)") }
        let settings: [SettingState]
        do { settings = try await service.settings(id: entry.id) } catch { throw await mcpFailure(error, entry: entry, verb: "Not read") }
        for setting in settings {
            let declaration = setting.declaration
            var text = "  \(setting.id): \(mcpCompact(setting.value))"
            var facts: [String] = []
            if let choices = declaration["choices"]?.array { facts.append(choices.map(mcpCompact).joined(separator: "|")) }
            if let low = declaration["min"]?.number, let high = declaration["max"]?.number {
                facts.append("\(MCPFormat.number(low, decimals: 4))–\(MCPFormat.number(high, decimals: 4))")
            }
            facts.append("applies at \(declaration["applies"]?.string ?? "request")")
            text += " (\(facts.joined(separator: ", ")))"
            if let title = declaration["title"]?.string { text += " — \(title)" }
            if setting.pending { text += "; changed, takes effect at the next load" }
            lines.append(text)
        }
        lines.append("Change with \(MCPNames.prefix)settings(model:\"\(entry.id)\", set:{key: value}); apply:true reloads to apply load settings now.")
        return mcpText(lines.joined(separator: "\n"))
    }
    func mcpAppSettings(_ values: JSONObject?) async throws -> CallTool.Result {
        var residency = await service.residencySettings(), memory = await service.memorySettings()
        if let values {
            let allowed = ["idle_unload_minutes", "manual_idle_unload_minutes", "when_full"]
            if let unknown = values.keys.sorted().first(where: { !allowed.contains($0) }) {
                throw MCPFailure(
                    .invalidInput, "Not changed: \(unknown) is not an app setting over MCP (\(allowed.joined(separator: ", "))). Launch at Login is set in the Arsenal menu.")
            }
            for key in ["idle_unload_minutes", "manual_idle_unload_minutes"] {
                guard let value = values[key] else { continue }
                guard let minutes = value.number, minutes >= 0, minutes <= 10080, minutes == minutes.rounded() else {
                    throw MCPFailure(.invalidInput, "Not changed: \(key) must be a whole number of minutes from 0 (never) to 10080.")
                }
                if key == "idle_unload_minutes" { residency.onDemandIdleMinutes = Int(minutes) } else { residency.manualIdleMinutes = Int(minutes) }
            }
            if let value = values["when_full"] {
                guard let raw = value.string, let choice = MemoryWhenFull(rawValue: raw) else { throw MCPFailure(.invalidInput, "Not changed: when_full must be refuse or queue.") }
                memory.whenFull = choice
            }
            try await service.setResidencySettings(residency)
            if values["when_full"] != nil { try await service.setMemorySettings(memory) }
            return mcpText("Changed \(values.keys.sorted().joined(separator: ", ")).")
        }
        return mcpText(
            [
                "App settings:",
                "  idle_unload_minutes: \(residency.onDemandIdleMinutes) — a model loaded for a request unloads after this many idle minutes (0 = never)",
                "  manual_idle_unload_minutes: \(residency.manualIdleMinutes) — a model loaded by hand unloads after this many idle minutes (0 = never)",
                "  when_full: \(memory.whenFull.rawValue) (refuse|queue) — what a request does when memory is short",
                "  Launch at Login: set in the Arsenal menu, not over MCP.",
                "Change with \(MCPNames.prefix)settings(set:{key: value}). The memory limit is fixed at 100/128 of this Mac's RAM (\(MCPFormat.memory(decimalGB: MacMemoryGuard.macLimitGB)) here)."
            ].joined(separator: "\n"))
    }
    func mcpSettingsAction(_ action: String, body: JSONObject) async throws -> CallTool.Result {
        switch action {
        case "refresh":
            do { try await service.refresh() } catch { throw await mcpFailure(error, entry: nil, verb: "Not refreshed") }
            return mcpText("Rescanned model folders and the catalog.")
        case "approve_code":
            let entry = try await mcpEntry(mcpModelID(body))
            guard entry.row.local, let current = entry.row.contentDigest else {
                return mcpText("\(entry.name) is not a local model with code to approve; nothing changed.")
            }
            let digest = body["content_digest"]?.string ?? current
            do { try await service.approveLocalCode(id: entry.id, contentDigest: digest, approvedBy: .mcp) } catch {
                throw await mcpFailure(error, entry: entry, verb: "Not approved")
            }
            let previous = entry.row.approval.flatMap { $0.digest != digest ? " It replaces the approval of digest \($0.digest.prefix(12))…." : nil } ?? ""
            let note = await service.pinnedFilesNote(id: entry.id).map { " " + $0 } ?? ""
            return mcpText(
                "Approved \(entry.name)'s local code (digest \(digest.prefix(12))…), on the user's behalf.\(previous) A change to its folder needs a new approval.\(note)")
        case "revoke_code":
            let entry = try await mcpEntry(mcpModelID(body))
            guard entry.row.local else { return mcpText("\(entry.name) is not a local model with code to approve; nothing changed.") }
            guard let approval = entry.row.approval else { return mcpText("\(entry.name)'s code is not approved; nothing changed.") }
            try await mcpBusy(entry, verb: "Not revoked")
            do { try await service.revokeLocalCode(id: entry.id) } catch { throw await mcpFailure(error, entry: entry, verb: "Not revoked") }
            return mcpText(
                "Revoked the approval of \(entry.name)'s local code (digest \(approval.digest.prefix(12))…)\(entry.loaded ? " and unloaded it" : ""). "
                    + "It runs again only after the user approves it: \(MCPNames.prefix)settings(model:\"\(entry.id)\", action:\"approve_code\").")
        case "rollback":
            let entry = try await mcpEntry(mcpModelID(body))
            do { try await service.rollbackRevision(id: entry.id) } catch { throw await mcpFailure(error, entry: entry, verb: "Not rolled back") }
            return mcpText("Rolled \(entry.name) back to its previous revision.")
        default: throw MCPFailure(.invalidInput, "Not changed: action must be refresh, approve_code, revoke_code or rollback.")
        }
    }

    func mcpPlayback(_ body: JSONObject) async throws -> CallTool.Result {
        let queue = await playback
        switch body["action"]?.string {
        case "stop":
            let before = await queue.snapshot()
            await queue.stop()
            let discarded = before.queued + (before.current == nil ? 0 : 1)
            return mcpText(discarded == 0 ? "Nothing was playing; the queue is empty." : "Playback stopped; \(MCPFormat.plural(discarded, "clip")) discarded. The files are kept.")
        case "mute":
            try await queue.setMuted(true)
            return mcpText("Muted. Speech is still saved as files; nothing plays until unmute.")
        case "unmute":
            try await queue.setMuted(false)
            return mcpText("Unmuted. Speech plays as it is made.")
        default:
            let snapshot = await queue.snapshot()
            return mcpText("Playback: \(snapshot.playing ? "playing" : "idle"), \(snapshot.queued) queued, \(snapshot.muted ? "muted" : "not muted").")
        }
    }
}
