// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import MCP

extension APIRouter {
    /// One entry's state at its selected dtype, as a short sentence.
    func mcpStateText(_ entry: CatalogEntry, dtype: String) -> String {
        let state = entry.row.states[dtype]
        let table = entry.added ? "On the user's table" : "Not on the user's table"
        let size = entry.sizesByDtype[dtype].flatMap { $0 > 0 ? MCPFormat.size($0) : nil }
        switch state?.state ?? .notInstalled {
        case .loaded: return "\(table); downloaded, loaded."
        case .ready, .loading: return "\(table); downloaded."
        case .downloading:
            let figures = state?.totalBytes.map { MCPFormat.progress(state?.bytes ?? 0, of: $0) } ?? (state?.progress.map { "\(Int($0 * 100))%" } ?? "")
            return "\(table); downloading\(figures.isEmpty ? "" : " (\(figures))")."
        case .deriving: return "\(table); converting to \(dtype)."
        case .updateAvailable: return "\(table); downloaded, with an update to download first\(size.map { " (\($0))" } ?? "")."
        case .failed: return "\(table); the last download failed\(state?.error.map { ": \($0)" } ?? "")."
        case .notInstalled: return "\(table); not downloaded\(size.map { " (\($0))" } ?? "")."
        }
    }
    /// Good for and limits on one line; each verification note ("Checked live on …") on its own "Checked:" line, since it is
    /// evidence that the model runs, not something it cannot do.
    func mcpNotes(_ entry: CatalogEntry) -> [String] {
        let checkedPrefix = "Checked live "
        let checked = entry.limits.filter { $0.hasPrefix(checkedPrefix) }
        let limits = entry.limits.filter { !$0.hasPrefix(checkedPrefix) }
        if entry.goodFor.isEmpty && limits.isEmpty && checked.isEmpty { return ["No good-for/limits notes yet."] }
        var parts: [String] = []
        if !entry.goodFor.isEmpty { parts.append("Good for: \(MCPFormat.phrases(entry.goodFor)).") }
        if !limits.isEmpty { parts.append("Limits: \(MCPFormat.phrases(limits)).") }
        var lines = parts.isEmpty ? [] : [parts.joined(separator: " ")]
        lines += checked.map { "Checked: live " + MCPFormat.phrases([String($0.dropFirst(checkedPrefix.count))]) + "." }
        return lines
    }

    func mcpFind(_ body: JSONObject) async throws -> CallTool.Result {
        let query = body["query"]?.string?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
        let task = body["task"]?.string?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
        let limit = min(50, max(1, Int(body["limit"]?.number ?? 8))), runsHere = body["runs_here"]?.bool ?? true
        let memory = mcpEnvironment.macMemory()
        var entries = await mcpCatalog()
        switch body["on_mac"]?.string {
        case "table": entries = entries.filter(\.added)
        case "installed": entries = entries.filter(\.installed)
        case "loaded": entries = entries.filter(\.loaded)
        default: break
        }
        if let commercial = body["commercial"]?.bool { entries = entries.filter { $0.commercial == commercial } }
        var exactLanguage: Set<String> = []
        if let language = body["language"]?.string?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty {
            let code = MCPSearch.languageCode(language)
            entries = entries.filter { MCPSearch.languageFit($0.languages, code: code) != .none }
            exactLanguage = Set(entries.filter { MCPSearch.languageFit($0.languages, code: code) == .exact }.map(\.id))
        }
        if let state = body["state"]?.string { entries = entries.filter { $0.row.state.rawValue == state } }
        var operations: [String: [String]] = [:]
        if query != nil || task != nil { for entry in entries { operations[entry.id] = await service.declaredOperationNames(id: entry.id) } }
        if let task { entries = entries.filter { MCPSearch.matches($0, operations: operations[$0.id] ?? [], task: task) } }
        var scores: [String: Int] = [:]
        if let query {
            scores = MCPSearch.relevant(entries, operations: operations, query: query)
            entries = entries.filter { scores[$0.id] != nil }
        }
        let fits = { (entry: CatalogEntry) in (self.mcpNeedGB(entry, dtype: entry.row.selectedPrecision) ?? 0) <= memory.limitGB }
        let tooLarge = runsHere ? entries.filter { !fits($0) }.count : 0
        if runsHere { entries = entries.filter(fits) }
        entries.sort { a, b in
            let sa = scores[a.id] ?? 0, sb = scores[b.id] ?? 0
            if sa != sb { return sa > sb }
            let la = exactLanguage.contains(a.id), lb = exactLanguage.contains(b.id)
            if la != lb { return la }
            for (x, y) in [(a.suggested, b.suggested), (a.added, b.added), (a.installed, b.installed), (a.loaded, b.loaded)] where x != y { return x }
            return a.name.localizedStandardCompare(b.name) == .orderedAscending
        }
        let shown = Array(entries.prefix(limit))
        var described: [String] = []
        if let query { described.append("for \"\(query)\"") }
        if let task { described.append("for task \"\(task)\"") }
        if let language = body["language"]?.string { described.append("in language \"\(language)\"") }
        if let state = body["state"]?.string { described.append("in state \(state)") }
        if let place = body["on_mac"]?.string { described.append(place == "table" ? "on the user's table" : place == "installed" ? "downloaded" : "loaded") }
        let description = described.isEmpty ? "" : " " + described.joined(separator: " ")
        if body["format"]?.string == "json" {
            return try mcpJSON([
                "models": .array(
                    shown.map { entry in
                        let dtype = entry.row.selectedPrecision
                        var object: JSONObject = [
                            "id": .string(entry.id), "name": .string(entry.name), "kind": .string(entry.kind), "purpose": .string(entry.row.purpose),
                            "summary": .string(entry.summary),
                            "dtype": .string(dtype), "state": .string(entry.row.states[dtype]?.state.rawValue ?? "notInstalled"), "added": .bool(entry.added),
                            "installed": .bool(entry.installed), "loaded": .bool(entry.loaded), "commercial": .bool(entry.commercial),
                            "good_for": .array(entry.goodFor.map(JSONValue.string)), "limits": .array(entry.limits.map(JSONValue.string))
                        ]
                        object["size_bytes"] = entry.sizesByDtype[dtype].map { .number(Double($0)) } ?? .null
                        object["needed_gb"] = mcpNeedGB(entry, dtype: dtype).map { MCPFormat.memoryValue(decimalGB: $0) } ?? .null
                        return .object(object)
                    }),
                "matched": .number(Double(entries.count)), "too_large": .number(Double(tooLarge)), "limit_gb": MCPFormat.memoryValue(decimalGB: memory.limitGB)
            ])
        }
        guard !shown.isEmpty else {
            let large = tooLarge > 0 ? " \(MCPFormat.plural(tooLarge, "model")) matched but won't fit this Mac; runs_here:false lists them." : ""
            return mcpText("No model\(description).\(large) Try other words, or \(MCPNames.prefix)find() with no arguments for every model.")
        }
        let onTable = entries.filter(\.added), downloaded = onTable.filter(\.installed)
        var header = "\(MCPFormat.plural(entries.count, "model"))\(description)"
        if !onTable.isEmpty {
            header +=
                " (\(onTable.count) on the user's table, \(downloaded.count == onTable.count ? (onTable.count == 1 ? "downloaded" : "all downloaded") : "\(downloaded.count) downloaded"))"
        }
        header += entries.count > shown.count ? "; the first \(shown.count) (pass limit for more):" : ":"
        var lines = [header]
        for entry in shown {
            let dtype = entry.row.selectedPrecision
            var facts: [String] = []
            if let size = entry.sizesByDtype[dtype], size > 0 { facts.append(MCPFormat.size(size)) }
            if let need = mcpNeedGB(entry, dtype: dtype) { facts.append("needs ~\(MCPFormat.memory(decimalGB: need)) RAM") }
            let purpose = entry.row.purpose.isEmpty ? entry.summary : entry.row.purpose
            if !entry.params.isEmpty { facts.insert("\(entry.params) parameters", at: 0) }
            if let languages = Self.mcpLanguages(entry.languages) { facts.append(languages) }
            lines.append("• \(entry.id) — \(entry.name): \(purpose).\(facts.isEmpty ? "" : " " + facts.joined(separator: ", ") + ".") \(mcpStateText(entry, dtype: dtype))")
            lines += mcpNotes(entry).map { "  " + $0 }
        }
        if tooLarge > 0 {
            lines.append(
                "\(MCPFormat.plural(tooLarge, "more model")) \(tooLarge == 1 ? "doesn't" : "don't") fit this Mac (over the \(MCPFormat.memory(decimalGB: memory.limitGB)) limit); runs_here:false lists them."
            )
        }
        lines.append("Details: \(MCPNames.prefix)model(model:\"ID\").")
        return mcpText(lines.joined(separator: "\n"))
    }

    func mcpModel(_ body: JSONObject) async throws -> CallTool.Result {
        let entry = try await mcpEntry(mcpModelID(body))
        let operations = await service.operations(id: entry.id)
        let primary = RunAdmission.primaryOperation(operations, preferred: nil)
        var shown = operations
        if let name = body["operation"]?.string {
            guard let operation = operations.first(where: { $0.name == name }) else {
                throw MCPFailure(.unsupported, "\(entry.name) has no operation \"\(name)\". It offers: \(operations.map(\.name).joined(separator: ", ")).")
            }
            shown = [operation]
        }
        let dtype = entry.row.selectedPrecision, row = entry.row, memory = mcpEnvironment.macMemory()
        let readme = body["docs"]?.bool == true ? (try? await service.modelREADME(id: entry.id)) : nil
        if body["format"]?.string == "json" {
            var object = try WireJSON.decode(JSONEncoder().encode(entry))
            object["row"] = nil
            object["operations"] = .array(
                shown.map { operation in
                    [
                        "name": .string(operation.name), "title": .string(operation.title), "primary": .bool(operation.name == primary), "output": .string(operation.output),
                        "input_schema": .object(operation.inputSchema), "schema_uri": .string("arsenal://models/\(entry.id)/operations/\(operation.name)/schema")
                    ]
                })
            object["needed_gb"] = mcpNeedGB(entry, dtype: dtype).map { MCPFormat.memoryValue(decimalGB: $0) } ?? .null
            object["free_gb"] = MCPFormat.memoryValue(decimalGB: memory.roomGB); object["limit_gb"] = MCPFormat.memoryValue(decimalGB: memory.limitGB)
            if let readme { object["readme"] = .string(readme) }
            return try mcpJSON(object)
        }
        var lines: [String] = []
        let purpose = row.purpose.isEmpty ? entry.summary : row.purpose
        let licence = entry.licenceName ?? entry.licence
        let commercial = entry.commercial ? "commercial use allowed" : "no commercial use" + (entry.licenceConditions.map { " (\($0))" } ?? "")
        lines.append(
            "\(entry.name) (\(entry.id)) — \(purpose). \(entry.publisher.isEmpty ? "" : entry.publisher + ", ")\(licence.isEmpty ? "licence not stated" : licence), \(commercial).")
        let tiers = entry.dtypes.map { tier -> String in
            var text = tier
            let state = row.states[tier]?.state ?? .notInstalled
            if let reason = row.dtypeUnavailableReason(tier) { return text + " unavailable (\(reason.trimmingCharacters(in: .punctuationCharacters)))" }
            switch state {
            case .ready, .loaded, .loading: text += " downloaded"
            case .downloading, .deriving: text += " downloading"
            case .updateAvailable: text += " downloaded (update pending)"
            case .failed: text += " failed"
            case .notInstalled: text += " not downloaded"
            }
            if let size = entry.sizesByDtype[tier], size > 0 { text += " (\(MCPFormat.size(size)))" }
            if row.precisionQualityWarnings[tier] != nil { text += ", quality may be lower" }
            if tier == dtype { text += ", selected" }
            return text
        }
        let loaded = row.residentDType.map { "Loaded at \($0)." } ?? "Not loaded."
        let need = mcpNeedGB(entry, dtype: dtype).map { "Needs ~\(MCPFormat.memory(decimalGB: $0)) RAM at \(dtype); " } ?? "RAM need not declared; "
        lines.append(
            "\(entry.added ? "On the user's table" : "Not on the user's table"); \(tiers.joined(separator: "; ")). \(loaded) \(need)\(MCPFormat.memory(decimalGB: memory.roomGB)) free under the \(MCPFormat.memory(decimalGB: memory.limitGB)) limit now."
        )
        if (row.states[dtype]?.state ?? .notInstalled) == .updateAvailable {
            lines.append("An update must be downloaded before it runs: \(MCPNames.prefix)download(model:\"\(entry.id)\") gives its size; download only with the user's yes.")
        }
        if row.revision?.pending != nil {
            lines.append("A newer revision is ready; Arsenal swaps it in by itself when the model is idle. settings action rollback returns to the previous one.")
        }
        if row.codeConsentRequired, let approval = row.approval, approval.digest != row.contentDigest {
            lines.append(
                "Its code changed since the user approved digest \(approval.digest.prefix(12))…; the new contents are \(row.contentDigest?.prefix(12) ?? "unknown")…. The user's yes is enough to approve them: \(MCPNames.prefix)settings(model:\"\(entry.id)\", action:\"approve_code\", content_digest:\"\(row.contentDigest ?? "")\")."
            )
        } else if row.codeConsentRequired {
            lines.append(
                "Runs local code nobody has approved (digest \(row.contentDigest?.prefix(12) ?? "unknown")…): approve only with the user's yes, via \(MCPNames.prefix)settings(model:\"\(entry.id)\", action:\"approve_code\")."
            )
        }
        let runners = entry.runners.filter(\.eligible)
        if runners.count > 1 {
            lines.append("Runners: " + runners.map { $0.name + ($0.selected ? " (selected)" : "") }.joined(separator: ", ") + "; \(MCPNames.prefix)settings set runner switches.")
        }
        lines += mcpFactLines(entry)
        lines += mcpNotes(entry)
        if shown.isEmpty {
            lines.append("Operations: none declared.")
        } else {
            lines.append("Operations:")
            for operation in shown {
                lines.append("  \(operation.name)\(operation.name == primary ? " (default)" : "") — \(operation.title); returns \(mcpOutputNoun(operation))")
                lines += mcpFieldLines(operation.inputSchema, indent: "    ", depth: 0)
            }
            let example = shown.first { $0.name == primary } ?? shown[0]
            let input = String(decoding: (try? WireJSON.encode(mcpExampleInput(example.inputSchema))) ?? Data("{}".utf8), as: UTF8.self)
            let op = example.name == primary ? "" : ", operation:\"\(example.name)\""
            lines.append("Run: \(MCPNames.prefix)run(model:\"\(entry.id)\"\(op), input:\(input))")
            lines.append("Full input schema: arsenal://models/\(entry.id)/operations/\(example.name)/schema")
        }
        if let readme {
            lines.append("README:\n" + readme)
        } else if body["docs"]?.bool == true {
            lines.append("README: unavailable for this model.")
        } else {
            lines.append("What it is for, exact input formats, limits and licence: \(MCPNames.prefix)model(model:\"\(entry.id)\", docs:true).")
        }
        return mcpText(lines.joined(separator: "\n"))
    }
    /// "languages: en, pl, de", "multilingual", or nil for a model with no language (images, audio cleanup).
    nonisolated static func mcpLanguages(_ codes: [String]) -> String? {
        let real = codes.filter { !["und", "zxx"].contains($0) }
        if real.isEmpty { return nil }
        if real == ["mul"] { return "multilingual" }
        return (real.count == 1 ? "language: " : "languages: ") + real.joined(separator: ", ")
    }
    /// Parameters, languages and the recorded figures per precision (the summary should print them). Each figure
    /// says how it was obtained (measured, quick check, estimate) and on which chip; unrecorded figures are left out, never zero.
    nonisolated func mcpFactLines(_ entry: CatalogEntry) -> [String] {
        var lines: [String] = []
        var head = ["Parameters: \(entry.params.isEmpty ? "not declared" : entry.params)."]
        if let languages = Self.mcpLanguages(entry.languages) { head.append(languages.prefix(1).uppercased() + languages.dropFirst() + ".") }
        lines.append(head.joined(separator: " "))
        let figures = entry.dtypes.compactMap { dtype -> String? in
            guard let stats = entry.row.precisionStats[dtype] else { return nil }
            let cells = stats.kind.metrics.compactMap { metric -> String? in
                let cell = stats.value(metric)
                guard let value = cell.value, [.measured, .quickCheck, .derived, .declared].contains(cell.state) else { return nil }
                return Self.mcpFigure(metric, value) + (cell.state == .measured ? "" : " (\(cell.label))")
            }
            guard !cells.isEmpty else { return nil }
            // The chip of the first recorded figure in the kind's metric order (deterministic, not dictionary order).
            let chip = stats.kind.metrics.lazy.compactMap { stats.value($0).value != nil ? stats.value($0).provenance?.chip : nil }.first
            return "\(dtype)\(chip.map { " on \($0)" } ?? ""): " + cells.joined(separator: ", ")
        }
        lines.append(figures.isEmpty ? "Figures: none recorded yet." : "Figures: " + figures.joined(separator: "; ") + ".")
        return lines
    }
    nonisolated static func mcpFigure(_ metric: StatMetric, _ value: Double) -> String {
        func number(_ v: Double) -> String { v >= 100 ? String(Int(v.rounded())) : v >= 10 ? String(format: "%.1f", v) : String(format: "%.2f", v) }
        switch metric {
        // Memory in Activity Monitor's GB, as the table and the "needs ~… RAM" lines show it (2 Nov: SeedVR2 read 38.7 here
        // and ~36 there for the same recorded peak).
        case .peakMemoryGB: return "peak memory \(number(MemoryDisplay.gib(decimalGB: value))) GB"
        case .loadPeakMemoryGB: return "load peak \(number(MemoryDisplay.gib(decimalGB: value))) GB"
        case .realTimeFactor: return "real-time factor \(number(value))"
        case .firstAudioSeconds: return "first audio \(number(value)) s"
        case .firstTokenSeconds: return "first token \(number(value)) s"
        case .maximumContextTokens: return "context \(Int(value)) tokens"
        default: return "\(number(value)) \(metric.unit)"
        }
    }
    nonisolated func mcpOutputNoun(_ operation: ModelOperation) -> String {
        switch operation.output {
        case "audio": return "audio files"
        case "image": return "images"
        case "file": return "files"
        case "text-stream", "text": return "text"
        default: return "data"
        }
    }

    func mcpStatus(_ body: JSONObject) async throws -> CallTool.Result {
        let memory = mcpEnvironment.macMemory(), workers = await service.memory(), activity = await service.activity(), rows = await service.rows()
        let playback = await self.playback.snapshot()
        let loaded = rows.filter { $0.residentDType != nil }.map { row -> (String, String, Double?, Bool) in
            let busy = activity.models.first { $0.id == row.id }.map { $0.running + $0.queued > 0 } ?? false
            return (row.id, row.residentDType ?? row.selectedPrecision, workers.models[row.id]?.footprintMB, busy)
        }
        var jobs: [(MCPJob, String)] = []
        for job in mcpJobs.values.sorted(by: { $0.created < $1.created }) where await job.status == .running { jobs.append((job, await job.statusLine())) }
        let jobModels = Set(jobs.map(\.0.model))
        let external = rows.filter { row in !jobModels.contains(row.id) && row.states.values.contains { $0.state == .downloading || $0.state == .deriving } }
        let history = body["history"]?.bool == true
        let recent = mcpJobRecords.values.filter { $0.status != .running }.sorted { ($0.finished ?? $0.created) > ($1.finished ?? $1.created) }.prefix(10)
        if body["format"]?.string == "json" {
            var object: JSONObject = [
                "memory": [
                    "in_use_gb": MCPFormat.memoryValue(decimalGB: memory.inUseGB), "limit_gb": MCPFormat.memoryValue(decimalGB: memory.limitGB),
                    "total_gb": MCPFormat.memoryValue(decimalGB: memory.totalGB)
                ],
                "loaded": .array(
                    loaded.map {
                        ["model": .string($0.0), "dtype": .string($0.1), "footprint_gb": $0.2.map { MCPFormat.memoryValue(decimalGB: $0 / 1000) } ?? .null, "busy": .bool($0.3)]
                    }),
                "downloads_outside_mcp": .array(external.map { .string($0.id) }),
                "playback": ["muted": .bool(playback.muted), "playing": .bool(playback.playing), "queued": .number(Double(playback.queued))]
            ]
            var values: [JSONValue] = []
            for (job, _) in jobs { values.append(.object(await mcpJobObject(job))) }
            object["jobs"] = .array(values)
            if history { object["recent"] = .array(recent.compactMap { try? JSONDecoder().decode(JSONValue.self, from: JSONEncoder.mcp.encode($0)) }) }
            return try mcpJSON(object)
        }
        var lines = [
            "Mac memory: \(MCPFormat.memory(decimalGB: memory.inUseGB)) of the \(MCPFormat.memory(decimalGB: memory.limitGB)) limit in use (\(MCPFormat.memory(decimalGB: memory.totalGB)) RAM)."
        ]
        lines.append(
            "Loaded: "
                + (loaded.isEmpty
                    ? "nothing."
                    : loaded.map { "\($0.0) \($0.1)" + ($0.2.map { " " + MCPFormat.memory(decimalGB: $0 / 1000) } ?? "") + ", \($0.3 ? "busy" : "idle")" }.joined(separator: " · ")
                        + "."))
        var running = jobs.map { "\($0.0.id) \($0.1)" }
        for row in external {
            let state = row.states.values.first { $0.state == .downloading || $0.state == .deriving }
            let figures = state?.totalBytes.map { ": " + MCPFormat.progress(state?.bytes ?? 0, of: $0) } ?? ""
            running.append("\(row.id) download started in the app\(figures) (\(MCPNames.prefix)cancel(model:\"\(row.id)\") stops it)")
        }
        lines.append("Running: " + (running.isEmpty ? "nothing." : running.joined(separator: " · ") + "."))
        let clip = playback.playing ? "playing" : "idle"
        lines.append("Playback: \(clip), \(playback.queued) queued, \(playback.muted ? "muted" : "not muted").")
        if history {
            lines.append(recent.isEmpty ? "Recent jobs (24 h): none." : "Recent jobs (24 h):")
            for record in recent {
                let title = record.kind == .run ? "\(record.operation ?? "run") with \(record.name)" : "\(record.kind.rawValue) of \(record.name)"
                lines.append("  \(record.id) \(record.status.rawValue) — \(title)" + (record.line.map { ": " + $0 } ?? ""))
            }
            let minute = activity.models.compactMap { model in model.recent.map { "\(model.id) \($0.finished) finished, \($0.failed) failed" } }
            if !minute.isEmpty { lines.append("Last minute: " + minute.joined(separator: " · ") + ".") }
        }
        return mcpText(lines.joined(separator: "\n"))
    }
}

extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
}

/// Input fields as readable lines: name, type or allowed values, required/default, the description's first clause.
func mcpFieldLines(_ schema: JSONObject, indent: String, depth: Int) -> [String] {
    guard depth < 3, let properties = schema["properties"]?.object else { return [] }
    let required = Set(schema["required"]?.array?.compactMap(\.string) ?? [])
    let names = properties.keys.sorted { a, b in required.contains(a) != required.contains(b) ? required.contains(a) : a < b }
    var lines: [String] = []
    for name in names {
        guard let field = properties[name]?.object else { continue }
        var line = indent + name + ": " + mcpTypeText(field)
        if required.contains(name) { line += " (required)" } else if let value = field["default"] { line += " (default \(mcpCompact(value)))" }
        if let description = field["description"]?.string { line += " — " + mcpClause(description) }
        lines.append(line)
        if field["x-arsenal-file"] == nil, field["type"]?.string == "object" || field["properties"] != nil {
            lines += mcpFieldLines(field, indent: indent + "  ", depth: depth + 1)
        }
        if field["type"]?.string == "array", let items = field["items"]?.object, items["properties"] != nil, items["x-arsenal-file"] == nil {
            lines += mcpFieldLines(items, indent: indent + "  ", depth: depth + 1)
        }
    }
    return lines
}
func mcpTypeText(_ field: JSONObject) -> String {
    if let file = field["x-arsenal-file"]?.object {
        let accept = file["accept"]?.array?.compactMap(\.string) ?? []
        return "file {path:\"/absolute/path\"}" + (accept.isEmpty ? "" : " (" + accept.joined(separator: ", ") + ")")
    }
    if let values = field["enum"]?.array {
        let shown = values.prefix(12).map(mcpCompact).joined(separator: "|")
        return values.count > 12 ? shown + "… (\(values.count - 12) more, see schema)" : shown
    }
    if let value = field["const"] { return mcpCompact(value) }
    if let options = (field["oneOf"] ?? field["anyOf"])?.array { return "one of: " + options.compactMap { $0.object.map(mcpTypeText) }.joined(separator: " or ") }
    let type = field["type"]?.string ?? field["type"]?.array?.compactMap(\.string).filter { $0 != "null" }.first ?? "value"
    switch type {
    case "boolean": return "true|false"
    case "integer", "number":
        let word = type == "integer" ? "whole number" : "number"
        switch (field["minimum"]?.number, field["maximum"]?.number) {
        case (let low?, let high?): return "\(word) \(MCPFormat.number(low, decimals: 4))–\(MCPFormat.number(high, decimals: 4))"
        case (let low?, nil): return "\(word) ≥ \(MCPFormat.number(low, decimals: 4))"
        case (nil, let high?): return "\(word) ≤ \(MCPFormat.number(high, decimals: 4))"
        default: return word
        }
    case "string": return field["maxLength"]?.number.map { "text (up to \(Int($0)) characters)" } ?? "text"
    case "array":
        let item = field["items"]?.object.map(mcpTypeText) ?? "values"
        return "list of " + item
    case "object": return field["properties"] == nil ? "object" : "object:"
    default: return type
    }
}
func mcpCompact(_ value: JSONValue) -> String {
    switch value {
    case .string(let text): return text
    case .number(let number): return MCPFormat.number(number, decimals: 6)
    case .bool(let flag): return flag ? "true" : "false"
    case .null: return "null"
    default: return String(decoding: (try? JSONEncoder().encode(value)) ?? Data(), as: UTF8.self)
    }
}
func mcpClause(_ text: String) -> String {
    let flat = text.replacingOccurrences(of: "\n", with: " ")
    let cut = flat.range(of: ". ").map { String(flat[..<$0.lowerBound]) } ?? flat
    let trimmed = cut.trimmingCharacters(in: CharacterSet(charactersIn: ". "))
    return trimmed.count > 140 ? String(trimmed.prefix(140)) + "…" : trimmed
}
/// The smallest input that names every required field, for the `Run:` example line.
func mcpExampleInput(_ schema: JSONObject) -> JSONObject {
    let properties = schema["properties"]?.object ?? [:]
    var input: JSONObject = [:]
    for name in schema["required"]?.array?.compactMap(\.string) ?? [] {
        guard let field = properties[name]?.object else { continue }
        input[name] = mcpExampleValue(field)
    }
    return input
}
func mcpExampleValue(_ field: JSONObject) -> JSONValue {
    if field["x-arsenal-file"] != nil { return ["path": "/path/to/file"] }
    if let first = field["enum"]?.array?.first { return first }
    if let value = field["const"] ?? field["default"] { return value }
    switch field["type"]?.string {
    case "boolean": return false
    case "integer", "number": return .number(field["minimum"]?.number ?? 1)
    case "array": return .array(field["items"]?.object.map { [mcpExampleValue($0)] } ?? [])
    case "object": return .object(mcpExampleInput(field))
    default: return "…"
    }
}
