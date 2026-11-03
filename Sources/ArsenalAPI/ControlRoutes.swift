// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import ArsenalBench

extension APIRouter {
    func control(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        do { try await performControl(request, connection) } catch let changed as GetPlanChanged {
            try await connection.respond(
                [
                    "error": [
                        "code": "get_plan_changed", "message": changed.localizedDescription,
                        "revised_plan": try JSONSerialization.jsonObject(with: JSONEncoder().encode(changed.revisedPlan))
                    ]
                ], status: 409)
        } catch {
            let id = (try? WireJSON.decode(request.body))?["id"]?.string
            throw serviceHTTPFailure(error, modelID: id)
        }
    }
    private func approvedGetPlan(_ body: JSONObject, id: String, dtype: String) async throws -> GetPlan {
        let current = try await service.getPlan(id: id, dtype: dtype)
        try current.requireKnownDownloadSize()
        let approved: GetPlan
        if let value = body["approved_plan"] {
            approved = try JSONDecoder().decode(GetPlan.self, from: JSONEncoder().encode(value))
        } else {
            // Old clients may ask for small weights, never tools or a large transfer.
            guard current.items.allSatisfy({ $0.kind == "weights" || $0.downloadBytes == 0 }),
                let bytes = current.totalDownloadBytes, bytes < 2_000_000_000
            else { throw GetPlanChanged(revisedPlan: current) }
            approved = current
        }
        guard approved.covers(current) else { throw GetPlanChanged(revisedPlan: current) }
        return approved
    }
    private func approveHTTP(_ body: JSONObject, id: String) async throws {
        let digest = try required(body, "content_digest")
        let source: ApprovalSource
        if let value = body["approved_by"] {
            guard let raw = value.string, ["cli", "api"].contains(raw), let parsed = ApprovalSource(rawValue: raw) else {
                throw HTTPFailure(422, "approved_by must be cli or api.")
            }
            source = parsed
        } else {
            source = .api
        }
        try await service.approveLocalCode(id: id, contentDigest: digest, approvedBy: source)
    }
    private func performControl(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        let path = String(request.path.dropFirst("/arsenal/v1/".count))
        if path.hasPrefix("models/") {
            let parts = path.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
            guard parts.count >= 3, ModelManifest.validID(parts[1]) else { throw HTTPFailure(404, "Model endpoint not found.") }
            let id = parts[1]
            if parts.count == 4, parts[2] == "ops" {
                try await universal(request, connection, id: id, op: parts[3]); return
            }
            guard parts.count == 3 else { throw HTTPFailure(404, "Model endpoint not found.") }
            if parts[2] == "get-plan", request.method == "GET" {
                guard let row = await service.catalog(query: .init()).first(where: { $0.id == id })?.row else { throw HTTPFailure(404, "Unknown model \(id).") }
                let plan = try await service.getPlan(id: id, dtype: request.query["dtype"] ?? row.selectedPrecision)
                try await connection.respondBytes(try JSONEncoder().encode(plan), type: "application/json"); return
            }
            if parts[2] == "reference", request.method == "GET" {
                let reference = try await service.reference(id: id, dtype: request.query["dtype"], mode: request.query["mode"])
                try await connection.respondBytes(try JSONEncoder().encode(reference), type: "application/json"); return
            }
            if parts[2] == "settings", request.method == "GET" {
                try await connection.respondBytes(try JSONEncoder().encode(try await service.settings(id: id)), type: "application/json"); return
            }
            if parts[2] == "runner", request.method == "GET" {
                guard let entry = await service.catalog(query: .init()).first(where: { $0.id == id }) else { throw HTTPFailure(404, "Unknown model \(id).") }
                try await connection.respond([
                    "runner": entry.row.runner, "override": entry.row.runnerOverride as Any? ?? NSNull(), "default": entry.row.defaultRunner as Any? ?? NSNull(),
                    "resident": entry.row.residentRunner as Any? ?? NSNull(), "needsReload": entry.row.needsReload,
                    "runners": try JSONSerialization.jsonObject(with: JSONEncoder().encode(entry.runners))
                ]); return
            }
            guard request.method == "POST" else { throw HTTPFailure(405, "Use GET or POST for model settings.") }
            let body = try object(request)
            switch parts[2] {
            case "tools":
                guard let on = body["on"]?.bool else { throw HTTPFailure(422, "on must be a boolean.") }
                try await service.setToolsShown(id: id, on: on)
            case "runner":
                guard let value = body["runner"], value == .null || value.string != nil else { throw HTTPFailure(422, "runner must be a name or null for the accepted default.") }
                try await service.setRunner(id: id, runner: value.string)
            case "rollback": try await service.rollbackRevision(id: id)
            case "delete-withdrawn":
                // The precisions on this Mac that the model no longer offers (`ModelRow.withdrawnTiers`), as the table's Remove.
                guard let before = await anyRow(id)?.withdrawnTiers else { throw HTTPFailure(404, "Unknown model \(id).") }
                try await service.deleteWithdrawnTiers(id: id)
                let after = await anyRow(id)?.withdrawnTiers ?? [:]
                try await connection.respond(["ok": true, "removed": before.keys.filter { after[$0] == nil }.sorted()]); return
            case "add": try await service.add(id: id)
            case "approve":
                try await approveHTTP(body, id: id)
            case "remove":
                if let value = body["delete_weights"], value.bool == nil { throw HTTPFailure(422, "delete_weights must be a boolean.") }
                try await service.remove(id: id, deleteWeights: body["delete_weights"]?.bool ?? false)
            case "settings":
                try await setModelSettings(id: id, body: body)
                if body["apply"]?.bool == true { try await service.applySettings(id: id) }
                try await connection.respondBytes(try JSONEncoder().encode(try await service.settings(id: id)), type: "application/json"); return
            default: throw HTTPFailure(404, "Model endpoint not found.")
            }
            try await connection.respond(["ok": true]); return
        }
        if request.method == "GET" {
            switch path {
            case "memory/settings": try await connection.respondBytes(try JSONEncoder().encode(await service.memorySettings()), type: "application/json")
            case "activity": try await connection.respondBytes(try activityData(await service.activity()), type: "application/json")
            case "activity/stream": try await streamActivity(connection, machine: !["0", "false"].contains(request.query["machine"] ?? ""))
            case "openapi.json": try await connection.respond(plain(try await openAPI()))
            case "models":
                let models = await rows().filter { request.query["kind"] == nil || $0["kind"] as? String == request.query["kind"] }.map { row in
                    var row = row
                    if row["needsReload"] == nil { row["needsReload"] = false }
                    if row["codeConsentRequired"] == nil { row["codeConsentRequired"] = false }
                    if row["contentDigest"] == nil { row["contentDigest"] = NSNull() }
                    if row["verification"] == nil { row["verification"] = "Not tried on this Mac yet" }
                    if row["downloadBytes"] == nil { row["downloadBytes"] = [row["selectedPrecision"] as? String ?? "bf16": row["sizeBytes"] ?? NSNull()] }
                    return row
                }
                try await connection.respond(models)
            case "status":
                let models = await service.rows(), memory = await service.memory()
                let memoryObject = try JSONSerialization.jsonObject(with: JSONEncoder().encode(memory))
                let loaded = models.filter { $0.residentDType != nil }.map { ["id": $0.id, "kind": $0.kind, "dtype": $0.residentDType ?? $0.selectedPrecision] }
                try await connection.respond([
                    "version": Self.releaseVersion(), "build": "development", "chip": HostChip.current, "os": ProcessInfo.processInfo.operatingSystemVersionString, "port": port,
                    "pid": ProcessInfo.processInfo.processIdentifier, "api": 1,
                    "base_urls": [
                        "openai": origin + "/v1", "anthropic": origin + "/providers/anthropic", "elevenlabs": origin + "/providers/elevenlabs",
                        "replicate": origin + "/providers/replicate", "typesafe": origin + "/providers/typesafe"
                    ], "counts": ["models": models.count, "loaded": loaded.count], "memory": memoryObject, "loaded_models": loaded
                ])
            case let path where path.hasPrefix("bench/"):
                guard let controller = benchmarkController else { throw HTTPFailure(503, "Benchmark controller is not configured.") }
                guard let job = await controller.status(String(path.dropFirst(6))) else { throw HTTPFailure(404, "Benchmark job not found.") }
                try await connection.respondBytes(try JSONEncoder().encode(job), type: "application/json")
            case "catalog", "catalog/browse":
                try await connection.respondBytes(try JSONEncoder().encode(await service.catalog(query: catalogQuery(request.query))), type: "application/json")
            case "catalog/status": try await connection.respondBytes(try JSONEncoder().encode(await service.catalogStatus()), type: "application/json")
            case let path where path.hasPrefix("get/"):
                let id = String(path.dropFirst(4))
                guard let row = await anyRow(id) else { throw HTTPFailure(404, "Unknown model \(id).") }
                try await connection.respondBytes(try WireJSON.encode(getProgressBody(row)), type: "application/json")
            default: throw HTTPFailure(404, "Control endpoint not found.")
            }
            return
        }
        guard request.method == "POST" else { throw HTTPFailure(405, "Use POST for control actions.") }
        if path == "memory/settings" {
            let body = try object(request)
            guard let raw = body["when_full"]?.string, let policy = MemoryWhenFull(rawValue: raw) else {
                throw HTTPFailure(422, "when_full must be refuse or queue.")
            }
            try await service.setMemorySettings(.init(whenFull: policy))
            try await connection.respondBytes(try JSONEncoder().encode(await service.memorySettings()), type: "application/json"); return
        }
        if path == "refresh" { try await refreshControl(connection); return }
        guard ["get", "cancel-get", "load", "unload", "keep-hot", "delete", "bench"].contains(path) else {
            throw HTTPFailure(404, "Control endpoint not found.")
        }
        let body = try object(request), id = try required(body, "id")
        switch path {
        case "get":
            guard let row = await anyRow(id) else { throw HTTPFailure(404, "Unknown model \(id).") }
            if row.codeConsentRequired { throw HTTPFailure(409, "This local model runs unreviewed code; inspect it and approve its current content_digest before Get.") }
            let precision = body["dtype"]?.string ?? body["precision"]?.string ?? row.selectedPrecision
            guard row.precisions.contains(precision) else { throw HTTPFailure(400, "That dtype is not offered by \(id).") }
            if let reason = row.dtypeUnavailableReason(precision) { throw HTTPFailure(422, reason, code: "unsupported_format") }
            // A gated model waiting for the user: the stored wait answers, so polling neither spends tries nor reopens pages.
            // Once Arsenal gave up, a Get is a fresh try, as a click on Get is (the token or access may have changed since).
            if let wait = row.states[precision]?.gate, !wait.gaveUp {
                throw HTTPFailure(403, GatedModelText.line(wait, name: row.name), code: "waiting_on_person", details: GatedModelText.details(wait))
            }
            let approved = try await approvedGetPlan(body, id: id, dtype: precision)
            // Get deliberately outlives this HTTP connection; the table and GET /get/{id} carry its progress.
            try startControlGet(id: id, dtype: precision, approved: approved)
            try await connection.respond(["id": id, "dtype": precision, "started": true], status: 202)
        case "cancel-get": await service.cancelGet(id: id); try await connection.respond(["ok": true])
        case "load":
            guard let row = await anyRow(id) else { throw HTTPFailure(404, "Unknown model \(id).") }
            if row.codeConsentRequired { throw HTTPFailure(409, "This local model runs unreviewed code; inspect it and approve its current content_digest before Load.") }
            if let precision = (body["dtype"]?.string ?? body["precision"]?.string), !row.precisions.contains(precision) {
                throw HTTPFailure(400, "That dtype is not offered by \(id).")
            }
            let dtype = body["dtype"]?.string ?? body["precision"]?.string ?? row.selectedPrecision
            if let reason = row.dtypeUnavailableReason(dtype) { throw HTTPFailure(422, reason, code: "unsupported_format") }
            try await service.load(id: id, precision: body["dtype"]?.string ?? body["precision"]?.string, fast: body["fast"]?.bool); try await connection.respond(["ok": true])
        case "unload": try await service.unload(id: id); try await connection.respond(["ok": true])
        case "keep-hot":
            guard let on = body["on"]?.bool else { throw HTTPFailure(422, "on must be a boolean.") }
            try await service.setKeepHot(id: id, on: on); try await connection.respond(["ok": true])
        case "delete": try await service.delete(id: id); try await connection.respond(["ok": true])
        case "bench":
            guard let controller = benchmarkController else { throw HTTPFailure(503, "Benchmark controller is not configured.") }
            guard let request = try? JSONDecoder().decode(BenchmarkRequest.self, from: request.body) else { throw HTTPFailure(422, "Invalid benchmark request.") }
            do { let job = try await controller.start(request); try await connection.respondBytes(try JSONEncoder().encode(job), type: "application/json", status: 202) } catch {
                throw HTTPFailure(422, error.localizedDescription)
            }
        default: throw HTTPFailure(404, "Control endpoint not found.")
        }
    }
    private func refreshControl(_ connection: HTTPConnection) async throws {
        try await service.refresh()
        let entries = await service.catalog(query: .init()), rows = await service.rows()
        try await connection.respond([
            "ok": true, "counts": ["catalog": entries.count, "models": rows.count, "loaded": rows.filter { $0.residentDType != nil }.count],
            "catalog": try JSONSerialization.jsonObject(with: JSONEncoder().encode(await service.catalogStatus()))
        ])
    }

}

extension APIRouter {
    /// `POST /arsenal/v1/models/<id>/settings`: the model's own settings, and dtype, keep_hot, show_in_menu and runner as
    /// MCP's arsenal_settings takes them, each checked before anything is applied.
    func setModelSettings(id: String, body: JSONObject) async throws {
        guard let values = body["values"]?.object else { throw HTTPFailure(422, "values must be an object of setting IDs and values.") }
        if let apply = body["apply"], apply.bool == nil { throw HTTPFailure(422, "apply must be a boolean.") }
        var declared = values, reserved: [(String, JSONValue)] = []
        if ModelSettingKeys.reserved.contains(where: { values[$0] != nil }) {
            guard let entry = await service.catalog(query: .init()).first(where: { $0.id == id }) else { throw HTTPFailure(404, "Unknown model \(id).") }
            do {
                let split = try ModelSettingKeys.split(values, entry: entry)
                declared = split.declared; reserved = split.reserved
            } catch {
                throw HTTPFailure(422, "Not changed: \(error.text)", code: error.unsupported ? "unsupported" : "invalid_request")
            }
        }
        var applied: [String] = []
        if !declared.isEmpty {
            try await service.setSettings(id: id, values: declared)
            applied = declared.keys.sorted().map { $0 == "__tools" ? "show_in_menu" : $0 }
        }
        for (key, value) in reserved {
            do { try await ModelSettingKeys.apply(key, value, id: id, service: service) } catch {
                // A key refused by the model's state now (a Get running): the answer says what did apply, as MCP's does.
                let failure = serviceHTTPFailure(error, modelID: id)
                guard !applied.isEmpty else { throw failure }
                throw HTTPFailure(
                    failure.status, "Applied \(applied.joined(separator: ", ")); \(key) was not: \(failure.message)", code: failure.code,
                    details: failure.details.merging(["applied": .array(applied.map(JSONValue.string))]) { _, new in new })
            }
            applied.append(key)
        }
    }
}
