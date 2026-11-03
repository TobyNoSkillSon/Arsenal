// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import MCP

extension APIRouter {
    /// Maps any service, HTTP or worker error onto the closed `[category]` list, with the next step in the same line.
    /// `verb` starts a refusal ("Not run", "Not loaded").
    func mcpFailure(_ error: any Error, entry: CatalogEntry?, verb: String = "Not run") async -> MCPFailure {
        if let failure = error as? MCPFailure { return failure }
        if error is CancellationError { return MCPFailure(.cancelled, "\(verb): the request was cancelled.") }
        if let gate = error as? HFGateError { return MCPFailure(.waitingOnPerson, GatedModelText.line(gate, name: entry?.name ?? gate.repo)) }
        if let missing = error as? HFMissingFileError {
            return MCPFailure(.download, "\(verb): \(missing.localizedDescription) Do not retry; tell the user the model's folder pins a file its repository lacks.")
        }
        if let changed = error as? GetPlanChanged {
            let size = changed.revisedPlan.totalDownloadBytes.map(MCPFormat.size) ?? "an unknown size"
            return MCPFailure(
                .download, "\(verb): the download plan changed while it was being prepared (now \(size)). Ask the user again before retrying with allow_download:true.")
        }
        let http = serviceHTTPFailure(error, modelID: entry?.id)
        let name = entry?.name ?? "the model"
        switch http.code {
        case "insufficient_memory":
            let refusal = LoadMemoryRefusal(details: http.details)
            guard let refusal else { return MCPFailure(.memory, "\(verb): \(http.message)") }
            let unload: String
            if refusal.unloadingIdleFits {
                // Arsenal's own idle models free enough: unloading them is the fix that needs no swap.
                let idle = refusal.idle.map { "\($0.id) (\(MCPFormat.memory(decimalGB: $0.gb)))" }.joined(separator: ", ")
                unload = "Unloading Arsenal's idle \(refusal.idle.count == 1 ? "model" : "models") \(idle) with \(MCPNames.prefix)unload frees enough; then retry"
            } else {
                let candidates = await mcpUnloadCandidates(except: refusal.model)
                unload = candidates.isEmpty ? "Close other apps" : "Unload \(candidates) with \(MCPNames.prefix)unload"
            }
            let inUse = max(0, refusal.limitGB - refusal.freeGB)
            return MCPFailure(
                .memory,
                "\(verb): \(refusal.name) needs ~\(MCPFormat.memory(decimalGB: refusal.neededGB)); \(MCPFormat.memory(decimalGB: refusal.freeGB)) free under the "
                    + "\(MCPFormat.memory(decimalGB: refusal.limitGB)) limit (\(MCPFormat.memory(decimalGB: inUse)) in use). \(unload), or retry with force:true only if the user "
                    + "accepts macOS swap slowing the whole Mac.")
        case "memory":
            let details = http.details
            let footprint = details["footprint_gb"]?.number.map { MCPFormat.memory(decimalGB: $0) }
            if details["reason"]?.string == "runaway" {
                let expected = details["expected_gb"]?.number.map { ", far past its expected \(MCPFormat.memory(decimalGB: $0))" } ?? ""
                return MCPFailure(
                    .memory,
                    "Stopped: \(name) grew to \(footprint ?? "an unexpected size")\(expected), and was unloaded. Nothing else changed. Retry once; if it happens again, tell the user."
                )
            }
            if details["reason"]?.string == "mac_memory_limit" {
                let inUse = details["mac_in_use_gb"]?.number.map { " (\(MCPFormat.memory(decimalGB: $0)) in use)" } ?? ""
                let limit = MCPFormat.memory(decimalGB: details["mac_limit_gb"]?.number ?? mcpEnvironment.macMemory().limitGB)
                return MCPFailure(
                    .memory,
                    "Stopped: the Mac passed the \(limit) limit\(inUse) while \(name) grew to \(footprint ?? "more than expected"). \(name) was unloaded; nothing else changed. Retry when memory is free."
                )
            }
            return MCPFailure(.memory, "\(verb): \(http.message)")
        case "busy", "updating", "shutting_down", "not_running", "config_changed", "catalog_key": return MCPFailure(.busy, "\(verb): \(http.message)")
        case "code_consent":
            return MCPFailure(
                .approval,
                "\(verb): \(name) runs local code nobody has approved yet. Only with the user's yes: \(MCPNames.prefix)settings(model:\"\(entry?.id ?? "ID")\", action:\"approve_code\")."
            )
        case "unsupported_format", "unsupported", "unsupported_op", "precision", "runner", "kind_order", "mode": return MCPFailure(.unsupported, "\(verb): \(http.message)")
        case "quarantined", "worker_exit", "protocol", "timeout", "hello_timeout", "invalid_output", "revision", "integrity", "execution":
            return MCPFailure(.failed, "\(verb): \(http.message)")
        case "not_installed", "update_available", "download_size", "missing_weights":
            return MCPFailure(.download, "\(verb): \(http.message)")
        case "not_found" where entry == nil: return MCPFailure(.unknownModel, http.message)
        case "invalid_request", "settings", "setting", "op_schema", "invalid_input": return MCPFailure(.invalidInput, "\(verb): \(http.message)")
        default: break
        }
        switch http.status {
        case 429: return MCPFailure(.limit, "\(verb): \(http.message)")
        case 507: return MCPFailure(.disk, "\(verb): \(http.message)")
        case 404: return MCPFailure(.notFound, "\(verb): \(http.message)")
        case 409: return MCPFailure(.busy, "\(verb): \(http.message)")
        case 400, 413, 415, 422: return MCPFailure(.invalidInput, "\(verb): \(http.message)")
        default: return MCPFailure(.failed, "\(verb): \(http.message)")
        }
    }
}
