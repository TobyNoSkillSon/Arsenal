// SPDX-License-Identifier: MIT
import Foundation

/// Revoking a local model's code approval (the settings card's Revoke, MCP `revoke_code`). Approval itself stays beside
/// discovery in RealModelsService: the owner's yes to a digest is enough, also after the runner changed.
extension RealModelsService {
    /// Forgets the owner's approval of a local model's code: the next Get or Load asks again, showing the contents' digest. A
    /// loaded model is unloaded first (refused while a request, Get or Load runs), so no unapproved code keeps running, and the
    /// revisions kept for rollback go too: a rollback must not bring back code without an approval.
    public func revokeLocalCode(id: String) async throws {
        try admitWork(); try await discoverLocalModels()
        guard let local = localModels[id] else { throw WorkerFailure(code: "not_found", message: "\(id) is not a local model; only local models have a code approval.") }
        guard config.localCodeConsent[id] != nil || config.localCodeApprovals[id] != nil else { return }
        guard gets[id] == nil, loadTasks[id] == nil else { throw WorkerFailure(code: "busy", message: "Finish Get or Load before revoking the approval.") }
        try requireIdleResident(id)
        if let swap = swapTasks[id] { swap.cancel(); await swap.value }
        if residents[id] != nil { try await unload(id: id) }
        previousRevisions[id] = nil; dormantRevisions[id] = nil
        let old = config.localCodeConsent[id], oldApproval = config.localCodeApprovals[id]
        config.localCodeConsent[id] = nil; config.localCodeApprovals[id] = nil
        do { try persist() } catch { config.localCodeConsent[id] = old; config.localCodeApprovals[id] = oldApproval; throw error }
        modelRows[id]?.approval = nil
        modelRows[id]?.needsCodeConsent = (try? Self.localCodeRequiresConsent(local)) ?? true
        updatePending(id); updateRevisionRow(id); changed()
    }
}
