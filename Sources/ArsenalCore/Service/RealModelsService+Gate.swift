// SPDX-License-Identifier: MIT
import Foundation

/// A Get refused by a gated Hugging Face model waits for a person: the model's page (and the token page, without a token)
/// opens once, the row counts down to an automatic retry, and a Get started meanwhile retries at once (`GateRetry`).
/// Nothing here suspends between a Get ending and its row being written, so a Get started meanwhile is never overwritten.
extension RealModelsService {
    /// Tests: the wait's bookkeeping (synchronous, safe from any context).
    func gateRetryForTests() -> GateRetry { gateRetry }
    /// The app's browser opener and, in tests, the retry clock.
    public func setGateHandling(retry: GateRetry? = nil, open: @escaping @Sendable ([URL]) -> Void) {
        if let retry { gateRetry = retry }
        openPages = open
    }
    /// A Get of `precision` met the gate: record the refusal, open the pages on the first one, show the wait.
    func gated(_ id: String, precision: String, gate: HFGateError) {
        let (wait, began) = gateRetry.refused(id, gate: gate) { [weak self] in await self?.automaticRetry(id, precision: precision) }
        if began { openPages(gate.pages) }
        clearGateRows(id, except: precision)
        modelRows[id]?.states[precision] = .init(
            state: loadedPreferences[id]?.precision == precision ? .loaded : .failed, error: gate.localizedDescription, phase: PrecisionState.gatedPhase,
            gate: wait)
        // No error event: the row says what to do and counts down, so no footer line repeats it on every retry.
        changed()
    }
    /// The counter ran out: the same Get a click starts. A refusal before the Get registers (Arsenal refreshing or
    /// preparing an update) counts as a try and waits again; any other one ends the wait with its reason on the row.
    func automaticRetry(_ id: String, precision: String) async {
        do { try await get(id: id, precision: precision) } catch {
            // The Get ran and settled the row itself, or another Get or a cancel owns it now.
            guard gets[id] == nil, let state = modelRows[id]?.states[precision], state.phase == PrecisionState.gatedPhase, let wait = state.gate,
                !gateRetry.hasPendingRetry(id), gateRetry.isWaiting(id), !(error is HFGateError), !(error is CancellationError)
            else { return }
            if let refusal = error as? WorkerFailure, ["busy", "updating"].contains(refusal.code) {
                let gate = HFGateError(repo: wait.repo, hasToken: wait.hasToken, mode: wait.mode, tokenRejected: wait.tokenRejected == true)
                gated(id, precision: precision, gate: gate)
                return
            }
            gateRetry.end(id)
            modelRows[id]?.states[precision] = .init(state: .failed, error: error.localizedDescription)
            changed()
        }
    }
    /// A wait is per model; a precision that waited before another one's Get keeps no stale counter.
    func clearGateRows(_ id: String, except precision: String? = nil) {
        for (p, state) in modelRows[id]?.states ?? [:] where p != precision && state.phase == PrecisionState.gatedPhase {
            resetInstalledState(id, precision: p)
        }
    }
    /// Cancel or Remove: the wait ends, and a waiting row says nothing more (it holds no weights, so it leaves the table).
    func endGateWait(_ id: String) {
        gateRetry.end(id)
        clearGateRows(id)
    }
    /// A token was just saved: every waiting Get retries now, and so does one whose wait gave up.
    public func retryGatedGets() async {
        await pinnedFilesTokenChanged() // a folder whose repository looked missing without a token is listed again
        let pending = gateRetry.takePendingRetries()
        for (_, retry) in pending { Task { await retry() } }
        for (id, row) in modelRows where pending[id] == nil && gets[id] == nil {
            guard let precision = row.precisions.first(where: { row.states[$0]?.phase == PrecisionState.gatedPhase && row.states[$0]?.gate?.gaveUp == true })
            else { continue }
            Task { [weak self] in _ = try? await self?.get(id: id, precision: precision) }
        }
    }
}
