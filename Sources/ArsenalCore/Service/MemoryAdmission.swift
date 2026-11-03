// SPDX-License-Identifier: MIT
import Foundation

extension RealModelsService {
    public func memorySettings() async -> MemorySettings { config.memory }
    public func setMemorySettings(_ settings: MemorySettings) async throws {
        config.memory = settings; try persist(); signalMemoryChange(); changed()
    }
    func reserveMemory(id: String, precision: String, estimateMB: Double, waitForMemory: Bool = true) async throws {
        let ticket = UUID()
        var creditedAvailable: Double?
        defer {
            if memoryAdmissionOrder.contains(ticket) {
                memoryAdmissionOrder.removeAll { $0 == ticket }; signalMemoryChange()
            }
        }
        while true {
            try admitWork(); if swapTasks[id] == nil { try admitNewWork() }
            let snapshot = memory()
            var candidates = Array(residents.values)
            for index in candidates.indices { candidates[index].busy += pendingRuns[candidates[index].id] ?? 0 }
            let reservations = loadReservations.values.reduce(0, +)
            let available = creditedAvailable ?? (probe.rawAvailableMB(loadedMB: snapshot.workersMB) - reservations)
            creditedAvailable = nil
            let need = estimateMB + MemoryProbe.headroomMB
            let reclaimable = candidates.filter { $0.id != id && $0.residency == .onDemand && !$0.keepHot }.reduce(0) { $0 + $1.memoryMB }
            let possible = need <= probe.totalMB && (config.residency.allowSwap || available + reservations + reclaimable >= need)
            guard possible else { throw insufficientMemory(id: id, need: need, available: available, waitForMemory: false) }
            let mayWait = waitForMemory && config.memory.whenFull == .queue
            if mayWait && !memoryAdmissionOrder.isEmpty && !memoryAdmissionOrder.contains(ticket) {
                memoryAdmissionOrder.append(ticket)
            }
            if mayWait, let first = memoryAdmissionOrder.first, first != ticket {
                markMemoryQueued(id: id, precision: precision)
                try await waitForMemoryChange(periodic: true); continue
            }
            let decision = ResidencyPolicy.admission(
                id: id, estimateMB: estimateMB, availableMB: available, residents: candidates, allowSwap: config.residency.allowSwap)
            let victims: [String]
            switch decision {
            case .admit(let selected):
                if selected.isEmpty { loadReservations[id] = need; return }
                victims = selected
            case .refuse:
                guard mayWait else { throw insufficientMemory(id: id, need: need, available: available, waitForMemory: waitForMemory) }
                // Only waiting requests may reclaim part of the need before a busy model becomes idle.
                victims = ResidencyPolicy.evictionOrder(candidates).filter {
                    $0.id != id && $0.residency == .onDemand && !$0.keepHot && $0.busy == 0
                }.prefix(1).map(\.id)
                if victims.isEmpty {
                    guard waitForMemory && config.memory.whenFull == .queue else {
                        throw insufficientMemory(id: id, need: need, available: available, waitForMemory: waitForMemory)
                    }
                    if !memoryAdmissionOrder.contains(ticket) { memoryAdmissionOrder.append(ticket) }
                    markMemoryQueued(id: id, precision: precision)
                    if swapTasks[id] == nil { try admitNewWork() } // Keep admission and waiter registration adjacent after any future queue hook.
                    try await waitForMemoryChange(periodic: true); continue
                }
            }
            // Credit known released footprints; VM counters can lag behind the worker's exit.
            let workersBefore = snapshot.workersMB
            var released = 0.0
            for victim in victims {
                guard let resident = residents[victim], resident.residency == .onDemand, !resident.keepHot,
                    resident.busy == 0, pendingRuns[victim, default: 0] == 0
                else { continue }
                released += resident.memoryMB
                try await unloadResident(id: victim)
            }
            let workersAfter = memory().workersMB
            let otherGrowth = max(0, workersAfter - (workersBefore - released))
            creditedAvailable = available + released - otherGrowth - (loadReservations.values.reduce(0, +) - reservations)
        }
    }
    func markMemoryQueued(id: String, precision: String) {
        if let tracking = ActivityRequestContext.tracking { activityAggregator.phase(tracking.id, "queued") }
        guard modelRows[id]?.states[precision]?.phase != "queued" else { return }
        modelRows[id]?.states[precision] = .init(state: .loading, phase: "queued"); changed()
        signalMemoryChange() // Immediate shared-load callers must notice when an active load starts waiting.
    }
    func insufficientMemory(id: String, need: Double, available: Double, waitForMemory: Bool) -> WorkerFailure {
        func amount(_ megabytes: Double) -> String {
            if megabytes >= 2000 { return MemoryDisplay.text(decimalGB: megabytes / 1000) }
            return "\(Int(max(0, megabytes))) MB"
        }
        let loaded = residents.keys.sorted().map { id in
            let resident = residents[id]
            let state = (resident?.busy ?? 0) + (pendingRuns[id] ?? 0) > 0 ? "generating" : "loaded"
            return "\(modelRows[id]?.name ?? id) (\(amount(resident?.memoryMB ?? 0))) is \(state)"
        }.joined(separator: ", ")
        let context = loaded.isEmpty ? "with no models loaded" : "while \(loaded)"
        let action = waitForMemory ? "choose Memory ▸ When a model doesn't fit ▸ Wait in queue, or unload a model" : "unload a model or choose a smaller dtype"
        return WorkerFailure(
            code: "insufficient_memory",
            message: "Not enough memory for \(modelRows[id]?.name ?? id) (needs \(amount(need)), \(amount(available)) free \(context)); \(action).")
    }
    func waitForMemoryChange(periodic: Bool = false) async throws {
        let token = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (reply: CheckedContinuation<Void, any Error>) in
                if Task.isCancelled {
                    reply.resume(throwing: CancellationError())
                } else {
                    if periodic {
                        do { try admitNewWork() } catch { reply.resume(throwing: error); return }
                    }
                    memoryWaiters[token] = reply
                    if periodic { memoryPeriodicWaiters.insert(token); startMemoryRecheck() }
                }
            }
        } onCancel: {
            Task { await self.cancelMemoryWaiter(token) }
        }
        try Task.checkCancellation()
    }
    private func startMemoryRecheck() {
        guard memoryRecheckTask == nil else { return }
        memoryRecheckTask = Task { [weak self, memoryRecheckSeconds] in
            do {
                try await Task.sleep(for: .seconds(memoryRecheckSeconds))
                try Task.checkCancellation(); await self?.signalMemoryChange()
            } catch {}
        }
    }
    private func cancelMemoryWaiter(_ token: UUID) {
        memoryWaiters.removeValue(forKey: token)?.resume(throwing: CancellationError())
        memoryPeriodicWaiters.remove(token)
        if memoryPeriodicWaiters.isEmpty { memoryRecheckTask?.cancel(); memoryRecheckTask = nil }
    }
    func signalMemoryChange() {
        let waiters = memoryWaiters.values; memoryWaiters.removeAll(); memoryPeriodicWaiters.removeAll()
        memoryRecheckTask?.cancel(); memoryRecheckTask = nil
        for waiter in waiters { waiter.resume() }
    }
    public func beginUpdate() async throws -> [String] {
        try admitWork(); try admitNewWork(); updating = true; signalMemoryChange()
        // Queued loads wake and refuse admission; current generations complete without cancellation.
        var wroteSnapshot = false
        do {
            while !runTasks.isEmpty || !loadTasks.isEmpty || !gets.isEmpty || !swapTasks.isEmpty {
                try requireUpdateGate()
                for task in Array(runTasks.values) { _ = try? await task.value }
                for task in Array(loadTasks.values) + Array(gets.values) { _ = try? await task.value }
                for task in Array(swapTasks.values) { await task.value }
                try Task.checkCancellation(); await Task.yield()
            }
            try Task.checkCancellation()
            try requireUpdateGate()
            try saveUpdateResidency(); wroteSnapshot = true
            try Task.checkCancellation()
            return residents.keys.sorted()
        } catch {
            updating = false; signalMemoryChange()
            if wroteSnapshot { try? clearUpdateResidency() }
            throw error
        }
    }
    private func requireUpdateGate() throws {
        guard updating else { throw WorkerFailure(code: "update_cancelled", message: "The update was cancelled.") }
    }
    public func cancelUpdate() async {
        updating = false; signalMemoryChange()
        do { try clearUpdateResidency() } catch { emitUpdateRestoreError(id: nil, error: error) }
    }
}
