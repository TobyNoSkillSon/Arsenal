// SPDX-License-Identifier: MIT
import Foundation

extension RealModelsService {
    /// A refresh read, not an inference job: never discover with notifications, load, or publish a failure event.
    public func voiceInventory(id: String) async throws -> JSONObject {
        let token = UUID()
        let task = Task { try await self.readVoiceInventory(id: id) }
        runTasks[token] = task
        defer { runTasks[token] = nil }
        return try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }

    private func readVoiceInventory(id: String) async throws -> JSONObject {
        try admitRun(id)
        if localModels[id] != nil { try await discoverLocalModels(notifyChanges: false) }
        let (_, operation) = try await requestOperation(id: id, op: "voices", notifyChanges: false)
        guard residents[id] != nil, !(await supervisor.hasExited(id: id)) else {
            throw WorkerFailure(code: "not_running", message: "Load the model to list its voices.")
        }
        let snapshot = residentSettings[id] ?? modelSettings
        let params = snapshot.residentRequest(id, params: [:], live: modelSettings, schema: operation?.inputSchema)
        if let operation { try OperationSchemaValidator.validate(.object(params), schema: .object(operation.inputSchema)) }
        residents[id]?.busy += 1; pendingRuns[id, default: 0] += 1
        timers[id]?.cancel(); timers[id] = nil
        defer {
            residents[id]?.busy -= 1; pendingRuns[id, default: 0] -= 1
            armTimer(id); signalMemoryChange(); scheduleRevisionSwap(id)
        }
        return try await supervisor.request(
            id: id, op: "voices", params: params, timeout: operation?.policy?.timeoutS ?? 180,
            firstResponseTimeout: operation?.policy?.firstResponseTimeoutS)
    }
}
