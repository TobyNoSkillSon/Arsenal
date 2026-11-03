// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire

extension RealModelsService {
    public func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject {
        try admitNewWork(); try admitWork(allowRefresh: residents[id] != nil || reloading.contains(id))
        let operationID = UUID()
        let aggregator = activityAggregator
        // A live session (a *_session operation's create, pushes, then finish or close) is one Activity entry: its pushes,
        // about five a second while someone speaks, run inside the entry its create opened instead of each adding one.
        let action = op.hasSuffix("_session") ? params["action"]?.string : nil
        let sessionKey = action == nil ? nil : params["session_id"]?.string.map { Self.sessionActivityKey(id, op, $0) }
        expireSessionActivities()
        let continuing = sessionKey.flatMap { sessionActivities[$0]?.activity }
        let activityID: UUID
        if let continuing {
            activityID = continuing; sessionActivities[sessionKey!]?.seen = Date()
        } else {
            let title = try await resolvedOperations(id: id).first { $0.name == op }?.title
            activityID = aggregator.begin(model: id, op: op, requester: ActivityRequestContext.requester, title: title)
        }
        // The entry stays open after a create that opened a session and after a push into an open entry.
        let keepsOpen: @Sendable (JSONObject) -> Bool = { result in
            action == "create" && result["session_id"]?.string != nil || action == "push" && continuing != nil
        }
        let tracking = ActivityRunTracking(id: activityID)
        let progress: ProgressTransform = { event in
            let decorated = tracking.decorate(event)
            aggregator.receive(activityID, .json(decorated)); return decorated
        }
        // A stream announces its audio's sample rate before the audio: Activity counts seconds made with it.
        var observed: RunSink?
        if let sink {
            observed = { @Sendable (frame: WireFrame) async throws in
                if case .json(let event) = frame, event["event"]?.string == "audio_begin" { aggregator.receive(activityID, frame) }
                try await sink(frame)
            }
        }
        let task = Task {
            do {
                let result = try await ActivityRequestContext.$tracking.withValue(tracking) {
                    try await self.performRun(id: id, op: op, params: params, progress: progress, sink: observed)
                }
                if !keepsOpen(result) { aggregator.finish(activityID, .finished(result)) }
                return tracking.decorate(result)
            } catch {
                aggregator.finish(activityID, error is CancellationError || Task.isCancelled ? .cancelled : .failed); throw error
            }
        }; runTasks[operationID] = task
        defer { runTasks[operationID] = nil }
        do {
            let result = try await withTaskCancellationHandler {
                try await task.value
            } onCancel: {
                task.cancel()
            }
            if action == "create", continuing == nil, let session = result["session_id"]?.string {
                sessionActivities[Self.sessionActivityKey(id, op, session)] = (activityID, Date())
            } else if let sessionKey, !keepsOpen(result) {
                sessionActivities[sessionKey] = nil
            }
            return result
        } catch {
            if let sessionKey { sessionActivities[sessionKey] = nil }
            throw error
        }
    }
    static func sessionActivityKey(_ model: String, _ op: String, _ session: String) -> String { model + "\n" + op + "\n" + session }
    /// A session whose client vanished without finish or close (a live stream pushes every fraction of a second) ends
    /// its entry after two idle minutes, so Activity never shows it running forever.
    private func expireSessionActivities(now: Date = Date()) {
        for (key, entry) in sessionActivities where now.timeIntervalSince(entry.seen) > 120 {
            sessionActivities[key] = nil; activityAggregator.finish(entry.activity, .cancelled)
        }
    }
    public func activity() async -> ActivitySnapshot {
        // Activity reads the service's event-fed state; it does not rescan folders four times a second.
        await activityAggregator.snapshot(rows: modelRows.values.sorted { $0.id < $1.id }, memory: memory())
    }
    public func activityStream(machine: Bool) async -> AsyncStream<ActivitySnapshot> {
        await activityAggregator.stream(machine: machine, snapshot: { [weak self] in await self?.activity() ?? .init() })
    }
}
