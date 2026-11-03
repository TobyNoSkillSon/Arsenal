// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

extension APIRouter {
    func activityData(_ snapshot: ActivitySnapshot) throws -> Data {
        let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(snapshot)
    }
    /// `?machine=0`: requests only; the machine is not read for this subscriber.
    func streamActivity(_ connection: HTTPConnection, machine: Bool = true) async throws {
        let stream = await service.activityStream(machine: machine)
        try await connection.beginStream()
        let clock = ContinuousClock()
        var previous: ContinuousClock.Instant?
        for await value in stream {
            try Task.checkCancellation()
            if let previous { try await clock.sleep(until: previous.advanced(by: .milliseconds(100))) }
            previous = clock.now
            try await connection.event(JSONSerialization.jsonObject(with: activityData(value)), name: "activity")
        }
    }
}
