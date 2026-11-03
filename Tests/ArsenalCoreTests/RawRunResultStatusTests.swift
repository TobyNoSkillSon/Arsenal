// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// The decisions routes answer with the worker's own error status. A runner that gives none must not turn a request
/// refused for the runner's state (queue full, not loaded, cancelled) into a 500.
struct RawRunResultStatusTests {
    func result(_ code: String, status: JSONValue? = nil) -> RawRunResult {
        var details: JSONObject = ["code": .string(code), "message": "m"]
        if let status { details["status"] = status }
        return RawRunResult(rawFinalFrame: Data(), body: Data(), ok: false, error: WorkerFailure(code: code, message: "m", details: details))
    }

    @Test func requestStateRefusalsAreConflictsOrRetriesWithoutAWorkerStatus() {
        #expect(result("busy").status == 409)
        #expect(result("not_loaded").status == 409)
        #expect(result("already_loaded").status == 409)
        #expect(result("cancelled").status == 400) // workerHTTPStatus maps it so on every other route
        #expect(result("runner_crashed").status == 500)
        #expect(RawRunResult(rawFinalFrame: Data(), body: Data(), ok: true).status == 200)
    }

    @Test func theWorkersOwnStatusWins() {
        #expect(result("cancelled", status: 409).status == 409)
        #expect(result("busy", status: 503).status == 503)
        #expect(result("busy", status: 42).status == 409) // out of range: ignored
    }
}
