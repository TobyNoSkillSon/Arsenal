// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

struct ControlGetProgress: Sendable {
    var inProgress = true
    var error: String?
    var code: String?
    var revisedPlan: GetPlan?
}
extension APIRouter {
    /// Reserve before replying 202: installed weights do not mean dependency preparation has completed.
    func startControlGet(id: String, dtype: String, approved: GetPlan) throws {
        guard controlGets[id]?.inProgress != true else { throw HTTPFailure(409, "Get is already running for \(id).", code: "busy") }
        controlGets[id] = ControlGetProgress()
        Task {
            do {
                try await service.get(id: id, precision: dtype, approvedPlan: approved)
                controlGets[id] = .init(inProgress: false)
            } catch let changed as GetPlanChanged {
                controlGets[id] = .init(inProgress: false, error: changed.localizedDescription, code: "get_plan_changed", revisedPlan: changed.revisedPlan)
            } catch {
                let failure = serviceHTTPFailure(error, modelID: id)
                controlGets[id] = .init(inProgress: false, error: failure.message, code: failure.code)
            }
        }
    }
    func getProgressBody(_ row: ModelRow) throws -> JSONObject {
        var body = try WireJSON.decode(JSONEncoder().encode(row))
        if let state = controlGets[row.id] {
            body["get_in_progress"] = .bool(state.inProgress)
            if let error = state.error { body["get_error"] = .string(error) }
            if let code = state.code { body["get_error_code"] = .string(code) }
            if let plan = state.revisedPlan { body["revised_plan"] = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(plan)) }
        }
        return body
    }
}
