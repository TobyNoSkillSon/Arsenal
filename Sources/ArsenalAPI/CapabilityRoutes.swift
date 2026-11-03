// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

extension APIRouter {
    func capabilityModel(_ body: JSONObject, operation: String) async throws -> String {
        if let supplied = body["model"], supplied != .null {
            let id = try required(body, "model")
            _ = try await resolve(id, kind: nil, operation: operation)
            guard await service.operations(id: id).contains(where: { $0.name == operation }) else {
                throw HTTPFailure(422, "Model \(id) does not declare \(operation); inspect \(MCPNames.prefix)model(model: ID).")
            }
            return id
        }
        var ids: [String] = []
        for row in await installedModels() {
            if let id = row["id"] as? String, await service.operations(id: id).contains(where: { $0.name == operation }) { ids.append(id) }
        }
        guard ids.count == 1 else { throw HTTPFailure(422, "Supply model: exactly one installed model must declare \(operation) when model is omitted.") }
        return ids[0]
    }
    func capabilityAlias(_ request: HTTPRequest, _ connection: HTTPConnection, operation: String) async throws {
        let input = try object(request)
        let id = try await capabilityModel(input, operation: operation)
        // Forward the body's own bytes without `model`, so ordered members keep their key order.
        let forwarded = HTTPRequest(method: "POST", target: request.target, headers: request.headers, body: try RawJSON(request.body).replacing(removing: ["model"]))
        try await universal(forwarded, connection, id: id, op: operation)
    }
    func capabilityRun(_ body: JSONObject, operation: String, params: JSONObject) async throws -> JSONObject {
        let id = try await capabilityModel(body, operation: operation)
        guard let declaration = await service.operations(id: id).first(where: { $0.name == operation }) else { throw HTTPFailure(422, "Operation is unavailable.") }
        try OperationSchemaValidator.validate(.object(try OrderedJSONMembers.split(params, schema: declaration.inputSchema).request), schema: .object(declaration.inputSchema))
        return try await run(id, op: operation, params: params)
    }
}
