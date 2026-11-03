// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

struct DecisionReply: Sendable { let status: Int; let body: Data; let bits: String? }

extension APIRouter {
    func decisions(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        let path = request.path.replacingOccurrences(of: "/providers/typesafe", with: "")
        if request.method == "GET", path == "/v1/models" {
            var models = await rows().filter { $0["kind"] as? String == "decisions" }.map {
                ["name": $0["id"] ?? "", "description": $0["summary"] ?? "", "release_date": "2026-10-02"]
            }
            let installedIDs = Set(await installedModels().compactMap { $0["id"] as? String })
            if installedIDs.isSuperset(of: ["laya-english", "laya-multilingual"]) {
                models.insert(["name": "auto", "description": "Route English or multilingual decisions", "release_date": "2026-10-02"], at: 0)
            }
            try await connection.respond(["models": models]); return
        }
        guard ["/v1/systemone", "/v1/judge"].contains(path) else { throw HTTPFailure(404, "TypeSafe endpoint not supported.") }
        guard request.method == "POST" else { throw HTTPFailure(405, "Use POST.") }
        guard request.headers["content-type"]?.lowercased().hasPrefix("application/json") == true else { throw HTTPFailure(415, "Send application/json.") }
        let parsed = try? RawJSON(request.body), requestedModel = parsed?["model"]?.string
        let model: String
        if let requestedModel, requestedModel != "auto" {
            model = requestedModel
        } else {
            let installed = await installedModels()
            if installed.contains(where: { $0["id"] as? String == "laya-english" }) { model = "laya-english" } else { model = try await defaultModel(kind: "decisions") }
        }
        let reply = try await decisionCall(model: model, op: path.hasSuffix("judge") ? "judge" : "decide", body: request.body)
        try await connection.respondBytes(reply.body, type: "application/json", status: reply.status, headers: reply.bits.map { ["x-verdict-bits": $0] } ?? [:])
    }
    func decisionCall(model: String, op: String, body: Data, depth: Int = 0) async throws -> DecisionReply {
        guard depth <= 4 else { throw HTTPFailure(500, "Decision model routing did not settle.") }
        _ = try await resolve(model, kind: "decisions")
        guard let text = String(data: body, encoding: .utf8) else { throw HTTPFailure(422, "Body must be UTF-8 JSON.") }
        // Preserve question/option/state order and NFC/NFD-distinct labels in both directions.
        let result = try await service.runRawResult(id: model, op: op, params: ["body": .string(text)])
        if result.ok {
            let raw = try RawJSON(result.body)
            // Runners that report `bits` need no row lookup (each rows() call refreshes folders and install state).
            var bits = raw["bits"]?.int.map(String.init)
            if bits == nil {
                let row = await service.rows().first { $0.id == model }
                bits = row.map { $0.selectedPrecision == "int4" ? "4" : $0.selectedPrecision == "int8" ? "8" : "16" }
            }
            return DecisionReply(status: 200, body: try raw.replacing(removing: ["bits"]), bits: bits)
        }
        if result.error?.code == "wrong_model" {
            if op == "decide", let needed = result.error?.details["model"]?.string {
                return try await decisionCall(model: needed, op: op, body: body, depth: depth + 1)
            }
            if op == "judge", let routes = result.error?.details["routes"]?.object {
                let original = try RawJSON(body)
                guard let items = try original["items"]?.elements(), !items.isEmpty else { throw HTTPFailure(400, "items must be a nonempty array.") }
                var groups: [String: [Int]] = [:], assigned: Set<Int> = []
                for (target, values) in routes {
                    guard let indexes = values.array?.compactMap({ $0.number }), indexes.count == values.array?.count else {
                        throw HTTPFailure(500, "Invalid decision route indexes.")
                    }
                    for number in indexes {
                        let index = try jsonInteger(.number(number), name: "Decision route index", range: 0...(items.count - 1), status: 500)
                        guard assigned.insert(index).inserted else { throw HTTPFailure(500, "Invalid or repeated decision route index.") }
                        groups[target, default: []].append(index)
                    }
                }
                for index in items.indices where !assigned.contains(index) { groups[model, default: []].append(index) }
                var output = [Data?](repeating: nil, count: items.count), precisions: Set<String> = []
                for target in groups.keys.sorted() {
                    let indexes = groups[target]!
                    let subbody = try original.replacing(["items": RawJSON.array(indexes.map { items[$0].data })])
                    let reply = try await decisionCall(model: target, op: op, body: subbody, depth: depth + 1)
                    if reply.status != 200 { return reply }
                    if let bits = reply.bits { precisions.insert(bits) }
                    guard let results = try RawJSON(reply.body)["results"]?.elements(), results.count == indexes.count else {
                        throw HTTPFailure(500, "Decision batch returned the wrong item count.")
                    }
                    for (index, item) in zip(indexes, results) { output[index] = item.data }
                }
                guard output.allSatisfy({ $0 != nil }) else { throw HTTPFailure(500, "Decision batch lost an item.") }
                return DecisionReply(
                    status: 200, body: Data("{\"results\":".utf8) + RawJSON.array(output.compactMap { $0 }) + Data("}".utf8), bits: precisions.count == 1 ? precisions.first : nil)
            }
        }
        return DecisionReply(status: result.status, body: result.body, bits: nil)
    }
}
