// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

struct HTTPFailure: Error, Sendable {
    let status: Int
    let message: String
    let code: String?
    let details: JSONObject
    init(_ status: Int, _ message: String, code: String? = nil, details: JSONObject = [:]) {
        self.status = status; self.message = message; self.code = code; self.details = details
    }
}

struct HTTPRequest: Sendable {
    let method: String
    let target: String
    let headers: [String: String]
    let body: Data
    var path: String { String(target.split(separator: "?", maxSplits: 1).first ?? "") }
    var query: [String: String] {
        let parts = URLComponents(string: "http://localhost" + target)?.queryItems ?? []
        return parts.reduce(into: [:]) { $0[$1.name] = $1.value ?? "" }
    }
}

func jsonData(_ object: Any) -> Data { (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .fragmentsAllowed])) ?? Data("null".utf8) }

func errorBody(_ failure: HTTPFailure, path: String, requestID: String) -> Data {
    guard failure.details.isEmpty else {
        let base = errorBody(HTTPFailure(failure.status, failure.message, code: failure.code), path: path, requestID: requestID)
        var object = (try? JSONSerialization.jsonObject(with: base) as? [String: Any]) ?? [:]
        object["details"] = plain(failure.details)
        object["status"] = failure.status
        if let code = failure.code { object["code"] = code }
        return jsonData(object)
    }
    let message = failure.message
    if path.contains("/anthropic/") {
        return jsonData(["type": "error", "error": ["type": failure.status >= 500 ? "api_error" : "invalid_request_error", "message": message], "request_id": requestID])
    }
    if path.contains("/elevenlabs/") { return jsonData(["detail": ["status": "invalid_request", "code": failure.code ?? "invalid_request", "message": message]]) }
    if path.contains("/replicate/") { return jsonData(["detail": message]) }
    if path.contains("systemone") || path.contains("/typesafe/") {
        if failure.status == 422 { return jsonData(["detail": [["loc": ["body"], "msg": message, "type": "value_error"]]]) }
        return jsonData(["error": ["type": failure.status == 409 ? "conflict_error" : "invalid_request_error", "message": message]])
    }
    if path.hasPrefix("/arsenal/") || path.hasSuffix("/judge") {
        var object = ["error": message]; if let code = failure.code { object["code"] = code }; return jsonData(object)
    }
    return jsonData([
        "error": ["message": message, "type": failure.status >= 500 ? "server_error" : "invalid_request_error", "param": NSNull(), "code": failure.code as Any? ?? NSNull()]
    ])
}
