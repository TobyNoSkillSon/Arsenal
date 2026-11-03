// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

// Check the JSON type and domain before any integer conversion. Int(exactly:) also
// protects ranges whose Double upper bound rounds beyond Int.max.
func jsonInteger(_ value: JSONValue?, name: String, range: ClosedRange<Int>, default fallback: Int? = nil, status: Int = 422) throws -> Int {
    if value == nil, let fallback { return fallback }
    guard let number = value?.number, number.isFinite, number.rounded() == number,
        number >= Double(range.lowerBound), number <= Double(range.upperBound),
        let integer = Int(exactly: number), range.contains(integer)
    else {
        throw HTTPFailure(status, "\(name) must be an integer from \(range.lowerBound) to \(range.upperBound).")
    }
    return integer
}

func tokenCount(_ value: JSONValue?) throws -> Int {
    try jsonInteger(value, name: "Worker token count", range: 0...1_000_000_000, default: 0, status: 500)
}

func chatUsage(_ result: JSONObject) throws -> [String: Int] {
    let prompt = try tokenCount(result["usage"]?["prompt_tokens"])
    let completion = try tokenCount(result["usage"]?["completion_tokens"])
    return ["prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion]
}

func workerHTTPStatus(_ error: WorkerFailure) -> Int {
    let mapped =
        [
            "invalid_output": 502, "revision_gate": 409, "revision": 409, "unknown_voice": 404, "not_found": 404, "not_installed": 404, "invalid_request": 422, "bad_request": 400,
            "unsupported": 400, "unsupported_op": 400, "unsupported_feature": 400, "unsupported_option": 400,
            "context_length_exceeded": 400, "cancelled": 400, "canceled": 400, "dtype_mismatch": 400,
            "unsupported_format": 422, "setting": 422, "settings": 422, "code_consent": 409, "local_folder": 409, "mode": 400, "not_loaded": 409, "already_loaded": 409,
            "conflict": 409, "busy": 409,
            "precision": 422, "insufficient_memory": 409, "memory": 507
        ][error.code] ?? 500
    return (try? jsonInteger(error.details["status"], name: "Worker HTTP status", range: 400...599)) ?? mapped
}
