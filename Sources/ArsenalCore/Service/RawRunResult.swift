// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire

/// `body` is the exact worker body with transport id/ok/timing removed; nested labels/numbers are never decoded.
public struct RawRunResult: Sendable {
    public var rawFinalFrame: Data
    public var body: Data
    public var ok: Bool
    public var error: WorkerFailure?
    /// The worker's own `error.status`; without one, a runner's request-state refusals are the caller's conflict or a
    /// cancellation (`busy`, `not_loaded`: 409; `cancelled`: 400, as workerHTTPStatus maps it on every other route),
    /// and anything else a runner fault (500).
    public var status: Int {
        guard let value = error?.details["status"]?.number, let status = Int(exactly: value), (100...599).contains(status) else {
            return ok ? 200 : Self.fallbackStatuses[error?.code ?? ""] ?? 500
        }
        return status
    }
    static let fallbackStatuses = ["busy": 409, "not_loaded": 409, "already_loaded": 409, "cancelled": 400, "canceled": 400]
    public init(rawFinalFrame: Data, body: Data, ok: Bool, error: WorkerFailure? = nil) { self.rawFinalFrame = rawFinalFrame; self.body = body; self.ok = ok; self.error = error }
}
extension ModelsService {
    /// The entire final JSON frame, including failed replies, byte-for-byte as the worker wrote it.
    public func runRaw(id: String, op: String, params: JSONObject) async throws -> Data {
        do {
            var result = try await run(id: id, op: op, params: params, sink: nil)
            let raw = result.removeValue(forKey: "_raw_frame")?.string
            result["_raw_body"] = nil
            return try raw.map { Data($0.utf8) } ?? WireJSON.encode(result)
        } catch let failure as WorkerFailure {
            guard let raw = failure.rawFrame else { throw failure }
            return Data(raw.utf8)
        }
    }
    public func runRawResult(id: String, op: String, params: JSONObject) async throws -> RawRunResult {
        let frame = try await runRaw(id: id, op: op, params: params)
        let metadata = try WireJSON.decode(frame)
        let ok = metadata["ok"]?.bool != false
        if ok { return RawRunResult(rawFinalFrame: frame, body: try RawJSONObject.removing(["id", "ok", "timing"], from: frame), ok: true) }
        let details = metadata["error"]?.object ?? [:]
        let errorData = try RawJSONObject.member("error", from: frame) ?? Data("{}".utf8)
        let body = try RawJSONObject.member("body", from: errorData) ?? errorData
        return RawRunResult(
            rawFinalFrame: frame, body: body, ok: false,
            error: WorkerFailure(
                code: details["code"]?.string ?? "worker", message: details["message"]?.string ?? "Worker request failed.", details: details,
                rawBody: String(decoding: body, as: UTF8.self), rawFrame: String(decoding: frame, as: UTF8.self)))
    }
}
