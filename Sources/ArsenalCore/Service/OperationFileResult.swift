// SPDX-License-Identifier: MIT
import Foundation

/// JSON file responses preserve the runner payload and every per-file field. The service adds
/// url (job-scoped HTTP URL), mime, bytes and sha256; path remains the verified local path.
/// Callers should inspect `payload["files"]` rather than decoding away model-specific metadata.
public struct OperationFileResult: Codable, Sendable, Equatable {
    public var payload: JSONObject
    public init(payload: JSONObject) { self.payload = payload }
    public var files: [JSONObject] { payload["files"]?.array?.compactMap(\.object) ?? [] }
    public init(from decoder: any Decoder) throws { payload = try JSONObject(from: decoder) }
    public func encode(to encoder: any Encoder) throws { try payload.encode(to: encoder) }
}
