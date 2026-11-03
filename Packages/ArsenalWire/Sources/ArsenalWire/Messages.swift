// SPDX-License-Identifier: MIT
import Foundation
public struct WorkerHello: Codable, Sendable, Equatable {
    public var event = "hello"
    public var proto = 1
    public var worker: String
    public var runner: String
    public var pid: Int
    public var build: String
    public var kinds: [String]
    public var ops: [String]
    public init(worker: String, runner: String = "native", pid: Int = Int(ProcessInfo.processInfo.processIdentifier), build: String = "dev", kinds: [String], ops: [String]) {
        self.worker = worker; self.runner = runner; self.pid = pid; self.build = build; self.kinds = kinds; self.ops = ops
    }
    public func validate(kind: String, runner: String? = nil) throws {
        guard event == "hello", proto == 1, pid > 0, kinds.contains(kind), runner == nil || self.runner == runner else { throw WireError.invalidHandshake }
    }
    public var json: JSONObject { get throws { try WireJSON.decode(JSONEncoder().encode(self)) } }
}
public struct WorkerMemory: Codable, Sendable, Equatable {
    public var footprintMB: Double
    public var mlxActiveMB: Double
    public var mlxCacheMB: Double
    enum CodingKeys: String, CodingKey { case footprintMB = "footprint_mb", mlxActiveMB = "mlx_active_mb", mlxCacheMB = "mlx_cache_mb" }
    public init(footprintMB: Double = 0, mlxActiveMB: Double = 0, mlxCacheMB: Double = 0) {
        self.footprintMB = footprintMB; self.mlxActiveMB = mlxActiveMB; self.mlxCacheMB = mlxCacheMB
    }
}
public struct WorkerRequest: Codable, Sendable {
    public var id: Int; public var op: String; public var params: JSONObject
    public init(id: Int, op: String, params: JSONObject = [:]) { self.id = id; self.op = op; self.params = params }
    public var json: JSONObject { var v = params; v["id"] = .number(Double(id)); v["op"] = .string(op); return v }
}
public struct WorkerFailure: Error, Codable, Sendable, Equatable, LocalizedError {
    public var code: String; public var message: String; public var details: JSONObject; public var rawBody: String?; public var rawFrame: String?
    public init(code: String, message: String, details: JSONObject = [:], rawBody: String? = nil, rawFrame: String? = nil) {
        self.code = code; self.message = message; self.details = details; self.rawBody = rawBody; self.rawFrame = rawFrame
    }
    public var errorDescription: String? { message }
}
