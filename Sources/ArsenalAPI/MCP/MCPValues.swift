// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import MCP

func mcpValue<T: Encodable>(_ value: T) throws -> MCP.Value { try JSONDecoder().decode(MCP.Value.self, from: JSONEncoder().encode(value)) }
func wireObject(_ value: [String: MCP.Value]) throws -> JSONObject { try WireJSON.decode(JSONEncoder().encode(value)) }
func mcpWireValue(_ value: JSONValue) -> MCP.Value {
    switch value {
    case .null: return .null
    case .bool(let value): return .bool(value)
    case .string(let value): return .string(value)
    case .number(let value): return .double(value)
    case .array(let values): return .array(values.map(mcpWireValue))
    case .object(let values): return .object(values.mapValues(mcpWireValue))
    }
}
