// SPDX-License-Identifier: MIT
import Foundation

/// Stable serializer used by `pin`; unknown contributor fields survive without evaluating them.
public enum TOMLWriter {
    public static func encode(_ table: [String: TOMLValue]) -> String { emit(table, path: []) }
    private static func quote(_ string: String) -> String {
        guard let bytes = try? JSONEncoder().encode(string) else { preconditionFailure("Encoding a plain string cannot fail") }
        return String(data: bytes, encoding: .utf8)!.replacingOccurrences(of: "\\/", with: "/")
    }
    private static func key(_ key: String) -> String { key.range(of: "^[A-Za-z0-9_-]+$", options: .regularExpression) != nil ? key : quote(key) }
    private static func inline(_ value: TOMLValue) -> String {
        switch value {
        case .string(let s): return quote(s)
        case .integer(let n): return String(n)
        case .decimal(let n): return String(n)
        case .bool(let b): return b ? "true" : "false"
        case .array(let a): return "[" + a.map(inline).joined(separator: ", ") + "]"
        case .table(let t): return "{ " + t.keys.sorted().map { key($0) + " = " + inline(t[$0]!) }.joined(separator: ", ") + " }"
        }
    }
    private static func isNested(_ value: TOMLValue) -> Bool {
        if case .table = value { return true }
        if case .array(let a) = value, !a.isEmpty,
            a.allSatisfy({
                if case .table = $0 { return true }; return false
            })
        {
            return true
        }; return false
    }
    private static func emit(_ table: [String: TOMLValue], path: [String]) -> String {
        var s = table.keys.sorted().filter { !isNested(table[$0]!) }.map { key($0) + " = " + inline(table[$0]!) + "\n" }.joined()
        for name in table.keys.sorted() where isNested(table[name]!) {
            let nestedPath = path + [name]; let header = nestedPath.map(key).joined(separator: ".")
            switch table[name]! {
            case .table(let nested): s += "\n[\(header)]\n" + emit(nested, path: nestedPath)
            case .array(let a): for case .table(let nested) in a { s += "\n[[\(header)]]\n" + emit(nested, path: nestedPath) }
            default: break
            }
        }; return s
    }
}
