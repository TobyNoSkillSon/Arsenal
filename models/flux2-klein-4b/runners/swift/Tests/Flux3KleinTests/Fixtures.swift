// SPDX-License-Identifier: MIT
// The tiny mflux capture (lab/flux3/scripts/capture_tiny.py, mflux 9669af51): a random-initialised checkpoint in the
// vendor's layout under snapshot/, mflux's outputs on fixed inputs in components.safetensors, and case.json.
import Foundation

enum TinyFixture {
    static var root: URL { Bundle.module.resourceURL!.appendingPathComponent("Fixtures/tiny") }
    static var snapshot: URL { root.appendingPathComponent("snapshot") }
    static var components: URL { root.appendingPathComponent("components.safetensors") }

    static func json(_ name: String) throws -> [String: Any] {
        try need(try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent(name))) as? [String: Any], "\(name) is not an object")
    }
    static var caseJSON: [String: Any] { get throws { try json("case.json") } }

    /// Host read of a float32 or int32 tensor from a safetensors file (no MLX, so default tests stay off the GPU).
    static func hostTensor(_ name: String, file: URL = components) throws -> (shape: [Int], values: [Float]) {
        let data = try Data(contentsOf: file)
        let n = data.prefix(8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
        let header = try need(try JSONSerialization.jsonObject(with: data.subdata(in: 8..<(8 + Int(n)))) as? [String: Any], "bad header")
        let entry = try need(header[name] as? [String: Any], "no \(name)")
        let shape = try need(entry["shape"] as? [Int], "no shape"), offsets = try need(entry["data_offsets"] as? [Int], "no offsets")
        let dtype = try need(entry["dtype"] as? String, "no dtype")
        let body = data.subdata(in: (8 + Int(n) + offsets[0])..<(8 + Int(n) + offsets[1]))
        let values: [Float] = body.withUnsafeBytes { raw in
            switch dtype {
            case "F32": return Array(raw.bindMemory(to: Float.self))
            case "I32": return raw.bindMemory(to: Int32.self).map(Float.init)
            default: return []
            }
        }
        guard !values.isEmpty else { throw FixtureError("\(name) is \(dtype); host reads take F32 or I32") }
        return (shape, values)
    }
}

func need<T>(_ value: T?, _ message: String) throws -> T {
    guard let value else { throw FixtureError(message) }
    return value
}
struct FixtureError: Error, CustomStringConvertible { let description: String; init(_ d: String) { description = d } }
